"""
Video summarizer backend (no UI code).

Stages:
  1. Source      job_from_file() / download_url()
  2. Transcript  prepare()           audio extraction + Whisper word timestamps (cached)
  3. Summary     compute_summary()   BERT extractive summary -> timed segments + thumbnails
  4. Video       render()            cut segments, join, add subtitles, export text files

Every long step takes a `progress(fraction, description)` callback.
"""
from __future__ import annotations

import bisect
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import imageio_ffmpeg

WORKSPACE = Path(os.environ.get("VIDEOSUM_WORKSPACE", "workspace")).resolve()
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

WHISPER_MODELS = {
    # UI name: (MLX repo for Apple Silicon, Hugging Face repo for other machines)
    "small (fast)": ("mlx-community/whisper-small-mlx", "openai/whisper-small"),
    "large-v3-turbo (accurate)": ("mlx-community/whisper-large-v3-turbo", "openai/whisper-large-v3-turbo"),
}
EMBED_BATCH_SIZE = 32
RENDER_WORKERS = 3

ProgressFn = Callable[[float, str], None]


def _no_progress(fraction: float, desc: str = "") -> None:
    pass


# ----------------------------------------------------------------------------- FFmpeg helpers
def ffmpeg_run(args, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", *map(str, args)], capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-2000:])
    return r


def probe(path) -> dict:
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if not m:
        raise ValueError("This file could not be read as a video.")
    res = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    return {
        "duration": int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]),
        "has_audio": "Audio:" in err,
        "has_video": "Video:" in err,
        "resolution": f"{res[1]}x{res[2]}" if res else "unknown",
    }


def load_audio_16k(path) -> np.ndarray:
    r = ffmpeg_run(["-v", "error", "-i", path, "-map", "0:a:0", "-vn",
                    "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def extract_frame(video, t: float, out: Path, width: int = 320) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_run(["-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", video,
                    "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", out])
    return out


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


def quick_hash(path, block: int = 4 << 20) -> str:
    """Fast content fingerprint (size + first and last 4 MB), so the same video re-uploaded reuses its cache."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(block))
        if size > 2 * block:
            f.seek(-block, os.SEEK_END)
            h.update(f.read(block))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- Jobs
@dataclass
class Job:
    job_id: str
    video_path: Path
    title: str
    work_dir: Path
    duration: float
    has_audio: bool
    resolution: str
    size_mb: float
    full_text: str = ""
    chunks: list = field(default_factory=list)
    transcript_key: str = ""
    embed_cache: dict = field(default_factory=dict)   # (model, min_chars, max_chars) -> (sentences, features)
    lock: threading.Lock = field(default_factory=threading.Lock)


JOBS: dict[str, Job] = {}


def get_job(job_id: Optional[str]) -> Job:
    if not job_id or job_id not in JOBS:
        raise ValueError("Add a video first (upload a file or download a link).")
    return JOBS[job_id]


def job_from_file(src_path, title: Optional[str] = None) -> Job:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(str(src))
    job_id = quick_hash(src)
    if job_id in JOBS:
        return JOBS[job_id]
    work = WORKSPACE / "jobs" / job_id
    work.mkdir(parents=True, exist_ok=True)
    dest = work / f"source{src.suffix.lower() or '.mp4'}"
    if not dest.exists():
        try:
            os.link(src, dest)            # instant, no extra disk space
        except OSError:
            shutil.copy2(src, dest)
    info = probe(dest)
    if not info["has_video"]:
        raise ValueError("This file has no video stream.")
    meta_file = work / "meta.json"
    if title is None and meta_file.exists():
        title = json.loads(meta_file.read_text()).get("title")
    title = title or src.stem
    meta_file.write_text(json.dumps({"title": title, "source": str(src)}, indent=2))
    job = Job(job_id, dest, title, work, info["duration"], info["has_audio"], info["resolution"],
              os.path.getsize(dest) / 1e6)
    JOBS[job_id] = job
    return job


def set_title(job: Job, title: str) -> Job:
    """Rename a stored video. The name is kept in the job folder so it survives a restart."""
    title = (title or "").strip()
    if title:
        job.title = title
        (job.work_dir / "meta.json").write_text(json.dumps({"title": title, "source": str(job.video_path)}, indent=2))
    return job


def source_thumbnail(job: Job) -> Path:
    return extract_frame(job.video_path, job.duration * 0.1, job.work_dir / "thumbs" / "source.jpg", width=640)


# ----------------------------------------------------------------------------- Download from a link
def is_youtube(url: str) -> bool:
    return bool(re.match(r"^(https?://)?([\w-]+\.)?(youtube\.com|youtu\.be)/", url.strip(), re.I))


def youtube_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([\w-]{11})", url)
    return m.group(1) if m else None


def find_js_runtime() -> Optional[dict]:
    # Apps started from an IDE often miss Homebrew's PATH, so yt-dlp can't see Deno. Add the usual places.
    parts = os.environ.get("PATH", "").split(os.pathsep)
    extra = ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".deno" / "bin")]
    os.environ["PATH"] = os.pathsep.join(parts + [d for d in extra if os.path.isdir(d) and d not in parts])
    for name in ("deno", "node"):
        path = shutil.which(name)
        if path:
            return {name: {"path": path}}
    return None


def download_url(url: str, max_height: int = 720, progress: ProgressFn = _no_progress,
                 cookies_browser: Optional[str] = None) -> Job:
    import yt_dlp

    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    downloads = WORKSPACE / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    vid = youtube_video_id(url) if is_youtube(url) else None
    if vid:
        existing = [p for p in downloads.glob(f"{vid}.*") if p.suffix.lower() in VIDEO_EXTS]
        meta = downloads / f"{vid}.json"
        if existing:
            progress(1.0, "Already downloaded")
            title = json.loads(meta.read_text()).get("title") if meta.exists() else None
            return job_from_file(existing[0], title)

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            kind = "audio" if (d.get("info_dict") or {}).get("vcodec") == "none" else "video"
            speed = d.get("speed") or 0
            if total:
                progress(min(0.99, done / total),
                         f"Downloading {kind}: {done / 1e6:.1f} / {total / 1e6:.1f} MB ({speed / 1e6:.1f} MB/s)")
            else:
                progress(0.0, f"Downloading {kind}: {done / 1e6:.1f} MB")
        elif d["status"] == "finished":
            progress(0.99, "Download finished, preparing file")

    opts = {
        "format": (f"bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]/"
                   f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"),
        "merge_output_format": "mp4",
        "outtmpl": str(downloads / "%(id)s.%(ext)s"),
        "ffmpeg_location": FFMPEG,
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    runtime = find_js_runtime()
    if runtime:
        opts["js_runtimes"] = runtime
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)

    progress(0.0, "Reading link")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as ex:
        hint = "" if runtime or not is_youtube(url) else " For YouTube, install Deno (brew install deno) and try again."
        raise RuntimeError(f"Download failed: {ex}.{hint}") from ex
    path = Path(info["requested_downloads"][0]["filepath"])
    (downloads / f"{info['id']}.json").write_text(json.dumps(
        {k: info.get(k) for k in ("id", "title", "channel", "duration", "webpage_url")}, indent=2))
    progress(1.0, "Download complete")
    return job_from_file(path, info.get("title"))


# ----------------------------------------------------------------------------- Transcription
@contextlib.contextmanager
def _whisper_progress(progress: ProgressFn, base: float = 0.10, span: float = 0.88):
    """Report mlx-whisper's own progress bar as a single, clean progress value.

    mlx-whisper (and the model downloader) create several tqdm bars. Letting Gradio track tqdm draws
    them all in one place, on top of each other, so we read the transcription bar ourselves instead.
    """
    try:
        import tqdm as tqdm_module
    except Exception:
        yield
        return
    original = tqdm_module.tqdm

    class Reporting(original):
        def update(self, n=1):
            super().update(n)
            try:
                if self.total:
                    done = min(1.0, self.n / self.total)
                    progress(base + span * done, f"Transcribing {done:.0%}")
            except Exception:
                pass

    tqdm_module.tqdm = Reporting          # mlx-whisper calls tqdm.tqdm(...) at run time
    try:
        yield
    finally:
        tqdm_module.tqdm = original


def _transcribe_mlx(audio, repo, language, progress: ProgressFn = _no_progress):
    import mlx_whisper
    with _whisper_progress(progress):
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, word_timestamps=True, language=language,
                                     condition_on_previous_text=False, verbose=False)
    chunks = [{"text": w["word"], "timestamp": [float(w["start"]), float(w["end"])]}
              for seg in res["segments"] for w in seg.get("words", [])]
    return res["text"].strip(), chunks


def _transcribe_hf(audio, repo, language, progress: ProgressFn = _no_progress):
    import torch
    from transformers import pipeline
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    asr = pipeline("automatic-speech-recognition", model=repo, device=device,
                   dtype=torch.float16 if device != "cpu" else torch.float32)
    out = asr({"raw": audio, "sampling_rate": 16000}, return_timestamps="word", chunk_length_s=30, batch_size=8,
              generate_kwargs={"language": language} if language else {})
    return out["text"].strip(), [{"text": c["text"], "timestamp": list(c["timestamp"])} for c in out["chunks"]]


def prepare(job: Job, whisper_choice: str, language: Optional[str], progress: ProgressFn = _no_progress) -> dict:
    """Extract audio and transcribe with word timestamps. Cached per video + model + language."""
    if not job.has_audio:
        raise ValueError("This video has no audio track, so there is nothing to transcribe.")
    mlx_repo, hf_repo = WHISPER_MODELS[whisper_choice]
    repo = mlx_repo if IS_APPLE_SILICON else hf_repo
    key = hashlib.md5(f"{repo}|{language}".encode()).hexdigest()[:10]
    cache = job.work_dir / f"transcript_{key}.json"
    t0 = time.perf_counter()
    with job.lock:
        if cache.exists():
            data = json.loads(cache.read_text())
            cached = True
        else:
            progress(0.02, "Extracting audio")
            audio = load_audio_16k(job.video_path)
            progress(0.08, f"Transcribing {len(audio) / 16000 / 60:.1f} min of audio with {repo.split('/')[-1]}"
                           " (first run downloads the model)")
            text, chunks = (_transcribe_mlx if IS_APPLE_SILICON else _transcribe_hf)(
                audio, repo, language, progress)
            data = {"text": text, "chunks": chunks}
            cache.write_text(json.dumps(data))
            cached = False
        job.full_text, job.chunks = data["text"], data["chunks"]
        if job.transcript_key != key:
            job.embed_cache.clear()
        job.transcript_key = key
    progress(1.0, "Transcript ready")
    return {"words": len(job.chunks), "cached": cached, "seconds": time.perf_counter() - t0,
            "model": repo, "language": language, "cache_path": str(cache), "text": job.full_text}


# ----------------------------------------------------------------------------- Summarization
_SUMMARIZERS: dict = {}
_SUMMARIZER_LOCK = threading.Lock()


def _get_summarizer(model_name: str):
    """bert-extractive-summarizer with batched GPU embeddings (same embeddings as the library)."""
    with _SUMMARIZER_LOCK:
        if model_name in _SUMMARIZERS:
            return _SUMMARIZERS[model_name]
        import torch
        import transformers
        for name in ("TransfoXLModel", "TransfoXLTokenizer"):   # removed in transformers 5, still imported
            try:
                getattr(transformers, name)
            except (ImportError, AttributeError):
                setattr(transformers, name, None)
        from summarizer import Summarizer
        from summarizer.transformer_embeddings.bert_embedding import BertEmbedding

        if not getattr(BertEmbedding, "_fast_patched", False):
            original = BertEmbedding.create_matrix

            def fast_create_matrix(self, content, hidden=-2, reduce_option="mean", hidden_concat=False):
                if not isinstance(hidden, int) or reduce_option != "mean":
                    return original(self, content, hidden, reduce_option, hidden_concat)
                dev = next(self.model.parameters()).device
                pad_id = self.tokenizer.pad_token_id or 0
                ids_list = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t))[:512] for t in content]
                order = sorted(range(len(content)), key=lambda i: len(ids_list[i]))
                result = [None] * len(content)
                with torch.inference_mode():
                    for b in range(0, len(order), EMBED_BATCH_SIZE):
                        idx = order[b:b + EMBED_BATCH_SIZE]
                        width = max(1, max(len(ids_list[i]) for i in idx))
                        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
                        mask = torch.zeros((len(idx), width), dtype=torch.long)
                        for r, i in enumerate(idx):
                            n = len(ids_list[i])
                            if n:
                                ids[r, :n] = torch.tensor(ids_list[i], dtype=torch.long)
                                mask[r, :n] = 1
                        out = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
                        h = out.hidden_states[hidden]
                        m = mask.to(dev).unsqueeze(-1).to(h.dtype)
                        emb = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
                        for r, i in enumerate(idx):
                            result[i] = emb[r]
                return np.stack(result) if result else np.zeros((0, self.model.config.hidden_size), np.float32)

            BertEmbedding.create_matrix = fast_create_matrix
            BertEmbedding._fast_patched = True

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        summ = Summarizer(model=model_name)
        summ.model.func.model.to(device)
        summ.model.func.device = torch.device(device)
        _SUMMARIZERS[model_name] = summ
        return summ


@dataclass
class SummarySettings:
    ratio: float = 0.2                     # share of candidate sentences
    num_sentences: Optional[int] = None    # overrides ratio when set
    min_chars: int = 40
    max_chars: int = 600
    pad: float = 0.25
    merge_gap: float = 1.0
    use_first: bool = True
    model: str = "bert-large-uncased"


def _select_indices(features: np.ndarray, ratio: float, num_sentences: Optional[int], use_first: bool,
                    random_state: int = 12345) -> list[int]:
    """Same selection as bert-extractive-summarizer's cluster_runner, on cached embeddings."""
    from summarizer.cluster_features import ClusterFeatures
    n = len(features)
    if n == 0:
        return []
    if not use_first:
        return list(ClusterFeatures(features, "kmeans", random_state=random_state).cluster(ratio, num_sentences))
    if n <= 1:
        return [0]
    ns = num_sentences - 1 if num_sentences else num_sentences
    rest = ClusterFeatures(features[1:], "kmeans", random_state=random_state).cluster(ratio, ns)
    return [0] + [i + 1 for i in rest]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def map_sentences_to_segments(sentences, chunks, duration, pad, merge_gap):
    """Locate each summary sentence in the timed transcript; returns merged segments with their text."""
    full, spans = "", []
    for i, ch in enumerate(chunks):
        t = _normalize(ch["text"])
        if not t:
            continue
        if full:
            full += " "
        spans.append((len(full), len(full) + len(t), i))
        full += t
    span_starts = [s for s, _, _ in spans]
    padded_full = f" {full} "

    def find_range(target, cursor):
        for start_at in (cursor, 0):
            idx = padded_full.find(f" {target} ", start_at)
            if idx >= 0:
                return idx, idx + len(target)
        tw = target.split(" ")
        n = min(5, len(tw))
        head, tail = " ".join(tw[:n]), " ".join(tw[-n:])
        h = padded_full.find(f" {head} ", cursor)
        if h < 0:
            h = padded_full.find(f" {head} ")
        if h < 0:
            return None
        t = padded_full.find(f" {tail} ", h)
        if t < 0 or t - h > len(target) * 2:
            return None
        return h, t + len(tail)

    found, unmatched, cursor = [], [], 0
    for sentence in sentences:
        target = _normalize(sentence)
        if not target:
            continue
        r = find_range(target, cursor)
        if r is None or not spans:
            unmatched.append(sentence)
            continue
        cursor = r[1]
        k0 = max(0, bisect.bisect_right(span_starts, r[0]) - 1)
        k1 = max(k0, bisect.bisect_left(span_starts, r[1]) - 1)
        start = chunks[spans[k0][2]]["timestamp"][0]
        end = chunks[spans[k1][2]]["timestamp"][1]
        if start is None:
            unmatched.append(sentence)
            continue
        end = duration if end is None else end
        found.append([max(0.0, start - pad), min(duration, end + pad), [sentence.strip()]])

    merged = []
    for s, e, texts in sorted(found, key=lambda x: x[0]):
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].extend(texts)
        else:
            merged.append([s, e, list(texts)])
    segments = [{"start": s, "end": e, "text": " ".join(t)} for s, e, t in merged if e - s > 0.05]
    return segments, unmatched


def compute_summary(job: Job, s: SummarySettings, progress: ProgressFn = _no_progress) -> dict:
    if not job.chunks:
        raise ValueError("Transcribe the video first.")
    t0 = time.perf_counter()
    progress(0.05, f"Loading {s.model}")
    summ = _get_summarizer(s.model)

    key = (s.model, int(s.min_chars), int(s.max_chars))
    with job.lock:
        if key not in job.embed_cache:
            progress(0.2, "Splitting sentences and computing embeddings")
            sentences = summ.sentence_handler(job.full_text, int(s.min_chars), int(s.max_chars))
            features = summ.model(sentences) if sentences else np.zeros((0, 1), np.float32)
            job.embed_cache[key] = (sentences, features)
        sentences, features = job.embed_cache[key]

    progress(0.6, "Choosing sentences")
    num = int(s.num_sentences) if s.num_sentences else None
    indices = _select_indices(features, s.ratio, num, s.use_first)
    summary_sentences = [sentences[i] for i in indices]

    progress(0.75, "Mapping sentences to video times")
    segments, unmatched = map_sentences_to_segments(summary_sentences, job.chunks, job.duration, s.pad, s.merge_gap)

    progress(0.85, "Extracting preview frames")
    thumbs_dir = job.work_dir / "thumbs"

    def thumb(seg):
        mid = (seg["start"] + seg["end"]) / 2
        return extract_frame(job.video_path, mid, thumbs_dir / f"seg_{mid:.2f}.jpg")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for seg, path in zip(segments, pool.map(thumb, segments)):
            seg["thumb"] = str(path)

    all_sentences = summ.sentence_handler(job.full_text, 0, 10 ** 9)
    kept = sum(seg["end"] - seg["start"] for seg in segments)
    progress(1.0, "Summary ready")
    return {
        "sentences": summary_sentences,
        "segments": segments,
        "unmatched": unmatched,
        "stats": {
            "transcript_sentences": len(all_sentences),
            "candidates": len(sentences),
            "too_short": sum(1 for x in all_sentences if len(x) <= s.min_chars),
            "too_long": sum(1 for x in all_sentences if len(x) >= s.max_chars),
            "chosen": len(summary_sentences),
            "summary_words": sum(len(x.split()) for x in summary_sentences),
            "transcript_words": len(job.full_text.split()),
            "kept_seconds": kept,
            "video_seconds": job.duration,
            "seconds": time.perf_counter() - t0,
        },
    }


# ----------------------------------------------------------------------------- Rendering
_ENCODER = None


def _works(args) -> bool:
    return ffmpeg_run(args, check=False).returncode == 0


def video_encoder() -> list:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        if IS_APPLE_SILICON:
            test = ["-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=0.2"]
            for enc in (["-c:v", "h264_videotoolbox", "-q:v", "65"], ["-c:v", "h264_videotoolbox", "-b:v", "10M"]):
                if _works([*test, *enc, "-f", "null", "-"]):
                    _ENCODER = enc
                    break
    return _ENCODER


def hw_decode(job: Job) -> list:
    if IS_APPLE_SILICON and _works(["-v", "error", "-hwaccel", "videotoolbox", "-t", "1",
                                    "-i", job.video_path, "-f", "null", "-"]):
        return ["-hwaccel", "videotoolbox"]
    return []


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def build_captions(chunks, segments, clip_durations, max_words=8, max_seconds=3.5):
    captions, offset = [], 0.0
    for seg, clip_len in zip(segments, clip_durations):
        s0, s1 = seg["start"], seg["end"]
        line = []

        def flush():
            if line:
                captions.append((line[0][0], line[-1][1], " ".join(w for _, _, w in line)))
                line.clear()

        for ch in chunks:
            a, b = ch["timestamp"]
            if a is None or not ch["text"].strip():
                continue
            b = s1 if b is None else b
            if b <= s0 or a >= s1:
                continue
            a, b, w = max(a, s0) - s0 + offset, min(b, s1) - s0 + offset, ch["text"].strip()
            if line and (len(line) >= max_words or b - line[0][0] > max_seconds):
                flush()
            line.append((a, b, w))
            if re.search(r"[.!?]$", w):
                flush()
        flush()
        offset += clip_len
    return captions


def render(job: Job, segments: list, progress: ProgressFn = _no_progress) -> dict:
    if not segments:
        raise ValueError("No segments selected.")
    t0 = time.perf_counter()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = job.work_dir / "renders" / stamp
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    enc, hw = video_encoder(), hw_decode(job)

    def cut(i, seg):
        out = clips_dir / f"clip_{i:04d}.mp4"
        ffmpeg_run(["-v", "error", "-y", *hw, "-ss", f"{seg['start']:.3f}", "-i", job.video_path,
                    "-t", f"{seg['end'] - seg['start']:.3f}", "-map", "0:v:0", "-map", "0:a:0",
                    *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                    "-avoid_negative_ts", "make_zero", out])
        return i, out

    paths = [None] * len(segments)
    progress(0.0, f"Cutting {len(segments)} clips ({enc[1]})")
    with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        futures = [pool.submit(cut, i, seg) for i, seg in enumerate(segments)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, p = fut.result()
            paths[i] = p
            progress(0.9 * done / len(segments), f"Cut clip {done} of {len(segments)}")

    progress(0.92, "Joining clips and adding subtitles")
    durations = [probe(p)["duration"] for p in paths]
    captions = build_captions(job.chunks, segments, durations)
    base = re.sub(r"[^\w\-]+", "_", job.title)[:60].strip("_") or "video"
    video_out = out_dir / f"{base}_summary.mp4"
    srt_out = out_dir / f"{base}_summary.srt"
    srt_out.write_text("".join(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n\n"
                               for i, (a, b, t) in enumerate(captions, 1)), encoding="utf-8")
    list_file = out_dir / "concat.txt"
    list_file.write_text("".join("file '" + p.resolve().as_posix().replace("'", "'\\''") + "'\n" for p in paths))
    subs = (["-i", srt_out, "-map", "0:v", "-map", "0:a", "-map", "1:0", "-c:s", "mov_text",
             "-metadata:s:s:0", "language=eng"] if captions else ["-map", "0:v", "-map", "0:a"])
    ffmpeg_run(["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_file, *subs,
                "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", video_out])
    shutil.rmtree(clips_dir, ignore_errors=True)
    list_file.unlink(missing_ok=True)

    summary_txt = out_dir / f"{base}_summary.txt"
    summary_txt.write_text("\n\n".join(f"[{fmt_time(s['start'])} - {fmt_time(s['end'])}]\n{s['text']}"
                                       for s in segments) + "\n", encoding="utf-8")
    transcript_txt = out_dir / f"{base}_transcript.txt"
    transcript_txt.write_text(job.full_text + "\n", encoding="utf-8")
    (out_dir / "segments.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "thumb"} for s in segments], indent=2))

    out_info = probe(video_out)
    progress(1.0, "Video ready")
    return {
        "video": str(video_out), "srt": str(srt_out), "summary_txt": str(summary_txt),
        "transcript_txt": str(transcript_txt), "duration": out_info["duration"],
        "has_audio": out_info["has_audio"], "clips": len(segments), "captions": len(captions),
        "encoder": enc[1], "seconds": time.perf_counter() - t0,
    }