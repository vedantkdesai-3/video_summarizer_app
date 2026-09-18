"""
Video Summarizer - web app with accounts, projects and a stored library.

Three screens:
    Projects   search, open, create and delete your projects
    Videos     search the videos inside one project, add new ones, open one
    Summarise  transcribe, tune the summary, create the summary video

First time:   python db.py add-user <username>
Run:          python app.py            then open http://127.0.0.1:7860 and sign in
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import gradio as gr

import db
import pipeline as P

PRESETS = {
    #            ratio %, min chars, max chars, pad, merge gap
    "Short":    (10, 40, 600, 0.25, 1.0),
    "Balanced": (20, 40, 600, 0.25, 1.0),
    "Detailed": (35, 20, 1500, 0.4, 2.5),
    "Extended": (50, 10, 3000, 0.5, 5.0),
}
LANGUAGES = {"English": "en", "Auto-detect": None, "Hindi": "hi", "Kannada": "kn", "Tamil": "ta",
             "Telugu": "te", "Marathi": "mr", "Spanish": "es", "French": "fr", "German": "de"}

SEGMENT_HEADERS = ["Keep", "#", "Start", "End", "Seconds", "What is said"]
PROJECT_HEADERS = ["#", "Project", "Videos", "Summaries", "Created", "Description"]
VIDEO_HEADERS = ["#", "Title", "Length", "Source", "Summaries", "Added", "Stored at"]
RENDER_HEADERS = ["#", "Created", "Length", "Clips", "File"]
TRANSCRIPT_HEADERS = ["Speech model", "Language", "Words", "Stored at"]


# ----------------------------------------------------------------------------- helpers
def _user(request: gr.Request):
    username = getattr(request, "username", None)
    if not username:
        raise gr.Error("Session expired. Reload the page and sign in again.")
    return db.user_id_for(username), username


def _short_date(iso: str) -> str:
    return (iso or "").replace("T", " ")[:16]


def _row_index(event: gr.SelectData):
    return event.index[0] if isinstance(event.index, (list, tuple)) else event.index


def _projects_frame(user_id: int, query: str = ""):
    rows = db.list_projects(user_id, query)
    frame = pd.DataFrame([[i + 1, r["name"], r["video_count"], r["render_count"],
                           _short_date(r["created_at"]), r["description"] or ""]
                          for i, r in enumerate(rows)], columns=PROJECT_HEADERS)
    return frame, [r["id"] for r in rows]


def _videos_frame(project_id, query: str = ""):
    rows = db.list_videos(project_id, query) if project_id else []
    frame = pd.DataFrame([[i + 1, r["title"], P.fmt_time(r["duration"] or 0),
                           "link" if r["source_type"] == "link" else "upload",
                           r["render_count"], _short_date(r["created_at"]), r["file_path"]]
                          for i, r in enumerate(rows)], columns=VIDEO_HEADERS)
    return frame, [r["id"] for r in rows]


def _video_preview_path(video_ids, row_index, user_id: int):
    """Return the stored source-video path for the selected row."""
    if not video_ids or row_index is None:
        return None
    index = int(row_index)
    if index < 0 or index >= len(video_ids):
        return None
    row = db.get_video(video_ids[index], user_id)
    if not row:
        return None
    path = Path(row["file_path"])
    return str(path) if path.exists() else None


def _renders_frame(video_id):
    rows = db.list_renders(video_id) if video_id else []
    frame = pd.DataFrame([[i + 1, _short_date(r["created_at"]), P.fmt_time(r["duration"] or 0),
                           r["clips"], Path(r["video_path"]).name]
                          for i, r in enumerate(rows)], columns=RENDER_HEADERS)
    return frame, [r["id"] for r in rows]


def _transcripts_frame(video_id):
    rows = db.list_transcripts(video_id) if video_id else []
    return pd.DataFrame([[r["model"], r["language"] or "auto", r["words"], _short_date(r["created_at"])]
                         for r in rows], columns=TRANSCRIPT_HEADERS)


def _selected_id(ids, row_index, what: str) -> int:
    if not ids:
        raise gr.Error(f"There are no {what} in this list yet.")
    if row_index is None:
        if len(ids) == 1:
            row_index = 0          # only one row: no need to make the user click it
        else:
            raise gr.Error(f"Click a row in the table to select a {what[:-1]} first.")
    index = int(row_index)
    if index >= len(ids):
        raise gr.Error("That row is no longer in the list. Refresh and try again.")
    return ids[index]


def _project_header(project) -> str:
    if not project:
        return ""
    return (f"### {project['name']}\n{project['description'] or ''}  \n"
            f"{project['video_count']} video(s) · {project['render_count']} summary video(s) · "
            f"created {_short_date(project['created_at'])}")


def _video_header(job: P.Job, project_name: str) -> str:
    return (f"### {job.title}\n"
            f"{P.fmt_time(job.duration)} · {job.resolution} · {job.size_mb:.1f} MB · "
            f"audio: {'yes' if job.has_audio else '**no**'} · project **{project_name}**")


def _session_line(user_id: int, username: str) -> str:
    stats = db.project_stats(user_id)
    return (f"Signed in as **{username}** · {stats['projects']} project(s) · "
            f"{stats['videos']} video(s) · {stats.get('transcripts', 0)} transcript(s) · "
            f"{stats['renders']} summary video(s)")


def _kept_markdown(segments, keep_flags, video_seconds) -> str:
    kept = [s for s, k in zip(segments, keep_flags) if k]
    secs = sum(s["end"] - s["start"] for s in kept)
    return (f"**Selected:** {len(kept)} of {len(segments)} segments · "
            f"**{P.fmt_time(secs)}** of {P.fmt_time(video_seconds)} "
            f"({secs / video_seconds:.0%} of the video)")


def _keep_flags(table, n):
    if table is None or len(table) == 0:
        return [True] * n
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    flags = [bool(v) if not isinstance(v, str) else v.lower() == "true" for v in frame.iloc[:, 0].tolist()]
    return (flags + [True] * n)[:n]


def _restore_cached_transcript(job: P.Job, transcript_row) -> None:
    """Restore word-level transcript data from the DB-backed cache after an app restart."""
    if not transcript_row:
        return
    cache_path = transcript_row["chunks_path"] or ""
    if not cache_path or not Path(cache_path).exists():
        return
    try:
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        job.full_text = data.get("text", transcript_row["text"] or "")
        job.chunks = data.get("chunks", [])
        # Make compute_summary usable after restarting the application.
        job.transcript_key = "restored-from-db-cache"
        job.embed_cache.clear()
    except Exception:
        # The DB transcript text remains usable even if its optional timestamp cache is missing.
        job.full_text = transcript_row["text"] or ""


def _open_summarise(job: P.Job, video_id: int, project_name: str, project_id: int, message: str = "", user_id: int | None = None):
    """Everything the Summarise screen needs when a video is opened, restored from SQLite."""
    videos, video_ids = _videos_frame(project_id)
    renders, render_ids = _renders_frame(video_id)
    transcripts = _transcripts_frame(video_id)

    # Restore the latest transcript into the in-memory Job so Preview works after a restart.
    stored = db.list_transcripts(video_id) if video_id else []
    latest_transcript = stored[0] if stored else None
    _restore_cached_transcript(job, latest_transcript)

    if latest_transcript:
        transcript_note = (f"Stored transcript: **{latest_transcript['words']:,} words** "
                           f"({latest_transcript['model']} / {latest_transcript['language'] or 'auto-detect'}, "
                           f"{_short_date(latest_transcript['created_at'])}). "
                           f"Loaded from the database/cache; you can preview the summary immediately.")
        transcript_text = latest_transcript["text"]
    else:
        transcript_note, transcript_text = "", ""

    # Restore the most recent generated summary preview, including settings, segments and selections.
    summary_row = db.latest_summary(video_id, user_id) if video_id and user_id else None
    stats_md = ""
    summary_md = ""
    gallery = None
    segments_table = None
    restored_segments = None
    restored_settings = None
    kept_md = ""
    render_interactive = False

    if summary_row:
        try:
            stats = json.loads(summary_row["stats_json"] or "{}")
            restored_settings = json.loads(summary_row["settings_json"] or "{}")
            restored_segments = json.loads(summary_row["segments_json"] or "[]")
            keep_flags = json.loads(summary_row["keep_json"] or "[]")
            if restored_segments:
                stats_md = (f"**{stats.get('chosen', len(restored_segments))} sentences** chosen from "
                            f"{stats.get('candidates', len(restored_segments))} candidates "
                            f"({stats.get('transcript_sentences', 0)} in transcript; "
                            f"{stats.get('too_short', 0)} too short, {stats.get('too_long', 0)} too long) · "
                            f"{stats.get('summary_words', 0):,} of {stats.get('transcript_words', 0):,} words")
                summary_md = summary_row["summary_text"] or ""
                gallery = [
                    (s["thumb"], f"#{i + 1}  {P.fmt_time(s['start'])}-{P.fmt_time(s['end'])}")
                    for i, s in enumerate(restored_segments) if s.get("thumb") and Path(s["thumb"]).exists()
                ]
                segments_table = pd.DataFrame(
                    [[bool(keep_flags[i]) if i < len(keep_flags) else True, i + 1,
                      P.fmt_time(s["start"]), P.fmt_time(s["end"]),
                      round(s["end"] - s["start"], 1), s["text"]]
                     for i, s in enumerate(restored_segments)], columns=SEGMENT_HEADERS)
                kept = keep_flags if keep_flags else [True] * len(restored_segments)
                kept_md = _kept_markdown(restored_segments, kept, job.duration)
                render_interactive = any(kept)
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            pass

    # Restore the latest generated render after an application restart.
    # The previous version restored only the video player, leaving the
    # "Output files" component empty even though the files were persisted.
    latest_render_video = None
    latest_render_files = None
    if video_id:
        render_rows = db.list_renders(video_id)
        if render_rows:
            latest = render_rows[0]
            video_path = latest["video_path"] or ""
            if video_path and Path(video_path).exists():
                latest_render_video = video_path

            persisted_files = [
                latest["video_path"],
                latest["srt_path"],
                latest["summary_path"],
                latest["transcript_path"],
            ]
            existing_files = [
                p for p in persisted_files
                if p and Path(p).exists()
            ]
            if existing_files:
                latest_render_files = existing_files

    return (gr.Tabs(selected="summarise"), job.job_id, video_id,
            _video_header(job, project_name), str(P.source_thumbnail(job)),
            gr.update(interactive=job.has_audio),
            transcript_note, transcript_text,
            gr.update(interactive=bool(job.chunks)), gr.update(interactive=render_interactive),
            restored_segments, restored_settings, latest_render_video, latest_render_files,
            stats_md, summary_md, gallery, segments_table, kept_md, "",
            renders, render_ids, videos, video_ids, transcripts, message)


# ----------------------------------------------------------------------------- screen 1: projects
def on_start(request: gr.Request):
    user_id, username = _user(request)
    db.default_project(user_id)      # every account starts with one project
    frame, ids = _projects_frame(user_id)
    return user_id, _session_line(user_id, username), frame, ids


def on_select_project(event: gr.SelectData):
    """Remember which row was clicked. The gr.SelectData annotation is what makes Gradio pass the click."""
    return _row_index(event)


def on_search_projects(query, request: gr.Request):
    user_id, _ = _user(request)
    frame, ids = _projects_frame(user_id, query)
    return frame, ids, None


def on_create_project(name, description, request: gr.Request):
    user_id, username = _user(request)
    try:
        project_id = db.create_project(user_id, name, description)
    except ValueError as ex:
        raise gr.Error(str(ex))
    project = db.get_project(project_id, user_id)
    projects, project_ids = _projects_frame(user_id)
    videos, video_ids = _videos_frame(project_id)
    return (gr.Tabs(selected="videos"), project_id, _project_header(project),
            videos, video_ids, "", projects, project_ids, "", "",
            _session_line(user_id, username), f"Project **{name.strip()}** created. Add your first video below.")


def on_open_project(project_ids, row_index, request: gr.Request):
    user_id, _ = _user(request)
    project_id = _selected_id(project_ids, row_index, "projects")
    rows = [r for r in db.list_projects(user_id) if r["id"] == project_id]
    if not rows:
        raise gr.Error("That project does not belong to this account.")
    videos, video_ids = _videos_frame(project_id)

    # When reopening a project, immediately expose a playable source video
    # when at least one stored video exists. The user can still select any
    # other row in the table to change the player.
    first_row = 0 if video_ids else None
    first_path = _video_preview_path(video_ids, first_row, user_id)
    first_detail = ""
    if video_ids:
        first_video = db.get_video(video_ids[0], user_id)
        if first_video:
            first_detail = f"**{first_video['title']}** · {first_video['file_path']}"

    return (gr.Tabs(selected="videos"), project_id, _project_header(rows[0]),
            videos, video_ids, first_row, first_detail, first_path)


def on_delete_project(project_ids, row_index, confirm, query, request: gr.Request):
    user_id, username = _user(request)
    project_id = _selected_id(project_ids, row_index, "projects")
    if not confirm:
        raise gr.Error("Tick 'Confirm delete' first. Deleting a project removes its video and summary records.")
    project = db.get_project(project_id, user_id)
    try:
        db.delete_project(project_id, user_id)
    except ValueError as ex:
        raise gr.Error(str(ex))
    frame, ids = _projects_frame(user_id, query)
    return (frame, ids, None, False, _session_line(user_id, username),
            f"Deleted project **{project['name']}**. The video files themselves are still on disk.")


# ----------------------------------------------------------------------------- screen 2: videos
def on_search_videos(project_id, query):
    frame, ids = _videos_frame(project_id, query)
    return frame, ids, None, None


def on_back_to_projects(query, request: gr.Request):
    user_id, username = _user(request)
    frame, ids = _projects_frame(user_id, query)
    return gr.Tabs(selected="projects"), frame, ids, _session_line(user_id, username)


def _project_name(project_id, user_id) -> str:
    project = db.get_project(project_id, user_id) if project_id else None
    if not project:
        raise gr.Error("Open a project first, on the Projects screen.")
    return project["name"]


def on_upload(file_path, project_id, custom_name, request: gr.Request, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("No file received.")
    user_id, _ = _user(request)
    name = _project_name(project_id, user_id)
    progress(0.3, desc="Reading video")
    try:
        job = P.job_from_file(file_path, (custom_name or "").strip() or None)
        P.set_title(job, custom_name)          # optional: blank keeps the automatic name
    except Exception as ex:
        raise gr.Error(str(ex))
    video_id = db.add_video(project_id, user_id, job, "upload", Path(file_path).name)
    if (custom_name or "").strip():
        db.rename_video(video_id, user_id, custom_name)
    return _open_summarise(job, video_id, name, project_id,
                           f"Added **{job.title}** to project **{name}**.", user_id)


def on_download(url, max_height, project_id, custom_name, request: gr.Request, progress=gr.Progress()):
    if not url or not url.strip():
        raise gr.Error("Paste a YouTube or video link first.")
    user_id, _ = _user(request)
    name = _project_name(project_id, user_id)
    try:
        job = P.download_url(url, int(max_height), progress=lambda f, d="": progress(f, desc=d))
        P.set_title(job, custom_name)          # optional: blank keeps the title from the link
    except Exception as ex:
        raise gr.Error(str(ex))
    video_id = db.add_video(project_id, user_id, job, "link", url.strip())
    if (custom_name or "").strip():
        db.rename_video(video_id, user_id, custom_name)
    return _open_summarise(job, video_id, name, project_id,
                           f"Added **{job.title}** to project **{name}**.", user_id)


def on_open_video(video_ids, row_index, project_id, request: gr.Request):
    """Re-open a stored video without uploading it again."""
    user_id, _ = _user(request)
    video_id = _selected_id(video_ids, row_index, "videos")
    row = db.get_video(video_id, user_id)
    if not row:
        raise gr.Error("That video does not belong to this account.")
    if not Path(row["file_path"]).exists():
        raise gr.Error(f"The stored file is missing: {row['file_path']}")
    job = P.job_from_file(row["file_path"], row["title"])
    return _open_summarise(job, row["id"], _project_name(project_id, user_id), project_id, "", user_id)


def on_delete_video(video_ids, row_index, confirm, project_id, query, request: gr.Request):
    user_id, username = _user(request)
    video_id = _selected_id(video_ids, row_index, "videos")
    if not confirm:
        raise gr.Error("Tick 'Confirm delete' first.")
    row = db.get_video(video_id, user_id)
    try:
        db.delete_video(video_id, user_id)
    except ValueError as ex:
        raise gr.Error(str(ex))
    frame, ids = _videos_frame(project_id, query)
    return (frame, ids, None, False, _session_line(user_id, username),
            f"Removed **{row['title']}** from this project. The file is still on disk.", None)


def on_rename_video(video_ids, row_index, new_name, project_id, query, request: gr.Request):
    user_id, _ = _user(request)
    video_id = _selected_id(video_ids, row_index, "videos")
    try:
        title = db.rename_video(video_id, user_id, new_name)
    except ValueError as ex:
        raise gr.Error(str(ex))
    row = db.get_video(video_id, user_id)
    if row and Path(row["file_path"]).exists():
        P.set_title(P.job_from_file(row["file_path"], title), title)
    frame, ids = _videos_frame(project_id, query)
    return frame, ids, "", f"Renamed to **{title}**."


def on_select_video(video_ids, event: gr.SelectData, request: gr.Request):
    user_id, _ = _user(request)
    index = _row_index(event)
    if not video_ids or index >= len(video_ids):
        return None, "", None
    video = db.get_video(video_ids[index], user_id)
    if not video:
        return index, "", None
    path = Path(video["file_path"])
    playable = str(path) if path.exists() else None
    detail = (f"**{video['title']}** · {video['file_path']}"
              if playable else
              f"**{video['title']}** · file not found: `{video['file_path']}`")
    return index, detail, playable


# ----------------------------------------------------------------------------- screen 3: summarise
def on_back_to_videos(project_id, query, request: gr.Request):
    user_id, _ = _user(request)
    rows = [r for r in db.list_projects(user_id) if r["id"] == project_id] if project_id else []
    videos, ids = _videos_frame(project_id, query)
    return gr.Tabs(selected="videos"), videos, ids, _project_header(rows[0] if rows else None)


def on_process(job_id, video_id, whisper_choice, language_name, request: gr.Request,
               progress=gr.Progress(track_tqdm=True)):
    user_id, username = _user(request)
    language = LANGUAGES[language_name]
    try:
        job = P.get_job(job_id)
        info = P.prepare(job, whisper_choice, language, progress=lambda f, d="": progress(f, desc=d))
    except Exception as ex:
        raise gr.Error(str(ex))

    # The full transcript is stored in the database against this video, speech model and language.
    if video_id:
        db.save_transcript(video_id, user_id, whisper_choice, language,
                           job.full_text, len(job.full_text.split()), info.get("cache_path", ""))
    note = "loaded from cache" if info["cached"] else f"transcribed in {info['seconds']:.0f} s"
    return (f"Transcript ready: **{len(job.full_text.split()):,} words** "
            f"({info['words']:,} timed words), {note}. Stored for **{whisper_choice}** / "
            f"**{language or 'auto-detect'}**.",
            job.full_text, gr.update(interactive=True),
            _transcripts_frame(video_id), _session_line(user_id, username))


def apply_preset(name):
    return PRESETS[name]


def toggle_size_mode(mode):
    by_count = mode == "Exact number of sentences"
    return gr.update(visible=not by_count), gr.update(visible=by_count)


def mark_stale():
    return gr.update(interactive=False), "Settings changed. Click **Preview summary** to update."


def on_preview(job_id, video_id, mode, ratio, num_sentences, min_chars, max_chars, pad, merge_gap, use_first, model,
               request: gr.Request, progress=gr.Progress()):
    if not job_id:
        raise gr.Error("Open a video first, on the Videos screen.")
    if not P.JOBS.get(job_id, None) or not P.JOBS[job_id].chunks:
        raise gr.Error("Press 'Extract audio & transcribe' first. If this video already has a stored "
                       "transcript, that button loads it from cache in a couple of seconds.")
    try:
        job = P.get_job(job_id)
        settings = P.SummarySettings(
            ratio=ratio / 100.0,
            num_sentences=int(num_sentences) if mode == "Exact number of sentences" else None,
            min_chars=int(min_chars), max_chars=int(max_chars), pad=float(pad), merge_gap=float(merge_gap),
            use_first=bool(use_first), model=model)
        result = P.compute_summary(job, settings, progress=lambda f, d="": progress(f, desc=d))
    except Exception as ex:
        raise gr.Error(str(ex))

    st, segs = result["stats"], result["segments"]
    if not segs:
        raise gr.Error("No sentences were selected. Lower 'Ignore sentences shorter than' or increase the summary size.")
    stats = (
        f"**{st['chosen']} sentences** chosen from {st['candidates']} candidates "
        f"({st['transcript_sentences']} in transcript; {st['too_short']} too short, {st['too_long']} too long) · "
        f"{st['summary_words']:,} of {st['transcript_words']:,} words · computed in {st['seconds']:.1f} s"
    )
    if result["unmatched"]:
        stats += f"  \n{len(result['unmatched'])} chosen sentence(s) could not be located in the video and were skipped."
    gallery = [(s["thumb"], f"#{i + 1}  {P.fmt_time(s['start'])}-{P.fmt_time(s['end'])}") for i, s in enumerate(segs)]
    table = pd.DataFrame(
        [[True, i + 1, P.fmt_time(s["start"]), P.fmt_time(s["end"]), round(s["end"] - s["start"], 1), s["text"]]
         for i, s in enumerate(segs)], columns=SEGMENT_HEADERS)
    summary_text = "\n\n".join(result["sentences"])
    keep_flags = [True] * len(segs)
    if video_id:
        user_id, _ = _user(request)
        db.save_summary(video_id, user_id, summary_text, st, vars(settings), segs, keep_flags)
    return (stats, summary_text, gallery, table, segs, vars(settings),
            _kept_markdown(segs, keep_flags, job.duration), gr.update(interactive=True))


def on_table_change(table, segments, job_id, request: gr.Request):
    if not segments or job_id not in P.JOBS:
        return ""
    keep_flags = _keep_flags(table, len(segments))
    job = P.JOBS[job_id]
    # Persist the user's segment selections as soon as they change.
    try:
        user_id, _ = _user(request)
        video = db.video_for_job(user_id, job_id)
        if video:
            latest = db.latest_summary(video["id"], user_id)
            if latest:
                db.save_summary(video["id"], user_id, latest["summary_text"],
                                json.loads(latest["stats_json"] or "{}"),
                                json.loads(latest["settings_json"] or "{}"), segments, keep_flags)
    except Exception:
        pass
    return _kept_markdown(segments, keep_flags, job.duration)


def on_render(job_id, video_id, segments, settings, table, project_id, request: gr.Request,
              progress=gr.Progress()):
    user_id, username = _user(request)
    try:
        job = P.get_job(job_id)
        if not segments:
            raise ValueError("Preview a summary first.")
        chosen = [s for s, k in zip(segments, _keep_flags(table, len(segments))) if k]
        if not chosen:
            raise ValueError("All segments are unticked. Tick at least one segment to keep.")
        out = P.render(job, chosen, progress=lambda f, d="": progress(f, desc=d))
    except Exception as ex:
        raise gr.Error(str(ex))

    if video_id:
        db.add_render(video_id, user_id, out, settings or {}, chosen)
    info = (f"**Done in {out['seconds']:.0f} s.** Summary video: **{P.fmt_time(out['duration'])}** "
            f"from {P.fmt_time(job.duration)} ({out['duration'] / job.duration:.0%}) · {out['clips']} clips · "
            f"{out['captions']} subtitle lines · encoder {out['encoder']}  \n"
            f"Saved to this video's history below.")
    files = [out["video"], out["srt"], out["summary_txt"], out["transcript_txt"]]
    renders, render_ids = _renders_frame(video_id)
    videos, video_ids = _videos_frame(project_id)
    return (info, gr.Video(value=out["video"], subtitles=out["srt"]), files,
            renders, render_ids, videos, video_ids, _session_line(user_id, username))


def on_select_render(render_ids, event: gr.SelectData, request: gr.Request):
    user_id, _ = _user(request)
    index = _row_index(event)
    if not render_ids or index >= len(render_ids):
        return None, None
    render = db.get_render(render_ids[index], user_id)
    if not render:
        return None, None
    files = [p for p in (render["video_path"], render["srt_path"], render["summary_path"],
                         render["transcript_path"]) if p and Path(p).exists()]
    return (render["video_path"] if Path(render["video_path"]).exists() else None), files


# -----------------------------------------------------------------------------
# layout
# The UI below intentionally keeps the existing component IDs and event wiring.
# Only presentation/layout is changed; database and pipeline logic are untouched.
CSS = """
/* =========================================================
   Video Summarizer — clean, aligned UI
   Presentation only: application logic/event wiring unchanged.
   ========================================================= */

:root {
    --vs-bg: #f5f7fb;
    --vs-card: #ffffff;
    --vs-border: #e2e6ef;
    --vs-text: #182230;
    --vs-muted: #667085;
    --vs-primary: #5b5bd6;
    --vs-primary-dark: #4545b5;
    --vs-danger: #c0392b;
    --vs-radius: 14px;
    --vs-shadow: 0 4px 18px rgba(16, 24, 40, 0.06);
}

/* ---------- Global sizing/alignment ---------- */

html, body {
    background: var(--vs-bg) !important;
}

body {
    margin: 0 !important;
}

.gradio-container {
    max-width: 1440px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 24px 30px 40px !important;
    box-sizing: border-box !important;
    color: var(--vs-text);
}

/* Prevent long text/widgets from forcing a row wider than the page. */
.gradio-row,
.gradio-column,
.gradio-column > *,
.gradio-row > * {
    min-width: 0 !important;
    box-sizing: border-box !important;
}

/* Keep columns aligned from their top edge. */
.gradio-row {
    align-items: stretch !important;
}

.gradio-column {
    align-items: stretch !important;
}

/* ---------- Header ---------- */

.vs-topbar {
    width: 100% !important;
    min-height: 82px !important;
    margin: 0 0 18px !important;
    padding: 18px 22px !important;
    background: var(--vs-card) !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--vs-shadow) !important;
    align-items: center !important;
}

.vs-brand {
    justify-content: center !important;
}

.vs-brand h1 {
    margin: 0 !important;
    font-size: 29px !important;
    line-height: 1.15 !important;
    letter-spacing: -0.6px !important;
}

.vs-brand p {
    margin: 6px 0 0 !important;
    color: var(--vs-muted) !important;
    font-size: 14px !important;
    line-height: 1.45 !important;
}

.vs-session {
    width: 100% !important;
    margin: 0 !important;
    padding: 10px 13px !important;
    background: #f8f9fc !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: 10px !important;
    color: var(--vs-muted) !important;
    font-size: 13px !important;
    line-height: 1.45 !important;
    text-align: right !important;
    box-sizing: border-box !important;
}

/* ---------- Main navigation ---------- */

.vs-tabs {
    width: 100% !important;
}

.vs-tabs > .tab-nav {
    display: flex !important;
    width: 100% !important;
    gap: 4px !important;
    margin: 0 0 18px !important;
    padding: 5px !important;
    background: var(--vs-card) !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04) !important;
    box-sizing: border-box !important;
}

.vs-tabs > .tab-nav button {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 10px 16px !important;
    border: 0 !important;
    border-radius: 8px !important;
    font-weight: 650 !important;
    color: var(--vs-muted) !important;
    white-space: nowrap !important;
}

.vs-tabs > .tab-nav button.selected {
    background: #ececff !important;
    color: var(--vs-primary-dark) !important;
}

.vs-tabs > .tabitem {
    width: 100% !important;
    border: 0 !important;
}

/* ---------- Reusable cards ---------- */

.vs-card,
.vs-step,
.vs-preview,
.vs-output {
    width: 100% !important;
    box-sizing: border-box !important;
    background: var(--vs-card) !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: var(--vs-radius) !important;
    box-shadow: var(--vs-shadow) !important;
}

.vs-card {
    margin: 0 0 16px !important;
    padding: 18px !important;
}

.vs-step {
    margin: 0 0 16px !important;
    padding: 17px !important;
}

.vs-preview {
    margin: 0 !important;
    padding: 17px !important;
    height: auto !important;
    align-self: stretch !important;
}

.vs-output {
    margin: 0 0 16px !important;
    padding: 17px !important;
}

.vs-soft-card {
    background: #fafaff !important;
    border: 1px solid #e8e9f4 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

.vs-section-title {
    width: 100% !important;
    margin: 2px 0 4px !important;
}

.vs-section-title h2,
.vs-section-title h3 {
    margin: 0 !important;
    line-height: 1.25 !important;
    letter-spacing: -0.25px !important;
}

.vs-note {
    margin: 0 0 12px !important;
    color: var(--vs-muted) !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}

/* ---------- Project/video headers ---------- */

.vs-project-header {
    width: 100% !important;
    margin: 0 0 16px !important;
    padding: 17px 20px !important;
    background: var(--vs-card) !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--vs-shadow) !important;
    box-sizing: border-box !important;
}

.vs-project-header h3 {
    margin: 0 0 5px !important;
    font-size: 21px !important;
    line-height: 1.25 !important;
}

.vs-project-header p {
    margin: 0 !important;
    color: var(--vs-muted) !important;
}

/* ---------- Summary hero ---------- */

.vs-hero {
    width: 100% !important;
    margin: 0 0 18px !important;
    padding: 20px 22px !important;
    background: linear-gradient(135deg, #5757cf 0%, #7474e8 100%) !important;
    border: 0 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(91, 91, 214, 0.18) !important;
    box-sizing: border-box !important;
    align-items: center !important;
}

.vs-hero h2,
.vs-hero h3,
.vs-hero p {
    color: white !important;
}

.vs-hero h2 {
    margin: 0 0 5px !important;
    line-height: 1.25 !important;
}

.vs-hero p {
    margin: 0 !important;
    opacity: 0.9 !important;
}

.vs-hero .image-container {
    width: 100% !important;
    min-height: 150px !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ---------- Step headings ---------- */

.vs-step-head {
    width: 100% !important;
    margin: 0 0 14px !important;
    padding: 10px 13px !important;
    background: #f7f7ff !important;
    border: 1px solid #e7e7f4 !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
}

.vs-step-head h3 {
    margin: 0 !important;
    font-size: 16px !important;
    line-height: 1.35 !important;
}

/* ---------- Inputs/buttons ---------- */

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .gr-input,
.gradio-container .gr-text-input {
    box-sizing: border-box !important;
    border-radius: 9px !important;
}

.gradio-container label {
    font-weight: 600 !important;
    color: #344054 !important;
}

.gradio-container button {
    min-height: 42px !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    box-sizing: border-box !important;
}

.gradio-container button.primary {
    border-radius: 9px !important;
}

.vs-danger button {
    color: var(--vs-danger) !important;
    border-color: #efc9c4 !important;
}

.vs-back {
    margin: 0 0 12px !important;
}

.vs-back button {
    min-height: 36px !important;
    padding-left: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: var(--vs-primary-dark) !important;
    font-weight: 650 !important;
}

/* Keep action rows aligned when labels have different heights. */
.vs-card > .gradio-row,
.vs-step > .gradio-row,
.vs-output > .gradio-row {
    align-items: end !important;
}

/* ---------- Dataframes ---------- */

.vs-table {
    width: 100% !important;
    margin: 0 !important;
    border: 1px solid var(--vs-border) !important;
    border-radius: 11px !important;
    overflow: hidden !important;
    background: white !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

.vs-table .table-wrap {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
    border: 0 !important;
}

.vs-table table {
    width: 100% !important;
    min-width: 760px !important;
    table-layout: auto !important;
    font-size: 13px !important;
}

.vs-table thead th {
    height: 40px !important;
    padding: 8px 10px !important;
    background: #f8f9fc !important;
    color: #475467 !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    vertical-align: middle !important;
}

.vs-table tbody td {
    padding: 9px 10px !important;
    vertical-align: middle !important;
}

/* Don't let long filesystem paths destroy the layout. */
.vs-table tbody td:last-child {
    max-width: 420px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* ---------- Summary content ---------- */

.vs-stat {
    width: 100% !important;
    margin: 0 0 12px !important;
    padding: 10px 12px !important;
    background: #f8f9fc !important;
    border: 1px solid #eaecf0 !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
}

#summary-text {
    width: 100% !important;
    max-height: 300px !important;
    overflow-y: auto !important;
    line-height: 1.65 !important;
    box-sizing: border-box !important;
}

.vs-gallery {
    width: 100% !important;
    margin: 0 0 14px !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.vs-gallery .grid-wrap {
    width: 100% !important;
}

/* ---------- Accordions ---------- */

.accordion {
    width: 100% !important;
    border-radius: 10px !important;
    border-color: var(--vs-border) !important;
    box-sizing: border-box !important;
}

/* ---------- Output/history ---------- */

.vs-output .video-container,
.vs-output .file-preview {
    width: 100% !important;
}

.vs-output > .gradio-row {
    align-items: stretch !important;
}

.vs-output .video-container,
.vs-output .file-preview {
    min-height: 0 !important;
}

/* ---------- Video library player ---------- */

.vs-video-library-row {
    width: 100% !important;
    align-items: stretch !important;
}

.vs-video-library-row > .gradio-column {
    min-width: 0 !important;
}

.vs-player-panel {
    background: #fafaff !important;
    border: 1px solid #e6e7f2 !important;
    border-radius: 12px !important;
    padding: 13px !important;
    box-sizing: border-box !important;
}

.vs-player-title h3 {
    margin: 0 0 9px !important;
    font-size: 15px !important;
}

.vs-player-panel .video-container,
.vs-player-panel video {
    width: 100% !important;
    max-width: 100% !important;
    border-radius: 9px !important;
    overflow: hidden !important;
}

.vs-player-panel .video-container {
    margin-bottom: 9px !important;
}

/* ---------- Responsive layout ---------- */

@media (max-width: 1000px) {
    .gradio-container {
        padding: 18px 18px 32px !important;
    }

    .vs-topbar {
        align-items: stretch !important;
    }

    .vs-session {
        margin-top: 12px !important;
        text-align: left !important;
    }

    .vs-hero .image-container {
        min-height: 120px !important;
    }

    .vs-video-library-row {
        flex-direction: column !important;
    }

    .vs-player-panel {
        width: 100% !important;
    }
}

@media (max-width: 760px) {
    .gradio-container {
        padding: 12px 12px 28px !important;
    }

    .vs-tabs > .tab-nav button {
        padding: 9px 8px !important;
        font-size: 13px !important;
    }

    .vs-card,
    .vs-step,
    .vs-preview,
    .vs-output {
        padding: 13px !important;
    }

    .vs-table table {
        min-width: 700px !important;
    }
}
"""

with gr.Blocks(title="Video Summarizer") as demo:
    user_state = gr.State(None)
    project_state = gr.State(None)
    job_state = gr.State(None)
    video_state = gr.State(None)
    segments_state = gr.State(None)
    settings_state = gr.State(None)
    project_ids_state = gr.State([])
    video_ids_state = gr.State([])
    render_ids_state = gr.State([])
    project_row_state = gr.State(None)
    video_row_state = gr.State(None)

    # -------------------------------------------------------------------------
    # App header
    with gr.Row(elem_classes="vs-topbar", equal_height=True):
        with gr.Column(scale=3, elem_classes="vs-brand"):
            gr.Markdown(
                "# Video Summarizer\n"
                "Turn long videos into searchable transcripts and concise summary clips.",
            )
        with gr.Column(scale=2):
            session_md = gr.Markdown(elem_classes="vs-session")

    with gr.Tabs(elem_classes="vs-tabs") as nav:
        # =====================================================================
        # SCREEN 1 — PROJECTS
        with gr.Tab("①  Projects", id="projects"):
            gr.Markdown(
                "## Your projects\n"
                "Organise videos into projects and keep every transcript and summary in one place.",
                elem_classes="vs-section-title",
            )
            gr.Markdown(
                "Search, open, create or remove a project. Select a row to work with it.",
                elem_classes="vs-note",
            )

            with gr.Row(elem_classes="vs-card"):
                project_search = gr.Textbox(
                    label="Search projects",
                    placeholder="Search by project name or description…",
                    scale=5,
                )
                projects_refresh = gr.Button("↻ Refresh", scale=1)

            with gr.Column(elem_classes="vs-card"):
                gr.Markdown("### Project library")
                projects_table = gr.Dataframe(
                    headers=PROJECT_HEADERS,
                    interactive=False,
                    wrap=True,
                    max_height=380,
                    show_label=False,
                    elem_classes="vs-table",
                )
                with gr.Row():
                    open_project_btn = gr.Button(
                        "Open selected project  →",
                        variant="primary",
                        scale=2,
                    )
                    delete_project_confirm = gr.Checkbox(
                        label="Confirm delete",
                        scale=1,
                    )
                    delete_project_btn = gr.Button(
                        "Delete selected",
                        variant="stop",
                        scale=1,
                        elem_classes="vs-danger",
                    )

            with gr.Column(elem_classes="vs-card"):
                gr.Markdown("### Create a project")
                gr.Markdown(
                    "Start a new workspace for a topic, course, client or collection of videos.",
                    elem_classes="vs-note",
                )
                with gr.Row():
                    new_project_name = gr.Textbox(
                        label="Project name",
                        placeholder="e.g. Capstone lectures",
                        scale=2,
                    )
                    new_project_desc = gr.Textbox(
                        label="Description",
                        placeholder="Optional description",
                        scale=3,
                    )
                    create_project_btn = gr.Button(
                        "＋ Create project",
                        variant="primary",
                        scale=1,
                    )
                projects_msg = gr.Markdown()

        # =====================================================================
        # SCREEN 2 — VIDEOS
        with gr.Tab("②  Videos", id="videos"):
            with gr.Row():
                back_to_projects_btn = gr.Button(
                    "← All projects",
                    elem_classes="vs-back",
                    scale=1,
                )
                gr.Markdown("", scale=5)

            project_header_md = gr.Markdown(
                "Open a project from the Projects screen.",
                elem_classes="vs-project-header",
            )

            with gr.Row(elem_classes="vs-card"):
                video_search = gr.Textbox(
                    label="Search videos",
                    placeholder="Search by title or source…",
                    scale=5,
                )
                videos_refresh = gr.Button("↻ Refresh", scale=1)

            with gr.Column(elem_classes="vs-card"):
                gr.Markdown("### Video library")
                with gr.Row(equal_height=False, elem_classes="vs-video-library-row"):
                    with gr.Column(scale=2, min_width=560):
                        videos_table = gr.Dataframe(
                            headers=VIDEO_HEADERS,
                            interactive=False,
                            wrap=True,
                            max_height=360,
                            show_label=False,
                            elem_classes="vs-table",
                        )
                    with gr.Column(scale=1, min_width=300, elem_classes="vs-player-panel"):
                        gr.Markdown("### ▶ Play selected video", elem_classes="vs-player-title")
                        selected_video_player = gr.Video(
                            label="",
                            interactive=False,
                            height=250,
                            show_label=False,
                        )
                        video_detail_md = gr.Markdown(
                            "Select a video row to play it.",
                            elem_classes="vs-note",
                        )

                with gr.Row():
                    open_video_btn = gr.Button(
                        "Open selected video  →",
                        variant="primary",
                        scale=2,
                    )
                    delete_video_confirm = gr.Checkbox(
                        label="Confirm delete",
                        scale=1,
                    )
                    delete_video_btn = gr.Button(
                        "Remove selected",
                        variant="stop",
                        scale=1,
                        elem_classes="vs-danger",
                    )

                with gr.Row():
                    rename_box = gr.Textbox(
                        label="Rename selected video",
                        placeholder="Enter a new video name…",
                        scale=4,
                    )
                    rename_btn = gr.Button("Rename", scale=1)

            with gr.Column(elem_classes="vs-card"):
                gr.Markdown("### ＋ Add a video")
                gr.Markdown(
                    "Upload a local video or download one from a YouTube/direct video link.",
                    elem_classes="vs-note",
                )
                video_name_box = gr.Textbox(
                    label="Video name (optional)",
                    placeholder="Leave blank to use the file name or link title",
                )

                with gr.Tabs():
                    with gr.Tab("Upload file"):
                        upload = gr.File(
                            label="Choose a video",
                            file_types=["video"],
                            type="filepath",
                        )

                    with gr.Tab("Video link"):
                        url_box = gr.Textbox(
                            label="YouTube or direct video link",
                            placeholder="https://www.youtube.com/watch?v=…",
                        )
                        with gr.Row():
                            max_height = gr.Dropdown(
                                [360, 480, 720, 1080],
                                value=720,
                                label="Maximum resolution",
                                scale=2,
                            )
                            download_btn = gr.Button(
                                "↓ Download & add",
                                variant="secondary",
                                scale=1,
                            )

            videos_msg = gr.Markdown()

        # =====================================================================
        # SCREEN 3 — SUMMARISE
        with gr.Tab("③  Summarise", id="summarise"):
            with gr.Row():
                back_to_videos_btn = gr.Button(
                    "← Back to videos",
                    elem_classes="vs-back",
                    scale=1,
                )
                gr.Markdown("", scale=5)

            with gr.Row(elem_classes="vs-hero", equal_height=False):
                with gr.Column(scale=4):
                    video_header_md = gr.Markdown(
                        "Open a video from the Videos screen to begin."
                    )
                    summarise_msg = gr.Markdown()
                with gr.Column(scale=1):
                    source_img = gr.Image(
                        label="Source preview",
                        interactive=False,
                        height=180,
                        scale=1,
                    )

            # -----------------------------------------------------------------
            # Step 1
            with gr.Column(elem_classes="vs-step"):
                gr.Markdown(
                    "### Step 1 · Transcribe\n"
                    "Extract the audio and generate a word-timestamped transcript.",
                    elem_classes="vs-step-head",
                )
                with gr.Row():
                    whisper_choice = gr.Radio(
                        list(P.WHISPER_MODELS),
                        value="small (fast)",
                        label="Speech model",
                        scale=2,
                    )
                    language = gr.Dropdown(
                        list(LANGUAGES),
                        value="English",
                        label="Language",
                        scale=1,
                    )
                    process_btn = gr.Button(
                        "Extract audio & transcribe",
                        variant="primary",
                        interactive=False,
                        scale=2,
                    )
                transcript_md = gr.Markdown()

                with gr.Accordion("View full transcript", open=False):
                    transcript_box = gr.Textbox(
                        show_label=False,
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        placeholder=(
                            "The transcript appears here after you press "
                            "'Extract audio & transcribe'."
                        ),
                    )

                gr.Markdown(
                    "Stored transcripts for this video",
                    elem_classes="vs-note",
                )
                transcripts_table = gr.Dataframe(
                    headers=TRANSCRIPT_HEADERS,
                    interactive=False,
                    wrap=True,
                    max_height=170,
                    show_label=False,
                    elem_classes="vs-table",
                )

            # -----------------------------------------------------------------
            # Step 2
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=330, elem_classes="vs-step"):
                    gr.Markdown(
                        "### Step 2 · Tune the summary\n"
                        "Choose how much content to keep, then preview the selected clips.",
                        elem_classes="vs-step-head",
                    )
                    preset = gr.Radio(
                        list(PRESETS),
                        value="Balanced",
                        label="Summary preset",
                    )
                    size_mode = gr.Radio(
                        ["Percentage of sentences", "Exact number of sentences"],
                        value="Percentage of sentences",
                        label="Summary size",
                    )
                    ratio = gr.Slider(
                        5, 80, value=20, step=1,
                        label="Keep this % of sentences",
                    )
                    num_sentences = gr.Slider(
                        1, 300, value=20, step=1,
                        label="Number of sentences",
                        visible=False,
                    )

                    with gr.Accordion("Advanced settings", open=False):
                        min_chars = gr.Slider(
                            0, 120, value=40, step=5,
                            label="Ignore sentences shorter than (characters)",
                        )
                        max_chars = gr.Slider(
                            200, 5000, value=600, step=100,
                            label="Ignore sentences longer than (characters)",
                        )
                        pad = gr.Slider(
                            0, 2, value=0.25, step=0.05,
                            label="Extra time around each sentence (s)",
                        )
                        merge_gap = gr.Slider(
                            0, 10, value=1.0, step=0.5,
                            label="Join clips closer than (s)",
                        )
                        use_first = gr.Checkbox(
                            value=True,
                            label="Always include the first sentence",
                        )
                        model = gr.Dropdown(
                            ["bert-large-uncased", "bert-base-uncased"],
                            value="bert-large-uncased",
                            label="Embedding model (base is ~3x faster)",
                        )

                    preview_btn = gr.Button(
                        "Preview summary",
                        variant="primary",
                    )

                with gr.Column(scale=2, elem_classes="vs-preview"):
                    stats_md = gr.Markdown(elem_classes="vs-stat")
                    with gr.Accordion("Summary text", open=True):
                        summary_md = gr.Markdown(elem_id="summary-text")

                    gallery = gr.Gallery(
                        label="Selected segments",
                        columns=5,
                        height=250,
                        object_fit="cover",
                        allow_preview=True,
                        elem_classes="vs-gallery",
                    )

                    segments_table = gr.Dataframe(
                        headers=SEGMENT_HEADERS,
                        datatype=["bool", "number", "str", "str", "number", "str"],
                        interactive=True,
                        static_columns=[1, 2, 3, 4, 5],
                        wrap=True,
                        max_height=350,
                        label="Untick segments you do not want in the final video",
                        elem_classes="vs-table",
                    )
                    kept_md = gr.Markdown()

            # -----------------------------------------------------------------
            # Step 3
            with gr.Column(elem_classes="vs-output"):
                gr.Markdown(
                    "### Step 3 · Create the summary video\n"
                    "Review the selected segments above, then generate the final video.",
                    elem_classes="vs-step-head",
                )
                with gr.Row():
                    render_btn = gr.Button(
                        "✓ Confirm & create video",
                        variant="primary",
                        interactive=False,
                        scale=2,
                    )
                    render_md = gr.Markdown(scale=3)

                with gr.Row(equal_height=False):
                    out_video = gr.Video(
                        label="Summary video",
                        interactive=False,
                        scale=2,
                    )
                    out_files = gr.File(
                        label="Output files",
                        file_count="multiple",
                        interactive=False,
                        scale=1,
                    )

            # -----------------------------------------------------------------
            # History
            with gr.Column(elem_classes="vs-card"):
                gr.Markdown(
                    "### Earlier summary videos",
                    elem_classes="vs-section-title",
                )
                gr.Markdown(
                    "Select a previous render to play it again or access its generated files.",
                    elem_classes="vs-note",
                )
                renders_table = gr.Dataframe(
                    headers=RENDER_HEADERS,
                    interactive=False,
                    wrap=True,
                    max_height=210,
                    show_label=False,
                    elem_classes="vs-table",
                )
                with gr.Row(equal_height=False):
                    history_video = gr.Video(
                        label="Stored summary video",
                        interactive=False,
                        scale=2,
                    )
                    history_files = gr.File(
                        label="Files",
                        file_count="multiple",
                        interactive=False,
                        scale=1,
                    )

    # =========================================================================
    # Existing event wiring — intentionally unchanged.
    demo.load(on_start, None, [user_state, session_md, projects_table, project_ids_state])

    # screen 1
    project_search.input(on_search_projects, [project_search],
                         [projects_table, project_ids_state, project_row_state])
    projects_refresh.click(on_search_projects, [project_search],
                           [projects_table, project_ids_state, project_row_state], api_name="refresh_projects")
    projects_table.select(on_select_project, None, [project_row_state])
    open_project_btn.click(on_open_project, [project_ids_state, project_row_state],
                           [nav, project_state, project_header_md, videos_table, video_ids_state,
                            video_row_state, video_detail_md, selected_video_player], api_name="open_project")
    create_project_btn.click(on_create_project, [new_project_name, new_project_desc],
                             [nav, project_state, project_header_md, videos_table, video_ids_state,
                              video_search, projects_table, project_ids_state,
                              new_project_name, new_project_desc, session_md, videos_msg],
                             api_name="create_project")
    delete_project_btn.click(on_delete_project,
                             [project_ids_state, project_row_state, delete_project_confirm, project_search],
                             [projects_table, project_ids_state, project_row_state, delete_project_confirm,
                              session_md, projects_msg], api_name="delete_project")

    # screen 2
    video_search.input(on_search_videos, [project_state, video_search],
                       [videos_table, video_ids_state, video_row_state, selected_video_player])
    videos_refresh.click(on_search_videos, [project_state, video_search],
                         [videos_table, video_ids_state, video_row_state, selected_video_player], api_name="refresh_videos")
    videos_table.select(on_select_video, [video_ids_state], [video_row_state, video_detail_md, selected_video_player])
    back_to_projects_btn.click(on_back_to_projects, [project_search],
                               [nav, projects_table, project_ids_state, session_md])
    delete_video_btn.click(on_delete_video,
                           [video_ids_state, video_row_state, delete_video_confirm, project_state, video_search],
                           [videos_table, video_ids_state, video_row_state, delete_video_confirm,
                            session_md, videos_msg, selected_video_player], api_name="delete_video")

    open_outputs = [nav, job_state, video_state, video_header_md, source_img, process_btn,
                    transcript_md, transcript_box, preview_btn, render_btn,
                    segments_state, settings_state, out_video, out_files,
                    stats_md, summary_md, gallery, segments_table, kept_md, render_md,
                    renders_table, render_ids_state, videos_table, video_ids_state,
                    transcripts_table, summarise_msg]
    upload.upload(on_upload, [upload, project_state, video_name_box], open_outputs, api_name="upload")
    download_btn.click(on_download, [url_box, max_height, project_state, video_name_box], open_outputs,
                       api_name="download")
    url_box.submit(on_download, [url_box, max_height, project_state, video_name_box], open_outputs)
    rename_btn.click(on_rename_video,
                     [video_ids_state, video_row_state, rename_box, project_state, video_search],
                     [videos_table, video_ids_state, rename_box, videos_msg], api_name="rename_video")
    open_video_btn.click(on_open_video, [video_ids_state, video_row_state, project_state], open_outputs,
                         api_name="open_video")

    # screen 3
    back_to_videos_btn.click(on_back_to_videos, [project_state, video_search],
                             [nav, videos_table, video_ids_state, project_header_md])
    process_btn.click(on_process, [job_state, video_state, whisper_choice, language],
                      [transcript_md, transcript_box, preview_btn, transcripts_table, session_md],
                      api_name="transcribe")
    preset.input(apply_preset, [preset], [ratio, min_chars, max_chars, pad, merge_gap]) \
          .then(mark_stale, None, [render_btn, stats_md])
    size_mode.input(toggle_size_mode, [size_mode], [ratio, num_sentences])
    for comp in (size_mode, ratio, num_sentences, min_chars, max_chars, pad, merge_gap, use_first, model):
        comp.input(mark_stale, None, [render_btn, stats_md])
    preview_btn.click(on_preview,
                      [job_state, video_state, size_mode, ratio, num_sentences, min_chars, max_chars, pad, merge_gap,
                       use_first, model],
                      [stats_md, summary_md, gallery, segments_table, segments_state, settings_state,
                       kept_md, render_btn], api_name="preview")
    segments_table.input(on_table_change, [segments_table, segments_state, job_state], [kept_md])
    render_btn.click(on_render,
                     [job_state, video_state, segments_state, settings_state, segments_table, project_state],
                     [render_md, out_video, out_files, renders_table, render_ids_state,
                      videos_table, video_ids_state, session_md], api_name="render")
    renders_table.select(on_select_render, [render_ids_state], [history_video, history_files])


if __name__ == "__main__":
    if not db.list_users():
        raise SystemExit("No accounts yet. Create one first:\n    python db.py add-user <username>")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Soft(), css=CSS,
        allowed_paths=[str(P.WORKSPACE)],
        auth=db.check_password,
        auth_message="Sign in to the Video Summarizer",
    )
