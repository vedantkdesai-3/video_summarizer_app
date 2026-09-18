"""
Database layer for the Video Summarizer app.

SQLite, one file at workspace/app.db. Four tables:

    users     -> login accounts (password stored as a PBKDF2 hash, never in clear text)
    projects  -> a named folder of work belonging to one user
    videos    -> every video added to a project, with where its file is stored
    renders   -> every summary video created from a video, with its settings

Command line helpers:

    python db.py add-user <username>     create an account (asks for the password)
    python db.py list-users
    python db.py reset-password <username>
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

WORKSPACE = Path(os.environ.get("VIDEOSUM_WORKSPACE", "workspace")).resolve()
DB_PATH = WORKSPACE / "app.db"

_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name     TEXT,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id      TEXT NOT NULL,              -- content fingerprint used by the pipeline
    title       TEXT NOT NULL,
    source_type TEXT NOT NULL,              -- 'upload' or 'link'
    source_ref  TEXT,                       -- original file name or URL
    file_path   TEXT NOT NULL,              -- where the video is stored on disk
    duration    REAL,
    resolution  TEXT,
    size_mb     REAL,
    has_audio   INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE (project_id, job_id)
);

CREATE TABLE IF NOT EXISTS renders (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_path     TEXT NOT NULL,
    srt_path       TEXT,
    summary_path   TEXT,
    transcript_path TEXT,
    duration       REAL,
    clips          INTEGER,
    settings_json  TEXT,
    segments_json  TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,           -- speech model used
    language    TEXT,                    -- language code, NULL means auto-detect
    words       INTEGER,
    text        TEXT NOT NULL,           -- the full transcript
    chunks_path TEXT,                    -- JSON file holding the word-level timestamps
    created_at  TEXT NOT NULL,
    UNIQUE (video_id, model, language)
);

CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    summary_text  TEXT NOT NULL,
    stats_json    TEXT,
    settings_json TEXT,
    segments_json TEXT,
    keep_json     TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_project ON videos(project_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_video ON transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_renders_video ON renders(video_id);
CREATE INDEX IF NOT EXISTS idx_summaries_video ON summaries(video_id);
"""


def connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA foreign_keys = ON")
        _CONN.executescript(SCHEMA)
        _CONN.commit()
    return _CONN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(sql: str, params: tuple = ()) -> int:
    with _LOCK:
        conn = connect()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def _rows(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with _LOCK:
        return connect().execute(sql, params).fetchall()


def _row(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    rows = _rows(sql, params)
    return rows[0] if rows else None


# ----------------------------------------------------------------------------- users
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(username: str, password: str, full_name: str = "") -> int:
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_hex(16)
    return _write(
        "INSERT INTO users (username, full_name, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
        (username, full_name or username, _hash(password, salt), salt, _now()),
    )


def set_password(username: str, password: str) -> None:
    user = get_user(username)
    if not user:
        raise ValueError(f"No such user: {username}")
    salt = secrets.token_hex(16)
    _write("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
           (_hash(password, salt), salt, user["id"]))


def get_user(username: str) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM users WHERE username = ?", (username,))


def check_password(username: str, password: str) -> bool:
    """Passed to Gradio as the login check."""
    user = get_user(username or "")
    if not user:
        return False
    return secrets.compare_digest(_hash(password or "", user["salt"]), user["password_hash"])


def list_users() -> list[sqlite3.Row]:
    return _rows("SELECT id, username, full_name, created_at FROM users ORDER BY username")


def user_id_for(username: str) -> int:
    user = get_user(username)
    if not user:
        raise ValueError(f"No such user: {username}")
    return user["id"]


# ----------------------------------------------------------------------------- projects
def create_project(user_id: int, name: str, description: str = "") -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the project a name.")
    existing = _row("SELECT id FROM projects WHERE user_id = ? AND name = ? COLLATE NOCASE", (user_id, name))
    if existing:
        raise ValueError(f"You already have a project called '{name}'.")
    return _write("INSERT INTO projects (user_id, name, description, created_at) VALUES (?,?,?,?)",
                  (user_id, name, description, _now()))


def list_projects(user_id: int, query: str = "") -> list[sqlite3.Row]:
    """All projects of a user, newest first. `query` filters on name and description."""
    sql = """SELECT p.*,
                    (SELECT COUNT(*) FROM videos v WHERE v.project_id = p.id) AS video_count,
                    (SELECT COUNT(*) FROM renders r
                       JOIN videos v2 ON v2.id = r.video_id WHERE v2.project_id = p.id) AS render_count
             FROM projects p WHERE p.user_id = ?"""
    params: tuple = (user_id,)
    if query and query.strip():
        like = f"%{query.strip()}%"
        sql += " AND (p.name LIKE ? OR IFNULL(p.description, '') LIKE ?)"
        params += (like, like)
    return _rows(sql + " ORDER BY p.created_at DESC", params)


def delete_project(project_id: int, user_id: int) -> None:
    """Removes the project and its video and summary records. Files on disk are left alone."""
    project = get_project(project_id, user_id)
    if not project:
        raise ValueError("That project does not belong to this account.")
    _write("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))


def default_project(user_id: int) -> sqlite3.Row:
    """Every user gets a starting project so they can begin without setting one up."""
    projects = list_projects(user_id)
    if projects:
        return projects[0]
    create_project(user_id, "My first project", "Created automatically")
    return list_projects(user_id)[0]


def get_project(project_id: int, user_id: int) -> Optional[sqlite3.Row]:
    """Return a project with the same aggregate fields exposed by list_projects()."""
    return _row(
        """SELECT p.*,
                      (SELECT COUNT(*) FROM videos v WHERE v.project_id = p.id) AS video_count,
                      (SELECT COUNT(*) FROM renders r
                         JOIN videos v2 ON v2.id = r.video_id WHERE v2.project_id = p.id) AS render_count
               FROM projects p
               WHERE p.id = ? AND p.user_id = ?""",
        (project_id, user_id),
    )


# ----------------------------------------------------------------------------- videos
def add_video(project_id: int, user_id: int, job, source_type: str, source_ref: str = "") -> int:
    """`job` is a pipeline.Job. Adding the same video to the same project twice reuses the row."""
    existing = _row("SELECT id FROM videos WHERE project_id = ? AND job_id = ?", (project_id, job.job_id))
    if existing:
        return existing["id"]
    return _write(
        """INSERT INTO videos (project_id, user_id, job_id, title, source_type, source_ref, file_path,
                               duration, resolution, size_mb, has_audio, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (project_id, user_id, job.job_id, job.title, source_type, source_ref, str(job.video_path),
         job.duration, job.resolution, job.size_mb, int(job.has_audio), _now()),
    )


def list_videos(project_id: int, query: str = "") -> list[sqlite3.Row]:
    """All videos in a project, newest first. `query` filters on title and source."""
    sql = """SELECT v.*, (SELECT COUNT(*) FROM renders r WHERE r.video_id = v.id) AS render_count
             FROM videos v WHERE v.project_id = ?"""
    params: tuple = (project_id,)
    if query and query.strip():
        like = f"%{query.strip()}%"
        sql += " AND (v.title LIKE ? OR IFNULL(v.source_ref, '') LIKE ?)"
        params += (like, like)
    return _rows(sql + " ORDER BY v.created_at DESC", params)


def delete_video(video_id: int, user_id: int) -> None:
    """Removes the video record and its summary records. Files on disk are left alone."""
    video = get_video(video_id, user_id)
    if not video:
        raise ValueError("That video does not belong to this account.")
    _write("DELETE FROM videos WHERE id = ? AND user_id = ?", (video_id, user_id))


def get_video(video_id: int, user_id: int) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM videos WHERE id = ? AND user_id = ?", (video_id, user_id))


def rename_video(video_id: int, user_id: int, title: str) -> str:
    title = (title or "").strip()
    if not title:
        raise ValueError("Give the video a name.")
    video = get_video(video_id, user_id)
    if not video:
        raise ValueError("That video does not belong to this account.")
    _write("UPDATE videos SET title = ? WHERE id = ? AND user_id = ?", (title, video_id, user_id))
    return title


def video_by_job(project_id: int, job_id: str) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM videos WHERE project_id = ? AND job_id = ?", (project_id, job_id))


# ----------------------------------------------------------------------------- renders
def add_render(video_id: int, user_id: int, out: dict, settings: dict, segments: list) -> int:
    return _write(
        """INSERT INTO renders (video_id, user_id, video_path, srt_path, summary_path, transcript_path,
                                duration, clips, settings_json, segments_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (video_id, user_id, out["video"], out.get("srt"), out.get("summary_txt"), out.get("transcript_txt"),
         out.get("duration"), out.get("clips"), json.dumps(settings),
         json.dumps([{k: v for k, v in s.items() if k != "thumb"} for s in segments]), _now()),
    )


def list_renders(video_id: int) -> list[sqlite3.Row]:
    return _rows("SELECT * FROM renders WHERE video_id = ? ORDER BY created_at DESC", (video_id,))


def get_render(render_id: int, user_id: int) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM renders WHERE id = ? AND user_id = ?", (render_id, user_id))


# ----------------------------------------------------------------------------- transcripts
def save_transcript(video_id: int, user_id: int, model: str, language: Optional[str],
                    text: str, words: int, chunks_path: str = "") -> int:
    """One stored transcript per video, speech model and language. Re-running replaces it."""
    lang = language or ""
    existing = _row("SELECT id FROM transcripts WHERE video_id = ? AND model = ? AND language = ?",
                    (video_id, model, lang))
    if existing:
        _write("""UPDATE transcripts SET words = ?, text = ?, chunks_path = ?, created_at = ? WHERE id = ?""",
               (words, text, chunks_path, _now(), existing["id"]))
        return existing["id"]
    return _write(
        """INSERT INTO transcripts (video_id, user_id, model, language, words, text, chunks_path, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (video_id, user_id, model, lang, words, text, chunks_path, _now()),
    )


def get_transcript(video_id: int, model: str, language: Optional[str]) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM transcripts WHERE video_id = ? AND model = ? AND language = ?",
                (video_id, model, language or ""))


def list_transcripts(video_id: int) -> list[sqlite3.Row]:
    return _rows("SELECT * FROM transcripts WHERE video_id = ? ORDER BY created_at DESC", (video_id,))


def save_summary(video_id: int, user_id: int, summary_text: str, stats: dict,
                 settings: dict, segments: list, keep_flags: list) -> int:
    """Persist the latest generated summary preview, including segment metadata and selections."""
    clean_segments = [{k: v for k, v in seg.items() if k != "thumb"} for seg in (segments or [])]
    # Keep thumbnail paths separately because they are generated files that should survive restart.
    thumbs = [seg.get("thumb", "") for seg in (segments or [])]
    for seg, thumb in zip(clean_segments, thumbs):
        if thumb:
            seg["thumb"] = thumb
    return _write(
        """INSERT INTO summaries
           (video_id, user_id, summary_text, stats_json, settings_json, segments_json, keep_json, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (video_id, user_id, summary_text or "", json.dumps(stats or {}), json.dumps(settings or {}),
         json.dumps(clean_segments), json.dumps([bool(x) for x in (keep_flags or [])]), _now()),
    )


def video_for_job(user_id: int, job_id: str) -> Optional[sqlite3.Row]:
    """Return a user's video record by its pipeline content fingerprint."""
    return _row("SELECT * FROM videos WHERE user_id = ? AND job_id = ? ORDER BY id DESC LIMIT 1",
                (user_id, job_id))


def latest_summary(video_id: int, user_id: int) -> Optional[sqlite3.Row]:
    return _row("SELECT * FROM summaries WHERE video_id = ? AND user_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (video_id, user_id))


def list_summaries(video_id: int, user_id: int) -> list[sqlite3.Row]:
    return _rows("SELECT * FROM summaries WHERE video_id = ? AND user_id = ? ORDER BY created_at DESC, id DESC",
                 (video_id, user_id))


def project_stats(user_id: int) -> dict:
    row = _row(
        """SELECT (SELECT COUNT(*) FROM projects WHERE user_id = ?) AS projects,
                  (SELECT COUNT(*) FROM videos   WHERE user_id = ?) AS videos,
                  (SELECT COUNT(*) FROM renders  WHERE user_id = ?) AS renders,
                  (SELECT COUNT(*) FROM transcripts WHERE user_id = ?) AS transcripts""",
        (user_id, user_id, user_id, user_id))
    return dict(row) if row else {"projects": 0, "videos": 0, "renders": 0, "transcripts": 0}


# ----------------------------------------------------------------------------- CLI
def _cli():
    import getpass
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "add-user" and len(args) >= 2:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.")
            return
        full_name = input("Full name (optional): ").strip()
        create_user(args[1], password, full_name)
        print(f"Created user '{args[1]}'. Database: {DB_PATH}")
    elif cmd == "reset-password" and len(args) >= 2:
        password = getpass.getpass("New password: ")
        set_password(args[1], password)
        print("Password updated.")
    elif cmd == "list-users":
        for u in list_users():
            print(f"{u['id']:>3}  {u['username']:<20} {u['full_name'] or '':<25} {u['created_at']}")
        if not list_users():
            print("No users yet. Create one with:  python db.py add-user <username>")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli()
