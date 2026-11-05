"""SQLite-Zugriff. Jeder Aufruf oeffnet eine eigene Verbindung, damit Threads sich nicht stoeren."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS vods (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    streamer TEXT NOT NULL DEFAULT '',
    time_range TEXT,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS streamers (
    name TEXT PRIMARY KEY,
    cam_rect TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    vod_id TEXT NOT NULL,
    id TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    candidate TEXT NOT NULL,
    spec TEXT NOT NULL,
    preview_path TEXT NOT NULL DEFAULT '',
    output_path TEXT NOT NULL DEFAULT '',
    text_path TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (vod_id, id)
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    vod_id TEXT NOT NULL,
    clip_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    step TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    finished_at REAL
);
"""

# Renderjobs laufen vor neuen VODs: ein VOD kann Stunden dauern, ein Clip nur Minuten.
KIND_ORDER = "CASE kind WHEN 'preview' THEN 0 WHEN 'render' THEN 1 ELSE 2 END"


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    # VODs

    def add_vod(
        self, vod_id: str, source: str, streamer: str, time_range: list[float] | None
    ) -> None:
        with self._connect() as con:
            con.execute(
                """INSERT INTO vods (id, source, streamer, time_range, status, created_at)
                   VALUES (?, ?, ?, ?, 'queued', ?)
                   ON CONFLICT(id) DO UPDATE SET status='queued', error='',
                       streamer=excluded.streamer, time_range=excluded.time_range""",
                (vod_id, source, streamer, json.dumps(time_range), time.time()),
            )

    def get_vod(self, vod_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM vods WHERE id=?", (vod_id,)).fetchone()
        return _vod(row) if row else None

    def list_vods(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM vods ORDER BY created_at DESC").fetchall()
        return [_vod(r) for r in rows]

    def update_vod(self, vod_id: str, **fields: Any) -> None:
        if "time_range" in fields:
            fields["time_range"] = json.dumps(fields["time_range"])
        _update(self, "vods", "id=?", (vod_id,), fields)

    def delete_vod(self, vod_id: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM clips WHERE vod_id=?", (vod_id,))
            con.execute("DELETE FROM jobs WHERE vod_id=?", (vod_id,))
            con.execute("DELETE FROM vods WHERE id=?", (vod_id,))

    # Streamer-Profile: Facecam relativ zur Bildgroesse, weil VODs verschieden aufgeloest sein koennen

    def get_streamer_cam(self, name: str) -> list[float] | None:
        if not name:
            return None
        with self._connect() as con:
            row = con.execute("SELECT cam_rect FROM streamers WHERE name=?", (name,)).fetchone()
        return json.loads(row["cam_rect"]) if row else None

    def set_streamer_cam(self, name: str, rect: list[float]) -> None:
        with self._connect() as con:
            con.execute(
                """INSERT INTO streamers (name, cam_rect, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET cam_rect=excluded.cam_rect, updated_at=excluded.updated_at""",
                (name, json.dumps(rect), time.time()),
            )

    # Clips

    def replace_proposed_clips(self, vod_id: str, clips: list[dict]) -> None:
        """Neue Auswahl: Vorschlaege werden ersetzt, Bearbeitetes bleibt erhalten."""
        with self._connect() as con:
            con.execute(
                "DELETE FROM clips WHERE vod_id=? AND status IN ('proposed', 'rejected')", (vod_id,)
            )
            for clip in clips:
                con.execute(
                    """INSERT OR IGNORE INTO clips (vod_id, id, status, score, candidate, spec)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        vod_id,
                        clip["id"],
                        clip.get("status", "proposed"),
                        clip.get("score", 0),
                        json.dumps(clip["candidate"]),
                        json.dumps(clip["spec"]),
                    ),
                )

    def get_clip(self, vod_id: str, clip_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM clips WHERE vod_id=? AND id=?", (vod_id, clip_id)
            ).fetchone()
        return _clip(row) if row else None

    def list_clips(self, vod_id: str) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM clips WHERE vod_id=? ORDER BY score DESC, id", (vod_id,)
            ).fetchall()
        return [_clip(r) for r in rows]

    def list_finished_clips(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM clips WHERE status='done' ORDER BY rowid DESC"
            ).fetchall()
        return [_clip(r) for r in rows]

    def update_clip(self, vod_id: str, clip_id: str, **fields: Any) -> None:
        if "spec" in fields:
            fields["spec"] = json.dumps(fields["spec"])
        _update(self, "clips", "vod_id=? AND id=?", (vod_id, clip_id), fields)

    # Jobs

    def enqueue(self, kind: str, vod_id: str, clip_id: str = "") -> int:
        with self._connect() as con:
            existing = con.execute(
                """SELECT id FROM jobs WHERE kind=? AND vod_id=? AND clip_id=?
                   AND status IN ('queued', 'running')""",
                (kind, vod_id, clip_id),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = con.execute(
                "INSERT INTO jobs (kind, vod_id, clip_id, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
                (kind, vod_id, clip_id, time.time()),
            )
            return int(cursor.lastrowid)

    def claim_next_job(self) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                f"SELECT * FROM jobs WHERE status='queued' ORDER BY {KIND_ORDER}, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            con.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
        return {**dict(row), "status": "running"}

    def get_job(self, job_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_active_jobs(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                f"""SELECT * FROM jobs WHERE status IN ('queued', 'running', 'cancelling')
                    ORDER BY status='queued', {KIND_ORDER}, id"""
            ).fetchall()
        return [dict(r) for r in rows]

    def update_job(self, job_id: int, **fields: Any) -> None:
        _update(self, "jobs", "id=?", (job_id,), fields)

    def finish_job(self, job_id: int, status: str, error: str = "") -> None:
        self.update_job(job_id, status=status, error=error, finished_at=time.time())

    def request_cancel(self, job_id: int) -> bool:
        """Wartende Jobs enden sofort, laufende beim naechsten Fortschrittswert."""
        with self._connect() as con:
            con.execute(
                "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                (time.time(), job_id),
            )
            cursor = con.execute(
                "UPDATE jobs SET status='cancelling' WHERE id=? AND status='running'", (job_id,)
            )
            return cursor.rowcount > 0

    def requeue_interrupted_jobs(self) -> int:
        """Nach einem Neustart: Ergebnisse liegen auf der Platte, der Job setzt dort wieder an."""
        with self._connect() as con:
            con.execute(
                "UPDATE jobs SET status='cancelled', finished_at=? WHERE status='cancelling'",
                (time.time(),),
            )
            cursor = con.execute("UPDATE jobs SET status='queued' WHERE status='running'")
            return cursor.rowcount


def _update(store: Store, table: str, where: str, args: tuple, fields: dict[str, Any]) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{name}=?" for name in fields)
    with store._connect() as con:
        con.execute(
            f"UPDATE {table} SET {assignments} WHERE {where}",
            (*fields.values(), *args),
        )


def _vod(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["time_range"] = json.loads(data["time_range"]) if data["time_range"] else None
    return data


def _clip(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["candidate"] = json.loads(data["candidate"])
    data["spec"] = json.loads(data["spec"])
    return data
