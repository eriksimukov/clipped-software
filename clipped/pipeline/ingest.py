"""Import: holt das VOD (Datei oder Link), Audiospur, Kapitel und Chat-Replay."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from clipped.config import Settings
from clipped.pipeline.chat import parse_chat_file
from clipped.pipeline.common import (
    VodPaths,
    probe_duration,
    read_json,
    run_ffmpeg,
    vod_id_for,
    write_json,
)

Progress = Callable[[str, float], None]

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
CHAT_LANGS = ["rechat", "live_chat"]


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source.strip()))


def _download(url: str, paths: VodPaths, progress: Progress) -> dict:
    import yt_dlp

    def hook(status: dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if total:
                progress("download", status.get("downloaded_bytes", 0) / total)

    options = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(paths.root / "source.%(ext)s"),
        "writesubtitles": True,
        "subtitleslangs": CHAT_LANGS,
        # Ein fehlender Chat darf den Download nicht abbrechen.
        "ignoreerrors": "only_download",
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    if info is None:
        raise RuntimeError(f"Download fehlgeschlagen: {url}")

    video = next(
        (p for p in paths.root.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm"}),
        None,
    )
    if video is None:
        raise RuntimeError(f"Keine Videodatei nach dem Download gefunden: {url}")

    return {
        "video": str(video),
        "title": info.get("title") or "",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "chapters": [
            {"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title", "")}
            for c in info.get("chapters") or []
        ],
        "platform": info.get("extractor_key") or "",
    }


def _collect_chat(paths: VodPaths) -> int:
    """Gibt die Anzahl der Nachrichten zurueck; 0 heisst: es gibt kein Chat-Signal."""
    messages: list[dict] = []
    for lang in CHAT_LANGS:
        for path in paths.root.glob(f"source.{lang}.*"):
            try:
                messages = parse_chat_file(path)
            except (ValueError, KeyError) as exc:
                print(f"Chat-Datei {path.name} nicht lesbar: {exc}")
    write_json(paths.chat, messages)
    return len(messages)


def ingest(
    source: str,
    settings: Settings,
    streamer: str | None = None,
    progress: Progress = lambda step, value: None,
) -> VodPaths:
    """Idempotent: bereits vorhandene Ergebnisse werden wiederverwendet."""
    source = source.strip().strip('"')
    paths = VodPaths.for_vod(settings.paths.data_dir, vod_id_for(source))

    if paths.meta.exists():
        meta = read_json(paths.meta)
    else:
        if is_url(source):
            meta = _download(source, paths, progress)
        else:
            video = Path(source)
            if not video.is_file():
                raise FileNotFoundError(f"Datei nicht gefunden: {source}")
            # Lokale Dateien bleiben, wo sie sind, damit kein Speicher doppelt belegt wird.
            meta = {
                "video": str(video.resolve()),
                "title": video.stem,
                "uploader": "",
                "chapters": [],
                "platform": "local",
            }
        meta["source"] = source
        meta["streamer"] = streamer or meta.get("uploader", "")
        meta["duration"] = probe_duration(Path(meta["video"]))
        write_json(paths.meta, meta)

    if not paths.chat.exists():
        meta["chat_messages"] = _collect_chat(paths)
        write_json(paths.meta, meta)

    if not paths.audio.exists():
        progress("audio", 0.0)
        tmp = paths.audio.with_name("audio.tmp.wav")
        run_ffmpeg(["-i", meta["video"], "-vn", "-ac", "1", "-ar", "16000", str(tmp)])
        tmp.replace(paths.audio)
        progress("audio", 1.0)

    return paths
