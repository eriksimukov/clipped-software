"""Import: holt das VOD (Datei oder Link), Audiospur, Kapitel und Chat-Replay."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

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
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CHAT_LANGS = ["rechat", "live_chat"]


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source.strip()))


def link_problem(source: str) -> str | None:
    """Findet Eingaben, die wie ein Link aussehen, aber keiner sind."""
    if is_url(source) and re.search(r"\s", source.strip()):
        return (
            "Der Link darf keine Leerzeichen enthalten. Zusatzoptionen wie --cookies-from-browser "
            "gehören nicht in dieses Feld, die Anmeldung stellst du unter Einstellungen ein."
        )
    return None


def has_login(settings: Settings) -> bool:
    return bool(settings.download.cookies_file or settings.download.cookies_from_browser)


def ytdlp_options(settings: Settings) -> dict:
    """Gemeinsame Grundeinstellungen fuer Video und Chat."""
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Sonst landen Farbcodes der Konsole in den Fehlermeldungen der Oberflaeche.
        "color": "never",
        # YouTube verlangt inzwischen JavaScript. Beide Laufzeiten startet yt-dlp abgeschottet.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    download = settings.download
    if download.cookies_file:
        options["cookiefile"] = download.cookies_file
    elif download.cookies_from_browser:
        options["cookiesfrombrowser"] = (download.cookies_from_browser,)
    return options


def explain_download_error(message: str, logged_in: bool) -> str:
    """Uebersetzt die haeufigen Faelle in etwas, womit man weiterarbeiten kann."""
    clean = ANSI_RE.sub("", message).strip().removeprefix("ERROR:").strip()
    lowered = clean.lower()
    if "confirm your age" in lowered or "age-restricted" in lowered:
        if logged_in:
            return (
                "Dieses Video ist altersbeschränkt, und die hinterlegte Anmeldung wurde nicht "
                "akzeptiert. Sie ist vielleicht abgelaufen oder das Konto ist nicht altersbestätigt. "
                "Exportiere die Cookies neu oder gib die Videodatei direkt an."
            )
        return (
            "Dieses Video ist altersbeschränkt, YouTube liefert es nur mit Anmeldung. Hinterlege "
            "unter Einstellungen eine Anmeldung oder gib die Videodatei direkt an."
        )
    if "not a bot" in lowered:
        return (
            "YouTube hält diesen Zugriff für einen Bot und verlangt eine Anmeldung. Versuche es "
            "später noch einmal, hinterlege unter Einstellungen eine Anmeldung oder gib die "
            "Videodatei direkt an."
        )
    if "could not find" in lowered and "cookies database" in lowered:
        return (
            "Der eingestellte Browser hat auf diesem PC kein Profil. Wähle unter Einstellungen einen "
            "anderen Browser oder nutze eine Cookie-Datei."
        )
    if "could not copy" in lowered and "cookie" in lowered or "dpapi" in lowered:
        return (
            "Der Browser gibt seine Cookies nicht frei. Chrome und Edge sperren sie unter Windows "
            "häufig. Nutze unter Einstellungen stattdessen eine Cookie-Datei."
        )
    if "private video" in lowered or "members-only" in lowered or "join this channel" in lowered:
        return f"Dieses Video ist nicht öffentlich zugänglich. Originalmeldung: {clean}"
    return f"Download fehlgeschlagen: {clean}"


def _download_video(url: str, paths: VodPaths, progress: Progress, settings: Settings) -> dict:
    import yt_dlp

    def hook(status: dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if total:
                progress("download", status.get("downloaded_bytes", 0) / total)

    options = {
        **ytdlp_options(settings),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        # H.264 dekodiert auf dem Prozessor am schnellsten und laeuft in jedem Browser.
        "format_sort": ["res:1080", "fps", "vcodec:h264", "acodec:m4a"],
        "merge_output_format": "mp4",
        "outtmpl": str(paths.root / "source.%(ext)s"),
        "continuedl": True,
        "progress_hooks": [hook],
    }
    progress("download", 0.0)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.YoutubeDLError as exc:
        raise RuntimeError(explain_download_error(str(exc), has_login(settings))) from exc

    # requested_downloads nennt die fertige Datei; im Ordner koennten noch Teilformate liegen.
    downloads = (info or {}).get("requested_downloads") or []
    filepath = downloads[0].get("filepath") if downloads else None
    if not filepath or not Path(filepath).is_file():
        raise RuntimeError(f"Keine Videodatei nach dem Download gefunden: {url}")

    return {
        "video": str(Path(filepath)),
        "title": info.get("title") or "",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "chapters": [
            {"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title", "")}
            for c in info.get("chapters") or []
        ],
        "platform": info.get("extractor_key") or "",
    }


def _download_chat(url: str, paths: VodPaths, settings: Settings) -> None:
    """Eigener Lauf: Scheitert der Chat, darf das weder Video noch Clips gefaehrden.

    Fehlt das Replay, faellt nur das Chat-Signal weg und die anderen Signale uebernehmen.
    """
    import yt_dlp

    options = {
        **ytdlp_options(settings),
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": CHAT_LANGS,
        "outtmpl": str(paths.root / "source.%(ext)s"),
        "ignoreerrors": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except yt_dlp.utils.YoutubeDLError as exc:
        print(f"Chat-Replay nicht verfuegbar: {exc}")


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


def set_time_range(paths: VodPaths, time_range: list[float] | None) -> None:
    """Merkt den manuellen Zeitbereich. Bei einer Aenderung wird die Auswahl neu aufgebaut."""
    meta = read_json(paths.meta)
    if meta.get("time_range") == time_range:
        return
    meta["time_range"] = time_range
    write_json(paths.meta, meta)
    paths.candidates.unlink(missing_ok=True)


def ingest(
    source: str,
    settings: Settings,
    streamer: str | None = None,
    progress: Progress = lambda step, value: None,
) -> VodPaths:
    """Idempotent: bereits vorhandene Ergebnisse werden wiederverwendet."""
    source = source.strip().strip('"')
    problem = link_problem(source)
    if problem:
        raise ValueError(problem)
    paths = VodPaths.for_vod(settings.paths.data_dir, vod_id_for(source))

    if paths.meta.exists():
        meta = read_json(paths.meta)
        if streamer and meta.get("streamer") != streamer:
            meta["streamer"] = streamer
            write_json(paths.meta, meta)
    else:
        if is_url(source):
            meta = _download_video(source, paths, progress, settings)
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
        if is_url(source):
            progress("chat", 0.0)
            _download_chat(source, paths, settings)
        meta["chat_messages"] = _collect_chat(paths)
        write_json(paths.meta, meta)

    if not paths.audio.exists():
        progress("audio", 0.0)
        tmp = paths.audio.with_name("audio.tmp.wav")
        run_ffmpeg(["-i", meta["video"], "-vn", "-ac", "1", "-ar", "16000", str(tmp)])
        tmp.replace(paths.audio)
        progress("audio", 1.0)

    return paths
