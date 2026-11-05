import functools
import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from clipped.config import Settings
from clipped.pipeline.common import VodPaths, read_json, vod_id_for, write_json
from clipped.pipeline.ingest import (
    _download_chat,
    explain_download_error,
    ingest,
    is_url,
    link_problem,
    set_time_range,
    ytdlp_options,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _settings(data_dir: Path) -> Settings:
    settings = Settings()
    settings.paths.data_dir = data_dir
    return settings


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg fehlt")
    folder = tmp_path_factory.mktemp("served")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=15:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=3",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(folder / "stream.mp4"),
        ],
        check=True,
    )
    return folder


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture
def video_url(video_dir):
    handler = functools.partial(_QuietHandler, directory=str(video_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/stream.mp4"
    server.shutdown()
    server.server_close()


def test_is_url():
    assert is_url("https://twitch.tv/videos/1")
    assert is_url(" HTTP://x.test")
    assert not is_url("G:/videos/a.mp4")
    assert not is_url("file:///c:/secret.mp4")


def test_local_file_stays_in_place_and_gets_audio(tmp_path, video_dir):
    source = video_dir / "stream.mp4"
    paths = ingest(f'"{source}"', _settings(tmp_path), "Name")
    meta = read_json(paths.meta)
    assert Path(meta["video"]) == source.resolve()
    assert meta["streamer"] == "Name"
    assert meta["title"] == "stream"
    assert meta["chat_messages"] == 0
    assert 2.5 < meta["duration"] < 3.5
    assert paths.audio.stat().st_size > 1000


def test_second_ingest_reuses_results_and_updates_streamer(tmp_path, video_dir):
    source = str(video_dir / "stream.mp4")
    settings = _settings(tmp_path)
    paths = ingest(source, settings, "First")
    modified = paths.audio.stat().st_mtime
    ingest(source, settings, "Second")
    assert paths.audio.stat().st_mtime == modified
    assert read_json(paths.meta)["streamer"] == "Second"


def test_missing_local_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest("C:/gibt/es/nicht.mp4", _settings(tmp_path))


def test_download_from_link_end_to_end(tmp_path, video_url):
    """Echter yt-dlp-Lauf gegen einen lokalen Server, ohne externe Plattform."""
    steps = []
    paths = ingest(video_url, _settings(tmp_path), None, lambda step, value: steps.append(step))
    meta = read_json(paths.meta)
    assert Path(meta["video"]).is_file()
    assert Path(meta["video"]).parent == paths.root
    assert meta["source"] == video_url
    assert meta["chat_messages"] == 0
    assert paths.audio.exists()
    assert "download" in steps
    assert "chat" in steps


def test_failed_download_raises_readable_error(tmp_path):
    with pytest.raises(RuntimeError, match="Download fehlgeschlagen"):
        ingest("http://127.0.0.1:9/nichts.mp4", _settings(tmp_path))


def test_unreachable_chat_service_does_not_raise(tmp_path):
    paths = VodPaths.for_vod(tmp_path, "x")
    _download_chat("http://127.0.0.1:9/nichts", paths, Settings())
    assert list(paths.root.iterdir()) == []


def test_chat_replay_is_collected(tmp_path, video_dir):
    source = str(video_dir / "stream.mp4")
    settings = _settings(tmp_path)
    paths = VodPaths.for_vod(tmp_path, vod_id_for(source))
    (paths.root / "source.rechat.json").write_text(
        json.dumps({"comments": [{"content_offset_seconds": 1.0, "message": {"body": "hi"}}]}),
        encoding="utf-8",
    )
    ingest(source, settings)
    assert read_json(paths.meta)["chat_messages"] == 1
    assert read_json(paths.chat) == [{"t": 1.0, "text": "hi"}]


def test_broken_chat_file_is_ignored(tmp_path, video_dir):
    source = str(video_dir / "stream.mp4")
    paths = VodPaths.for_vod(tmp_path, vod_id_for(source))
    (paths.root / "source.rechat.json").write_text("{ kaputt", encoding="utf-8")
    ingest(source, _settings(tmp_path))
    assert read_json(paths.chat) == []


def test_changing_time_range_discards_selection_only_when_different(tmp_path):
    paths = VodPaths.for_vod(tmp_path, "x")
    write_json(paths.meta, {"time_range": [1, 2]})
    write_json(paths.candidates, [])
    set_time_range(paths, [1, 2])
    assert paths.candidates.exists()
    set_time_range(paths, [1, 3])
    assert not paths.candidates.exists()
    assert read_json(paths.meta)["time_range"] == [1, 3]


# Links und Anmeldung


def test_link_with_extra_options_is_refused(tmp_path):
    source = "https://www.youtube.com/watch?v=abc --cookies-from-browser"
    assert "Leerzeichen" in link_problem(source)
    assert link_problem("https://www.youtube.com/watch?v=abc") is None
    assert link_problem("G:/mein ordner/stream.mp4") is None
    with pytest.raises(ValueError, match="Leerzeichen"):
        ingest(source, _settings(tmp_path))


def test_no_login_is_used_by_default():
    options = ytdlp_options(Settings())
    assert "cookiefile" not in options
    assert "cookiesfrombrowser" not in options


def test_cookie_file_wins_over_browser():
    settings = Settings()
    settings.download.cookies_from_browser = "firefox"
    assert ytdlp_options(settings)["cookiesfrombrowser"] == ("firefox",)
    settings.download.cookies_file = "C:/c.txt"
    options = ytdlp_options(settings)
    assert options["cookiefile"] == "C:/c.txt"
    assert "cookiesfrombrowser" not in options


def test_options_enable_javascript_and_disable_colors():
    options = ytdlp_options(Settings())
    assert set(options["js_runtimes"]) == {"deno", "node"}
    assert options["color"] == "never"


def test_unknown_browser_is_rejected():
    with pytest.raises(ValueError, match="Unbekannter Browser"):
        Settings.model_validate({"download": {"cookies_from_browser": "netscape"}})


AGE_ERROR = (
    "\x1b[0;31mERROR:\x1b[0m [youtube] abc: Sign in to confirm your age. "
    "This video may be inappropriate for some users."
)


def test_age_restriction_is_explained_without_color_codes():
    message = explain_download_error(AGE_ERROR, logged_in=False)
    assert "altersbeschränkt" in message
    assert "Einstellungen" in message
    assert "\x1b" not in message


def test_age_restriction_with_login_points_to_expired_cookies():
    assert "nicht akzeptiert" in explain_download_error(AGE_ERROR, logged_in=True)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ERROR: [youtube] abc: Sign in to confirm you\u2019re not a bot", "Bot"),
        ("ERROR: Could not copy Chrome cookie database. See link", "Cookie-Datei"),
        ("ERROR: Failed to decrypt with DPAPI. See link", "Cookie-Datei"),
        ("ERROR: [youtube] abc: Private video. Sign in", "nicht öffentlich"),
        ("ERROR: [generic] x: Unable to download webpage", "Download fehlgeschlagen: [generic]"),
    ],
)
def test_common_download_errors_are_explained(raw, expected):
    assert expected in explain_download_error(raw, logged_in=False)
