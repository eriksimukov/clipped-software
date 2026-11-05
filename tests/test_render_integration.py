"""Echte FFmpeg-Laeufe auf kurzen, erzeugten Videos. Prueft das fertige Ergebnis, nicht die Filter."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from clipped.config import Channel, Settings
from clipped.pipeline.common import probe_video
from clipped.pipeline.render import _correct_loudness, _run_with_progress, render_clip
from clipped.pipeline.spec import ClipSpec, Effects

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg fehlt")

WORDS = [
    "okay",
    "chat",
    "look",
    "at",
    "this",
    "helicopter.",
    "no",
    "way!",
    "that",
    "was",
    "insane.",
]


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("source")
    video = folder / "stream with space.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=30",
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )  # fmt: skip
    words = [
        {"start": 5.0 + i * 0.6, "end": 5.5 + i * 0.6, "word": word, "p": 0.9}
        for i, word in enumerate(WORDS)
    ]
    segments = [{"start": 5.0, "end": 11.5, "text": " ".join(WORDS), "words": words}]
    return {"meta": {"video": str(video)}, "segments": segments}


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.paths.data_dir = tmp_path
    settings.render.video_encoder = "libx264"
    settings.branding.channels = [
        Channel(platform="tiktok", handle="@me"),
        Channel(platform="kick", handle="@me"),
    ]
    return settings


def _spec(**values) -> ClipSpec:
    base = {
        "id": "c1",
        "start": 4.0,
        "end": 14.0,
        "peak": 8.0,
        "hook_text": "No way he did that",
        "layout": "split",
        "cam_rect": (40, 480, 320, 180),
        "credit": "Streamer {x}",
    }
    return ClipSpec(**{**base, **values})


def _streams(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _loudness(path: Path) -> float:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", result.stderr)[-1])


def _true_peak(path: Path) -> float:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", result.stderr)[-1])


@pytest.mark.parametrize(
    "layout, cam",
    [
        ("split", (40, 480, 320, 180)),
        ("gameplay", (40, 480, 320, 180)),
        ("gameplay", None),
        ("blur", None),
    ],
)
def test_render_produces_upload_ready_video(tmp_path, source, layout, cam):
    out = tmp_path / "export dir" / "clip.mp4"
    values = []
    render_clip(
        _spec(layout=layout, cam_rect=cam),
        source["meta"],
        source["segments"],
        _settings(tmp_path),
        out,
        tmp_path / "work",
        on_progress=values.append,
    )
    data = _streams(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert video["r_frame_rate"] == "30/1"
    assert (audio["codec_name"], audio["sample_rate"], audio["channels"]) == ("aac", "48000", 2)
    # Clip (10 s) + Endkarte (2,5 s). Er beginnt fast mit dem Hoehepunkt, also ohne Teaser.
    assert 12.0 < float(data["format"]["duration"]) < 13.5
    assert values[-1] == 1.0
    assert not list(out.parent.glob("*.tmp.mp4"))


def test_loudness_hits_target(tmp_path, source):
    out = tmp_path / "clip.mp4"
    render_clip(
        _spec(), source["meta"], source["segments"], _settings(tmp_path), out, tmp_path / "work"
    )
    assert abs(_loudness(out) - -14.0) < 1.0
    assert _true_peak(out) < -0.5


def test_preview_is_small_and_without_teaser_shorter(tmp_path, source):
    out = tmp_path / "clip.mp4"
    render_clip(
        _spec(teaser=False, effects=Effects(zoom=False, shake=False, flash=False, sfx=False)),
        source["meta"],
        source["segments"],
        _settings(tmp_path),
        out,
        tmp_path / "work",
        preview=True,
    )
    info = probe_video(out)
    assert (info.width, info.height) == (540, 960)
    assert 12.0 < info.duration < 13.0


def test_source_without_audio_still_renders(tmp_path):
    video = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=8",
            "-c:v", "libx264", "-preset", "ultrafast", str(video),
        ],
        check=True,
    )  # fmt: skip
    out = tmp_path / "clip.mp4"
    render_clip(
        _spec(start=1.0, end=6.0, peak=3.0, cam_rect=None, layout="gameplay"),
        {"video": str(video)},
        [],
        _settings(tmp_path),
        out,
        tmp_path / "work",
    )
    assert probe_video(out).has_audio


def test_cancel_stops_ffmpeg_and_leaves_no_partial_file(tmp_path, source):
    class Cancelled(Exception):
        pass

    def cancel(_value):
        raise Cancelled

    out = tmp_path / "export" / "clip.mp4"
    with pytest.raises(Cancelled):
        render_clip(
            _spec(), source["meta"], source["segments"], _settings(tmp_path), out, tmp_path / "work",
            on_progress=cancel,
        )  # fmt: skip
    assert not out.exists()
    assert not list(out.parent.glob("*"))


def test_cancel_releases_output_file_immediately(tmp_path):
    """Direkt nach dem Abbruch muss die Datei loeschbar sein; ein Shim-Kill allein schafft das nicht."""
    output = tmp_path / "long.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y", "-progress", "pipe:1",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=120",
        "-c:v", "libx264", "-preset", "ultrafast", str(output),
    ]  # fmt: skip

    def cancel(_value):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_with_progress(command, tmp_path, 120.0, cancel)
    output.unlink(missing_ok=True)
    assert not output.exists()


def test_failed_render_reports_ffmpeg_message(tmp_path, source):
    with pytest.raises(RuntimeError, match="Rendern fehlgeschlagen"):
        settings = _settings(tmp_path)
        settings.render.video_encoder = "gibt_es_nicht"
        render_clip(
            _spec(),
            source["meta"],
            source["segments"],
            settings,
            tmp_path / "c.mp4",
            tmp_path / "work",
        )


def test_quiet_source_is_corrected_to_target(tmp_path):
    """Der Filter im Rendern allein trifft bei leisem Ton nicht; die zweite Korrektur schon."""
    video = tmp_path / "quiet.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=20",
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=20,volume=0.01",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video),
        ],
        check=True,
    )  # fmt: skip
    out = tmp_path / "clip.mp4"
    render_clip(
        _spec(start=1.0, end=15.0, peak=5.0, cam_rect=None, layout="gameplay"),
        {"video": str(video)},
        [],
        _settings(tmp_path),
        out,
        tmp_path / "work",
    )
    assert abs(_loudness(out) - -14.0) < 1.0
    assert _true_peak(out) < -0.5
    assert not list(out.parent.glob("*.loud.mp4"))
    assert probe_video(out).width == 1080


def test_silent_clip_is_not_amplified(tmp_path):
    source = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=320x240:r=15:d=3",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
    )  # fmt: skip
    before = source.stat().st_mtime
    _correct_loudness(source, -14.0)
    assert source.stat().st_mtime == before
