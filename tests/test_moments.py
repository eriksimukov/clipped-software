import json
import shutil
import subprocess

import httpx
import numpy as np
import pytest

from clipped.config import Llm, Settings
from clipped.pipeline import moments as moments_module
from clipped.pipeline import select_clips as select_module
from clipped.pipeline.candidates import (
    candidates_from_moments,
    teaser_from_line,
    window_from_moment,
)
from clipped.pipeline.common import VideoInfo, VodPaths, read_json, write_json
from clipped.pipeline.facecam import best_game_crop, persistent_edges
from clipped.pipeline.lines import build_lines, reaction_flags
from clipped.pipeline.llm import Assessment
from clipped.pipeline.moments import (
    Moment,
    build_chunks,
    clean_moment,
    format_chunk,
    merge_moments,
    scan_chunk,
    scan_moments,
)
from clipped.pipeline.signals import cam_motion, motion_per_second

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _word(word, start, end=None):
    return {"word": word, "start": start, "end": end if end is not None else start + 0.3}


def _lines(count, length=3.0, gap=0.2):
    return [
        {"start": i * (length + gap), "end": i * (length + gap) + length, "text": f"line {i}"}
        for i in range(count)
    ]


def _moment(start, end, best=None, strength=8, story="He finds out something specific"):
    return {
        "story": story,
        "start_line": start,
        "end_line": end,
        "best_line": start if best is None else best,
        "strength": strength,
    }


# Zeilen


def test_lines_break_at_pauses_and_sentence_ends():
    segments = [
        {
            "words": [
                _word("oh", 0.0),
                _word("my", 0.3),
                _word("god.", 0.6),
                _word("look", 1.0),
                _word("here", 1.3),
                _word("later", 3.0),
            ]
        }
    ]
    assert [line["text"] for line in build_lines(segments)] == ["oh my god.", "look here", "later"]


def test_lines_have_a_maximum_length():
    words = [_word(f"w{i}", i * 0.2, i * 0.2 + 0.1) for i in range(40)]
    assert all(len(line["text"].split()) <= 16 for line in build_lines([{"words": words}]))


def test_lines_join_across_segments_when_speech_continues():
    segments = [{"words": [_word("first", 0.0)]}, {"words": [_word("second", 0.35)]}]
    assert [line["text"] for line in build_lines(segments)] == ["first second"]


def test_reaction_flags_use_the_highest_value_in_the_line():
    lines = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 3.0, "end": 4.0, "text": "b"}]
    assert reaction_flags(lines, [0.1, 0.9, 0.1, 0.1, 0.1], 0.45) == [True, False]
    assert reaction_flags(lines, [], 0.45) == [False, False]


# Abschnitte und Modellantworten


def test_chunks_overlap_and_cover_everything():
    chunks = build_chunks(130)
    assert chunks[0] == (0, 60)
    assert chunks[1][0] == 60 - moments_module.CHUNK_OVERLAP_LINES
    assert chunks[-1][1] == 130
    assert build_chunks(10) == [(0, 10)]
    assert build_chunks(0) == []


def test_format_chunk_marks_reactions_with_real_line_numbers():
    text = format_chunk(_lines(5), [False, False, True, False, False], 1, 3)
    assert text.splitlines() == ["1   line 1", "2 ! line 2"]


def test_clean_moment_fixes_lines_and_drops_weak_ones():
    cleaned = clean_moment(
        Moment(story="s", start_line=30, end_line=5, best_line=99, strength=8), 10, 40
    )
    assert (cleaned.start_line, cleaned.end_line, cleaned.best_line) == (10, 30, 10)
    assert (
        clean_moment(Moment(story="s", start_line=1, end_line=2, best_line=1, strength=3), 0, 9)
        is None
    )
    assert (
        clean_moment(Moment(story=" ", start_line=1, end_line=2, best_line=1, strength=9), 0, 9)
        is None
    )
    assert (
        clean_moment(Moment(story="s", start_line=50, end_line=60, best_line=50, strength=9), 0, 9)
        is None
    )


def test_merge_keeps_the_stronger_of_overlapping_moments():
    merged = merge_moments(
        [_moment(10, 14, strength=6), _moment(11, 15, strength=9), _moment(40, 42, strength=5)]
    )
    assert [(m["start_line"], m["strength"]) for m in merged] == [(11, 9), (40, 5)]


def _chat_response(content):
    return httpx.Response(
        200,
        text=json.dumps({"message": {"content": content}}),
        request=httpx.Request("POST", "http://x"),
    )


def test_scan_chunk_reads_moments_and_retries_invalid_answers(monkeypatch):
    answers = [
        _chat_response("kein json"),
        _chat_response(
            json.dumps(
                {
                    "moments": [
                        {
                            "story": "He finds out",
                            "start_line": 2,
                            "end_line": 3,
                            "best_line": 3,
                            "strength": 8,
                        }
                    ]
                }
            )
        ),
    ]
    payloads = []

    def fake_post(cfg, payload):
        payloads.append(payload)
        return answers.pop(0)

    monkeypatch.setattr(moments_module, "post_chat", fake_post)
    found = scan_chunk(_lines(5), [False] * 5, 0, 5, Llm())
    assert [m.best_line for m in found] == [3]
    assert len(payloads) == 2
    assert "3   line 3" in payloads[0]["messages"][1]["content"]


def test_scan_moments_goes_through_all_chunks(monkeypatch):
    seen = []

    def fake_chunk(lines, flags, start, end, cfg):
        seen.append((start, end))
        return [
            Moment(story="s", start_line=start, end_line=start + 1, best_line=start, strength=7)
        ]

    monkeypatch.setattr(moments_module, "scan_chunk", fake_chunk)
    values = []
    found = scan_moments(_lines(130), [False] * 130, Llm(), values.append)
    assert seen == build_chunks(130)
    assert len(found) == len(seen)
    assert values[-1] == 1.0


def test_moment_schema_limits_results():
    with pytest.raises(ValueError):
        moments_module.ScanResult.model_validate(
            {"moments": [_moment(1, 2, strength=5) for _ in range(5)]}
        )


# Fenster


def test_short_moment_gets_reaction_and_context():
    lines = _lines(20)
    first, last = window_from_moment(lines, _moment(10, 10), 12, 60)
    assert first < 10 < last
    assert lines[last]["end"] - lines[10]["end"] >= 4
    assert lines[10]["start"] - lines[first]["start"] >= 4
    assert lines[last]["end"] - lines[first]["start"] >= 12


def test_window_stops_at_long_pause():
    lines = [
        {"start": 0.0, "end": 3.0, "text": "before"},
        {"start": 30.0, "end": 33.0, "text": "a"},
        {"start": 33.2, "end": 36.0, "text": "b"},
        {"start": 36.2, "end": 39.0, "text": "c"},
        {"start": 90.0, "end": 93.0, "text": "after"},
    ]
    assert window_from_moment(lines, _moment(1, 3, 2), 12, 60) == (1, 3)


def test_too_long_moment_is_trimmed_around_the_best_line():
    lines = _lines(40)
    first, last = window_from_moment(lines, _moment(0, 39, 20), 12, 30)
    assert first <= 20 <= last
    assert lines[last]["end"] - lines[first]["start"] <= 30


def test_isolated_tiny_moment_is_dropped():
    lines = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 60.0, "end": 62.0, "text": "b"},
        {"start": 120.0, "end": 121.0, "text": "c"},
    ]
    assert window_from_moment(lines, _moment(1, 1), 12, 60) is None


def test_teaser_is_the_best_line_and_short_lines_get_the_next_one():
    lines = [
        {"start": 10.0, "end": 10.5, "text": "no way"},
        {"start": 10.7, "end": 13.0, "text": "they added streamers"},
        {"start": 13.2, "end": 16.0, "text": "more"},
    ]
    assert teaser_from_line(lines, 0, 2) == (10.0, 13.0)
    assert teaser_from_line(lines, 1, 2) == (10.7, 13.0)


def test_teaser_never_runs_past_the_clip():
    lines = [{"start": 10.0, "end": 10.5, "text": "a"}, {"start": 10.7, "end": 13.0, "text": "b"}]
    assert teaser_from_line(lines, 0, 0) == (10.0, 10.5)


def test_candidates_carry_story_teaser_and_energy():
    lines = _lines(30)
    curve = [0.1] * 100
    reaction = [0.0] * 100
    reaction[32:35] = [0.9, 0.9, 0.9]
    moments = [_moment(10, 11, best=10, strength=9, story="Specific thing")]
    [candidate] = candidates_from_moments(moments, lines, curve, reaction, 12, 60, 10)
    assert candidate["story"] == "Specific thing"
    assert candidate["peak"] == lines[10]["start"]
    assert candidate["teaser_start"] == lines[10]["start"]
    assert candidate["reaction_score"] == pytest.approx(0.9)
    assert "line 10" in candidate["text"]


def test_candidates_drop_overlaps_and_respect_limit():
    lines = _lines(60)
    moments = [
        _moment(10, 12, strength=9),
        _moment(11, 13, strength=5),
        _moment(40, 42, strength=7),
    ]
    result = candidates_from_moments(moments, lines, [0.1] * 300, [], 12, 60, 10)
    assert [c["strength"] for c in result] == [9, 7]
    assert len(candidates_from_moments(moments, lines, [0.1] * 300, [], 12, 60, 1)) == 1


def test_candidates_ignore_lines_outside_the_transcript():
    assert candidates_from_moments([_moment(50, 60)], _lines(5), [0.1] * 30, [], 12, 60, 10) == []


# Ablauf


def _assessment():
    return Assessment(
        reason="r",
        reaction=7,
        hook=7,
        payoff=7,
        emotion=7,
        standalone=7,
        gta6_relevance=8,
        hooks=["a", "b", "c"],
        title="t",
        description="d",
        hashtags=[],
    )


def _spoken_segments(count=30):
    return [
        {
            "start": i * 3.0,
            "end": i * 3.0 + 2.5,
            "text": f"sentence {i}.",
            "words": [
                {"start": i * 3.0, "end": i * 3.0 + 1.0, "word": "sentence", "p": 0.9},
                {"start": i * 3.0 + 1.1, "end": i * 3.0 + 2.5, "word": f"{i}.", "p": 0.9},
            ],
        }
        for i in range(count)
    ]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    paths = VodPaths.for_vod(tmp_path, "x")
    (paths.audio).write_bytes(b"")
    write_json(paths.transcript, {"segments": _spoken_segments()})
    write_json(paths.chat, [])
    write_json(
        paths.meta,
        {
            "video": "x.mp4",
            "duration": 90.0,
            "title": "GTA 6",
            "cam": {"rect": None, "layout": "gameplay"},
        },
    )
    write_json(paths.signals, {"curve": [0.2] * 91, "reaction": [], "parts": {}})
    monkeypatch.setattr(
        select_module, "probe_video", lambda path: VideoInfo(1920, 1080, True, 90.0)
    )
    monkeypatch.setattr(select_module, "assess", lambda *a, **k: _assessment())
    monkeypatch.setattr(select_module, "ensure_running", lambda cfg: None)
    monkeypatch.setattr(select_module, "unload", lambda cfg: None)
    return paths


def test_pipeline_builds_candidates_from_model_moments(pipeline, monkeypatch):
    calls = []

    def fake_scan(lines, flags, cfg, progress):
        calls.append(len(lines))
        return [_moment(5, 8, best=6, strength=9, story="Specific thing")]

    monkeypatch.setattr(select_module, "scan_moments", fake_scan)
    monkeypatch.setattr(select_module, "MIN_MOMENTS", 1)
    clips = select_module.select_clips(pipeline, Settings())
    assert [c["story"] for c in clips] == ["Specific thing"]
    assert clips[0]["status"] == "assessed"
    assert calls == [30]
    assert read_json(pipeline.moments)[0]["story"] == "Specific thing"


def test_saved_moments_are_not_scanned_again(pipeline, monkeypatch):
    write_json(pipeline.moments, [_moment(5, 8, best=6, story="Saved")])

    def no_scan(*_args):
        raise AssertionError("darf nicht erneut lesen")

    monkeypatch.setattr(select_module, "scan_moments", no_scan)
    monkeypatch.setattr(select_module, "MIN_MOMENTS", 1)
    assert select_module.select_clips(pipeline, Settings())[0]["story"] == "Saved"


def test_too_few_moments_are_filled_up_from_signals(pipeline, monkeypatch):
    monkeypatch.setattr(select_module, "scan_moments", lambda *a: [])
    clips = select_module.select_clips(pipeline, Settings())
    assert clips
    assert all(c["story"] == "" and c["strength"] == select_module.FALLBACK_STRENGTH for c in clips)


def test_no_gta6_means_no_candidates_and_no_model_call(pipeline, monkeypatch):
    meta = read_json(pipeline.meta)
    meta["title"] = "Fortnite"
    write_json(pipeline.meta, meta)

    def no_scan(*_args):
        raise AssertionError("ohne GTA 6 kein Modellaufruf")

    monkeypatch.setattr(select_module, "scan_moments", no_scan)
    assert select_module.select_clips(pipeline, Settings()) == []


# Overlays im Bildausschnitt


def _stream_with_chat(count=24):
    """Spiel in jedem Frame anders hell, oben rechts bleibt der Chat mit festen Kanten."""
    rng = np.random.default_rng(1)
    frames = np.stack(
        [np.full((540, 960), level, dtype=np.uint8) for level in rng.integers(40, 200, count)]
    )
    # Chat: Text-Zeilen ab x=696, y bis 45
    for y in range(4, 45, 8):
        frames[:, y : y + 3, 696:] = 255
    return frames


def test_persistent_edges_find_the_overlay_only():
    presence = persistent_edges(_stream_with_chat())
    assert presence[3:45, 700:900].max() > 0.9
    assert presence[100:400, 100:900].max() == 0.0


def test_game_crop_avoids_chat_and_facecam():
    presence = persistent_edges(_stream_with_chat())
    cam = (0, 341, 275, 199)
    crop = best_game_crop(presence, cam, 1080 / 1280)
    height = 540 * (1 - crop["top"] - crop["bottom"])
    width = height * 1080 / 1280
    left = crop["center"] * 960 - width / 2
    assert left >= cam[2] - 2
    assert left + width <= 697 or crop["top"] * 540 >= 45


def test_game_crop_stays_full_size_without_overlays():
    presence = np.zeros((540, 960))
    assert best_game_crop(presence, None, 1080 / 1280) == {"center": 0.5, "top": 0.0, "bottom": 0.0}


def test_game_crop_ignores_single_noisy_pixels():
    presence = np.zeros((540, 960))
    presence[300, 480] = 1.0
    assert best_game_crop(presence, None, 1080 / 1280) == {"center": 0.5, "top": 0.0, "bottom": 0.0}


# Bewegung in der Facecam


def test_scene_cuts_do_not_count_as_reaction():
    cam = np.array([1.0, 1.0, 1.0, 1.0, 30.0, 30.0, 1.0, 1.0])
    outside = np.array([1.0, 1.0, 1.0, 1.0, 40.0, 1.0, 1.0, 1.0])
    seconds = motion_per_second(cam, outside)
    assert len(seconds) == 2
    # Der erste Sprung ist ein Schnitt, der zweite echte Bewegung nur in der Cam.
    assert seconds[1] == pytest.approx((1.0 + 30.0 + 1.0 + 1.0) / 4)


def test_motion_of_empty_video_is_empty():
    assert len(motion_per_second(np.array([]), np.array([]))) == 0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg fehlt")
def test_cam_motion_finds_movement_inside_the_facecam(tmp_path):
    """Ein Block bewegt sich nur zwischen Sekunde 3 und 5 in der Facecam."""
    video = tmp_path / "cam.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=gray:s=1280x720:r=30:d=8",
            "-vf", _box_filter(),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )  # fmt: skip
    info = VideoInfo(1280, 720, False, 8.0)
    values = cam_motion(str(video), [0, 480, 400, 240], info)
    assert len(values) >= 7
    assert values[3:5].mean() > values[6:7].mean() + 1
    assert values[3:5].mean() > values[0:2].mean() + 1


def _box_filter() -> str:
    # Springt in Sekunde 3 bis 5 hin und her, sonst steht der Block still.
    gate = "between(t,3,5)*eq(mod(floor(t*3),2),1)"
    return (
        f"drawbox=x=40:y=520:w=120:h=120:color=white:t=fill:enable='not({gate})',"
        f"drawbox=x=240:y=520:w=120:h=120:color=white:t=fill:enable='{gate}'"
    )
