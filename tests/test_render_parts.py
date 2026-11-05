import itertools
import re

import numpy as np
import pytest

from clipped.pipeline.censor import censor_text, contains_slur
from clipped.pipeline.common import VideoInfo
from clipped.pipeline.effects import (
    PartEffects,
    Pulse,
    flash_filter,
    motion_filter,
    plan_effects,
    rhythm_pulses,
    zoom_expr,
)
from clipped.pipeline.export import clean_hashtags, export_text, slugify
from clipped.pipeline.facecam import Face, cluster_faces, find_cam_rect, locate_cam, sample_times
from clipped.pipeline.llm import Assessment, clean_assessment
from clipped.pipeline.render import (
    SfxInput,
    build_filter_graph,
    build_parts,
    fit_aspect,
    game_crop,
    resolve_layout,
    teaser_range,
)
from clipped.pipeline.sfx import available_sounds, ensure_starter_pack, pick_sound
from clipped.pipeline.spec import ClipSpec, Effects, TextEdit, spec_from_candidate
from clipped.pipeline.subtitles import (
    Part,
    apply_text_edits,
    ass_color,
    ass_time,
    build_ass,
    chunk_words,
    flatten_words,
    wrap_hook,
)

INFO = VideoInfo(width=1920, height=1080, has_audio=True, duration=100.0)


def _word(word, start, end=None):
    return {"word": word, "start": start, "end": end if end is not None else start + 0.3}


def _spec(**values):
    base = {"id": "c000", "start": 10.0, "end": 40.0, "peak": 25.0}
    return ClipSpec(**{**base, **values})


# Untertitel


def test_ass_time_format_and_rounding():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(61.5) == "0:01:01.50"
    assert ass_time(3661.256) == "1:01:01.26"
    assert ass_time(-1) == "0:00:00.00"


def test_ass_color_swaps_channels():
    assert ass_color("#FFE500") == "&H00E5FF&"


def test_chunks_respect_word_and_char_limits():
    words = [_word(w, i * 0.4) for i, w in enumerate(["a", "b", "c", "d", "e"])]
    assert [len(c) for c in chunk_words(words)] == [3, 2]
    long_words = [_word("extraordinary", 0.0), _word("situations", 0.4)]
    assert [len(c) for c in chunk_words(long_words)] == [1, 1]


def test_chunks_break_on_sentence_end_and_pause():
    words = [_word("done.", 0.0), _word("next", 0.3), _word("after", 2.0)]
    assert [[w["text"] for w in c] for c in chunk_words(words)] == [["DONE"], ["NEXT"], ["AFTER"]]


def test_chunks_strip_ass_control_characters():
    chunks = chunk_words([_word(r"{\pos(1,1)}hi", 0.0), _word("{}", 0.2)])
    assert [w["text"] for c in chunks for w in c] == ["POS(1,1)HI"]


def test_apply_text_edit_replaces_words_in_range():
    words = [_word("foo", 1.0), _word("bar", 2.0), _word("baz", 3.0)]
    result = apply_text_edits(words, [TextEdit(start=1.0, end=2.5, text="hello big world")])
    assert [w["word"] for w in result] == ["hello", "big", "world", "baz"]
    assert result[0]["start"] == 1.0
    assert result[2]["end"] == pytest.approx(2.5)


def test_apply_text_edit_with_empty_text_removes_words():
    words = [_word("foo", 1.0), _word("baz", 3.0)]
    assert [
        w["word"] for w in apply_text_edits(words, [TextEdit(start=0.5, end=2, text="  ")])
    ] == ["baz"]


def test_flatten_words_filters_by_start():
    segments = [{"start": 0, "end": 5, "text": "x", "words": [_word("a", 1.0), _word("b", 4.0)]}]
    assert [w["word"] for w in flatten_words(segments, 2.0, 5.0)] == ["b"]


def test_wrap_hook_limits_lines_and_uppercases():
    assert wrap_hook("he jumped the bridge") == r"HE JUMPED THE\NBRIDGE"
    assert (
        wrap_hook("one two three four five six seven eight nine ten eleven twelve thirteen").count(
            r"\N"
        )
        == 2
    )


def _events(ass):
    return [line for line in ass.splitlines() if line.startswith("Dialogue")]


def _subtitle_times(ass):
    pattern = re.compile(r"Dialogue: \d,(\d+):(\d+):([\d.]+),(\d+):(\d+):([\d.]+),Sub")
    times = []
    for line in _events(ass):
        match = pattern.match(line)
        if match:
            g = match.groups()
            times.append(
                (
                    int(g[0]) * 3600 + int(g[1]) * 60 + float(g[2]),
                    int(g[3]) * 3600 + int(g[4]) * 60 + float(g[5]),
                )
            )
    return times


def test_ass_maps_words_to_each_part_without_overlap():
    words = [
        _word(w, 10.0 + i * 0.5)
        for i, w in enumerate(["one", "two", "three", "four", "five", "six"])
    ]
    parts = [Part(11.0, 12.0, 0.0), Part(10.0, 14.0, 1.0)]
    ass = build_ass(parts, words, "split", "my hook", 2.5, "@me", "Streamer")
    times = _subtitle_times(ass)
    # Teaser: Woerter ab 11.0 liegen bei 0.0; Hauptteil: Wort bei 10.0 liegt bei 1.0.
    assert times[0][0] == 0.0
    assert any(start == 1.0 for start, _ in times)
    for (_, end), (start, _) in itertools.pairwise(times):
        assert end <= start + 1e-6
    assert all(end <= 5.0 for _, end in times)


def test_ass_marks_active_word_and_places_hook_and_marks():
    ass = build_ass(
        [Part(0.0, 2.0, 0.0)],
        [_word("hello", 0.0), _word("there", 0.5)],
        "split",
        "hook",
        2.0,
        "@me",
        "Name",
    )
    assert r"{\c&H00E5FF&}HELLO{\c&HFFFFFF&} THERE" in ass
    assert r"HELLO {\c&H00E5FF&}THERE" in ass
    assert r"\an8\pos(540,560)" in ass
    assert "@me" in ass
    assert "via Name" in ass


def test_ass_omits_empty_hook_watermark_and_credit():
    ass = build_ass([Part(0.0, 2.0, 0.0)], [], "split", "", 2.0, "", "")
    assert _events(ass) == []


# Effekte


def test_zoom_expression_contains_every_pulse():
    expr = zoom_expr([Pulse(1, 2, 0.1), Pulse(3, 4, 0.2)], "t")
    assert expr.startswith("1+0.1000*clip((t-1.000)/0.120,0,1)")
    assert "0.2000" in expr


def test_motion_filter_empty_without_effects():
    assert motion_filter(PartEffects(), 30) == ""


def test_motion_filter_has_eight_corner_values():
    result = motion_filter(PartEffects(pulses=[Pulse(0, 1, 0.1)]), 30)
    for name in ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"):
        assert f"{name}='" in result
    assert "eval=frame" in result


def test_rhythm_pulses_stay_away_from_peak():
    pulses = rhythm_pulses(40.0, 20.0, [float(b) for b in range(2, 40, 2)])
    for pulse in pulses:
        assert pulse.end <= 18.5 + 0.01 or pulse.start >= 22.6 - 0.01
        assert pulse.end - pulse.start >= 1.0


def test_plan_places_events_in_output_time():
    parts = [Part(24.0, 26.0, 0.0), Part(10.0, 40.0, 2.0)]
    plan = plan_effects(parts, 25.0, [], Effects(), endcard_start=32.0)
    impact = next(s for s in plan.sfx if s.category == "impact")
    assert impact.time == pytest.approx(2.0 + 15.0)
    assert [f.time for f in plan.flashes] == pytest.approx([2.0, 17.0])
    assert plan.parts[1].shakes[0].start == pytest.approx(15.0)
    assert plan.parts[0].pulses[0].amount == pytest.approx(0.10)


def test_plan_respects_disabled_effects():
    parts = [Part(10.0, 40.0, 0.0)]
    off = Effects(zoom=False, shake=False, flash=False, sfx=False)
    plan = plan_effects(parts, 25.0, [12.0], off, 30.0)
    assert plan.flashes == []
    assert plan.sfx == []
    assert plan.parts[0].pulses == []
    assert plan.parts[0].shakes == []


def test_shake_without_zoom_still_hides_frame_edges():
    plan = plan_effects([Part(10.0, 40.0, 0.0)], 25.0, [], Effects(zoom=False), 30.0)
    assert len(plan.parts[0].pulses) == 1


def test_no_riser_when_peak_is_too_early():
    plan = plan_effects([Part(10.0, 40.0, 0.0)], 10.5, [], Effects(), 30.0)
    assert "riser" not in [s.category for s in plan.sfx]


def test_flash_filter_empty_without_flashes():
    assert flash_filter([]) == ""


# Layout und Filtergraph


def test_fit_aspect_keeps_ratio_and_stays_inside():
    x, y, w, h = fit_aspect((40, 770, 480, 270), 1080 / 640, (1920, 1080))
    assert abs(w / h - 1080 / 640) < 0.02
    assert x >= 40 and y >= 770 and x + w <= 520 and y + h <= 1040
    assert all(v % 2 == 0 for v in (x, y, w, h))


def test_fit_aspect_keeps_top_edge_when_trimming_height():
    # Koepfe sitzen oben: Beschnitten wird unten, nicht mittig.
    assert fit_aspect((0, 682, 632, 398), 1080 / 640, (1920, 1080)) == (0, 682, 632, 374)


def test_game_crop_clamps_to_frame():
    assert game_crop(INFO, 0.0, 0.84)[0] == 0
    x, _, w, _ = game_crop(INFO, 1.0, 0.84)
    assert x + w <= 1920
    assert w == 906


def test_resolve_layout_falls_back_without_cam():
    assert resolve_layout(_spec(layout="split")) == "gameplay"
    assert resolve_layout(_spec(layout="split", cam_rect=(0, 0, 100, 100))) == "split"
    assert resolve_layout(_spec(layout="blur")) == "blur"


def test_teaser_range_snaps_to_words_and_limits():
    words = [_word("a", 24.0, 24.5), _word("b", 24.6, 25.0), _word("c", 25.1, 26.8)]
    first, last = teaser_range(25.0, 10.0, 40.0, words)
    assert first == 24.0
    assert last == 26.8
    first, last = teaser_range(10.1, 10.0, 40.0, [])
    assert first == 10.0
    assert 1.2 <= last - first <= 3.0


def test_build_parts_with_and_without_teaser():
    parts = build_parts(_spec(), [])
    assert len(parts) == 2
    assert parts[1].out_start == pytest.approx(parts[0].duration)
    assert (parts[1].src_start, parts[1].src_end) == (10.0, 40.0)
    assert len(build_parts(_spec(teaser=False), [])) == 1


@pytest.mark.parametrize(
    ("layout", "cam", "expected"),
    [
        ("split", (40, 770, 480, 270), "vstack"),
        ("gameplay", (40, 770, 480, 270), "overlay=60:250"),
        ("gameplay", None, "scale=1080:1920"),
        ("blur", None, "boxblur"),
    ],
)
def test_filter_graph_contains_layout_and_all_outputs(layout, cam, expected):
    spec = _spec(layout=layout, cam_rect=cam)
    parts = build_parts(spec, [])
    plan = plan_effects(parts, spec.peak, [], spec.effects, parts[-1].out_end)
    graph = build_filter_graph(
        spec,
        layout,
        INFO,
        parts,
        plan,
        [SfxInput(path=__import__("pathlib").Path("x.wav"), time=1.5, volume=0.5)],
        30,
        2.5,
        -14.0,
        preview=False,
    )
    assert expected in graph
    assert "concat=n=3:v=1:a=1[vc][ac]" in graph
    assert "[vout]" in graph and "[aout]" in graph
    assert "adelay=1500:all=1" in graph
    assert "loudnorm=I=-14.0" in graph
    assert "scale=540:960" not in graph


def test_filter_graph_preview_scales_and_handles_missing_audio():
    spec = _spec(teaser=False)
    info = VideoInfo(width=1920, height=1080, has_audio=False, duration=100.0)
    parts = build_parts(spec, [])
    plan = plan_effects(parts, spec.peak, [], Effects(sfx=False), parts[-1].out_end)
    graph = build_filter_graph(
        spec, "gameplay", info, parts, plan, [], 30, 2.5, -14.0, preview=True
    )
    assert "scale=540:960" in graph
    assert "anullsrc" in graph
    assert "[0:a]" not in graph
    assert "amix" not in graph


# Facecam


def test_cluster_finds_stable_face_and_ignores_moving_ones():
    stable = Face(100, 400, 40, 50)
    moving = [Face(500 + 80 * i, 100, 40, 50) for i in range(10)]
    detections = [
        [stable, moving[i]] if i % 2 == 0 else [Face(102, 402, 42, 50), moving[i]]
        for i in range(10)
    ]
    face = cluster_faces(detections)
    assert face is not None
    assert abs(face.x - 100) <= 2


def test_cluster_returns_none_for_rare_faces():
    detections = [[Face(10, 10, 20, 20)]] + [[] for _ in range(15)]
    assert cluster_faces(detections) is None
    assert cluster_faces([[] for _ in range(5)]) is None


def _stream_frames(rect, count=12, size=(540, 960)):
    """Spielbild, das in jedem Frame anders hell ist, mit gleichbleibender Facecam darueber."""
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 255, count, dtype=np.uint8)
    frames = np.stack([np.full(size, level, dtype=np.uint8) for level in levels])
    x, y, w, h = rect
    frames[:, y : y + h, x : x + w] = 90
    return frames


def test_cam_rect_snaps_to_border_exactly():
    frames = _stream_frames((30, 380, 240, 135))
    assert find_cam_rect(frames, Face(130, 420, 40, 50)) == (30, 380, 240, 135)


def test_cam_rect_ignores_static_edges_inside_the_room():
    frames = _stream_frames((30, 380, 240, 135))
    # Tuerrahmen im Zimmer: starke Kante, aber auf beiden Seiten kein bewegtes Bild.
    frames[:, 380:515, 80] = 255
    assert find_cam_rect(frames, Face(130, 420, 40, 50)) == (30, 380, 240, 135)


def test_cam_rect_uses_frame_edges_when_no_border_exists():
    # Freigestellter Streamer in der Ecke: keine Kante, links und oben ist der Bildrand nah.
    frames = np.full((12, 540, 960), 90, dtype=np.uint8)
    assert find_cam_rect(frames, Face(100, 100, 40, 50)) == (0, 0, 188, 166)


def test_cam_rect_reaches_bottom_for_corner_cam_with_top_border():
    frames = np.stack([np.full((540, 960), level, dtype=np.uint8) for level in range(20, 240, 20)])
    frames[:, 341:, :260] = 90
    x, y, _w, h = find_cam_rect(frames, Face(70, 380, 70, 90))
    assert (x, y) == (0, 341)
    assert y + h == 540


def test_cam_rect_falls_back_in_the_middle_of_the_frame():
    frames = np.full((12, 540, 960), 90, dtype=np.uint8)
    assert find_cam_rect(frames, Face(460, 200, 40, 50)) == (412, 190, 136, 76)


def test_locate_cam_layouts():
    frames = _stream_frames((30, 380, 240, 135))
    small = [[Face(130, 420, 40, 50)] for _ in range(12)]
    assert locate_cam(small, frames) == ((30, 380, 240, 135), "split")

    big = [[Face(300, 100, 300, 300)] for _ in range(12)]
    assert locate_cam(big, frames) == (None, "blur")
    assert locate_cam([[] for _ in range(12)], frames) == (None, "gameplay")


def test_locate_cam_without_visible_borders():
    frames = np.full((12, 540, 960), 90, dtype=np.uint8)
    rect, layout = locate_cam([[Face(100, 100, 40, 50)] for _ in range(12)], frames)
    assert layout == "split"
    # Keine Kante zu finden: Links und oben reicht der Rahmen bis zum Bildrand, sonst Standardgroesse.
    assert rect == (0, 0, 188, 166)


def test_sample_times_are_inside_range():
    times = sample_times(10.0, 20.0, 4)
    assert len(times) == 4 and all(10 < t < 20 for t in times)
    assert sample_times(5.0, 5.0) == []


# Sounds


def test_starter_pack_covers_all_categories_and_user_files_join(tmp_path):
    ensure_starter_pack(tmp_path)
    for category in ("whoosh", "impact", "riser", "pop"):
        assert available_sounds(tmp_path, category)
    (tmp_path / "impact").mkdir()
    own = tmp_path / "impact" / "mine.mp3"
    own.write_bytes(b"x")
    (tmp_path / "impact" / "notes.txt").write_text("x")
    sounds = available_sounds(tmp_path, "impact")
    assert own in sounds
    assert all(s.suffix != ".txt" for s in sounds)
    assert pick_sound(tmp_path, "impact", "seed") == pick_sound(tmp_path, "impact", "seed")
    assert pick_sound(tmp_path, "unknown", "seed") is None


# Modellausgabe


def test_clean_assessment_removes_markdown_and_hashtags():
    raw = Assessment(
        reaction=1,
        hook=1,
        payoff=1,
        emotion=1,
        standalone=1,
        gta6_relevance=1,
        reason="r",
        hooks=[" Is GTA 6 *this* good?", "b", "c"],
        title="**T**",
        description="Wild moment. #GTA6 #Gaming",
        hashtags=["#GTA6"],
    )
    cleaned = clean_assessment(raw)
    assert cleaned.hooks[0] == "Is GTA 6 this good?"
    assert cleaned.title == "T"
    assert cleaned.description == "Wild moment."
    assert cleaned.hashtags == ["#GTA6"]


# Export


def test_hashtags_are_normalized_and_deduplicated():
    assert clean_hashtags(["GTA6", "#Vice City", "", "#gta6", "lol!"]) == [
        "#gta6",
        "#vicecity",
        "#lol",
    ]


def test_slugify_handles_empty_and_symbols():
    assert slugify("He JUMPED!! the bridge?") == "he-jumped-the-bridge"
    assert slugify("???") == "clip"


def test_export_text_has_sections_and_credit():
    spec = _spec(title="Title", description="Desc", credit="Name", hashtags=["fun"])
    text = export_text(spec, {"source": "https://twitch.tv/videos/1"})
    assert "TITEL\nTitle" in text
    assert "Credit: Name\nOriginal: https://twitch.tv/videos/1" in text
    assert "#gta6 #fun" in text


def test_censor_masks_slurs_by_default_and_profanity_on_request():
    assert censor_text("You know I got a nigga outside") == "You know I got a n**** outside"
    assert censor_text("what the fuck bro") == "what the fuck bro"
    assert censor_text("what the fuck bro", "all") == "what the f*** bro"
    assert censor_text("a nigga", "none") == "a nigga"


def test_censor_leaves_harmless_words_alone():
    for text in ("spicy night in Nigeria", "flame retardant", "a raccoon on the bridge"):
        assert censor_text(text, "all") == text
    assert contains_slur("NIGGERS")
    assert not contains_slur("Nigeria")


def test_export_text_masks_title_and_description():
    spec = _spec(title="nigga moment", description="wow nigga", hashtags=[])
    text = export_text(spec, {})
    assert "n**** moment" in text
    assert "wow n****" in text
    assert "nigga" not in text


def test_spec_uses_hook_when_model_gives_no_title():
    candidate = {
        "id": "c",
        "start": 0.0,
        "end": 1.0,
        "peak": 0.5,
        "assessment": {"hooks": ["first", "b", "c"], "title": ""},
    }
    assert spec_from_candidate(candidate, {}).title == "first"


def test_spec_from_candidate_uses_assessment_and_cam():
    candidate = {
        "id": "c001",
        "start": 1.0,
        "end": 30.0,
        "peak": 5.0,
        "assessment": {
            "hooks": ["first", "second", "third"],
            "title": "T",
            "description": "D",
            "hashtags": ["x"],
        },
    }
    spec = spec_from_candidate(
        candidate, {"streamer": "S", "cam": {"rect": [1, 2, 3, 4], "layout": "split"}}
    )
    assert spec.hook_text == "first"
    assert spec.cam_rect == (1, 2, 3, 4)
    assert spec.layout == "split"
    assert spec.credit == "S"
    bare = spec_from_candidate({"id": "c", "start": 0.0, "end": 1.0, "peak": 0.5}, {})
    assert bare.layout == "gameplay" and bare.hook_text == ""
