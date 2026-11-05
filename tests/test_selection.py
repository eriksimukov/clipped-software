import wave

import numpy as np
import pytest

from clipped.config import Settings
from clipped.pipeline import select_clips as select_module
from clipped.pipeline.candidates import build_candidates, make_window, overlap_ratio, pick_peaks
from clipped.pipeline.common import VideoInfo, VodPaths, read_json, write_json
from clipped.pipeline.llm import Assessment, final_score, streamer_talk
from clipped.pipeline.signals import combine, smooth, spike_score
from clipped.pipeline.topic_filter import GTA6_RE, build_mask


def _segments(count, length=5.0, gap=0.0):
    return [
        {"start": i * (length + gap), "end": i * (length + gap) + length, "text": f"s{i}"}
        for i in range(count)
    ]


def test_regex_matches_gta6_but_not_gta5():
    for text in ("GTA 6 day one", "gta vi", "Grand Theft Auto VI", "GTA6", "back in Vice City"):
        assert GTA6_RE.search(text), text
    for text in ("GTA 5 RP", "gta v online", "GTA VII", "Grand Theft Auto V"):
        assert not GTA6_RE.search(text), text


def test_mask_prefers_range_then_chapters_then_title():
    meta = {
        "time_range": [10, 20],
        "title": "GTA 6",
        "chapters": [{"start": 0, "end": 5, "title": "GTA VI"}],
    }
    mask, _, _ = build_mask(meta, [], 100)
    assert mask.sum() == 10 and mask[10] and not mask[20]

    del meta["time_range"]
    mask, _, _ = build_mask(meta, [], 100)
    assert mask.sum() == 5

    meta["chapters"] = []
    mask, _, _ = build_mask(meta, [], 100)
    assert mask.all()


def test_mask_excludes_other_game_chapters_when_gta6_chapter_exists():
    meta = {
        "title": "GTA 6 then Fortnite",
        "chapters": [
            {"start": 0, "end": 50, "title": "Fortnite"},
            {"start": 50, "end": 100, "title": "Grand Theft Auto VI"},
        ],
    }
    mask, _, _ = build_mask(meta, [], 100)
    assert not mask[10] and mask[60]


def test_mask_uses_mentions_when_no_metadata():
    segments = [{"start": 1000, "end": 1005, "text": "this is gta 6 finally"}]
    mask, reason, confirmed = build_mask({"title": "chill stream"}, segments, 3000)
    assert confirmed is False
    assert mask[1000] and mask[500] and not mask[100] and not mask[2000]
    assert "1 Erwähnung im" in reason


def test_mask_empty_when_no_gta6_anywhere():
    mask, _, _ = build_mask({"title": "GTA 5 RP"}, [{"start": 0, "end": 5, "text": "hi"}], 100)
    assert not mask.any()


def test_spike_score_flags_outlier_only():
    values = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 20.0])
    scores = spike_score(values, block=100)
    assert scores[-1] == 1.0
    assert scores[:-1].max() < 0.5


def test_spike_score_handles_constant_input():
    assert spike_score(np.zeros(10)).tolist() == [0.0] * 10


def test_combine_redistributes_chat_weight():
    parts = {name: np.ones(3) for name in ("audio", "hype", "speech")}
    curve, weights = combine(parts)
    assert set(weights) == {"audio", "hype", "speech"}
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert np.allclose(curve, 1.0)


def test_pick_peaks_respects_mask_and_distance():
    curve = np.array([0, 9, 8, 0, 0, 0, 7, 0, 10, 0], dtype=float)
    mask = np.ones(10, dtype=bool)
    mask[8] = False
    assert pick_peaks(curve, mask, min_distance=3, limit=5) == [1, 6]


def test_window_snaps_to_segments_and_respects_limits():
    segments = _segments(40)
    start, end = make_window(segments, 100.0, 20, 60)
    assert start % 5 == 0 and end % 5 == 0
    assert 20 <= end - start <= 60
    assert start < 100 < end
    assert end - 100 >= 4


def test_window_stops_at_long_silence():
    segments = [
        {"start": 0, "end": 5, "text": "a"},
        {"start": 100, "end": 130, "text": "b"},
    ]
    assert make_window(segments, 110.0, 20, 60) == (100, 130)


def test_window_none_when_minimum_unreachable():
    assert make_window([{"start": 0, "end": 5, "text": "a"}], 2.0, 20, 60) is None
    assert make_window([], 2.0, 20, 60) is None


def test_window_none_when_single_segment_exceeds_max():
    assert make_window([{"start": 0, "end": 90, "text": "a"}], 10.0, 20, 60) is None


def test_build_candidates_drops_overlaps():
    segments = _segments(100)
    curve = np.zeros(500)
    curve[100] = 1.0  # Fenster 85-105
    curve[110] = 0.9  # Fenster 95-115, ueberlappt zur Haelfte und faellt weg
    curve[300] = 0.8
    result = build_candidates(segments, curve, np.ones(500, dtype=bool), 20, 60, 10)
    assert [c["peak"] for c in result] == [100.0, 300.0]
    assert overlap_ratio((0, 10), (5, 15)) == 0.5


def test_build_candidates_keeps_small_overlap():
    segments = _segments(100)
    curve = np.zeros(500)
    curve[100] = 1.0  # Fenster 85-105
    curve[112] = 0.9  # Fenster 100-120, nur 25 Prozent Ueberlappung
    result = build_candidates(segments, curve, np.ones(500, dtype=bool), 20, 60, 10)
    assert [c["peak"] for c in result] == [100.0, 112.0]


def test_window_extends_to_peak_in_speech_pause():
    segments = [
        {"start": 0.0, "end": 25.0, "text": "a"},
        {"start": 100.0, "end": 110.0, "text": "b"},
    ]
    assert make_window(segments, 30.0, 20, 60) == (0.0, 34.0)


def test_window_rejects_peak_far_from_speech():
    segments = [
        {"start": 0.0, "end": 25.0, "text": "a"},
        {"start": 100.0, "end": 110.0, "text": "b"},
    ]
    assert make_window(segments, 60.0, 20, 60) is None


def test_smooth_keeps_length_of_short_input():
    assert len(smooth(np.ones(2))) == 2
    assert len(smooth(np.ones(50))) == 50


def test_final_score_bounds():
    base = {
        "reason": "r",
        "hooks": ["a", "b", "c"],
        "title": "t",
        "description": "d",
        "hashtags": [],
    }
    top = Assessment(
        reaction=10, hook=10, payoff=10, emotion=10, standalone=10, gta6_relevance=10, **base
    )
    low = Assessment(
        reaction=0, hook=0, payoff=0, emotion=0, standalone=0, gta6_relevance=0, **base
    )
    assert final_score(top, 1.0, 1.0, 10) == 100.0
    assert final_score(low, 0.0, 0.0, 0) == 0.0


def test_only_metadata_or_user_confirm_the_topic():
    assert build_mask({"time_range": [0, 5]}, [], 10).confirmed
    assert build_mask({"title": "GTA 6 first look"}, [], 10).confirmed
    chapters = [{"start": 0, "end": 5, "title": "Grand Theft Auto VI"}]
    assert build_mask({"chapters": chapters}, [], 10).confirmed
    mention = [{"start": 1, "end": 2, "text": "gta 6 soon"}]
    assert not build_mask({"title": "chat"}, mention, 10).confirmed


def _prepare_vod(tmp_path, meta):
    """Kleines VOD auf der Platte: 60 Sekunden Audio, ein Satz je 5 Sekunden."""
    paths = VodPaths.for_vod(tmp_path, "x")
    samples = np.zeros(16000 * 60, dtype=np.int16)
    samples[16000 * 20 : 16000 * 22] = 20000  # Lautes Ereignis als Hoehepunkt
    with wave.open(str(paths.audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    write_json(paths.transcript, {"segments": _segments(12)})
    write_json(paths.chat, [])
    write_json(
        paths.meta,
        {"video": "x.mp4", "duration": 60.0, "cam": {"rect": None, "layout": "gameplay"}, **meta},
    )
    return paths


def _fake_assessment(relevance):
    base = {
        "reason": "r",
        "hooks": ["a", "b", "c"],
        "title": "t",
        "description": "d",
        "hashtags": [],
    }
    return Assessment(
        reaction=5, hook=5, payoff=5, emotion=5, standalone=5, gta6_relevance=relevance, **base
    )


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []
    relevance = {"value": 1}

    def fake_assess(candidate, streamer, cfg, attempts=3, section_confirmed=False):
        calls.append(section_confirmed)
        return _fake_assessment(relevance["value"])

    monkeypatch.setattr(select_module, "assess", fake_assess)
    monkeypatch.setattr(select_module, "ensure_running", lambda cfg: None)
    monkeypatch.setattr(select_module, "unload", lambda cfg: None)
    # Ohne Woerter im Testtranskript findet das Modell nichts; die Signale fuellen die Liste auf.
    monkeypatch.setattr(select_module, "scan_moments", lambda *args: [])
    monkeypatch.setattr(
        select_module, "probe_video", lambda path: VideoInfo(1920, 1080, True, 60.0)
    )
    return calls, relevance


def test_manual_range_is_never_sorted_out_by_the_model(tmp_path, fake_llm):
    calls, _ = fake_llm
    paths = _prepare_vod(tmp_path, {"time_range": [0, 60]})
    clips = select_module.select_clips(paths, Settings())
    assert clips
    assert {c["status"] for c in clips} == {"assessed"}
    assert all(calls)


def test_model_can_sort_out_clips_without_manual_range(tmp_path, fake_llm):
    calls, _ = fake_llm
    paths = _prepare_vod(tmp_path, {"title": "GTA 6 stream"})
    clips = select_module.select_clips(paths, Settings())
    assert {c["status"] for c in clips} == {"off_topic"}
    assert all(calls)


def test_topic_mentions_do_not_confirm_the_section_for_the_model(tmp_path, fake_llm):
    calls, relevance = fake_llm
    relevance["value"] = 9
    paths = _prepare_vod(tmp_path, {"title": "chat"})
    segments = _segments(12)
    segments[3]["text"] = "gta 6 is close"
    write_json(paths.transcript, {"segments": segments})
    clips = select_module.select_clips(paths, Settings())
    assert {c["status"] for c in clips} == {"assessed"}
    assert not any(calls)
    assert read_json(paths.meta)["topic_confirmed"] is False


def test_finished_assessments_survive_a_restart(tmp_path, fake_llm):
    calls, _ = fake_llm
    paths = _prepare_vod(tmp_path, {"time_range": [0, 60]})
    select_module.select_clips(paths, Settings())
    count = len(calls)
    assert count > 0
    select_module.select_clips(paths, Settings())
    assert len(calls) == count


def test_clip_with_slur_is_flagged_and_downgraded(tmp_path, fake_llm):
    _, relevance = fake_llm
    relevance["value"] = 9
    paths = _prepare_vod(tmp_path, {"title": "GTA 6"})
    segments = _segments(12)
    for segment in segments:
        segment["text"] = "you know i got a nigga outside"
    write_json(paths.transcript, {"segments": segments})
    clips = select_module.select_clips(paths, Settings())
    assert clips
    for clip in clips:
        assert clip["warnings"] == ["slur"]
        expected = (
            final_score(
                _fake_assessment(9), clip["signal_score"], clip["talk_score"], clip["strength"]
            )
            * select_module.SLUR_SCORE_FACTOR
        )
        assert clip["score"] == pytest.approx(expected, abs=0.1)


def test_streamer_talk_tells_reactions_from_dialogue():
    reaction = "oh my god chat look at this bro, no way"
    dialogue = "Lucia, Jason, please meet Andres de Leon. Should we go, gentlemen?"
    assert streamer_talk(reaction) == 1.0
    assert streamer_talk(dialogue) == 0.0
    assert streamer_talk("Chad, look at the cars") == 0.5


def test_streamer_talk_lifts_the_score():
    assessment = _fake_assessment(9)
    assert final_score(assessment, 0.5, 1.0) > final_score(assessment, 0.5, 0.0)
