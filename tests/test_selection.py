import numpy as np

from clipped.pipeline.candidates import build_candidates, make_window, overlap_ratio, pick_peaks
from clipped.pipeline.llm import Assessment, final_score
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
    mask, _ = build_mask(meta, [], 100)
    assert mask.sum() == 10 and mask[10] and not mask[20]

    del meta["time_range"]
    mask, _ = build_mask(meta, [], 100)
    assert mask.sum() == 5

    meta["chapters"] = []
    mask, _ = build_mask(meta, [], 100)
    assert mask.all()


def test_mask_excludes_other_game_chapters_when_gta6_chapter_exists():
    meta = {
        "title": "GTA 6 then Fortnite",
        "chapters": [
            {"start": 0, "end": 50, "title": "Fortnite"},
            {"start": 50, "end": 100, "title": "Grand Theft Auto VI"},
        ],
    }
    mask, _ = build_mask(meta, [], 100)
    assert not mask[10] and mask[60]


def test_mask_uses_mentions_when_no_metadata():
    segments = [{"start": 1000, "end": 1005, "text": "this is gta 6 finally"}]
    mask, reason = build_mask({"title": "chill stream"}, segments, 3000)
    assert mask[1000] and mask[500] and not mask[100] and not mask[2000]
    assert "1 Erwaehnungen" in reason


def test_mask_empty_when_no_gta6_anywhere():
    mask, _ = build_mask({"title": "GTA 5 RP"}, [{"start": 0, "end": 5, "text": "hi"}], 100)
    assert not mask.any()


def test_spike_score_flags_outlier_only():
    values = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 20.0])
    scores = spike_score(values, block=100)
    assert scores[-1] == 1.0
    assert scores[:-1].max() < 0.5


def test_spike_score_handles_constant_input():
    assert spike_score(np.zeros(10)).tolist() == [0.0] * 10


def test_combine_redistributes_chat_weight():
    parts = {name: np.ones(3) for name in ("audio", "chat", "hype", "speech")}
    curve, weights = combine(parts, has_chat=False)
    assert "chat" not in weights
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
    top = Assessment(hook=10, payoff=10, emotion=10, standalone=10, gta6_relevance=10, **base)
    low = Assessment(hook=0, payoff=0, emotion=0, standalone=0, gta6_relevance=0, **base)
    assert final_score(top, 1.0) == 100.0
    assert final_score(low, 0.0) == 0.0
