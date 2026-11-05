"""Macht aus der Signalkurve Kandidaten-Fenster, die an Satzgrenzen beginnen und enden."""

from __future__ import annotations

import numpy as np

# Laengere Sprechpausen beenden ein Fenster: dort passiert fuer den Zuschauer nichts.
MAX_GAP_SECONDS = 10.0
# So viel Nachlauf braucht ein Hoehepunkt, damit die Reaktion noch im Clip ist.
MIN_TAIL_SECONDS = 4.0
MAX_OVERLAP = 0.3


def pick_peaks(curve: np.ndarray, mask: np.ndarray, min_distance: int, limit: int) -> list[int]:
    """Hoechste Werte zuerst; zu nahe Nachbarn werden uebersprungen."""
    order = np.argsort(-curve, kind="stable")
    peaks: list[int] = []
    for index in order:
        index = int(index)
        if curve[index] <= 0 or len(peaks) >= limit:
            break
        if mask[index] and all(abs(index - p) >= min_distance for p in peaks):
            peaks.append(index)
    return peaks


def _segment_at(segments: list[dict], second: float) -> int:
    distances = [
        0.0
        if s["start"] <= second <= s["end"]
        else min(abs(s["start"] - second), abs(s["end"] - second))
        for s in segments
    ]
    return int(np.argmin(distances))


def make_window(
    segments: list[dict], peak: float, min_seconds: float, max_seconds: float
) -> tuple[float, float] | None:
    """Erweitert vom Segment des Hoehepunkts aus; None, wenn die Mindestlaenge nicht erreichbar ist."""
    if not segments:
        return None
    first = last = _segment_at(segments, peak)
    wanted_lead = min_seconds * 0.6

    def can_extend_back() -> bool:
        return (
            first > 0
            and segments[first]["start"] - segments[first - 1]["end"] <= MAX_GAP_SECONDS
            and segments[last]["end"] - segments[first - 1]["start"] <= max_seconds
        )

    def can_extend_forward() -> bool:
        return (
            last < len(segments) - 1
            and segments[last + 1]["start"] - segments[last]["end"] <= MAX_GAP_SECONDS
            and segments[last + 1]["end"] - segments[first]["start"] <= max_seconds
        )

    while True:
        start, end = segments[first]["start"], segments[last]["end"]
        too_short = end - start < min_seconds
        if peak - start < wanted_lead and can_extend_back():
            first -= 1
        elif (end - peak < MIN_TAIL_SECONDS or too_short) and can_extend_forward():
            last += 1
        elif too_short and can_extend_back():
            first -= 1
        else:
            break

    start, end = segments[first]["start"], segments[last]["end"]
    # Der Hoehepunkt kann in einer Sprechpause liegen (Explosion, Schrei ohne Worte). Weit weg von
    # jeder Sprache lohnt der Clip nicht, kurz daneben wird das Fenster bis zu ihm gezogen.
    if peak - end > MAX_GAP_SECONDS or start - peak > MAX_GAP_SECONDS:
        return None
    if peak > end:
        next_start = segments[last + 1]["start"] if last + 1 < len(segments) else float("inf")
        end = min(peak + MIN_TAIL_SECONDS, next_start)
    if peak < start:
        previous_end = segments[first - 1]["end"] if first > 0 else 0.0
        start = max(peak - 1.0, previous_end)
    if end - start < min_seconds or end - start > max_seconds:
        return None
    return start, end


def overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    shared = min(a[1], b[1]) - max(a[0], b[0])
    return max(shared, 0.0) / min(a[1] - a[0], b[1] - b[0])


def build_candidates(
    segments: list[dict],
    curve: np.ndarray,
    mask: np.ndarray,
    min_seconds: float,
    max_seconds: float,
    limit: int,
) -> list[dict]:
    # Mehr Peaks als noetig, weil Fenster an Pausen oder Ueberlappungen scheitern koennen.
    peaks = pick_peaks(curve, mask, int(min_seconds // 2), limit * 8)
    chosen: list[dict] = []
    for peak in peaks:
        window = make_window(segments, float(peak), min_seconds, max_seconds)
        if window is None:
            continue
        if any(overlap_ratio(window, (c["start"], c["end"])) > MAX_OVERLAP for c in chosen):
            continue
        text = " ".join(
            s["text"] for s in segments if s["start"] >= window[0] and s["end"] <= window[1]
        )
        chosen.append(
            {
                "start": window[0],
                "end": window[1],
                "peak": float(peak),
                "signal_score": float(curve[peak]),
                "text": text,
            }
        )
        if len(chosen) >= limit:
            break
    return chosen
