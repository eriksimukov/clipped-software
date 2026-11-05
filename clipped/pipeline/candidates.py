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


# Fenster aus den Momenten des Modells
CONTEXT_GAP_SECONDS = 6.0
WANTED_AROUND_BEST = 4.0
MIN_WINDOW_SECONDS = 6.0
TEASER_MIN_SECONDS = 2.0
TEASER_MAX_SECONDS = 7.0


def window_from_moment(
    lines: list[dict], moment: dict, min_seconds: float, max_seconds: float
) -> tuple[int, int] | None:
    """Gibt erste und letzte Zeile zurueck. Das Modell waehlt oft zu knapp, ohne Reaktion und Kontext."""
    first, last, best = moment["start_line"], moment["end_line"], moment["best_line"]

    def duration() -> float:
        return lines[last]["end"] - lines[first]["start"]

    def can_extend_forward() -> bool:
        return (
            last + 1 < len(lines)
            and lines[last + 1]["start"] - lines[last]["end"] <= CONTEXT_GAP_SECONDS
            and lines[last + 1]["end"] - lines[first]["start"] <= max_seconds
        )

    def can_extend_back() -> bool:
        return (
            first > 0
            and lines[first]["start"] - lines[first - 1]["end"] <= CONTEXT_GAP_SECONDS
            and lines[last]["end"] - lines[first - 1]["start"] <= max_seconds
        )

    while True:
        after = lines[last]["end"] - lines[best]["end"]
        before = lines[best]["start"] - lines[first]["start"]
        # Zuerst die Reaktion nach dem besten Satz, dann der Kontext davor, dann die Mindestlaenge.
        if after < WANTED_AROUND_BEST and can_extend_forward():
            last += 1
        elif before < WANTED_AROUND_BEST and can_extend_back():
            first -= 1
        elif duration() < min_seconds and can_extend_forward():
            last += 1
        elif duration() < min_seconds and can_extend_back():
            first -= 1
        else:
            break

    while duration() > max_seconds and first < last:
        if best - first > last - best:
            first += 1
        elif last > best:
            last -= 1
        else:
            first += 1
    if duration() < MIN_WINDOW_SECONDS or duration() > max_seconds:
        return None
    return first, last


def teaser_from_line(lines: list[dict], best: int, last: int) -> tuple[float, float]:
    """Der Teaser ist die staerkste Zeile; zu kurze Zeilen bekommen die naechste dazu.

    Er endet immer an einem Wortende, denn im Test wirkte er sonst abgeschnitten. Ist die Zeile sehr
    kurz, etwa ein einzelner Ausruf, laeuft das Bild noch weiter, damit er nicht vorbeihuscht.
    """
    start = lines[best]["start"]
    end_index = best
    while (
        lines[end_index]["end"] - start < TEASER_MIN_SECONDS
        and end_index < last
        and lines[end_index + 1]["end"] - start <= TEASER_MAX_SECONDS
    ):
        end_index += 1
    end = lines[end_index]["end"]
    if end - start > TEASER_MAX_SECONDS:
        fitting = [
            e for e in lines[end_index].get("word_ends", []) if e - start <= TEASER_MAX_SECONDS
        ]
        end = fitting[-1] if fitting else start + TEASER_MAX_SECONDS
    if end - start < TEASER_MIN_SECONDS:
        end = min(start + TEASER_MIN_SECONDS, lines[last]["end"])
    return start, end


def _top_mean(values: list[float], start: float, end: float, count: int = 3) -> float:
    window = sorted(values[int(start) : int(end) + 1], reverse=True)[:count]
    return sum(window) / len(window) if window else 0.0


def candidates_from_moments(
    moments: list[dict],
    lines: list[dict],
    curve: list[float],
    reaction: list[float],
    min_seconds: float,
    max_seconds: float,
    limit: int,
) -> list[dict]:
    chosen: list[dict] = []
    for moment in sorted(moments, key=lambda m: m["strength"], reverse=True):
        if not 0 <= moment["start_line"] <= moment["end_line"] < len(lines):
            continue
        window = window_from_moment(lines, moment, min_seconds, max_seconds)
        if window is None:
            continue
        first, last = window
        start, end = lines[first]["start"], lines[last]["end"]
        if any(overlap_ratio((start, end), (c["start"], c["end"])) > MAX_OVERLAP for c in chosen):
            continue
        best = min(max(moment["best_line"], first), last)
        teaser_start, teaser_end = teaser_from_line(lines, best, last)
        chosen.append(
            {
                "start": start,
                "end": end,
                "peak": lines[best]["start"],
                "story": moment["story"],
                "strength": moment["strength"],
                "teaser_start": teaser_start,
                "teaser_end": teaser_end,
                "signal_score": _top_mean(curve, start, end),
                "reaction_score": _top_mean(reaction, start, end),
                "text": " ".join(lines[i]["text"] for i in range(first, last + 1)),
            }
        )
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda c: c["start"])
