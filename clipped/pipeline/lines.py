"""Zerlegt das Transkript in kurze Zeilen, die an Sprechpausen und Satzenden enden.

Whisper-Segmente sind fuer Schnitte zu grob: Eines enthaelt oft Trailer-Dialog und Streamer-Reaktion
zugleich. Zeilen sind die Einheit, in der das Modell Momente benennt und der Schnitt beginnt und endet.
"""

from __future__ import annotations

import re

from clipped.pipeline.subtitles import remove_noise

# Ab dieser Pause beginnt eine neue Zeile.
LINE_GAP_SECONDS = 0.5
MAX_LINE_WORDS = 12
# Keine Zeile aus ein oder zwei Woertern, nur weil dort gerade die groesste Pause liegt.
MIN_SPLIT_WORDS = 3
SENTENCE_END = re.compile(r"[.?!]$")


def _split_long_run(words: list[dict]) -> list[list[dict]]:
    """Trennt an der groessten Pause. Eine feste Wortzahl riss Saetze auseinander."""
    if len(words) <= MAX_LINE_WORDS:
        return [words]
    positions = range(MIN_SPLIT_WORDS - 1, len(words) - MIN_SPLIT_WORDS)
    index = max(positions, key=lambda i: words[i + 1]["start"] - words[i]["end"])
    return _split_long_run(words[: index + 1]) + _split_long_run(words[index + 1 :])


def build_lines(segments: list[dict]) -> list[dict]:
    words = remove_noise([word for segment in segments for word in segment.get("words", [])])
    runs: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if current and (
            word["start"] - current[-1]["end"] >= LINE_GAP_SECONDS
            or SENTENCE_END.search(current[-1]["word"].strip())
        ):
            runs.append(current)
            current = []
        current.append(word)
    if current:
        runs.append(current)

    return [
        {
            "start": part[0]["start"],
            "end": part[-1]["end"],
            "text": " ".join(w["word"].strip() for w in part),
            "word_ends": [w["end"] for w in part],
        }
        for run in runs
        for part in _split_long_run(run)
    ]


def reaction_flags(lines: list[dict], reaction: list[float], threshold: float) -> list[bool]:
    """Markiert Zeilen, in denen der Streamer sichtbar reagiert, gemessen an der Facecam."""
    flags = []
    for line in lines:
        first, last = int(line["start"]), int(line["end"]) + 1
        values = reaction[first:last]
        flags.append(bool(values) and max(values) >= threshold)
    return flags
