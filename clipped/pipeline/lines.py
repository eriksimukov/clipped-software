"""Zerlegt das Transkript in kurze Zeilen, die an Sprechpausen und Satzenden enden.

Whisper-Segmente sind fuer Schnitte zu grob: Eines enthaelt oft Trailer-Dialog und Streamer-Reaktion
zugleich. Zeilen sind die Einheit, in der das Modell Momente benennt und der Schnitt beginnt und endet.
"""

from __future__ import annotations

import re

# Ab dieser Pause beginnt eine neue Zeile.
LINE_GAP_SECONDS = 0.5
MAX_LINE_WORDS = 16
SENTENCE_END = re.compile(r"[.?!]$")


def build_lines(segments: list[dict]) -> list[dict]:
    lines: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            lines.append(
                {
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": " ".join(w["word"].strip() for w in current),
                }
            )
            current.clear()

    for segment in segments:
        for word in segment.get("words", []):
            if current and (
                word["start"] - current[-1]["end"] >= LINE_GAP_SECONDS
                or SENTENCE_END.search(current[-1]["word"].strip())
                or len(current) >= MAX_LINE_WORDS
            ):
                flush()
            current.append(word)
    flush()
    return lines


def reaction_flags(lines: list[dict], reaction: list[float], threshold: float) -> list[bool]:
    """Markiert Zeilen, in denen der Streamer sichtbar reagiert, gemessen an der Facecam."""
    flags = []
    for line in lines:
        first, last = int(line["start"]), int(line["end"]) + 1
        values = reaction[first:last]
        flags.append(bool(values) and max(values) >= threshold)
    return flags
