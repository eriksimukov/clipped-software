"""Grenzt ein, welche Teile eines VODs ueberhaupt GTA 6 betreffen."""

from __future__ import annotations

import re

import numpy as np

GTA6_RE = re.compile(
    r"\b(gta|grand theft auto)\s*[-:]?\s*(6|vi|six)\b|\bgta6\b|\bvice city\b|\bleonida\b",
    re.IGNORECASE,
)
# Um eine Erwaehnung herum zaehlt das Gespraech noch zum Thema.
MENTION_RADIUS_SECONDS = 600


def build_mask(meta: dict, segments: list[dict], seconds: int) -> tuple[np.ndarray, str]:
    """Gibt je Sekunde an, ob sie infrage kommt, und nennt den Grund der Entscheidung."""
    mask = np.zeros(seconds, dtype=bool)

    time_range = meta.get("time_range")
    if time_range:
        mask[max(int(time_range[0]), 0) : int(time_range[1])] = True
        return mask, "manueller Zeitbereich"

    chapters = meta.get("chapters") or []
    matching = [c for c in chapters if GTA6_RE.search(c.get("title") or "")]
    if matching:
        for chapter in matching:
            mask[int(chapter["start"]) : int(chapter["end"])] = True
        return mask, f"{len(matching)} GTA-6-Kapitel"

    if GTA6_RE.search(meta.get("title") or ""):
        mask[:] = True
        return mask, "VOD-Titel nennt GTA 6"

    hits = 0
    for segment in segments:
        if GTA6_RE.search(segment["text"]):
            hits += 1
            start = max(int(segment["start"]) - MENTION_RADIUS_SECONDS, 0)
            mask[start : int(segment["end"]) + MENTION_RADIUS_SECONDS] = True
    return mask, f"{hits} Erwaehnungen im Transkript"
