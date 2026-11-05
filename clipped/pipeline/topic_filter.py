"""Grenzt ein, welche Teile eines VODs ueberhaupt GTA 6 betreffen."""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

GTA6_RE = re.compile(
    r"\b(gta|grand theft auto)\s*[-:]?\s*(6|vi|six)\b|\bgta6\b|\bvice city\b|\bleonida\b",
    re.IGNORECASE,
)
# Um eine Erwaehnung herum zaehlt das Gespraech noch zum Thema.
MENTION_RADIUS_SECONDS = 600


class Topic(NamedTuple):
    # Je Sekunde: kommt sie infrage?
    mask: np.ndarray
    reason: str
    # True, wenn Nutzer oder Metadaten sagen, dass hier GTA 6 laeuft. Blosse Erwaehnungen im
    # Gespraech reichen dafuer nicht.
    confirmed: bool


def build_mask(meta: dict, segments: list[dict], seconds: int) -> Topic:
    """Staerkere Quellen gewinnen: Zeitbereich vor Kapiteln vor Titel vor Erwaehnungen."""
    mask = np.zeros(seconds, dtype=bool)

    time_range = meta.get("time_range")
    if time_range:
        mask[max(int(time_range[0]), 0) : int(time_range[1])] = True
        return Topic(mask, "manueller Zeitbereich", True)

    chapters = meta.get("chapters") or []
    matching = [c for c in chapters if GTA6_RE.search(c.get("title") or "")]
    if matching:
        for chapter in matching:
            mask[int(chapter["start"] or 0) : int(chapter["end"] or seconds)] = True
        return Topic(mask, f"{len(matching)} GTA-6-Kapitel", True)

    if GTA6_RE.search(meta.get("title") or ""):
        mask[:] = True
        return Topic(mask, "VOD-Titel nennt GTA 6", True)

    hits = 0
    for segment in segments:
        if GTA6_RE.search(segment["text"]):
            hits += 1
            start = max(int(segment["start"]) - MENTION_RADIUS_SECONDS, 0)
            mask[start : int(segment["end"]) + MENTION_RADIUS_SECONDS] = True
    return Topic(mask, f"{hits} Erwähnung{'' if hits == 1 else 'en'} im Transkript", False)
