"""Soundeffekte aus dem eigenen Ordner des Nutzers.

Das fruehere synthetische Starter-Pack ist entfernt: Im Test klangen die Sounds fremd und nicht wie in
echten Clips. Ohne eigene Dateien bleibt der Originalton deshalb unveraendert.
Eigene Dateien kommen nach assets/sfx/<kategorie>/, zum Beispiel assets/sfx/impact/boom.mp3.
"""

from __future__ import annotations

import random
from pathlib import Path

CATEGORIES = ("whoosh", "impact")
AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def available_sounds(sfx_dir: Path, category: str) -> list[Path]:
    folder = sfx_dir / category
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES)


def pick_sound(sfx_dir: Path, category: str, seed: str) -> Path | None:
    """Mit demselben Seed entsteht beim erneuten Rendern derselbe Clip."""
    sounds = available_sounds(sfx_dir, category)
    return random.Random(seed).choice(sounds) if sounds else None
