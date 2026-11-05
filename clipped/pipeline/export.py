"""Legt den fertigen Clip samt kopierbarer Texte im Export-Ordner ab."""

from __future__ import annotations

import re
from pathlib import Path

from clipped.pipeline.spec import ClipSpec

SLUG_RE = re.compile(r"[^a-z0-9]+")
FIXED_HASHTAGS = ["#gta6"]
MAX_HASHTAGS = 8


def slugify(text: str, fallback: str = "clip", limit: int = 50) -> str:
    return SLUG_RE.sub("-", text.lower()).strip("-")[:limit].strip("-") or fallback


def clean_hashtags(tags: list[str]) -> list[str]:
    """Einheitlich mit Raute, ohne Leerzeichen und ohne Dopplungen."""
    result: list[str] = []
    for tag in [*FIXED_HASHTAGS, *tags]:
        word = re.sub(r"[^\w]", "", tag)
        if not word:
            continue
        formatted = f"#{word.lower()}"
        if formatted not in result:
            result.append(formatted)
    return result[:MAX_HASHTAGS]


def export_text(spec: ClipSpec, meta: dict) -> str:
    credit_lines = []
    if spec.credit:
        credit_lines.append(f"Credit: {spec.credit}")
    if meta.get("source", "").startswith("http"):
        credit_lines.append(f"Original: {meta['source']}")
    credit = "\n".join(credit_lines)
    description = "\n\n".join(p for p in (spec.description.strip(), credit) if p)
    return (
        f"TITEL\n{spec.title.strip()}\n\n"
        f"BESCHREIBUNG\n{description}\n\n"
        f"HASHTAGS\n{' '.join(clean_hashtags(spec.hashtags))}\n"
    )


def export_paths(data_dir: Path, vod_id: str, spec: ClipSpec) -> tuple[Path, Path]:
    folder = data_dir / "export" / vod_id
    stem = f"{spec.id}-{slugify(spec.title or spec.hook_text)}"
    return folder / f"{stem}.mp4", folder / f"{stem}.txt"
