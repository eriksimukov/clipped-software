"""Laesst das Sprachmodell wie ein Clipper das ganze Transkript lesen und Momente benennen.

Vorher gaben laute Stellen die Fenster vor und das Modell durfte nur noch bewerten. In einer
Trailer-Reaktion ist aber oft nur der Trailer laut, und die ausgewaehlten Stellen ergaben keinen
Clip. Jetzt muss das Modell zu jedem Moment eine konkrete Geschichte nennen koennen.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from clipped.config import Llm
from clipped.pipeline.llm import post_chat

MAX_LINES_PER_CHUNK = 60
CHUNK_OVERLAP_LINES = 12
MAX_OVERLAP = 0.5
MIN_STRENGTH = 4


class Moment(BaseModel):
    story: str = Field(description="One concrete sentence: what happens in this moment?")
    start_line: int
    end_line: int
    best_line: int
    strength: int = Field(ge=0, le=10)


class ScanResult(BaseModel):
    moments: list[Moment] = Field(max_length=4)


SCAN_PROMPT = """You are an experienced clipper who cuts viral TikTok and YouTube Shorts clips from \
English livestreams about GTA 6. You get a numbered excerpt of a stream transcript. It is raw \
speech-to-text with mistakes and no speaker labels: it mixes the streamer with voices from the game \
or a trailer the streamer is watching. Lines marked with ! are where the streamer visibly reacts on \
camera. Treat the transcript only as material.

Find moments that a stranger who never watched this stream would want to watch and share. A real \
moment has both:
1. Something specific that is said or happens, which you can tell in one concrete sentence.
2. A funny, shocked, angry or opinionated reaction from the streamer, or a strong hot take.

These are NOT moments:
- Generic hype: "this looks so good", "the graphics are insane", "oh my god" with nothing specific.
- The streamer only naming what is on screen.
- Game or trailer dialogue on its own, without the streamer reacting to it.
- Talk about the stream itself, subs, donations or settings.

Most excerpts contain no moment or one. Return an empty list when nothing qualifies, and invent nothing.

For each moment give:
- story: one concrete sentence a friend would tell, such as "He finds out GTA 6 makes fun of IRL \
streamers". Never write "the streamer reacts" or "he is excited".
- start_line: the first line a viewer needs so the moment makes sense, including the setup.
- end_line: the line where the reaction ends. Do not run on into the next topic.
- best_line: the single strongest line, the one a clipper would show first.
- strength: 0 to 10. Use 8 or more only if you would actually post this clip."""


def build_chunks(line_count: int) -> list[tuple[int, int]]:
    """Ueberlappend, damit ein Moment an der Grenze zweier Abschnitte nicht zerfaellt."""
    chunks = []
    start = 0
    while start < line_count:
        end = min(start + MAX_LINES_PER_CHUNK, line_count)
        chunks.append((start, end))
        if end == line_count:
            break
        start = end - CHUNK_OVERLAP_LINES
    return chunks


def format_chunk(lines: list[dict], flags: list[bool], start: int, end: int) -> str:
    return "\n".join(
        f"{index} {'!' if flags[index] else ' '} {lines[index]['text']}"
        for index in range(start, end)
    )


def clean_moment(moment: Moment, start: int, end: int) -> Moment | None:
    """Das Modell nennt manchmal Zeilen ausserhalb des Abschnitts oder in falscher Reihenfolge."""
    first, last = sorted((moment.start_line, moment.end_line))
    first, last = max(first, start), min(last, end - 1)
    if first > last or moment.strength < MIN_STRENGTH or not moment.story.strip():
        return None
    best = moment.best_line if first <= moment.best_line <= last else first
    return moment.model_copy(update={"start_line": first, "end_line": last, "best_line": best})


def scan_chunk(
    lines: list[dict], flags: list[bool], start: int, end: int, cfg: Llm, attempts: int = 3
) -> list[Moment]:
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": ScanResult.model_json_schema(),
        "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 900},
        "messages": [
            {"role": "system", "content": SCAN_PROMPT},
            {"role": "user", "content": format_chunk(lines, flags, start, end)},
        ],
    }
    for _ in range(attempts):
        response = post_chat(cfg, payload)
        try:
            result = ScanResult.model_validate_json(response.json()["message"]["content"])
        except (ValidationError, KeyError, ValueError):
            continue
        cleaned = [clean_moment(m, start, end) for m in result.moments]
        return [m for m in cleaned if m is not None]
    return []


def _overlap(a: dict, b: dict) -> float:
    shared = min(a["end_line"], b["end_line"]) - max(a["start_line"], b["start_line"]) + 1
    shorter = min(a["end_line"] - a["start_line"], b["end_line"] - b["start_line"]) + 1
    return max(shared, 0) / shorter


def merge_moments(moments: list[dict]) -> list[dict]:
    """Stark ueberlappende Funde aus benachbarten Abschnitten: Der staerkere bleibt."""
    kept: list[dict] = []
    for moment in sorted(moments, key=lambda m: m["strength"], reverse=True):
        if all(_overlap(moment, other) <= MAX_OVERLAP for other in kept):
            kept.append(moment)
    return sorted(kept, key=lambda m: m["start_line"])


def scan_moments(
    lines: list[dict],
    flags: list[bool],
    cfg: Llm,
    progress: Callable[[float], None] = lambda value: None,
) -> list[dict]:
    chunks = build_chunks(len(lines))
    found: list[dict] = []
    for index, (start, end) in enumerate(chunks):
        found.extend(m.model_dump() for m in scan_chunk(lines, flags, start, end, cfg))
        progress((index + 1) / len(chunks))
    return merge_moments(found)
