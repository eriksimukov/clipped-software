"""Erzeugt ASS-Untertitel mit Wort-Highlight sowie Hook, Wasserzeichen und Credit."""

from __future__ import annotations

import re
from dataclasses import dataclass

from clipped.pipeline.spec import TextEdit

SENTENCE_END = re.compile(r"[.?!]$")
UNSAFE_ASS = re.compile(r"[{}\\]")

MAX_WORDS_PER_CHUNK = 3
MAX_CHARS_PER_CHUNK = 15
MAX_GAP_IN_CHUNK = 0.6
# So lange bleibt der letzte Text einer Gruppe stehen, wenn danach eine Pause folgt.
LINGER_SECONDS = 0.15
HOOK_MAX_CHARS_PER_LINE = 14
HOOK_MAX_LINES = 3


@dataclass(frozen=True)
class Part:
    """Ein Stueck des Quellvideos und wo es im fertigen Clip liegt."""

    src_start: float
    src_end: float
    out_start: float

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_end(self) -> float:
        return self.out_start + self.duration


@dataclass(frozen=True)
class TextLayout:
    """Vertikale Positionen; alles bleibt innerhalb der Flaechen, die die Apps nicht ueberdecken."""

    hook_y: int
    sub_y: int
    watermark_y: int = 1450
    credit_y: int = 170


LAYOUT_TEXT = {
    "split": TextLayout(hook_y=700, sub_y=1130),
    "gameplay": TextLayout(hook_y=660, sub_y=1130),
    "blur": TextLayout(hook_y=420, sub_y=1340),
}


def flatten_words(segments: list[dict], start: float, end: float) -> list[dict]:
    """Alle Woerter, die im Zeitraum beginnen."""
    return [
        word
        for segment in segments
        if segment["end"] >= start and segment["start"] <= end
        for word in segment.get("words", [])
        if start <= word["start"] < end
    ]


def apply_text_edits(words: list[dict], edits: list[TextEdit]) -> list[dict]:
    """Verteilt korrigierten Text ueber den Zeitraum, gewichtet nach Wortlaenge."""
    result = list(words)
    for edit in edits:
        result = [w for w in result if not edit.start <= w["start"] < edit.end]
        tokens = edit.text.split()
        total = sum(len(t) for t in tokens)
        cursor = edit.start
        for token in tokens:
            length = (edit.end - edit.start) * len(token) / total
            result.append({"start": cursor, "end": cursor + length, "word": token})
            cursor += length
    return sorted(result, key=lambda w: w["start"])


def _display(word: str) -> str:
    return UNSAFE_ASS.sub("", word).strip().strip(".,;:").upper()


def chunk_words(words: list[dict]) -> list[list[dict]]:
    """Gruppiert Woerter zu kurzen Einblendungen; Satzenden und Pausen beenden eine Gruppe."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for word in words:
        text = _display(word["word"])
        if not text:
            continue
        item = {**word, "text": text}
        if current and (
            len(current) >= MAX_WORDS_PER_CHUNK
            or chars + 1 + len(text) > MAX_CHARS_PER_CHUNK
            or word["start"] - current[-1]["end"] > MAX_GAP_IN_CHUNK
            or SENTENCE_END.search(current[-1]["word"].strip())
        ):
            chunks.append(current)
            current, chars = [], 0
        current.append(item)
        chars += len(text) + (1 if chars else 0)
    if current:
        chunks.append(current)
    return chunks


def wrap_hook(text: str) -> str:
    """Bricht die Hook um und schneidet ab, was nicht in die erlaubten Zeilen passt."""
    lines: list[str] = []
    current = ""
    for token in UNSAFE_ASS.sub("", text).upper().split():
        candidate = f"{current} {token}".strip()
        if len(candidate) > HOOK_MAX_CHARS_PER_LINE and current:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return r"\N".join(lines[:HOOK_MAX_LINES])


def ass_time(seconds: float) -> str:
    centiseconds = round(max(seconds, 0.0) * 100)
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def ass_color(hex_rgb: str) -> str:
    """ASS erwartet Blau, Gruen, Rot statt Rot, Gruen, Blau."""
    value = hex_rgb.lstrip("#")
    return f"&H{value[4:6]}{value[2:4]}{value[0:2]}&".upper()


def _subtitle_events(
    part: Part, words: list[dict], y: int, highlight: str, next_part_start: float
) -> list[str]:
    inside = [w for w in words if part.src_start <= w["start"] < part.src_end]
    chunks = chunk_words(inside)
    events: list[str] = []
    for index, chunk in enumerate(chunks):
        # Sonst stapeln sich Gruppen, wenn das letzte Wort laenger dauert als der Abstand.
        next_chunk_start = (
            chunks[index + 1][0]["start"] if index + 1 < len(chunks) else part.src_end
        )
        for position, word in enumerate(chunk):
            start = word["start"]
            if position + 1 < len(chunk):
                end = chunk[position + 1]["start"]
            else:
                end = min(word["end"] + LINGER_SECONDS, next_chunk_start)
            out_start = part.out_start + (start - part.src_start)
            out_end = min(part.out_start + (end - part.src_start), next_part_start)
            if out_end <= out_start:
                continue
            tokens = [
                f"{{\\c{highlight}}}{w['text']}{{\\c&HFFFFFF&}}" if i == position else w["text"]
                for i, w in enumerate(chunk)
            ]
            pop = r"\fscx86\fscy86\t(0,80,\fscx100\fscy100)" if position == 0 else ""
            text = " ".join(tokens)
            events.append(
                f"Dialogue: 1,{ass_time(out_start)},{ass_time(out_end)},Sub,,0,0,0,,"
                f"{{\\an5\\pos(540,{y}){pop}}}{text}"
            )
    return events


def build_ass(
    parts: list[Part],
    words: list[dict],
    layout: str,
    hook_text: str,
    hook_until: float,
    watermark: str,
    credit: str,
    highlight_color: str = "#FFE500",
    subtitle_size: int = 148,
) -> str:
    positions = LAYOUT_TEXT[layout]
    body_end = parts[-1].out_end
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Anton,{subtitle_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,1,0,1,8,3,5,60,60,0,1
Style: Hook,Anton,140,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,1,0,1,10,4,5,60,60,0,1
Style: Mark,Anton,48,&H40FFFFFF,&H000000FF,&H80000000,&H80000000,0,0,0,0,100,100,1,0,1,3,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    highlight = ass_color(highlight_color)
    for index, part in enumerate(parts):
        following = parts[index + 1].out_start if index + 1 < len(parts) else body_end
        events.extend(_subtitle_events(part, words, positions.sub_y, highlight, following))

    hook = wrap_hook(hook_text)
    if hook:
        events.append(
            f"Dialogue: 2,{ass_time(0)},{ass_time(min(hook_until, body_end))},Hook,,0,0,0,,"
            f"{{\\an5\\pos(540,{positions.hook_y})\\fad(0,120)"
            f"\\fscx70\\fscy70\\t(0,110,\\fscx100\\fscy100)}}{hook}"
        )
    if watermark:
        clean = UNSAFE_ASS.sub("", watermark)
        events.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(body_end)},Mark,,0,0,0,,"
            f"{{\\an5\\pos(540,{positions.watermark_y})}}{clean}"
        )
    if credit:
        clean = UNSAFE_ASS.sub("", credit)
        events.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(body_end)},Mark,,0,0,0,,"
            f"{{\\an7\\pos(50,{positions.credit_y})\\fs40}}via {clean}"
        )
    return header + "\n".join(events) + "\n"
