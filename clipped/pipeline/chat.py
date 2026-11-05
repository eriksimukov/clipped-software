"""Liest Chat-Replays von Twitch (rechat) und YouTube (live_chat) in ein einheitliches Format."""

from __future__ import annotations

import json
from pathlib import Path


def parse_twitch_rechat(text: str) -> list[dict]:
    data = json.loads(text)
    comments = data.get("comments", []) if isinstance(data, dict) else data
    messages = []
    for comment in comments:
        offset = comment.get("content_offset_seconds")
        if offset is None:
            continue
        body = (comment.get("message") or {}).get("body", "")
        messages.append({"t": float(offset), "text": body})
    return messages


def parse_youtube_live_chat(text: str) -> list[dict]:
    """YouTube liefert eine JSON-Zeile je Aktion; Nicht-Textnachrichten werden uebersprungen."""
    messages = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            replay = json.loads(line)["replayChatItemAction"]
            offset = float(replay["videoOffsetTimeMsec"]) / 1000
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        for action in replay.get("actions", []):
            item = action.get("addChatItemAction", {}).get("item", {})
            renderer = item.get("liveChatTextMessageRenderer")
            if not renderer:
                continue
            runs = renderer.get("message", {}).get("runs", [])
            body = "".join(run.get("text", "") for run in runs)
            messages.append({"t": offset, "text": body})
    return messages


def parse_chat_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parser = parse_youtube_live_chat if "live_chat" in path.name else parse_twitch_rechat
    messages = parser(text)
    messages.sort(key=lambda m: m["t"])
    return messages
