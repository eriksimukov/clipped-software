"""Gemeinsame Helfer: Arbeitsordner je VOD, atomares Speichern, FFmpeg-Aufrufe."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def vod_id_for(source: str) -> str:
    """Stabile ID: dieselbe Quelle landet immer im selben Ordner und wird fortgesetzt."""
    return hashlib.sha1(source.strip().encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class VodPaths:
    root: Path

    @classmethod
    def for_vod(cls, data_dir: Path, vod_id: str) -> "VodPaths":
        root = data_dir / "vods" / vod_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def chat(self) -> Path:
        return self.root / "chat.json"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def signals(self) -> Path:
        return self.root / "signals.json"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates.json"


def write_json(path: Path, data: Any) -> None:
    """Schreibt erst in eine Temp-Datei, damit ein Abbruch keine halbe Datei hinterlaesst."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg fehlgeschlagen: {result.stderr.strip()[-2000:]}")


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe fehlgeschlagen fuer {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())
