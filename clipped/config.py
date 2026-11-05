"""Laedt config/settings.toml und faellt auf settings.example.toml zurueck."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
ASSETS_DIR = ROOT / "assets"


class Paths(BaseModel):
    data_dir: Path = Path("G:/clipped-data")


class Clips(BaseModel):
    min_seconds: int = 20
    max_seconds: int = 60
    candidates_per_vod: int = 35
    language: str = "en"


class Transcribe(BaseModel):
    model: str = "large-v3"
    compute_type: str = "int8_float16"
    device: str = "cuda"


class Llm(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"


class Render(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_encoder: str = "h264_nvenc"
    target_lufs: float = -14.0


class Channel(BaseModel):
    platform: str
    handle: str


class Branding(BaseModel):
    watermark: str = "@deinhandle"
    endcard_title: str = "Mehr GTA 6 Clips"
    endcard_seconds: float = 2.5
    channels: list[Channel] = Field(default_factory=list)


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    clips: Clips = Field(default_factory=Clips)
    transcribe: Transcribe = Field(default_factory=Transcribe)
    llm: Llm = Field(default_factory=Llm)
    render: Render = Field(default_factory=Render)
    branding: Branding = Field(default_factory=Branding)


def load_settings(path: Path | None = None) -> Settings:
    """Liest die Einstellungen; ohne settings.toml gilt die Beispieldatei."""
    candidates = [path] if path else [CONFIG_DIR / "settings.toml", CONFIG_DIR / "settings.example.toml"]
    for candidate in candidates:
        if candidate and candidate.exists():
            with candidate.open("rb") as handle:
                return Settings.model_validate(tomllib.load(handle))
    return Settings()
