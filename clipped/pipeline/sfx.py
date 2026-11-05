"""Soundeffekte: synthetisches, lizenzfreies Starter-Pack plus eigene Dateien des Nutzers.

Eigene Dateien kommen nach assets/sfx/<kategorie>/, zum Beispiel assets/sfx/impact/boom.mp3.
"""

from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

CATEGORIES = ("whoosh", "impact", "riser", "pop")
AUDIO_SUFFIXES = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
RATE = 48000


def _one_pole_lowpass(noise: np.ndarray, cutoff_hz: np.ndarray) -> np.ndarray:
    coefficient = 1 - np.exp(-2 * np.pi * cutoff_hz / RATE)
    output = np.zeros_like(noise)
    state = 0.0
    for index, sample in enumerate(noise):
        state += coefficient[index] * (sample - state)
        output[index] = state
    return output


def _normalized_stereo(left: np.ndarray, right: np.ndarray, peak: float = 0.8) -> np.ndarray:
    stereo = np.stack([left, right], axis=1)
    return stereo / max(float(np.max(np.abs(stereo))), 1e-6) * peak


def _whoosh(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    progress = np.linspace(0, 1, count)
    bell = np.sin(np.pi * progress) ** 2
    noise = _one_pole_lowpass(rng.standard_normal(count), 300 + 6000 * bell)
    left = noise * bell * np.cos(progress * np.pi / 2)
    right = noise * bell * np.sin(progress * np.pi / 2)
    return _normalized_stereo(left, right)


def _impact(seconds: float, start_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    time = np.arange(count) / RATE
    frequency = 38 + (start_hz - 38) * np.exp(-time * 16)
    phase = 2 * np.pi * np.cumsum(frequency) / RATE
    body = np.tanh(2.2 * np.sin(phase)) * np.exp(-time * 3.8)
    click = _one_pole_lowpass(rng.standard_normal(count), np.full(count, 2400.0)) * np.exp(
        -time * 55
    )
    mono = body + 0.5 * click
    return _normalized_stereo(mono, mono, peak=0.9)


def _riser(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * RATE)
    progress = np.linspace(0, 1, count)
    noise = _one_pole_lowpass(rng.standard_normal(count), 400 + 9000 * progress**2)
    frequency = 180 + 900 * progress**2
    tone = np.sin(2 * np.pi * np.cumsum(frequency) / RATE)
    mono = (0.7 * noise / max(float(np.max(np.abs(noise))), 1e-6) + 0.3 * tone) * progress**2.2
    return _normalized_stereo(mono, mono)


def _pop(seconds: float, start_hz: float) -> np.ndarray:
    count = int(seconds * RATE)
    time = np.arange(count) / RATE
    frequency = 260 + (start_hz - 260) * np.exp(-time * 45)
    mono = np.sin(2 * np.pi * np.cumsum(frequency) / RATE) * np.exp(-time * 26)
    return _normalized_stereo(mono, mono, peak=0.7)


def _write_wav(path: Path, stereo: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(pcm.tobytes())


def ensure_starter_pack(sfx_dir: Path) -> None:
    """Erzeugt fehlende Dateien; sie sind per Code definiert und landen nicht im Repo."""
    generated = sfx_dir / "generated"
    recipes = {
        "whoosh/whoosh_a.wav": lambda: _whoosh(0.45, 1),
        "whoosh/whoosh_b.wav": lambda: _whoosh(0.6, 2),
        "impact/impact_a.wav": lambda: _impact(1.1, 150.0, 3),
        "impact/impact_b.wav": lambda: _impact(1.4, 210.0, 4),
        "riser/riser_a.wav": lambda: _riser(1.4, 5),
        "pop/pop_a.wav": lambda: _pop(0.14, 950.0),
        "pop/pop_b.wav": lambda: _pop(0.18, 700.0),
    }
    for relative, recipe in recipes.items():
        path = generated / relative
        if not path.exists():
            _write_wav(path, recipe())


def available_sounds(sfx_dir: Path, category: str) -> list[Path]:
    folders = [sfx_dir / category, sfx_dir / "generated" / category]
    return sorted(
        file
        for folder in folders
        if folder.is_dir()
        for file in folder.iterdir()
        if file.suffix.lower() in AUDIO_SUFFIXES
    )


def pick_sound(sfx_dir: Path, category: str, seed: str) -> Path | None:
    """Mit demselben Seed entsteht beim erneuten Rendern derselbe Clip."""
    sounds = available_sounds(sfx_dir, category)
    return random.Random(seed).choice(sounds) if sounds else None
