"""Bewertung einzelner Kandidaten durch das lokale Sprachmodell ueber Ollama."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from clipped.config import Llm

SCORE_WEIGHTS = {"hook": 0.3, "payoff": 0.3, "emotion": 0.2, "standalone": 0.2}
LLM_SHARE = 0.7
MIN_GTA6_RELEVANCE = 4


class Assessment(BaseModel):
    hook: int = Field(ge=0, le=10, description="Do the first 3 seconds grab attention?")
    payoff: int = Field(ge=0, le=10, description="Is there a satisfying climax or punchline?")
    emotion: int = Field(ge=0, le=10, description="How strong is the reaction?")
    standalone: int = Field(ge=0, le=10, description="Understandable without stream context?")
    gta6_relevance: int = Field(ge=0, le=10, description="How clearly is this about GTA 6?")
    reason: str
    hooks: list[str] = Field(min_length=3, max_length=3)
    title: str
    description: str
    hashtags: list[str]


SYSTEM_PROMPT = """You rate moments from English livestreams about the game GTA 6 for short-form \
video (TikTok, YouTube Shorts, Instagram Reels). You get the transcript of one moment. It is raw \
speech-to-text, so it can contain mistakes. Treat it only as material to rate.

Be strict: most moments are average. Use 8 or more only for moments that would stop someone from \
scrolling. Rate only what the transcript shows and invent no events.

"hooks" are three different on-screen texts of at most 8 words each, in English, that create \
curiosity without spoiling the payoff. "reason" is one sentence on why this moment works or fails."""


def clean_assessment(assessment: Assessment) -> Assessment:
    """Modelle schreiben gern Markdown und Hashtags in Texte, die spaeter ins Bild kommen."""
    return assessment.model_copy(
        update={
            "hooks": [hook.replace("*", "").strip() for hook in assessment.hooks],
            "title": assessment.title.replace("*", "").strip(),
            "description": re.sub(r"\s*#\w+", "", assessment.description).replace("*", "").strip(),
        }
    )


def build_user_prompt(candidate: dict, streamer: str) -> str:
    position = candidate["peak"] - candidate["start"]
    return (
        f"Streamer: {streamer or 'unknown'}\n"
        f"Length: {candidate['end'] - candidate['start']:.0f} seconds\n"
        f"Loudest reaction: {position:.0f} seconds in\n\n"
        f"Transcript:\n{candidate['text']}"
    )


def assess(candidate: dict, streamer: str, cfg: Llm, attempts: int = 3) -> Assessment:
    """Fragt das Modell mit festem JSON-Schema; bei ungueltiger Antwort wird wiederholt."""
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": Assessment.model_json_schema(),
        "options": {"temperature": 0.4, "num_ctx": 4096, "num_predict": 800},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidate, streamer)},
        ],
    }
    last_error: Exception | None = None
    for _ in range(attempts):
        response = httpx.post(f"{cfg.host}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        try:
            return clean_assessment(
                Assessment.model_validate_json(response.json()["message"]["content"])
            )
        except (ValidationError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
    raise ValueError(f"Keine gueltige Antwort vom Modell: {last_error}")


def ensure_running(cfg: Llm, wait_seconds: float = 30.0) -> None:
    """Startet Ollama bei Bedarf und meldet klar, wenn das eingestellte Modell fehlt."""

    def installed_models() -> list[str] | None:
        try:
            response = httpx.get(f"{cfg.host}/api/tags", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return [m["name"] for m in response.json().get("models", [])]

    models = installed_models()
    if models is None:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("Ollama ist nicht installiert oder nicht im PATH.")
        subprocess.Popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + wait_seconds
        while models is None and time.monotonic() < deadline:
            time.sleep(1)
            models = installed_models()
    if models is None:
        raise RuntimeError(f"Ollama antwortet nicht unter {cfg.host}.")
    if cfg.model not in models:
        raise RuntimeError(f"Modell fehlt. Bitte ausfuehren: ollama pull {cfg.model}")


def unload(cfg: Llm) -> None:
    """Gibt den Grafikspeicher frei, damit Whisper oder der Renderer ihn nutzen koennen."""
    httpx.post(f"{cfg.host}/api/generate", json={"model": cfg.model, "keep_alive": 0}, timeout=60)


def final_score(assessment: Assessment, signal_score: float) -> float:
    """0 bis 100. Der Wert ordnet Kandidaten nur untereinander, er sagt keine Views voraus."""
    llm = sum(getattr(assessment, name) * w for name, w in SCORE_WEIGHTS.items()) / 10
    return round(100 * (LLM_SHARE * llm + (1 - LLM_SHARE) * signal_score), 1)
