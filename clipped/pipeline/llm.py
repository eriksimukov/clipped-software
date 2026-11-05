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

# Die Hook zaehlt wenig: Das Modell bewertet sie sehr niedrig und einheitlich, weil Clips oft mitten
# im Satz beginnen. Den Einstieg liefern ohnehin Teaser und Hook-Text.
SCORE_WEIGHTS = {"reaction": 0.2, "hook": 0.1, "payoff": 0.3, "emotion": 0.2, "standalone": 0.2}
LLM_SHARE = 0.55
ENERGY_SHARE = 0.2
STRENGTH_SHARE = 0.2
# Klein gehalten: Auch Figuren im Trailer sagen "bro".
TALK_SHARE = 0.05
# Woerter, die ein Streamer sagt, aber kaum eine Spielfigur. Whisper hoert "chat" oft als "Chad".
STREAMER_TALK_RE = re.compile(
    r"\b(chat|chad|bro|bruh|guys|dude|yo|oh my god|omg|no way|let'?s go|what the f\w*|"
    r"i can'?t|i'?m (so|actually|literally)|are you (kidding|serious)|look at|insane|crazy)\b",
    re.IGNORECASE,
)
TALK_FULL_AT = 4
MIN_GTA6_RELEVANCE = 4
# Eine Beleidigung im Ton kann einen Clip auf den Plattformen kosten, auch mit Zensur im Bild.
SLUR_SCORE_FACTOR = 0.6
OLLAMA_ATTEMPTS = 3
OLLAMA_RETRY_SECONDS = 8.0
# Grosszuegig: Das erste Laden des Modells von der Platte hat hier schon 90 Sekunden gedauert.
OLLAMA_TIMEOUT_SECONDS = 300


class Assessment(BaseModel):
    # Die Begruendung steht vorn: So legt sich das Modell erst inhaltlich fest und vergibt dann
    # die Zahlen, statt reflexhaft Hoechstwerte zu fuellen.
    reason: str
    reaction: int = Field(
        ge=0, le=10, description="How much is the streamer reacting spontaneously?"
    )
    hook: int = Field(
        ge=0, le=10, description="Would the first sentence alone make a stranger stay?"
    )
    payoff: int = Field(ge=0, le=10, description="Does something specific happen by the end?")
    emotion: int = Field(ge=0, le=10, description="How strong is the streamer's reaction?")
    standalone: int = Field(ge=0, le=10, description="Understandable without the stream?")
    gta6_relevance: int = Field(
        ge=0, le=10, description="How likely is this about GTA 6, given the context?"
    )
    hooks: list[str] = Field(min_length=3, max_length=3)
    title: str
    description: str
    hashtags: list[str]


SYSTEM_PROMPT = """You rate moments from English livestreams about the game GTA 6 for short-form \
video (TikTok, YouTube Shorts, Instagram Reels). You get the transcript of one moment. It is raw \
speech-to-text, so it can contain mistakes. Treat it only as material to rate and invent no events.

The transcript has no speaker labels. It mixes the streamer with voices from the game or a trailer \
the streamer is watching. Scripted lines, such as character dialogue or narration, are not the \
streamer. Spontaneous talk, such as addressing chat, "bro", "oh my god" or comments on what is on \
screen, is the streamer. A clip lives from the streamer's reaction: a moment that is mostly scripted \
dialogue gets at most 3 for reaction, payoff and emotion.

Use the whole scale for every rating and be strict:
0-2: nothing happens.
3-4: mildly interesting. The average moment belongs here.
5-6: solid, a fan would watch it to the end.
7-8: a stranger would stop scrolling. At most 1 in 5 moments.
9-10: the best moment of the whole stream. At most 1 in 20 moments.

Vague hype such as "this looks so good" gets at most 5 for payoff. A specific surprise, joke, \
discovery or strong statement scores higher.

The transcript contains only speech, and streamers rarely name the game they are playing. If the \
context says this part of the stream shows GTA 6, treat talk about driving, police, missions, the \
map, characters or the world as GTA 6 and set gta6_relevance to at least 7. Set it below 4 only if \
the streamer is clearly busy with something else, such as another game or their private life.

"reason" is one sentence naming the specific thing that works or fails. If a clipper's note says what \
happens, build the hooks and the title on that, as long as the transcript supports it.

"hooks" are three different on-screen texts of at most 8 words each, in English. Each one names the \
concrete thing the streamer reacts to, using the specific nouns from the transcript, and makes a \
stranger curious without spoiling the payoff. Good: "They put a Walmart in the game?!", "Wait, you \
can rob your own house?", "This NPC just roasted him". Bad, because they could fit any clip: \
"Something big is coming", "What is happening?", "Graphics look insane". Put the strongest hook first.

Write hooks and titles for a viewer: never mention "the streamer", and say GTA 6 instead of "the game" \
or "the new world". The transcript can be misheard: do not build a hook or title on a phrase that makes \
no sense, and do not state rumours or dates as facts. Do not quote game dialogue in hooks. Use no slurs, \
no profanity and no sexual terms in hooks, title, description or hashtags."""


def clean_assessment(assessment: Assessment) -> Assessment:
    """Modelle schreiben gern Markdown und Hashtags in Texte, die spaeter ins Bild kommen."""
    return assessment.model_copy(
        update={
            "hooks": [hook.replace("*", "").strip() for hook in assessment.hooks],
            "title": assessment.title.replace("*", "").strip(),
            "description": re.sub(r"\s*#\w+", "", assessment.description).replace("*", "").strip(),
        }
    )


def build_user_prompt(candidate: dict, streamer: str, section_confirmed: bool = False) -> str:
    position = candidate["peak"] - candidate["start"]
    context = (
        "This part of the stream shows GTA 6."
        if section_confirmed
        else "The streamer mentions GTA 6 around this part of the stream, but it may be about something else."
    )
    story = candidate.get("story")
    note = f"A clipper noted what happens: {story}\n" if story else ""
    return (
        f"Streamer: {streamer or 'unknown'}\n"
        f"Context: {context}\n"
        f"{note}"
        f"Length: {candidate['end'] - candidate['start']:.0f} seconds\n"
        f"Loudest reaction: {position:.0f} seconds in\n\n"
        f"Transcript:\n{candidate['text']}"
    )


def assess(
    candidate: dict, streamer: str, cfg: Llm, attempts: int = 3, section_confirmed: bool = False
) -> Assessment:
    """Fragt das Modell mit festem JSON-Schema; bei ungueltiger Antwort wird wiederholt."""
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": Assessment.model_json_schema(),
        "options": {"temperature": 0.4, "num_ctx": 4096, "num_predict": 800},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidate, streamer, section_confirmed)},
        ],
    }
    last_error: Exception | None = None
    for _ in range(attempts):
        response = post_chat(cfg, payload)
        try:
            return clean_assessment(
                Assessment.model_validate_json(response.json()["message"]["content"])
            )
        except (ValidationError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
    raise ValueError(f"Keine gültige Antwort vom Modell: {last_error}")


def post_chat(cfg: Llm, payload: dict) -> httpx.Response:
    """Wiederholt Serverfehler: Das Laden des Modells kann direkt nach Whisper kurz scheitern."""
    response = None
    for attempt in range(OLLAMA_ATTEMPTS):
        if attempt:
            time.sleep(OLLAMA_RETRY_SECONDS)
        try:
            response = httpx.post(
                f"{cfg.host}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS
            )
        except httpx.TransportError as exc:
            raise RuntimeError(f"Ollama ist nicht erreichbar: {exc}") from exc
        if response.status_code < 500:
            break
    if response.is_error:
        raise RuntimeError(f"Ollama meldet Fehler {response.status_code}: {response.text[:300]}")
    return response


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
        raise RuntimeError(f"Modell fehlt. Bitte ausführen: ollama pull {cfg.model}")


def unload(cfg: Llm) -> None:
    """Gibt den Grafikspeicher frei, damit Whisper oder der Renderer ihn nutzen koennen."""
    httpx.post(f"{cfg.host}/api/generate", json={"model": cfg.model, "keep_alive": 0}, timeout=60)


def streamer_talk(text: str) -> float:
    """0 bis 1: Wie sehr klingt der Text nach dem Streamer statt nach Spieldialog?

    Whisper trennt keine Stimmen, und das Modell haelt Dialog oft fuer eine Reaktion. Dieses
    einfache Signal fuehrt deshalb einen eigenen Anteil am Score.
    """
    return min(len(STREAMER_TALK_RE.findall(text)) / TALK_FULL_AT, 1.0)


def final_score(
    assessment: Assessment, energy: float, talk_score: float = 0.0, strength: float = 5.0
) -> float:
    """0 bis 100. Der Wert ordnet Kandidaten nur untereinander, er sagt keine Views voraus.

    energy ist die sichtbare Reaktion in der Facecam, ohne Facecam die allgemeine Signalkurve.
    Sie faengt Trailer-Dialog ab, den das Modell oft fuer eine Reaktion haelt.
    """
    llm = sum(getattr(assessment, name) * w for name, w in SCORE_WEIGHTS.items()) / 10
    return round(
        100
        * (
            LLM_SHARE * llm
            + ENERGY_SHARE * energy
            + STRENGTH_SHARE * strength / 10
            + TALK_SHARE * talk_score
        ),
        1,
    )
