"""Vergleicht lokale Sprachmodelle an denselben, von Hand beschrifteten Beispiel-Momenten.

Aufruf: uv run python scripts/compare_models.py qwen3:8b gemma4:e4b llama3.1:8b

Die Beispiele sind ausgedacht, nicht aus echten VODs. Der Vergleich zeigt also, ob ein Modell
zuverlaessig gueltiges JSON liefert, schnell genug ist und starke von schwachen Momenten trennt.
Er zeigt nicht, wie gut die Auswahl auf echtem Material wird.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipped.config import Llm, load_settings
from clipped.pipeline.llm import SCORE_WEIGHTS, assess, unload

# kind: strong = starker Moment, weak = langweilig, off = kein GTA 6
SAMPLES = [
    (
        "strong",
        (
            "Okay chat watch this, I'm gonna jump the bridge in Vice City. Three, two, one. NO WAY! "
            "He landed on the police helicopter! Bro the helicopter just exploded and I'm still alive! "
            "That is the craziest thing I have ever seen in GTA 6. Clip that, clip that right now!"
        ),
    ),
    (
        "strong",
        (
            "Wait, is that Lucia? They just killed Lucia in the first mission? Are you serious? "
            "No no no no, Rockstar, you cannot do this to me. I have waited thirteen years for this game "
            "and you kill her in the first hour? I'm actually shaking right now, chat."
        ),
    ),
    (
        "strong",
        (
            "Oh my god, oh my god, look at the graphics, look at the water! Every single person "
            "on this beach is doing something different. That guy is filming me! He just posted me on the "
            "in-game social media! I am on the news now! GTA 6 is not real, this cannot be a video game."
        ),
    ),
    (
        "strong",
        (
            "Chat I just found out you can rob the same bank twice. So I go back, right, same "
            "bank, the teller recognizes me. She recognizes me! She pushes the alarm before I even pull the "
            "gun. And now the whole of Leonida has a five star wanted level on me for walking into a bank."
        ),
    ),
    (
        "strong",
        (
            "I have forty seconds left and two hundred thousand dollars in the trunk. Come on, come "
            "on, come on. Train! There's a train! I'm not gonna make it. I MADE IT! I went under the train! "
            "Let's go! Let's go! Did you see that? GTA 6 is the greatest game ever made!"
        ),
    ),
    (
        "weak",
        (
            "So yeah, I'm just gonna drive over to the next mission marker now. It's kind of far, "
            "let me set the waypoint. Okay. I think I need to change the radio station. This one's fine. "
            "Yeah, so I had lunch earlier, nothing special. Anyway, we're driving in GTA 6."
        ),
    ),
    (
        "weak",
        (
            "Let me check the settings real quick, I want to turn the camera sensitivity down a "
            "little bit. There we go. And I'm going to lower the music volume. Maybe a bit more. Okay, that's "
            "better. Let's see the map. Alright, we still have a couple of side missions in Vice City."
        ),
    ),
    (
        "weak",
        (
            "Thanks for the five gifted, appreciate you. Thanks for the follow. Welcome in everybody, "
            "we've been playing GTA 6 for about two hours now. If you're new here, the schedule is in the "
            "description. Yeah. Thanks for the sub. Okay, where was I, oh yeah, the clothing store."
        ),
    ),
    (
        "off",
        (
            "Okay we're back on Fortnite now. This new season is pretty good, the map changed a lot. "
            "I got a triple kill last game and my duo was carrying me. Let me queue up again. Oh my god, "
            "no way, I just hit the craziest sniper shot across the whole map! Chat did you see that?"
        ),
    ),
    (
        "off",
        (
            "My cat knocked over my whole setup this morning. Like the monitor, the microphone, "
            "everything on the floor. No way, right? And then my landlord called and said the rent is going "
            "up again, so that was a great start. Anyway, how is everybody doing today, chat?"
        ),
    ),
]


def weighted(assessment) -> float:
    return sum(getattr(assessment, name) * w for name, w in SCORE_WEIGHTS.items())


def evaluate(model: str, host: str) -> dict:
    cfg = Llm(host=host, model=model)
    scores: dict[str, list[float]] = {"strong": [], "weak": [], "off": []}
    relevance: dict[str, list[int]] = {"strong": [], "weak": [], "off": []}
    durations, hook_words, examples = [], [], []
    failures = 0

    for kind, text in SAMPLES:
        candidate = {"start": 0.0, "end": 30.0, "peak": 15.0, "text": text}
        started = time.monotonic()
        try:
            result = assess(candidate, "TestStreamer", cfg, attempts=1)
        except ValueError:
            failures += 1
            continue
        durations.append(time.monotonic() - started)
        scores[kind].append(weighted(result))
        relevance[kind].append(result.gta6_relevance)
        hook_words.extend(len(hook.split()) for hook in result.hooks)
        if kind == "strong" and len(examples) < 2:
            examples.append(result.hooks[0])

    loaded = httpx.get(f"{host}/api/ps", timeout=10).json().get("models", [])
    gpu_share = None
    if loaded and loaded[0].get("size"):
        gpu_share = loaded[0].get("size_vram", 0) / loaded[0]["size"]
    unload(cfg)

    mean = lambda values: round(statistics.mean(values), 2) if values else None
    return {
        "model": model,
        "failures": failures,
        "seconds_first": round(durations[0], 1) if durations else None,
        "seconds_median": round(statistics.median(durations), 1) if durations else None,
        "score_strong": mean(scores["strong"]),
        "score_weak": mean(scores["weak"]),
        "score_off": mean(scores["off"]),
        "gta6_strong": mean(relevance["strong"]),
        "gta6_off": mean(relevance["off"]),
        "hooks_too_long": sum(1 for words in hook_words if words > 8),
        "hooks_total": len(hook_words),
        "gpu_share": round(gpu_share, 2) if gpu_share is not None else None,
        "examples": examples,
    }


def main() -> None:
    host = load_settings().llm.host
    for model in sys.argv[1:]:
        print(json.dumps(evaluate(model, host), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
