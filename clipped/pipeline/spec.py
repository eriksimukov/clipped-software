"""Beschreibt einen Clip so, wie er gerendert wird. Die Oberflaeche bearbeitet genau dieses Objekt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Layout = Literal["split", "gameplay", "blur"]


class Effects(BaseModel):
    zoom: bool = True
    # Shake und Flash wirkten im Test aufgesetzt; sie bleiben als Option im Editor.
    shake: bool = False
    flash: bool = False
    # Greift nur, wenn eigene Dateien in assets/sfx liegen.
    sfx: bool = True


class TextEdit(BaseModel):
    """Ersetzt alle Woerter, die in diesem Zeitraum beginnen, durch den neuen Text."""

    start: float
    end: float
    text: str


class ClipSpec(BaseModel):
    id: str
    start: float
    end: float
    peak: float
    hook_text: str = ""
    teaser: bool = True
    layout: Layout = "split"
    # Facecam im Quellvideo in Pixeln: x, y, Breite, Hoehe.
    cam_rect: tuple[int, int, int, int] | None = None
    # Mitte des Gameplay-Ausschnitts, 0 ist der linke und 1 der rechte Bildrand.
    game_center: float = Field(default=0.5, ge=0.0, le=1.0)
    # Anteil des Quellbilds, der oben und unten wegfaellt, etwa wegen Chat oder Overlays.
    game_top: float = Field(default=0.0, ge=0.0, le=0.4)
    game_bottom: float = Field(default=0.0, ge=0.0, le=0.4)
    # Stelle fuer den Teaser. Ohne Angabe gilt der Satz um den Hoehepunkt.
    teaser_start: float | None = None
    teaser_end: float | None = None
    credit: str = ""
    effects: Effects = Field(default_factory=Effects)
    text_edits: list[TextEdit] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)


def spec_from_candidate(candidate: dict, meta: dict) -> ClipSpec:
    """Startwerte fuer einen frisch bewerteten Kandidaten."""
    assessment = candidate.get("assessment") or {}
    hooks = assessment.get("hooks") or []
    cam = meta.get("cam") or {}
    game = cam.get("game") or {}
    return ClipSpec(
        id=candidate["id"],
        start=candidate["start"],
        end=candidate["end"],
        peak=candidate["peak"],
        hook_text=hooks[0] if hooks else "",
        layout=cam.get("layout", "gameplay"),
        cam_rect=tuple(cam["rect"]) if cam.get("rect") else None,
        game_center=game.get("center", 0.5),
        game_top=game.get("top", 0.0),
        game_bottom=game.get("bottom", 0.0),
        teaser_start=candidate.get("teaser_start"),
        teaser_end=candidate.get("teaser_end"),
        credit=meta.get("streamer", ""),
        title=assessment.get("title") or (hooks[0] if hooks else ""),
        description=assessment.get("description", ""),
        hashtags=assessment.get("hashtags", []),
    )
