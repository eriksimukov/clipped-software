"""Beschreibt einen Clip so, wie er gerendert wird. Die Oberflaeche bearbeitet genau dieses Objekt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Layout = Literal["split", "gameplay", "blur"]


class Effects(BaseModel):
    zoom: bool = True
    shake: bool = True
    flash: bool = True
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
    return ClipSpec(
        id=candidate["id"],
        start=candidate["start"],
        end=candidate["end"],
        peak=candidate["peak"],
        hook_text=hooks[0] if hooks else "",
        layout=cam.get("layout", "gameplay"),
        cam_rect=tuple(cam["rect"]) if cam.get("rect") else None,
        credit=meta.get("streamer", ""),
        title=assessment.get("title", ""),
        description=assessment.get("description", ""),
        hashtags=assessment.get("hashtags", []),
    )
