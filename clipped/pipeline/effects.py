"""Plant Zooms, Shake, Flashes und Sounds und uebersetzt sie in FFmpeg-Ausdruecke."""

from __future__ import annotations

from dataclasses import dataclass, field

from clipped.pipeline.spec import Effects
from clipped.pipeline.subtitles import Part

PEAK_ZOOM = 0.12
PEAK_HOLD_SECONDS = 1.6
RHYTHM_ZOOM = 0.04
RHYTHM_INTERVAL_SECONDS = 6.0
TEASER_ZOOM = 0.06
SHAKE_SECONDS = 0.4
SHAKE_PIXELS = 14.0
# Ohne Zoom waere beim Wackeln der Bildrand sichtbar.
SHAKE_MIN_ZOOM = 0.04


@dataclass(frozen=True)
class Pulse:
    start: float
    end: float
    amount: float
    rise: float = 0.12
    fall: float = 0.25


@dataclass(frozen=True)
class Shake:
    start: float
    duration: float = SHAKE_SECONDS
    pixels: float = SHAKE_PIXELS


@dataclass(frozen=True)
class Flash:
    time: float
    duration: float
    amount: float


@dataclass(frozen=True)
class SfxEvent:
    time: float
    category: str
    volume: float


@dataclass
class PartEffects:
    pulses: list[Pulse] = field(default_factory=list)
    shakes: list[Shake] = field(default_factory=list)


@dataclass
class EffectPlan:
    parts: list[PartEffects]
    flashes: list[Flash]
    sfx: list[SfxEvent]


def rhythm_pulses(duration: float, peak: float, boundaries: list[float]) -> list[Pulse]:
    """Abwechselnd leicht ran und zurueck an Satzgrenzen. Das haelt das Bild in Bewegung.

    Rund um den Hoehepunkt bleibt es ruhig, damit dessen Zoom heraussticht.
    """
    quiet_start, quiet_end = peak - 1.5, peak + PEAK_HOLD_SECONDS + 1.0
    cuts: list[float] = []
    last = 0.0
    for boundary in sorted(boundaries):
        if boundary - last >= RHYTHM_INTERVAL_SECONDS and boundary < duration - 1.0:
            cuts.append(boundary)
            last = boundary
    pulses = []
    for begin, finish in zip(cuts[::2], cuts[1::2] + [duration], strict=False):
        if begin < quiet_end and finish > quiet_start:
            finish = quiet_start
        if finish - begin >= 1.0:
            pulses.append(Pulse(begin, finish, RHYTHM_ZOOM, rise=0.1, fall=0.1))
    return pulses


def plan_effects(
    parts: list[Part], peak: float, boundaries: list[float], effects: Effects, endcard_start: float
) -> EffectPlan:
    """peak und boundaries sind Quellzeiten; das letzte Stueck ist immer der eigentliche Clip."""
    main = parts[-1]
    teaser = parts[0] if len(parts) > 1 else None
    peak_rel = min(max(peak - main.src_start, 0.0), main.duration)
    peak_out = main.out_start + peak_rel

    plans = [PartEffects() for _ in parts]
    main_plan = plans[-1]
    flashes: list[Flash] = []
    sfx: list[SfxEvent] = []

    if effects.zoom:
        relative = [b - main.src_start for b in boundaries if main.src_start < b < main.src_end]
        main_plan.pulses.extend(rhythm_pulses(main.duration, peak_rel, relative))
        main_plan.pulses.append(Pulse(peak_rel - 0.05, peak_rel + PEAK_HOLD_SECONDS, PEAK_ZOOM))
        if teaser:
            plans[0].pulses.append(Pulse(-1.0, teaser.duration + 1.0, TEASER_ZOOM))

    if effects.shake:
        main_plan.shakes.append(Shake(peak_rel))
        if not effects.zoom:
            main_plan.pulses.append(
                Pulse(peak_rel - 0.1, peak_rel + SHAKE_SECONDS, SHAKE_MIN_ZOOM, rise=0.05, fall=0.1)
            )

    if effects.flash:
        flashes.append(Flash(peak_out, 0.14, 0.55))
        if teaser:
            flashes.append(Flash(main.out_start, 0.12, 0.7))

    if effects.sfx:
        # Sparsam: Echte Clips leben vom Originalton.
        if teaser:
            sfx.append(SfxEvent(max(main.out_start - 0.22, 0.0), "whoosh", 0.4))
        sfx.append(SfxEvent(peak_out, "impact", 0.5))

    return EffectPlan(
        plans, sorted(flashes, key=lambda f: f.time), sorted(sfx, key=lambda s: s.time)
    )


def _pulse_term(pulse: Pulse, time_var: str, scale: float) -> str:
    return (
        f"{pulse.amount * scale:.4f}"
        f"*clip(({time_var}-{pulse.start:.3f})/{pulse.rise:.3f},0,1)"
        f"*(1-clip(({time_var}-{pulse.end:.3f})/{pulse.fall:.3f},0,1))"
    )


def zoom_expr(pulses: list[Pulse], time_var: str, scale: float = 1.0) -> str:
    return "+".join(["1", *(_pulse_term(p, time_var, scale) for p in pulses)])


def _shake_expr(shakes: list[Shake], time_var: str, wave: str) -> str:
    terms = [
        f"{s.pixels:.1f}*{wave}({time_var}*{61 if wave == 'sin' else 47})"
        f"*between({time_var},{s.start:.3f},{s.start + s.duration:.3f})"
        f"*(1-({time_var}-{s.start:.3f})/{s.duration:.3f})"
        for s in shakes
    ]
    return "+".join(terms) if terms else "0"


def motion_filter(plan: PartEffects, fps: int, zoom_scale: float = 1.0, shake: bool = True) -> str:
    """perspective statt zoompan: rechnet subpixelgenau und ruckelt bei Zooms nicht."""
    shakes = plan.shakes if shake else []
    if not plan.pulses and not shakes:
        return ""
    time_var = f"(in/{fps})"
    zoom = zoom_expr(plan.pulses, time_var, zoom_scale)
    inset_x = f"(W*(1-1/({zoom}))/2)"
    inset_y = f"(H*(1-1/({zoom}))/2)"
    shift_x = _shake_expr(shakes, time_var, "sin")
    shift_y = _shake_expr(shakes, time_var, "cos")
    left, right = f"{inset_x}+({shift_x})", f"W-{inset_x}+({shift_x})"
    top, bottom = f"{inset_y}+({shift_y})", f"H-{inset_y}+({shift_y})"
    return (
        f"perspective=x0='{left}':y0='{top}':x1='{right}':y1='{top}':"
        f"x2='{left}':y2='{bottom}':x3='{right}':y3='{bottom}':"
        "interpolation=linear:eval=frame"
    )


def flash_filter(flashes: list[Flash]) -> str:
    if not flashes:
        return ""
    terms = [
        f"{f.amount:.2f}*gte(t,{f.time:.3f})*clip(1-(t-{f.time:.3f})/{f.duration:.3f},0,1)"
        for f in flashes
    ]
    return f"eq=brightness='{'+'.join(terms)}':eval=frame"
