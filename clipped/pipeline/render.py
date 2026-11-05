"""Setzt alles zum fertigen Hochformat-Clip zusammen und rendert ueber FFmpeg."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from clipped.config import ASSETS_DIR, Settings
from clipped.pipeline.censor import censor_text
from clipped.pipeline.common import VideoInfo, probe_video
from clipped.pipeline.effects import (
    EffectPlan,
    PartEffects,
    flash_filter,
    motion_filter,
    plan_effects,
)
from clipped.pipeline.endcard import compose_endcard
from clipped.pipeline.sfx import ensure_starter_pack, pick_sound
from clipped.pipeline.spec import ClipSpec
from clipped.pipeline.subtitles import Part, apply_text_edits, build_ass, flatten_words

WIDTH, HEIGHT = 1080, 1920
CAM_HEIGHT = 640
GAME_HEIGHT = HEIGHT - CAM_HEIGHT
FACECAM_ZOOM_SCALE = 0.5
OVERLAY_CAM_WIDTH = 360
OVERLAY_CAM_POSITION = (60, 250)
TEASER_LEAD = 0.6
TEASER_TAIL = 1.6
TEASER_MIN, TEASER_MAX = 1.2, 3.0
HOOK_AFTER_TEASER = 1.5
HOOK_WITHOUT_TEASER = 3.0
SFX_DIR = ASSETS_DIR / "sfx"
FONT_FILE = ASSETS_DIR / "fonts" / "Anton-Regular.ttf"
LOUDNESS_TOLERANCE = 0.5
# Der harte Einstieg in den ersten Millisekunden erzeugte im fertigen Ton Spitzen ueber 0 dBFS.
AUDIO_FADE_IN = "afade=t=in:st=0:d=0.02"
# Darunter ist praktisch nur Stille: Eine Verstaerkung wuerde nur Rauschen anheben.
SILENCE_LUFS = -50.0
REMOVE_ATTEMPTS = 10
REMOVE_RETRY_SECONDS = 0.3

Progress = Callable[[float], None]


def _even(value: float) -> int:
    """Videofilter brauchen gerade Werte; die Position 0 muss dabei erhalten bleiben."""
    return max(int(value) // 2 * 2, 0)


def fit_aspect(
    rect: tuple[int, int, int, int], aspect: float, frame: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Groesster Ausschnitt mit dem Seitenverhaeltnis innerhalb des Rechtecks.

    Horizontal zentriert, vertikal an der Oberkante: Koepfe sitzen oben im Kamerabild.
    """
    x, y, w, h = rect
    if w / h > aspect:
        new_w, new_h = h * aspect, h
    else:
        new_w, new_h = w, w / aspect
    new_x = min(max(x + (w - new_w) / 2, 0), frame[0] - new_w)
    new_y = min(max(y, 0), frame[1] - new_h)
    return _even(new_x), _even(new_y), _even(new_w), _even(new_h)


def game_crop(info: VideoInfo, center: float, aspect: float) -> tuple[int, int, int, int]:
    """Volle Hoehe des Quellbilds, horizontal um den gewaehlten Punkt."""
    w = min(_even(info.height * aspect), _even(info.width))
    x = min(max(center * info.width - w / 2, 0), info.width - w)
    return _even(x), 0, w, _even(info.height)


def _crop(rect: tuple[int, int, int, int]) -> str:
    x, y, w, h = rect
    return f"crop={w}:{h}:{x}:{y}"


def _chain(*filters: str) -> str:
    return ",".join(f for f in filters if f)


def part_video_graph(
    index: int, spec: ClipSpec, layout: str, info: VideoInfo, effects: PartEffects, fps: int
) -> list[str]:
    """Filter fuer ein Stueck des Quellvideos; das Ergebnis heisst [v<index>]."""
    start = f"[{index}:v]fps={fps},setpts=PTS-STARTPTS"
    game_motion = motion_filter(effects, fps)

    if layout == "split" and spec.cam_rect:
        game = game_crop(info, spec.game_center, WIDTH / GAME_HEIGHT)
        cam = fit_aspect(spec.cam_rect, WIDTH / CAM_HEIGHT, (info.width, info.height))
        cam_motion = motion_filter(effects, fps, zoom_scale=FACECAM_ZOOM_SCALE, shake=False)
        return [
            f"{start},split=2[c{index}][g{index}]",
            _chain(
                f"[c{index}]{_crop(cam)}",
                f"scale={WIDTH}:{CAM_HEIGHT}:flags=lanczos",
                "setsar=1",
                cam_motion,
            )
            + f"[co{index}]",
            _chain(
                f"[g{index}]{_crop(game)}",
                f"scale={WIDTH}:{GAME_HEIGHT}:flags=lanczos",
                "setsar=1",
                game_motion,
            )
            + f"[go{index}]",
            f"[co{index}][go{index}]vstack=inputs=2,setsar=1[v{index}]",
        ]

    if layout == "blur":
        return [
            f"{start},split=2[b{index}][f{index}]",
            (
                f"[b{index}]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},boxblur=24:2,eq=brightness=-0.12[bg{index}]"
            ),
            _chain(f"[f{index}]scale={WIDTH}:-2:flags=lanczos", "setsar=1", game_motion)
            + f"[fg{index}]",
            f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,setsar=1[v{index}]",
        ]

    # Gameplay im Vollbild; eine vorhandene Facecam liegt klein darueber.
    game = game_crop(info, spec.game_center, WIDTH / HEIGHT)
    if spec.cam_rect:
        x, y = OVERLAY_CAM_POSITION
        return [
            f"{start},split=2[c{index}][g{index}]",
            f"[c{index}]{_crop(spec.cam_rect)},scale={OVERLAY_CAM_WIDTH}:-2:flags=lanczos,setsar=1[co{index}]",
            _chain(
                f"[g{index}]{_crop(game)}",
                f"scale={WIDTH}:{HEIGHT}:flags=lanczos",
                "setsar=1",
                game_motion,
            )
            + f"[go{index}]",
            f"[go{index}][co{index}]overlay={x}:{y},setsar=1[v{index}]",
        ]
    return [
        _chain(start, _crop(game), f"scale={WIDTH}:{HEIGHT}:flags=lanczos", "setsar=1", game_motion)
        + f"[v{index}]"
    ]


def part_audio_graph(index: int, duration: float, has_audio: bool) -> str:
    if not has_audio:
        return f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f}[a{index}]"
    fade_start = max(duration - 0.04, 0.0)
    return (
        f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.02,afade=t=out:st={fade_start:.3f}:d=0.04[a{index}]"
    )


@dataclass(frozen=True)
class SfxInput:
    path: Path
    time: float
    volume: float


def build_filter_graph(
    spec: ClipSpec,
    layout: str,
    info: VideoInfo,
    parts: list[Part],
    plan: EffectPlan,
    sfx_inputs: list[SfxInput],
    fps: int,
    endcard_seconds: float,
    target_lufs: float,
    preview: bool,
    subtitle_file: str = "subs.ass",
    fonts_dir: str = "fonts",
) -> str:
    """Eingaben: erst die Stuecke, dann die Endkarte, dann die Sounds."""
    statements: list[str] = []
    count = len(parts)
    for index, (part, effects) in enumerate(zip(parts, plan.parts, strict=True)):
        statements.extend(part_video_graph(index, spec, layout, info, effects, fps))
        statements.append(part_audio_graph(index, part.duration, info.has_audio))

    endcard = count
    statements.append(f"[{endcard}:v]fps={fps},format=yuv420p,setsar=1,fade=t=in:st=0:d=0.2[ve]")
    statements.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={endcard_seconds:.3f}[ae]")
    pairs = "".join(f"[v{i}][a{i}]" for i in range(count))
    statements.append(f"{pairs}[ve][ae]concat=n={count + 1}:v=1:a=1[vc][ac]")

    scale = "scale=540:960" if preview else ""
    statements.append(
        _chain(
            "[vc]" + (flash_filter(plan.flashes) or "null"),
            f"ass={subtitle_file}:fontsdir={fonts_dir}",
            "format=yuv420p",
            scale,
        )
        + "[vout]"
    )

    mix_inputs = "[ac]"
    for offset, sound in enumerate(sfx_inputs):
        stream = count + 1 + offset
        delay = round(sound.time * 1000)
        statements.append(
            f"[{stream}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={sound.volume:.2f},adelay={delay}:all=1[s{offset}]"
        )
        mix_inputs += f"[s{offset}]"
    mix = (
        f"{mix_inputs}amix=inputs={len(sfx_inputs) + 1}:normalize=0:duration=first:dropout_transition=0,"
        if sfx_inputs
        else "[ac]"
    )
    statements.append(
        f"{mix}loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11,aresample=48000,{AUDIO_FADE_IN}[aout]"
    )
    return ";\n".join(statements)


def teaser_range(peak: float, start: float, end: float, words: list[dict]) -> tuple[float, float]:
    """Kurzer Ausschnitt um den Hoehepunkt, moeglichst an Wortgrenzen."""
    first = max(peak - TEASER_LEAD, start)
    last = min(peak + TEASER_TAIL, end)
    nearby = [w for w in words if first - 0.5 <= w["start"] <= last + 0.5]
    starts = [w["start"] for w in nearby if w["start"] <= first]
    ends = [w["end"] for w in nearby if w["end"] >= last]
    if starts:
        first = max(starts)
    if ends:
        last = min(ends)
    if last - first < TEASER_MIN:
        last = min(first + TEASER_MIN, end)
    if last - first > TEASER_MAX:
        last = first + TEASER_MAX
    return first, last


def build_parts(spec: ClipSpec, words: list[dict]) -> list[Part]:
    duration = spec.end - spec.start
    if not spec.teaser or duration < TEASER_MIN * 2:
        return [Part(spec.start, spec.end, 0.0)]
    first, last = teaser_range(spec.peak, spec.start, spec.end, words)
    return [Part(first, last, 0.0), Part(spec.start, spec.end, last - first)]


def resolve_layout(spec: ClipSpec) -> str:
    """Ohne bekannte Facecam kann das geteilte Layout nicht funktionieren."""
    return "gameplay" if spec.layout == "split" and not spec.cam_rect else spec.layout


def encoder_args(encoder: str, preview: bool) -> list[str]:
    if encoder.endswith("_nvenc"):
        quality = ["-preset", "p1", "-cq", "30"] if preview else ["-preset", "p5", "-cq", "20"]
        return [
            "-c:v",
            encoder,
            *quality,
            "-rc",
            "vbr",
            "-b:v",
            "0",
            "-maxrate",
            "24M",
            "-bufsize",
            "48M",
            "-profile:v",
            "high",
        ]
    return [
        "-c:v",
        encoder,
        "-preset",
        "ultrafast" if preview else "medium",
        "-crf",
        "30" if preview else "20",
    ]


def _run_with_progress(cmd: list[str], cwd: Path, total: float, on_progress: Progress) -> None:
    log_path = cwd / "ffmpeg.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=log, text=True)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if line.startswith("out_time_us=") and total > 0:
                    try:
                        value = min(int(line.split("=", 1)[1]) / 1_000_000 / total, 1.0)
                    except ValueError:
                        continue
                    on_progress(value)
        except BaseException:
            # Abbruch aus dem Fortschritts-Callback: FFmpeg darf nicht weiterlaufen.
            _kill_process_tree(process)
            raise
        code = process.wait()
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError(f"Rendern fehlgeschlagen ({code}): {tail}")


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Der scoop-Aufruf von ffmpeg ist nur ein Shim; das echte FFmpeg laeuft als Kindprozess."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False
        )
    process.kill()
    process.wait()


def _remove_partial_file(path: Path) -> None:
    """Windows gibt die Datei erst kurz nach dem Prozessende frei."""
    for _ in range(REMOVE_ATTEMPTS):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(REMOVE_RETRY_SECONDS)
    # Darf den eigentlichen Grund des Abbruchs nicht ueberdecken; beim naechsten Lauf wird
    # dieselbe Datei ohnehin ueberschrieben.
    print(f"Temporaere Datei konnte nicht entfernt werden: {path}")


def measure_loudness(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vn",
            "-af",
            "ebur128",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", result.stderr)
    return float(found[-1]) if found else None


def _correct_loudness(path: Path, target: float) -> None:
    """Der Lautheitsfilter im Rendern verfehlt auf echtem Ton das Ziel um bis zu 4 LU.

    Das Messen am fertigen Ton und eine feste Verstaerkung treffen es zuverlaessig und aendern die
    Dynamik nicht. Das Bild wird dabei nur kopiert, das kostet wenige Sekunden.
    """
    measured = measure_loudness(path)
    if measured is None or measured < SILENCE_LUFS or abs(measured - target) <= LOUDNESS_TOLERANCE:
        return
    corrected = path.with_name(path.stem + ".loud.mp4")
    gain = target - measured
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-map", "0", "-c:v", "copy",
        "-af", f"volume={gain:.2f}dB,aresample=192000,alimiter=limit=0.75:level=false,aresample=48000,{AUDIO_FADE_IN}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", str(corrected),
    ]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Lautstaerke-Korrektur fehlgeschlagen: {result.stderr[-1000:]}")
        corrected.replace(path)
    finally:
        _remove_partial_file(corrected)


def _extract_frame(video: str, time: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(time, 0):.3f}",
        "-i",
        video,
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def render_clip(
    spec: ClipSpec,
    meta: dict,
    segments: list[dict],
    settings: Settings,
    out_path: Path,
    work_dir: Path,
    preview: bool = False,
    on_progress: Progress = lambda value: None,
) -> Path:
    """Rendert in einen eigenen Arbeitsordner; FFmpeg laeuft mit relativen Pfaden darin.

    Laufwerks-Doppelpunkte und Leerzeichen in Pfaden sind sonst in Filtern eine Fehlerquelle.
    """
    video = meta["video"]
    info = probe_video(Path(video))
    fps = settings.render.fps
    endcard_seconds = settings.branding.endcard_seconds

    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "fonts").mkdir(parents=True)
    shutil.copy(FONT_FILE, work_dir / "fonts" / FONT_FILE.name)

    words = apply_text_edits(flatten_words(segments, spec.start, spec.end), spec.text_edits)
    parts = build_parts(spec, words)
    censor_level = settings.style.censor
    visible_words = [{**w, "word": censor_text(w["word"], censor_level)} for w in words]
    layout = resolve_layout(spec)
    body_end = parts[-1].out_end
    teaser_length = parts[0].duration if len(parts) > 1 else 0.0
    hook_until = teaser_length + HOOK_AFTER_TEASER if teaser_length else HOOK_WITHOUT_TEASER

    style = settings.style
    (work_dir / "subs.ass").write_text(
        build_ass(
            parts,
            visible_words,
            layout,
            censor_text(spec.hook_text, censor_level),
            hook_until,
            settings.branding.watermark,
            spec.credit,
            style.highlight_color,
            style.subtitle_size,
        ),
        encoding="utf-8",
    )

    boundaries = [s["start"] for s in segments if spec.start < s["start"] < spec.end]
    plan = plan_effects(parts, spec.peak, boundaries, spec.effects, body_end)

    sfx_inputs: list[SfxInput] = []
    ensure_starter_pack(SFX_DIR)
    for number, event in enumerate(plan.sfx):
        sound = pick_sound(SFX_DIR, event.category, f"{spec.id}-{number}")
        if sound:
            sfx_inputs.append(SfxInput(sound, event.time, event.volume))

    last_frame = work_dir / "last.png"
    _extract_frame(video, spec.end - 0.1, last_frame)
    compose_endcard(
        last_frame,
        settings.branding.endcard_title,
        settings.branding.channels,
        work_dir / "endcard.png",
    )

    graph = build_filter_graph(
        spec,
        layout,
        info,
        parts,
        plan,
        sfx_inputs,
        fps,
        endcard_seconds,
        settings.render.target_lufs,
        preview,
    )
    (work_dir / "graph.txt").write_text(graph, encoding="utf-8")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y", "-progress", "pipe:1"]
    for part in parts:
        cmd += ["-ss", f"{part.src_start:.3f}", "-t", f"{part.duration:.3f}", "-i", video]
    cmd += [
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-t",
        f"{endcard_seconds:.3f}",
        "-i",
        "endcard.png",
    ]
    for sound in sfx_inputs:
        cmd += ["-i", str(sound.path)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out_path.with_name(out_path.stem + ".tmp.mp4")
    cmd += [
        "-/filter_complex",
        "graph.txt",
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        *encoder_args(settings.render.video_encoder, preview),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(temp_out.resolve()),
    ]
    try:
        _run_with_progress(cmd, work_dir, body_end + endcard_seconds, on_progress)
        if not preview:
            _correct_loudness(temp_out, settings.render.target_lufs)
        temp_out.replace(out_path)
    finally:
        # Nach Abbruch oder Fehler bleibt sonst eine halbe Datei im Export-Ordner liegen.
        _remove_partial_file(temp_out)
    on_progress(1.0)
    return out_path
