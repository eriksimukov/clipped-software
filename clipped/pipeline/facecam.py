"""Findet die Facecam: ein Gesicht, das ueber viele Frames an derselben Stelle bleibt.

Gesichter von Spielfiguren wandern und fallen deshalb raus. Der Rand der Facecam ist eine Kante,
die in jedem Frame an derselben Stelle liegt und hinter der sich das Bild staendig aendert.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from clipped.config import ASSETS_DIR
from clipped.pipeline.common import VideoInfo

MODEL_PATH = ASSETS_DIR / "facecam" / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 960
SAMPLE_FRAMES = 24
MIN_FACE_SCORE = 0.7
MIN_PRESENCE = 0.4
# Ab dieser Gesichtsbreite ist es ein Vollbild-Kamerabild statt einer eingeblendeten Facecam.
FULLSCREEN_FACE_RATIO = 0.18
MAX_CAM_AREA_RATIO = 0.4
# Suchweite ab dem Gesicht, in Gesichtsbreiten beziehungsweise -hoehen.
SEARCH_SIDEWAYS = 3.5
SEARCH_VERTICAL = 2.5
# Breite des Kamerabilds, wenn kein Rand erkennbar ist. Eng genug, dass bei freigestellten
# Streamern kein eingebrannter Text aus dem Spiel mit ins Bild rutscht.
CAM_WIDTH_IN_FACES = 3.4
MIN_EDGE_STRENGTH = 8.0
# Eine Kante muss deutlich ueber dem Umfeld liegen, sonst ist sie nur Bildstruktur.
EDGE_PROMINENCE = 2.5
# Mittlere Helligkeitsstreuung hinter dem Rand. Kanten im Zimmer haben auf beiden Seiten Ruhe.
MIN_OUTER_MOTION = 10.0
OUTER_BAND_PIXELS = 6
MAX_EDGE_CANDIDATES = 8
# Overlays: Pixel, an denen in so vielen Frames eine Kante liegt, gehoeren nicht zum Spiel.
OVERLAY_EDGE_THRESHOLD = 24
OVERLAY_PRESENCE = 0.25
OVERLAY_CLOSE_PIXELS = 15
OVERLAY_MIN_AREA = 40
OVERLAY_TOP_ZONE = 0.15
OVERLAY_BOTTOM_ZONE = 0.1
CENTER_COST = 1.5
SPLIT_GAME_ASPECT = 1080 / 1280
FULL_GAME_ASPECT = 9 / 16
CROP_TOPS = tuple(i / 100 for i in range(0, 15, 1))
CROP_BOTTOMS = tuple(i / 100 for i in range(0, 9, 2))
CROP_CENTERS = tuple(i / 100 for i in range(44, 57))


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def cluster_faces(detections: list[list[Face]], min_presence: float = MIN_PRESENCE) -> Face | None:
    """Gibt das Gesicht zurueck, das am haeufigsten ungefaehr an derselben Stelle auftaucht."""
    clusters: list[dict] = []
    for frame_index, faces in enumerate(detections):
        for face in faces:
            cx, cy = face.center
            for cluster in clusters:
                reference = cluster["faces"][0]
                rx, ry = reference.center
                if abs(cx - rx) < reference.w and abs(cy - ry) < reference.h:
                    cluster["faces"].append(face)
                    cluster["frames"].add(frame_index)
                    break
            else:
                clusters.append({"faces": [face], "frames": {frame_index}})
    if not clusters:
        return None
    best = max(clusters, key=lambda c: len(c["frames"]))
    if len(best["frames"]) < max(2, min_presence * len(detections)):
        return None
    return Face(
        float(np.median([f.x for f in best["faces"]])),
        float(np.median([f.y for f in best["faces"]])),
        float(np.median([f.w for f in best["faces"]])),
        float(np.median([f.h for f in best["faces"]])),
    )


def _fallback_rect(face: Face, width: int, height: int) -> tuple[int, int, int, int]:
    """Uebliches 16:9-Kamerabild um das Gesicht, fuer Seiten ohne erkennbaren Rand."""
    w = min(face.w * CAM_WIDTH_IN_FACES, width)
    h = min(w * 9 / 16, height)
    cx, cy = face.center
    x = min(max(cx - w / 2, 0), width - w)
    y = min(max(cy - h * 0.45, 0), height - h)
    return int(x), int(y), int(w), int(h)


def _find_border(
    edges: np.ndarray, motion: np.ndarray, near: int, far: int, outward: int
) -> int | None:
    """Sucht vom Gesicht nach aussen; edges[i] ist die Kante zwischen Position i und i + 1."""
    low, high = sorted((near, far))
    low, high = max(low, 0), min(high, len(edges) - 1)
    if high <= low:
        return None
    band = edges[low : high + 1]
    threshold = max(MIN_EDGE_STRENGTH, EDGE_PROMINENCE * float(np.median(band)))
    for offset in np.argsort(-band)[:MAX_EDGE_CANDIDATES]:
        index = low + int(offset)
        if edges[index] < threshold:
            break
        # Die Kante selbst verschmiert ueber ein bis zwei Pixel, deshalb mit Abstand messen.
        if outward < 0:
            outer = motion[max(index - OUTER_BAND_PIXELS - 1, 0) : max(index - 1, 0)]
        else:
            outer = motion[index + 3 : index + 3 + OUTER_BAND_PIXELS]
        if len(outer) and float(outer.mean()) >= MIN_OUTER_MOTION:
            return index + 1
    return None


def find_cam_rect(gray_frames: np.ndarray, face: Face) -> tuple[int, int, int, int]:
    """Bestimmt jede Seite einzeln; ohne klaren Rand gilt dort das uebliche Kamerabild."""
    height, width = gray_frames.shape[1:]
    frames = gray_frames.astype(np.int16)
    across = np.median(np.abs(np.diff(frames, axis=2)), axis=0)
    down = np.median(np.abs(np.diff(frames, axis=1)), axis=0)
    motion = gray_frames.std(axis=0)

    # Profile auf Hoehe des Gesichts: Dort verlaeuft sicher ein Stueck jedes Rands.
    y0, y1 = max(int(face.y - face.h * 0.25), 0), min(int(face.y + face.h * 1.25), height)
    x0, x1 = max(int(face.x - face.w * 0.25), 0), min(int(face.x + face.w * 1.25), width)
    column_edges, column_motion = across[y0:y1].mean(axis=0), motion[y0:y1].mean(axis=0)
    row_edges, row_motion = down[:, x0:x1].mean(axis=1), motion[:, x0:x1].mean(axis=1)

    right_of_face, below_face = int(face.x + face.w), int(face.y + face.h)
    left = _find_border(
        column_edges,
        column_motion,
        int(face.x - face.w * 0.1),
        int(face.x - face.w * SEARCH_SIDEWAYS),
        -1,
    )
    right = _find_border(
        column_edges,
        column_motion,
        right_of_face + int(face.w * 0.1),
        right_of_face + int(face.w * SEARCH_SIDEWAYS),
        1,
    )
    top = _find_border(
        row_edges,
        row_motion,
        int(face.y - face.h * 0.05),
        int(face.y - face.h * SEARCH_VERTICAL),
        -1,
    )
    bottom = _find_border(
        row_edges,
        row_motion,
        below_face + int(face.h * 0.05),
        below_face + int(face.h * SEARCH_VERTICAL),
        1,
    )

    # Kameras sitzen meist in einer Ecke. Dort ist der Bildrand zugleich ihr Rand, also gibt es
    # keine Kante zu finden; freigestellte Streamer haben gar keine sichtbaren Raender.
    fx, fy, fw, fh = _fallback_rect(face, width, height)
    if left is None:
        left = 0 if face.x - face.w * SEARCH_SIDEWAYS <= 0 else fx
    if top is None:
        top = 0 if face.y - face.h * SEARCH_VERTICAL <= 0 else fy
    if right is None:
        right = width if right_of_face + face.w * SEARCH_SIDEWAYS >= width else fx + fw
    if bottom is None:
        bottom = height if below_face + face.h * SEARCH_VERTICAL >= height else fy + fh
    return left, top, right - left, bottom - top


def locate_cam(
    detections: list[list[Face]], gray_frames: np.ndarray
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Gibt Rechteck in Erkennungs-Pixeln und passendes Layout zurueck."""
    height, width = gray_frames.shape[1:]
    face = cluster_faces(detections)
    if face is None:
        return None, "gameplay"
    if face.w / width >= FULLSCREEN_FACE_RATIO:
        return None, "blur"

    rect = find_cam_rect(gray_frames, face)
    x, y, w, h = rect
    contains_face = (
        x <= face.x and y <= face.y and x + w >= face.x + face.w and y + h >= face.y + face.h
    )
    if not contains_face or w * h > MAX_CAM_AREA_RATIO * width * height:
        return _fallback_rect(face, width, height), "split"
    return rect, "split"


def persistent_edges(gray_frames: np.ndarray) -> np.ndarray:
    """Je Pixel: wie oft dort eine Kante liegt. Das Spiel aendert sich, Overlays bleiben."""
    frames = gray_frames.astype(np.int16)
    across = np.abs(np.diff(frames, axis=2))[:, :-1, :]
    down = np.abs(np.diff(frames, axis=1))[:, :, :-1]
    return (np.maximum(across, down) > OVERLAY_EDGE_THRESHOLD).mean(axis=0)


def overlay_mask(presence: np.ndarray) -> np.ndarray:
    """Flaechen, die zu Overlays gehoeren. Nur an den Bildraendern, wo Chat und Einblendungen sitzen.

    In der Bildmitte liegen eher eingebrannte Untertitel des gezeigten Videos, und die sind Inhalt.
    Schrift besteht aus einzelnen Kanten; geschlossen wird daraus eine zusammenhaengende Flaeche.
    Kleine Reste danach sind Stoerungen und zaehlen nicht.
    """
    height = presence.shape[0]
    closing = np.ones((OVERLAY_CLOSE_PIXELS, OVERLAY_CLOSE_PIXELS), np.uint8)
    closed = cv2.morphologyEx(
        (presence > OVERLAY_PRESENCE).astype(np.uint8), cv2.MORPH_CLOSE, closing
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros(presence.shape, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= OVERLAY_MIN_AREA:
            mask[labels == label] = True
    mask[int(height * OVERLAY_TOP_ZONE) : int(height * (1 - OVERLAY_BOTTOM_ZONE))] = False
    return mask


def best_game_crop(
    presence: np.ndarray, cam_rect: tuple[int, int, int, int] | None, aspect: float
) -> dict:
    """Waehlt den Ausschnitt des Spiels so, dass weder Chat noch Overlays noch Facecam darin liegen.

    Sonst schauen Reste hinein, im Jynxzi-Test etwa die Badges der Chat-Zeilen oben rechts.
    """
    height, width = presence.shape
    blocked = overlay_mask(presence)
    if cam_rect:
        x, y, w, h = cam_rect
        blocked[y : y + h, x : x + w] = True
    table = np.zeros((height + 1, width + 1))
    table[1:, 1:] = blocked.cumsum(axis=0).cumsum(axis=1)

    best = {"center": 0.5, "top": 0.0, "bottom": 0.0}
    best_cost = float("inf")
    for top in CROP_TOPS:
        for bottom in CROP_BOTTOMS:
            crop_h = height * (1 - top - bottom)
            crop_w = min(crop_h * aspect, width)
            for center in CROP_CENTERS:
                x0 = int(min(max(center * width - crop_w / 2, 0), width - crop_w))
                x1, y0 = int(x0 + crop_w), int(top * height)
                y1 = int(y0 + crop_h)
                count = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
                share = count / (crop_w * crop_h)
                # Jeder Overlay-Rest wiegt schwer; verlorenes Bild und Abstand zur Mitte kosten wenig.
                cost = share * 1000 + top + bottom + abs(center - 0.5) * CENTER_COST
                if cost < best_cost:
                    best_cost = cost
                    best = {"center": center, "top": top, "bottom": bottom}
    return best


def read_frame(video: Path | str, time: float, info: VideoInfo) -> np.ndarray | None:
    height = round(info.height * DETECT_WIDTH / info.width / 2) * 2
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{time:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={DETECT_WIDTH}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    expected = DETECT_WIDTH * height * 3
    if result.returncode != 0 or len(result.stdout) != expected:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, DETECT_WIDTH, 3)


def detect_cam(video: Path | str, times: list[float], info: VideoInfo) -> dict:
    """Ergebnis fuer meta.json: Rechteck in Quell-Pixeln (oder None) und vorgeschlagenes Layout."""
    frames = [f for f in (read_frame(video, t, info) for t in times) if f is not None]
    if len(frames) < 2:
        return {
            "rect": None,
            "layout": "gameplay",
            "game": {"center": 0.5, "top": 0.0, "bottom": 0.0},
        }

    height, width = frames[0].shape[:2]
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), MIN_FACE_SCORE)
    detections: list[list[Face]] = []
    for frame in frames:
        _, faces = detector.detect(frame)
        detections.append([Face(*map(float, f[:4])) for f in faces] if faces is not None else [])
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])

    rect, layout = locate_cam(detections, gray)
    aspect = SPLIT_GAME_ASPECT if layout == "split" else FULL_GAME_ASPECT
    game = best_game_crop(persistent_edges(gray), rect, aspect)
    if rect is None:
        return {"rect": None, "layout": layout, "game": game}
    scale = info.width / width
    x, y, w, h = (round(v * scale) for v in rect)
    return {"rect": [x, y, w, h], "layout": layout, "game": game}


def sample_times(start: float, end: float, count: int = SAMPLE_FRAMES) -> list[float]:
    """Gleichmaessig verteilt, damit Spielszenen sich zwischen den Frames deutlich aendern."""
    if end <= start:
        return []
    step = (end - start) / (count + 1)
    return [start + step * (i + 1) for i in range(count)]
