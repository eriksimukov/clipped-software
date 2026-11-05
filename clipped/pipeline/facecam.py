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
SAMPLE_FRAMES = 16
MIN_FACE_SCORE = 0.7
MIN_PRESENCE = 0.4
# Ab dieser Gesichtsbreite ist es ein Vollbild-Kamerabild statt einer eingeblendeten Facecam.
FULLSCREEN_FACE_RATIO = 0.18
MAX_CAM_AREA_RATIO = 0.4
# Suchweite ab dem Gesicht, in Gesichtsbreiten beziehungsweise -hoehen.
SEARCH_SIDEWAYS = 3.5
SEARCH_VERTICAL = 2.5
MIN_EDGE_STRENGTH = 8.0
# Eine Kante muss deutlich ueber dem Umfeld liegen, sonst ist sie nur Bildstruktur.
EDGE_PROMINENCE = 2.5
# Mittlere Helligkeitsstreuung hinter dem Rand. Kanten im Zimmer haben auf beiden Seiten Ruhe.
MIN_OUTER_MOTION = 10.0
OUTER_BAND_PIXELS = 6
MAX_EDGE_CANDIDATES = 8


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
    w = min(face.w * 4.0, width)
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

    fx, fy, fw, fh = _fallback_rect(face, width, height)
    new_left = left if left is not None else fx
    new_top = top if top is not None else fy
    new_right = right if right is not None else fx + fw
    new_bottom = bottom if bottom is not None else fy + fh
    return new_left, new_top, new_right - new_left, new_bottom - new_top


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
        return {"rect": None, "layout": "gameplay"}

    height, width = frames[0].shape[:2]
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), MIN_FACE_SCORE)
    detections: list[list[Face]] = []
    for frame in frames:
        _, faces = detector.detect(frame)
        detections.append([Face(*map(float, f[:4])) for f in faces] if faces is not None else [])
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])

    rect, layout = locate_cam(detections, gray)
    if rect is None:
        return {"rect": None, "layout": layout}
    scale = info.width / width
    x, y, w, h = (round(v * scale) for v in rect)
    return {"rect": [x, y, w, h], "layout": layout}


def sample_times(start: float, end: float, count: int = SAMPLE_FRAMES) -> list[float]:
    """Gleichmaessig verteilt, damit Spielszenen sich zwischen den Frames deutlich aendern."""
    if end <= start:
        return []
    step = (end - start) / (count + 1)
    return [start + step * (i + 1) for i in range(count)]
