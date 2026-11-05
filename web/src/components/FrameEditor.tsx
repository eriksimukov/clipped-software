import { PointerEvent, useRef, useState } from "react";
import { clamp } from "../util";

type Rect = [number, number, number, number];
type DragMode = "move" | "nw" | "ne" | "sw" | "se";

interface Props {
  image: string;
  width: number;
  height: number;
  cam: Rect | null;
  showCam: boolean;
  gameCenter: number;
  gameAspect: number;
  disabled: boolean;
  onCamChange: (rect: Rect | null) => void;
  onGameCenterChange: (center: number) => void;
}

const MIN_SIZE = 32;
const DEFAULT_CAM_SHARE = 0.25;

function evenRect([x, y, w, h]: Rect): Rect {
  return [Math.round(x / 2) * 2, Math.round(y / 2) * 2, Math.round(w / 2) * 2, Math.round(h / 2) * 2];
}

/** Standbild des Quellvideos mit Rahmen fuer Facecam und Bildausschnitt des Spiels. */
export function FrameEditor(props: Props) {
  const { image, width, height, cam, showCam, gameCenter, gameAspect, disabled } = props;
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ mode: DragMode; startX: number; startY: number; rect: Rect } | null>(null);
  const [draft, setDraft] = useState<Rect | null>(null);
  const shown = draft ?? cam;

  function toSource(event: PointerEvent): [number, number] {
    const rect = box.current!.getBoundingClientRect();
    return [((event.clientX - rect.left) / rect.width) * width, ((event.clientY - rect.top) / rect.height) * height];
  }

  function onDown(mode: DragMode, event: PointerEvent) {
    if (disabled || !shown) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const [startX, startY] = toSource(event);
    drag.current = { mode, startX, startY, rect: shown };
  }

  function onMove(event: PointerEvent) {
    const current = drag.current;
    if (!current) return;
    const [px, py] = toSource(event);
    const dx = px - current.startX;
    const dy = py - current.startY;
    let [x, y, w, h] = current.rect;
    if (current.mode === "move") {
      x = clamp(x + dx, 0, width - w);
      y = clamp(y + dy, 0, height - h);
    } else {
      let right = x + w;
      let bottom = y + h;
      if (current.mode.includes("w")) x = clamp(x + dx, 0, right - MIN_SIZE);
      if (current.mode.includes("n")) y = clamp(y + dy, 0, bottom - MIN_SIZE);
      if (current.mode.includes("e")) right = clamp(right + dx, x + MIN_SIZE, width);
      if (current.mode.includes("s")) bottom = clamp(bottom + dy, y + MIN_SIZE, height);
      w = right - x;
      h = bottom - y;
    }
    setDraft(evenRect([x, y, w, h]));
  }

  function onUp(event: PointerEvent) {
    if (!drag.current) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    drag.current = null;
    if (draft) props.onCamChange(draft);
    setDraft(null);
  }

  const gameWidth = Math.min(height * gameAspect, width);
  const gameLeft = clamp(gameCenter * width - gameWidth / 2, 0, width - gameWidth);
  const percent = (value: number, total: number) => `${(value / total) * 100}%`;

  return (
    <div>
      <div className="frame-editor" ref={box} style={{ aspectRatio: `${width} / ${height}` }}>
        <img src={image} alt="Standbild aus dem Quellvideo" draggable={false} />
        <div
          className="game-window"
          style={{ left: percent(gameLeft, width), width: percent(gameWidth, width) }}
          title="Dieser Ausschnitt des Spiels ist im Clip zu sehen"
        />
        {showCam && shown && (
          <div
            className="cam-box"
            style={{
              left: percent(shown[0], width),
              top: percent(shown[1], height),
              width: percent(shown[2], width),
              height: percent(shown[3], height),
            }}
            onPointerDown={(event) => onDown("move", event)}
            onPointerMove={onMove}
            onPointerUp={onUp}
            title="Facecam: ziehen zum Verschieben"
          >
            <span>Facecam</span>
            {(["nw", "ne", "sw", "se"] as const).map((corner) => (
              <i
                key={corner}
                className={corner}
                onPointerDown={(event) => onDown(corner, event)}
                onPointerMove={onMove}
                onPointerUp={onUp}
              />
            ))}
          </div>
        )}
      </div>
      <div className="frame-controls">
        <label>
          Ausschnitt des Spiels
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
            value={gameCenter}
            onChange={(event) => props.onGameCenterChange(Number(event.target.value))}
          />
        </label>
        {showCam && !cam && (
          <button
            className="secondary"
            disabled={disabled}
            onClick={() => {
              const w = width * DEFAULT_CAM_SHARE;
              props.onCamChange(evenRect([0, height - (w * 9) / 16, w, (w * 9) / 16]));
            }}
          >
            Facecam-Rahmen setzen
          </button>
        )}
        {showCam && cam && (
          <button className="link" disabled={disabled} onClick={() => props.onCamChange(null)}>
            Rahmen entfernen
          </button>
        )}
      </div>
    </div>
  );
}
