import { PointerEvent, useRef } from "react";
import { clamp, formatPrecise } from "../util";

type Handle = "start" | "peak" | "end";

interface Props {
  viewStart: number;
  viewEnd: number;
  start: number;
  end: number;
  peak: number;
  current: number;
  curve: number[];
  curveStart: number;
  disabled: boolean;
  onChange: (change: { start?: number; end?: number; peak?: number }) => void;
  onCommit: () => void;
  onSeek: (time: number) => void;
}

const MIN_LENGTH = 3;
const HEIGHT = 96;

/** Zeigt die Signalkurve um den Clip; die Griffe aendern Start, Ende und Hoehepunkt. */
export function Timeline(props: Props) {
  const { viewStart, viewEnd, start, end, peak, current, curve, curveStart, disabled } = props;
  const bar = useRef<HTMLDivElement>(null);
  const dragging = useRef<Handle | null>(null);
  const span = Math.max(viewEnd - viewStart, 1);
  const toPercent = (time: number) => ((time - viewStart) / span) * 100;

  function timeAt(event: PointerEvent): number {
    const rect = bar.current!.getBoundingClientRect();
    return viewStart + clamp((event.clientX - rect.left) / rect.width, 0, 1) * span;
  }

  function onHandleDown(handle: Handle, event: PointerEvent) {
    if (disabled) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragging.current = handle;
  }

  function onHandleMove(handle: Handle, event: PointerEvent) {
    if (dragging.current !== handle) return;
    const time = timeAt(event);
    if (handle === "start") {
      const next = Math.min(time, end - MIN_LENGTH);
      props.onChange({ start: next, peak: Math.max(peak, next) });
    } else if (handle === "end") {
      const next = Math.max(time, start + MIN_LENGTH);
      props.onChange({ end: next, peak: Math.min(peak, next) });
    } else {
      props.onChange({ peak: clamp(time, start, end) });
    }
  }

  function onHandleUp(event: PointerEvent) {
    if (!dragging.current) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    dragging.current = null;
    props.onCommit();
  }

  const points = curve
    .map((value, index) => {
      const time = curveStart + index;
      if (time < viewStart || time > viewEnd) return null;
      return `${toPercent(time).toFixed(2)},${(HEIGHT - value * (HEIGHT - 6)).toFixed(1)}`;
    })
    .filter(Boolean);

  const handle = (name: Handle, time: number, label: string) => (
    <div
      className={`tl-handle tl-${name}`}
      style={{ left: `${toPercent(time)}%` }}
      onPointerDown={(event) => onHandleDown(name, event)}
      onPointerMove={(event) => onHandleMove(name, event)}
      onPointerUp={onHandleUp}
      role="slider"
      aria-label={label}
      aria-valuetext={formatPrecise(time)}
      aria-valuenow={Math.round(time)}
      title={`${label}: ${formatPrecise(time)}`}
    />
  );

  return (
    <div className="timeline" ref={bar} onPointerDown={(event) => props.onSeek(timeAt(event))} style={{ height: HEIGHT }}>
      {points.length > 1 && (
        <svg viewBox={`0 0 100 ${HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
          <polyline points={`0,${HEIGHT} ${points.join(" ")} 100,${HEIGHT}`} />
        </svg>
      )}
      <div className="tl-range" style={{ left: `${toPercent(start)}%`, width: `${toPercent(end) - toPercent(start)}%` }} />
      <div className="tl-playhead" style={{ left: `${clamp(toPercent(current), 0, 100)}%` }} />
      {handle("start", start, "Start")}
      {handle("end", end, "Ende")}
      {handle("peak", peak, "Höhepunkt")}
    </div>
  );
}
