import { useCallback, useEffect, useRef, useState } from "react";

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const mm = String(minutes).padStart(hours ? 2 : 1, "0");
  const ss = String(secs).padStart(2, "0");
  return hours ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function formatPrecise(seconds: number): string {
  const tenths = Math.floor((seconds % 1) * 10);
  return `${formatTime(seconds)}.${tenths}`;
}

/** Akzeptiert "90", "1:30" und "1:02:03". Gibt null zurueck, wenn es keine Zeit ist. */
export function parseTime(text: string): number | null {
  const trimmed = text.trim();
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(trimmed)) return null;
  return trimmed.split(":").reduce((sum, part) => sum * 60 + Number(part), 0);
}

export interface Polled<T> {
  data: T | null;
  error: string | null;
  refresh: () => Promise<void>;
}

/** Holt Daten regelmaessig; ein gescheiterter Abruf behaelt die letzten Daten bei. */
export function usePoll<T>(load: () => Promise<T>, intervalMs: number, key: unknown = null): Polled<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;

  const refresh = useCallback(async () => {
    try {
      setData(await loadRef.current());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  useEffect(() => {
    setData(null);
    void refresh();
    const timer = window.setInterval(() => void refresh(), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, key, refresh]);

  return { data, error, refresh };
}

export function useHashRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => {
      setRoute(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

export const STEP_LABELS: Record<string, string> = {
  download: "Video wird heruntergeladen",
  chat: "Chat-Replay wird geholt",
  audio: "Audio wird ausgelesen",
  transcribe: "Sprache wird transkribiert",
  signals: "Höhepunkte und Facecam werden gesucht",
  assess: "Momente werden bewertet",
  render: "Clip wird gerendert",
  preview: "Vorschau wird gerendert",
};
