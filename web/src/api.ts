export type Layout = "split" | "gameplay" | "blur";

export interface Effects {
  zoom: boolean;
  shake: boolean;
  flash: boolean;
  sfx: boolean;
}

export interface TextEdit {
  start: number;
  end: number;
  text: string;
}

export interface ClipSpec {
  id: string;
  start: number;
  end: number;
  peak: number;
  hook_text: string;
  teaser: boolean;
  layout: Layout;
  cam_rect: [number, number, number, number] | null;
  game_center: number;
  credit: string;
  effects: Effects;
  text_edits: TextEdit[];
  title: string;
  description: string;
  hashtags: string[];
}

export interface Assessment {
  hook: number;
  payoff: number;
  emotion: number;
  standalone: number;
  gta6_relevance: number;
  reason: string;
  hooks: string[];
}

export interface Candidate {
  id: string;
  start: number;
  end: number;
  peak: number;
  signal_score: number;
  text: string;
  status: string;
  score?: number;
  error?: string;
  assessment?: Assessment;
}

export type ClipStatus =
  | "proposed"
  | "rejected"
  | "off_topic"
  | "failed"
  | "approved"
  | "rendering"
  | "done"
  | "render_failed";

export interface Clip {
  vod_id: string;
  id: string;
  status: ClipStatus;
  score: number;
  candidate: Candidate;
  spec: ClipSpec;
  preview_path: string;
  output_path: string;
  error: string;
}

export interface SubtitleLine {
  start: number;
  end: number;
  text: string;
}

export interface ClipDetail extends Clip {
  lines: SubtitleLine[];
  preview_version: number;
  video?: { width: number; height: number; duration: number };
}

export type VodStatus = "queued" | "processing" | "ready" | "failed" | "cancelled";

export interface Vod {
  id: string;
  source: string;
  title: string;
  streamer: string;
  time_range: [number, number] | null;
  status: VodStatus;
  error: string;
  created_at: number;
  clip_count?: number;
  done_count?: number;
}

export interface VodDetail extends Vod {
  meta: {
    duration: number | null;
    topic_reason: string | null;
    topic_seconds: number | null;
    chat_messages: number | null;
    platform: string | null;
    cam: { rect: number[] | null; layout: Layout } | null;
  };
  clips: Clip[];
}

export interface Job {
  id: number;
  kind: "process_vod" | "render" | "preview";
  vod_id: string;
  clip_id: string;
  status: "queued" | "running" | "cancelling";
  step: string;
  progress: number;
}

export interface Status {
  ollama: { running: boolean; model_ready: boolean };
  model: string;
  free_gb: number;
  data_dir: string;
  placeholder_handles: boolean;
}

export interface ExportItem {
  vod_id: string;
  vod_title: string;
  id: string;
  score: number;
  spec: ClipSpec;
  text: string;
  file: string;
}

export interface Channel {
  platform: string;
  handle: string;
}

export interface Settings {
  paths: { data_dir: string };
  clips: { min_seconds: number; max_seconds: number; candidates_per_vod: number; language: string };
  transcribe: { model: string; compute_type: string; device: string };
  llm: { host: string; model: string };
  render: { width: number; height: number; fps: number; video_encoder: string; target_lufs: number };
  branding: { watermark: string; endcard_title: string; endcard_seconds: number; channels: Channel[] };
  style: { highlight_color: string; subtitle_size: number };
  download: { cookies_file: string; cookies_from_browser: string };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!response.ok) {
    let message = `Fehler ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) {
        message = body.detail
          .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).join(".")}: ${d.msg}`)
          .join("; ");
      }
    } catch {
      // Keine JSON-Antwort: der Statuscode bleibt die beste Auskunft.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

const put = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "PUT", body: JSON.stringify(body) });

const clipUrl = (vodId: string, clipId: string) => `/api/vods/${vodId}/clips/${clipId}`;

export const api = {
  status: () => request<Status>("/api/status"),
  jobs: () => request<Job[]>("/api/jobs"),
  cancelJob: (id: number) => post<{ running: boolean }>(`/api/jobs/${id}/cancel`),

  vods: () => request<Vod[]>("/api/vods"),
  vod: (id: string) => request<VodDetail>(`/api/vods/${id}`),
  addVod: (source: string, streamer: string, timeRange: [number, number] | null) =>
    post<Vod>("/api/vods", { source, streamer, time_range: timeRange }),
  retryVod: (id: string) => post<Vod>(`/api/vods/${id}/retry`),
  setRange: (id: string, timeRange: [number, number] | null) =>
    post<Vod>(`/api/vods/${id}/range`, { time_range: timeRange }),
  deleteVod: (id: string) => request<unknown>(`/api/vods/${id}`, { method: "DELETE" }),
  signals: (id: string, start: number, end: number) =>
    request<{ curve: number[]; start: number }>(`/api/vods/${id}/signals?start=${start}&end=${end}`),

  clip: (vodId: string, clipId: string) => request<ClipDetail>(clipUrl(vodId, clipId)),
  saveClip: (vodId: string, clipId: string, spec: ClipSpec) =>
    put<ClipDetail>(clipUrl(vodId, clipId), spec),
  approve: (vodId: string, clipId: string) => post<ClipDetail>(`${clipUrl(vodId, clipId)}/approve`),
  reject: (vodId: string, clipId: string) => post<ClipDetail>(`${clipUrl(vodId, clipId)}/reject`),
  restore: (vodId: string, clipId: string) => post<ClipDetail>(`${clipUrl(vodId, clipId)}/restore`),
  preview: (vodId: string, clipId: string) => post<unknown>(`${clipUrl(vodId, clipId)}/preview`),

  exports: () => request<ExportItem[]>("/api/exports"),
  openFolder: (vodId?: string) => post<unknown>("/api/open-folder", { vod_id: vodId ?? null }),

  settings: () => request<Settings>("/api/settings"),
  saveSettings: (settings: Settings) =>
    put<{ settings: Settings; restart_required: boolean }>("/api/settings", settings),
};

export const urls = {
  video: (vodId: string) => `/api/vods/${vodId}/video`,
  frame: (vodId: string, t: number, width?: number) =>
    `/api/vods/${vodId}/frame?t=${t.toFixed(2)}${width ? `&w=${width}` : ""}`,
  previewFile: (vodId: string, clipId: string, version: string) =>
    `${clipUrl(vodId, clipId)}/preview.mp4?v=${encodeURIComponent(version)}`,
  outputFile: (vodId: string, clipId: string) => `${clipUrl(vodId, clipId)}/output.mp4`,
};

/** Laedt die Datei ohne Formular hoch, damit auch grosse VODs nicht im Temp-Ordner landen. */
export function uploadFile(file: File, onProgress: (share: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/uploads?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve((JSON.parse(xhr.responseText) as { path: string }).path);
        return;
      }
      let message = `Upload fehlgeschlagen (${xhr.status})`;
      try {
        message = (JSON.parse(xhr.responseText) as { detail: string }).detail;
      } catch {
        // Der Status reicht als Auskunft.
      }
      reject(new Error(message));
    };
    xhr.onerror = () => reject(new Error("Upload fehlgeschlagen: keine Verbindung zum Server."));
    xhr.send(file);
  });
}
