import { DragEvent, FormEvent, useRef, useState } from "react";
import { api, Job, uploadFile, Vod } from "../api";
import { parseTime, STEP_LABELS, usePoll } from "../util";

const STATUS_LABELS: Record<Vod["status"], string> = {
  queued: "Wartet",
  processing: "In Arbeit",
  ready: "Bereit",
  failed: "Fehlgeschlagen",
  cancelled: "Abgebrochen",
};

interface Props {
  jobs: Job[];
  onError: (message: string) => void;
}

export function ImportPage({ jobs, onError }: Props) {
  const vods = usePoll(api.vods, 3000);
  const status = usePoll(api.status, 10000);
  const [source, setSource] = useState("");
  const [streamer, setStreamer] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploadShare, setUploadShare] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  function readRange(): [number, number] | null | undefined {
    if (!from.trim() && !to.trim()) return null;
    const start = parseTime(from);
    const end = parseTime(to);
    if (start === null || end === null || end <= start) {
      onError("Der Zeitbereich braucht Start und Ende, zum Beispiel 1:20:00 bis 2:45:00.");
      return undefined;
    }
    return [start, end];
  }

  async function addVod(path: string, range: [number, number] | null) {
    const vod = await api.addVod(path, streamer, range);
    setSource("");
    window.location.hash = `#/vod/${vod.id}`;
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const range = readRange();
    if (range === undefined) return;
    setBusy(true);
    try {
      await addVod(source, range);
    } catch (reason) {
      onError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function upload(file: File) {
    const range = readRange();
    if (range === undefined) return;
    setUploadShare(0);
    try {
      const path = await uploadFile(file, setUploadShare);
      await addVod(path, range);
    } catch (reason) {
      onError((reason as Error).message);
    } finally {
      setUploadShare(null);
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void upload(file);
  }

  async function removeVod(vod: Vod) {
    const name = vod.title || vod.source;
    const confirmed = window.confirm(
      `„${name}" entfernen?\n\nDie Arbeitsdaten dieses VODs werden gelöscht: Download, Transkript, Vorschauen und Vorschläge. Fertige Clips im Export-Ordner und Original-Dateien bleiben erhalten.`,
    );
    if (!confirmed) return;
    try {
      await api.deleteVod(vod.id);
      await vods.refresh();
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  const list = vods.data ?? [];

  return (
    <div className="page narrow">
      {status.data && !status.data.ollama.model_ready && status.data.ollama.running && (
        <div className="banner warn">
          Das Sprachmodell fehlt. Führe einmal <code>ollama pull {status.data.model}</code> aus.
        </div>
      )}
      {status.data?.placeholder_handles && (
        <div className="banner warn">
          In Endkarte und Wasserzeichen stehen noch Platzhalter. <a href="#/settings">Kanäle eintragen</a>
        </div>
      )}
      {status.data && status.data.free_gb < 50 && (
        <div className="banner warn">Nur noch {status.data.free_gb} GB frei in {status.data.data_dir}.</div>
      )}

      <h1>VOD importieren</h1>
      <form className="card form" onSubmit={(event) => void submit(event)}>
        <label>
          Link oder Dateipfad
          <input
            value={source}
            onChange={(event) => setSource(event.target.value)}
            placeholder="https://www.twitch.tv/videos/… oder D:\Videos\stream.mp4"
            autoFocus
          />
        </label>
        <div className="form-row">
          <label>
            Streamer (für den Credit)
            <input value={streamer} onChange={(event) => setStreamer(event.target.value)} placeholder="Wird bei Links automatisch übernommen" />
          </label>
          <label>
            Nur von (optional)
            <input value={from} onChange={(event) => setFrom(event.target.value)} placeholder="1:20:00" />
          </label>
          <label>
            bis
            <input value={to} onChange={(event) => setTo(event.target.value)} placeholder="2:45:00" />
          </label>
        </div>
        <p className="hint">
          Der Zeitbereich überschreibt die automatische GTA-6-Erkennung. Das hilft, wenn im VOD auch andere Spiele laufen.
        </p>
        <div className="form-actions">
          <button className="primary" disabled={busy || !source.trim()}>
            Clips suchen
          </button>
        </div>
      </form>

      <div
        className={`dropzone${dragging ? " active" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {uploadShare === null ? (
          <>
            Videodatei hierher ziehen oder{" "}
            <button className="link" onClick={() => fileInput.current?.click()}>
              auswählen
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="video/*,.mkv,.flv,.ts"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
              }}
            />
          </>
        ) : (
          <>
            Wird kopiert: {Math.round(uploadShare * 100)} %
            <div className="progress">
              <div style={{ width: `${uploadShare * 100}%` }} />
            </div>
          </>
        )}
      </div>

      <h2>Deine VODs</h2>
      {vods.data && list.length === 0 && <p className="muted">Noch nichts importiert.</p>}
      <ul className="vod-list">
        {list.map((vod) => {
          const job = jobs.find((j) => j.vod_id === vod.id && j.kind === "process_vod" && j.status !== "queued");
          return (
            <li key={vod.id} className="card vod-row">
              <a className="vod-main" href={`#/vod/${vod.id}`}>
                <strong>{vod.title || vod.source}</strong>
                <span className="muted">
                  {vod.streamer || "Unbekannter Streamer"}
                  {vod.status === "ready" && ` · ${vod.clip_count ?? 0} ${vod.clip_count === 1 ? "Vorschlag" : "Vorschläge"} · ${vod.done_count ?? 0} fertig`}
                </span>
                {job && (
                  <span className="vod-job">
                    {STEP_LABELS[job.step] ?? "Wird vorbereitet"}: {Math.round(job.progress * 100)} %
                  </span>
                )}
                {vod.error && <span className="error-text">{vod.error}</span>}
              </a>
              <span className={`pill status-${vod.status}`}>{STATUS_LABELS[vod.status]}</span>
              <button className="link danger" onClick={() => void removeVod(vod)}>
                Entfernen
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
