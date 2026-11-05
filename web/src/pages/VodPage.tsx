import { useState } from "react";
import { api, Clip, ClipStatus, Job, urls } from "../api";
import { formatTime, parseTime, STEP_LABELS, usePoll } from "../util";

export const CLIP_STATUS_LABELS: Record<ClipStatus, string> = {
  proposed: "Vorschlag",
  rejected: "Abgelehnt",
  off_topic: "Kein GTA 6",
  failed: "Modell-Fehler",
  approved: "In Warteschlange",
  rendering: "Wird gerendert",
  done: "Fertig",
  render_failed: "Render-Fehler",
};

type Filter = "open" | "done" | "dismissed";

const FILTERS: { id: Filter; label: string; match: ClipStatus[] }[] = [
  { id: "open", label: "Vorschläge", match: ["proposed", "approved", "rendering", "render_failed"] },
  { id: "done", label: "Fertig", match: ["done"] },
  { id: "dismissed", label: "Aussortiert", match: ["rejected", "off_topic", "failed"] },
];

interface Props {
  vodId: string;
  jobs: Job[];
  onError: (message: string) => void;
}

function ScoreBars({ clip }: { clip: Clip }) {
  const a = clip.candidate.assessment;
  if (!a) return null;
  const rows: [string, number][] = [
    ...(a.reaction === undefined ? [] : ([["Reaktion", a.reaction]] as [string, number][])),
    ["Hook", a.hook],
    ["Payoff", a.payoff],
    ["Emotion", a.emotion],
    ["Verständlich", a.standalone],
  ];
  return (
    <div className="bars">
      {rows.map(([label, value]) => (
        <div key={label} title={`${label}: ${value} von 10`}>
          <span>{label}</span>
          <i>
            <b style={{ width: `${value * 10}%` }} />
          </i>
        </div>
      ))}
    </div>
  );
}

export function VodPage({ vodId, jobs, onError }: Props) {
  const vod = usePoll(() => api.vod(vodId), 2500, vodId);
  const [filter, setFilter] = useState<Filter>("open");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");

  async function run(action: () => Promise<unknown>) {
    try {
      await action();
      await vod.refresh();
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  async function applyRange() {
    const start = parseTime(from);
    const end = parseTime(to);
    if (start === null || end === null || end <= start) {
      onError("Der Zeitbereich braucht Start und Ende, zum Beispiel 1:20:00 bis 2:45:00.");
      return;
    }
    await run(() => api.setRange(vodId, [start, end]));
  }

  if (vod.error && !vod.data) return <div className="page"><div className="banner error">{vod.error}</div></div>;
  if (!vod.data) return <div className="page muted">Lade…</div>;
  const data = vod.data;
  const job = jobs.find((j) => j.vod_id === vodId && j.kind === "process_vod");
  const shown = data.clips.filter((clip) => FILTERS.find((f) => f.id === filter)!.match.includes(clip.status));
  const noCandidates = data.status === "ready" && data.clips.length === 0;

  return (
    <div className="page">
      <a className="back" href="#/">← Alle VODs</a>
      <h1>{data.title || data.source}</h1>
      <p className="muted vod-facts">
        {data.streamer || "Unbekannter Streamer"}
        {data.meta.duration != null && ` · ${formatTime(data.meta.duration)}`}
        {data.meta.chat_messages ? ` · ${data.meta.chat_messages.toLocaleString("de-DE")} Chat-Nachrichten` : data.meta.duration != null ? " · kein Chat-Replay" : ""}
        {data.meta.topic_reason && ` · GTA-6-Bereich: ${data.meta.topic_reason}`}
        {data.time_range && ` (${formatTime(data.time_range[0])} bis ${formatTime(data.time_range[1])})`}
      </p>

      {(data.status === "queued" || data.status === "processing") && (
        <div className="card progress-card">
          <strong>{job?.status === "queued" || !job ? "Wartet auf einen freien Platz" : (STEP_LABELS[job.step] ?? "Wird vorbereitet")}</strong>
          <div className="progress">
            <div style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
          </div>
          <p className="hint">Du kannst das Fenster schließen. Beim nächsten Start setzt die Arbeit dort wieder an.</p>
        </div>
      )}

      {data.status === "failed" && (
        <div className="banner error">
          {data.error}
          <button className="link" onClick={() => void run(() => api.retryVod(vodId))}>
            Erneut versuchen
          </button>
        </div>
      )}
      {data.status === "cancelled" && (
        <div className="banner warn">
          Abgebrochen.
          <button className="link" onClick={() => void run(() => api.retryVod(vodId))}>
            Fortsetzen
          </button>
        </div>
      )}

      {noCandidates && (
        <div className="banner warn">
          {data.meta.topic_seconds === 0
            ? "Im Titel, in den Kapiteln und im Gesprochenen wurde kein GTA 6 gefunden. Gib unten den passenden Zeitbereich an."
            : "In diesem Bereich gab es keine passenden Momente. Versuche einen anderen Zeitbereich."}
        </div>
      )}

      {data.status === "ready" && (
        <details className="card range-form" open={noCandidates}>
          <summary>Zeitbereich festlegen und neu auswerten</summary>
          <div className="form-row">
            <label>
              Von
              <input value={from} onChange={(e) => setFrom(e.target.value)} placeholder="1:20:00" />
            </label>
            <label>
              Bis
              <input value={to} onChange={(e) => setTo(e.target.value)} placeholder="2:45:00" />
            </label>
            <button className="secondary" onClick={() => void applyRange()}>
              Neu auswerten
            </button>
          </div>
        </details>
      )}

      {data.clips.length > 0 && (
        <>
          <div className="tabs" role="tablist">
            {FILTERS.map((item) => {
              const count = data.clips.filter((clip) => item.match.includes(clip.status)).length;
              return (
                <button key={item.id} role="tab" aria-selected={filter === item.id} onClick={() => setFilter(item.id)}>
                  {item.label} <span className="muted">{count}</span>
                </button>
              );
            })}
          </div>
          {shown.length === 0 && <p className="muted">Hier ist gerade nichts.</p>}
          <ul className="clip-grid">
            {shown.map((clip) => {
              const running = jobs.find((j) => j.clip_id === clip.id && j.vod_id === vodId && j.kind === "render" && j.status !== "queued");
              const href = `#/vod/${vodId}/clip/${clip.id}`;
              return (
                <li key={clip.id} className="card clip-card">
                  <a href={href} className="clip-thumb">
                    <img src={urls.frame(vodId, clip.spec.peak, 480)} alt="" loading="lazy" />
                    <span className="score" title="Ordnet die Clips dieses VODs untereinander; keine Vorhersage von Aufrufen">
                      {Math.round(clip.score)}
                    </span>
                    <span className="duration">{formatTime(clip.spec.end - clip.spec.start)}</span>
                  </a>
                  <div className="clip-body">
                    <a href={href} className="clip-hook">
                      {clip.spec.hook_text || clip.spec.title || "Ohne Hook"}
                    </a>
                    <p className="muted small">
                      {formatTime(clip.spec.start)} bis {formatTime(clip.spec.end)}
                      {clip.status !== "proposed" && (
                        <span className={`pill clip-${clip.status}`}>{CLIP_STATUS_LABELS[clip.status]}</span>
                      )}
                    </p>
                    {clip.candidate.warnings?.includes("slur") && (
                      <p className="small warn-text" title="Im Bild wird sie maskiert, im Ton bleibt sie hörbar. Der Score ist deshalb abgewertet.">
                        Enthält eine Beleidigung im Ton
                      </p>
                    )}
                    {(clip.candidate.story || clip.candidate.assessment?.reason) && (
                      <p className="small reason">{clip.candidate.story || clip.candidate.assessment?.reason}</p>
                    )}
                    {clip.error && <p className="small error-text">{clip.error}</p>}
                    {clip.candidate.error && <p className="small error-text">{clip.candidate.error}</p>}
                    <ScoreBars clip={clip} />
                    {running && (
                      <div className="progress">
                        <div style={{ width: `${running.progress * 100}%` }} />
                      </div>
                    )}
                    <div className="clip-actions">
                      <a className="button secondary" href={href}>
                        Bearbeiten
                      </a>
                      {(clip.status === "proposed" || clip.status === "render_failed") && (
                        <>
                          <button className="primary" onClick={() => void run(() => api.approve(vodId, clip.id))}>
                            Freigeben
                          </button>
                          <button className="link danger" onClick={() => void run(() => api.reject(vodId, clip.id))}>
                            Ablehnen
                          </button>
                        </>
                      )}
                      {["rejected", "off_topic", "failed"].includes(clip.status) && (
                        <button className="link" onClick={() => void run(() => api.restore(vodId, clip.id))}>
                          Zurückholen
                        </button>
                      )}
                      {clip.status === "done" && (
                        <a className="link" href={urls.outputFile(vodId, clip.id)} target="_blank" rel="noreferrer">
                          Ansehen
                        </a>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
