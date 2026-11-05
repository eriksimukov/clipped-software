import { useCallback, useEffect, useRef, useState } from "react";
import { api, ClipDetail, ClipSpec, Job, Layout, SubtitleLine, urls } from "../api";
import { FrameEditor } from "../components/FrameEditor";
import { Timeline } from "../components/Timeline";
import { formatPrecise, formatTime, STEP_LABELS } from "../util";
import { CLIP_STATUS_LABELS } from "./VodPage";

interface Props {
  vodId: string;
  clipId: string;
  jobs: Job[];
  onError: (message: string) => void;
}

type Tab = "hook" | "subs" | "layout" | "fx" | "text";

const TABS: { id: Tab; label: string }[] = [
  { id: "hook", label: "Hook" },
  { id: "subs", label: "Untertitel" },
  { id: "layout", label: "Layout" },
  { id: "fx", label: "Effekte" },
  { id: "text", label: "Texte" },
];

const LAYOUTS: { id: Layout; label: string; note: string }[] = [
  { id: "split", label: "Facecam oben, Spiel unten", note: "Braucht einen Facecam-Rahmen" },
  { id: "gameplay", label: "Spiel im Vollbild", note: "Facecam klein darüber, falls vorhanden" },
  { id: "blur", label: "16:9 mit unscharfem Rand", note: "Zeigt das ganze Bild" },
];

const EFFECTS: { id: keyof ClipSpec["effects"]; label: string; note: string }[] = [
  { id: "zoom", label: "Zooms", note: "Rhythmisch an Satzgrenzen und stark am Höhepunkt" },
  { id: "shake", label: "Shake", note: "Kurzes Wackeln am Höhepunkt" },
  { id: "flash", label: "Flash", note: "Helles Aufblitzen am Höhepunkt und beim Übergang" },
  { id: "sfx", label: "Sounds", note: "Whoosh, Riser, Impact und Pop" },
];

const BUSY = new Set(["approved", "rendering"]);
const SAVE_DELAY_MS = 700;
const VIEW_PADDING = 30;
// Breite durch Hoehe des sichtbaren Spielbilds im fertigen Clip.
const GAME_ASPECT: Record<Layout, number> = { split: 1080 / 1280, gameplay: 1080 / 1920, blur: 16 / 9 };

function SubtitleRow({
  line,
  active,
  disabled,
  onSeek,
  onCommit,
}: {
  line: SubtitleLine;
  active: boolean;
  disabled: boolean;
  onSeek: () => void;
  onCommit: (text: string) => void;
}) {
  const [value, setValue] = useState(line.text);
  useEffect(() => setValue(line.text), [line.text]);
  return (
    <li className={active ? "active" : undefined}>
      <button className="link time" onClick={onSeek} title="Im Video hierher springen">
        {formatTime(line.start)}
      </button>
      <input
        value={value}
        disabled={disabled}
        aria-label={`Untertitel bei ${formatTime(line.start)}`}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        onBlur={() => {
          if (value !== line.text) onCommit(value);
        }}
      />
    </li>
  );
}

export function ClipEditor({ vodId, clipId, jobs, onError }: Props) {
  const [clip, setClip] = useState<ClipDetail | null>(null);
  const [spec, setSpec] = useState<ClipSpec | null>(null);
  const [tab, setTab] = useState<Tab>("hook");
  const [saveState, setSaveState] = useState<"saved" | "pending" | "saving">("saved");
  const [current, setCurrent] = useState(0);
  const [curve, setCurve] = useState<{ curve: number[]; start: number }>({ curve: [], start: 0 });
  const [neighbours, setNeighbours] = useState<string[]>([]);
  const player = useRef<HTMLVideoElement>(null);
  const specRef = useRef<ClipSpec | null>(null);
  const saveTimer = useRef<number | null>(null);
  specRef.current = spec;

  const applyServer = useCallback((detail: ClipDetail) => {
    setClip(detail);
    setSpec(detail.spec);
  }, []);

  useEffect(() => {
    api.clip(vodId, clipId).then(applyServer).catch((e: Error) => onError(e.message));
    api
      .vod(vodId)
      .then((vod) => setNeighbours(vod.clips.filter((c) => c.status === "proposed").map((c) => c.id)))
      .catch(() => setNeighbours([]));
  }, [vodId, clipId, applyServer, onError]);

  const viewStart = spec ? Math.max(spec.start - VIEW_PADDING, 0) : 0;
  const viewEnd = spec ? Math.min(spec.end + VIEW_PADDING, clip?.video?.duration ?? spec.end + VIEW_PADDING) : 0;
  const hasSpec = spec !== null;
  useEffect(() => {
    if (!hasSpec) return;
    api
      .signals(vodId, Math.floor(viewStart), Math.ceil(viewEnd))
      .then(setCurve)
      .catch(() => setCurve({ curve: [], start: 0 }));
    // Die Kurve haengt nur am sichtbaren Bereich des Clips, wie er beim Oeffnen war.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vodId, clipId, hasSpec]);

  // Laufende Jobs aendern den Clip im Hintergrund: Status und Vorschaupfad nachladen.
  const relevantJobs = jobs.filter((job) => job.vod_id === vodId && job.clip_id === clipId);
  const activeJob = relevantJobs.find((job) => job.status !== "queued") ?? relevantJobs[0];
  const jobSignature = relevantJobs.map((job) => `${job.id}:${job.status}`).join(",");
  const lastSignature = useRef(jobSignature);
  useEffect(() => {
    if (lastSignature.current === jobSignature) return;
    lastSignature.current = jobSignature;
    api.clip(vodId, clipId).then(setClip).catch((e: Error) => onError(e.message));
  }, [jobSignature, vodId, clipId, onError]);

  const flush = useCallback(async () => {
    if (saveTimer.current !== null) {
      window.clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    const latest = specRef.current;
    if (!latest) return null;
    setSaveState("saving");
    try {
      const detail = await api.saveClip(vodId, clipId, latest);
      setClip(detail);
      // Eingaben, die waehrend des Speicherns dazukamen, nicht ueberschreiben.
      if (specRef.current === latest) {
        setSpec(detail.spec);
        setSaveState("saved");
      } else {
        setSaveState("pending");
      }
      return detail;
    } catch (reason) {
      setSaveState("pending");
      onError((reason as Error).message);
      return null;
    }
  }, [vodId, clipId, onError]);

  useEffect(
    () => () => {
      // Beim Verlassen nichts verlieren: offene Aenderung sofort speichern.
      if (saveTimer.current !== null && specRef.current) {
        window.clearTimeout(saveTimer.current);
        void api.saveClip(vodId, clipId, specRef.current).catch(() => undefined);
      }
    },
    [vodId, clipId],
  );

  function update(change: Partial<ClipSpec>, delay = SAVE_DELAY_MS) {
    setSpec((old) => (old ? { ...old, ...change } : old));
    setSaveState("pending");
    if (saveTimer.current !== null) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => void flush(), delay);
  }

  // Beim Ziehen nur lokal aendern; gespeichert wird beim Loslassen.
  function preview(change: Partial<ClipSpec>) {
    setSpec((old) => (old ? { ...old, ...change } : old));
    setSaveState("pending");
  }

  const seek = useCallback((time: number) => {
    if (player.current) player.current.currentTime = time;
    setCurrent(time);
  }, []);

  useEffect(() => {
    if (player.current && spec) seek(spec.start);
    // Nur beim Oeffnen zum Clipstart springen, nicht bei jeder Bearbeitung.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasSpec, seek]);

  function onTimeUpdate() {
    const element = player.current;
    if (!element || !spec) return;
    setCurrent(element.currentTime);
    if (!element.paused && element.currentTime >= spec.end) element.currentTime = spec.start;
  }

  async function approve() {
    await flush();
    try {
      setClip(await api.approve(vodId, clipId));
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  async function requestPreview() {
    await flush();
    try {
      await api.preview(vodId, clipId);
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  async function setState(action: () => Promise<ClipDetail>) {
    try {
      setClip(await action());
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  if (!clip || !spec) return <div className="page muted">Lade…</div>;

  const busy = BUSY.has(clip.status);
  const assessment = clip.candidate.assessment;
  const hooks = assessment?.hooks ?? [];
  const position = neighbours.indexOf(clipId);
  const next = position >= 0 ? neighbours[position + 1] : neighbours[0];
  const previewJob = relevantJobs.find((job) => job.kind === "preview");
  const renderJob = relevantJobs.find((job) => job.kind === "render");
  const video = clip.video;

  return (
    <div className="page editor">
      <div className="editor-head">
        <a className="back" href={`#/vod/${vodId}`}>
          ← Zur Liste
        </a>
        <span className="score inline" title="Ordnet die Clips dieses VODs untereinander; keine Vorhersage von Aufrufen">
          {Math.round(clip.score)}
        </span>
        <span className={`pill clip-${clip.status}`}>{CLIP_STATUS_LABELS[clip.status]}</span>
        <span className="muted small save-state">
          {saveState === "saved" ? "Gespeichert" : saveState === "saving" ? "Speichert…" : "Nicht gespeichert"}
        </span>
        <div className="spacer" />
        {next && next !== clipId && (
          <a className="link" href={`#/vod/${vodId}/clip/${next}`}>
            Nächster Vorschlag →
          </a>
        )}
      </div>

      {assessment?.reason && <p className="reason lead">{assessment.reason}</p>}
      {clip.error && <div className="banner error">{clip.error}</div>}
      {busy && (
        <div className="banner info">
          {renderJob && renderJob.status !== "queued"
            ? `${STEP_LABELS.render}: ${Math.round(renderJob.progress * 100)} %`
            : "Wartet in der Warteschlange."}{" "}
          Bearbeiten ist erst danach wieder möglich.
        </div>
      )}

      <div className="editor-grid">
        <section className="editor-source">
          <video
            ref={player}
            src={urls.video(vodId)}
            controls
            preload="metadata"
            onTimeUpdate={onTimeUpdate}
            onLoadedMetadata={() => seek(spec.start)}
          />
          <Timeline
            viewStart={viewStart}
            viewEnd={viewEnd}
            start={spec.start}
            end={spec.end}
            peak={spec.peak}
            current={current}
            curve={curve.curve}
            curveStart={curve.start}
            disabled={busy}
            onChange={preview}
            onCommit={() => void flush()}
            onSeek={seek}
          />
          <div className="time-buttons">
            <button className="secondary" disabled={busy} onClick={() => update({ start: Math.min(current, spec.end - 3), peak: Math.max(spec.peak, Math.min(current, spec.end - 3)) }, 0)}>
              Start: {formatPrecise(spec.start)}
            </button>
            <button className="secondary" disabled={busy} onClick={() => update({ peak: Math.min(Math.max(current, spec.start), spec.end) }, 0)}>
              Höhepunkt: {formatPrecise(spec.peak)}
            </button>
            <button className="secondary" disabled={busy} onClick={() => update({ end: Math.max(current, spec.start + 3), peak: Math.min(spec.peak, Math.max(current, spec.start + 3)) }, 0)}>
              Ende: {formatPrecise(spec.end)}
            </button>
            <span className="muted small">Länge {formatTime(spec.end - spec.start)}. Klick setzt den Wert auf die Position im Video.</span>
          </div>
        </section>

        <section className="editor-panel">
          <div className="tabs" role="tablist">
            {TABS.map((item) => (
              <button key={item.id} role="tab" aria-selected={tab === item.id} onClick={() => setTab(item.id)}>
                {item.label}
              </button>
            ))}
          </div>

          {tab === "hook" && (
            <div className="panel-body">
              <fieldset disabled={busy}>
                <legend>Vorschläge des Modells. Prüfe sie gegen das Gesagte, das Transkript kann Hörfehler enthalten.</legend>
                {hooks.map((hook) => (
                  <label key={hook} className="option">
                    <input type="radio" name="hook" checked={spec.hook_text === hook} onChange={() => update({ hook_text: hook }, 0)} />
                    {hook}
                  </label>
                ))}
                {hooks.length === 0 && <p className="muted small">Das Modell hat keine Hooks geliefert.</p>}
              </fieldset>
              <label>
                Eigener Text
                <input disabled={busy} value={spec.hook_text} onChange={(event) => update({ hook_text: event.target.value })} placeholder="Leer lassen für keine Hook" />
              </label>
              <label className="option">
                <input type="checkbox" disabled={busy} checked={spec.teaser} onChange={(event) => update({ teaser: event.target.checked }, 0)} />
                Höhepunkt kurz als Teaser voranstellen
              </label>
            </div>
          )}

          {tab === "subs" && (
            <div className="panel-body">
              <p className="hint">Korrigiere Verhörer direkt in der Zeile, eine leere Zeile entfällt. So kannst du auch Dialog aus dem Spiel entfernen. Großbuchstaben setzt der Renderer automatisch.</p>
              {clip.lines.length === 0 && <p className="muted small">In diesem Clip wurde nichts gesprochen.</p>}
              <ul className="sub-list">
                {clip.lines.map((line) => (
                  <SubtitleRow
                    key={line.start}
                    line={line}
                    disabled={busy}
                    active={current >= line.start && current < line.end}
                    onSeek={() => seek(line.start)}
                    onCommit={(text) => update({ text_edits: [...spec.text_edits, { start: line.start, end: line.end, text }] }, 0)}
                  />
                ))}
              </ul>
              {spec.text_edits.length > 0 && (
                <button className="link" disabled={busy} onClick={() => update({ text_edits: [] }, 0)}>
                  Alle Korrekturen zurücksetzen
                </button>
              )}
            </div>
          )}

          {tab === "layout" && (
            <div className="panel-body">
              <fieldset disabled={busy}>
                <legend>Layout</legend>
                {LAYOUTS.map((layout) => (
                  <label key={layout.id} className="option">
                    <input type="radio" name="layout" checked={spec.layout === layout.id} onChange={() => update({ layout: layout.id }, 0)} />
                    <span>
                      {layout.label} <span className="muted small">{layout.note}</span>
                    </span>
                  </label>
                ))}
              </fieldset>
              {spec.layout === "split" && !spec.cam_rect && (
                <div className="banner warn">Ohne Facecam-Rahmen wird dieser Clip als Vollbild gerendert. Setze unten einen Rahmen.</div>
              )}
              {video && (
                <FrameEditor
                  image={urls.frame(vodId, spec.peak)}
                  width={video.width}
                  height={video.height}
                  cam={spec.cam_rect}
                  showCam={spec.layout !== "blur"}
                  gameCenter={spec.game_center}
                  gameAspect={GAME_ASPECT[spec.layout]}
                  disabled={busy}
                  onCamChange={(rect) => update({ cam_rect: rect }, 0)}
                  onGameCenterChange={(center) => update({ game_center: center })}
                />
              )}
              <p className="hint">Änderungen am Facecam-Rahmen gelten für alle offenen Clips dieses VODs und werden für den Streamer gespeichert.</p>
            </div>
          )}

          {tab === "fx" && (
            <div className="panel-body">
              <fieldset disabled={busy}>
                <legend>Effekte</legend>
                {EFFECTS.map((effect) => (
                  <label key={effect.id} className="option">
                    <input
                      type="checkbox"
                      checked={spec.effects[effect.id]}
                      onChange={(event) => update({ effects: { ...spec.effects, [effect.id]: event.target.checked } }, 0)}
                    />
                    <span>
                      {effect.label} <span className="muted small">{effect.note}</span>
                    </span>
                  </label>
                ))}
              </fieldset>
              <label>
                Credit im Bild
                <input disabled={busy} value={spec.credit} onChange={(event) => update({ credit: event.target.value })} placeholder="Name des Streamers" />
              </label>
            </div>
          )}

          {tab === "text" && (
            <div className="panel-body">
              <label>
                Titel
                <input disabled={busy} value={spec.title} onChange={(event) => update({ title: event.target.value })} />
              </label>
              <label>
                Beschreibung
                <textarea disabled={busy} rows={4} value={spec.description} onChange={(event) => update({ description: event.target.value })} />
              </label>
              <label>
                Hashtags
                <input
                  disabled={busy}
                  value={spec.hashtags.join(" ")}
                  onChange={(event) => update({ hashtags: event.target.value.split(/\s+/).filter(Boolean) })}
                />
              </label>
            </div>
          )}
        </section>

        <section className="editor-preview">
          <div className="preview-actions">
            <button className="primary" disabled={busy} onClick={() => void approve()}>
              {clip.status === "done" ? "Neu rendern" : "Freigeben und rendern"}
            </button>
            <button className="secondary" disabled={busy || Boolean(previewJob)} onClick={() => void requestPreview()}>
              Vorschau rendern
            </button>
            {clip.status === "done" && (
              <a className="button secondary" href={urls.outputFile(vodId, clipId)} target="_blank" rel="noreferrer">
                Fertigen Clip ansehen
              </a>
            )}
            {["rejected", "off_topic", "failed"].includes(clip.status) ? (
              <button className="link" onClick={() => void setState(() => api.restore(vodId, clipId))}>
                Zurückholen
              </button>
            ) : (
              <button className="link danger" disabled={busy} onClick={() => void setState(() => api.reject(vodId, clipId))}>
                Ablehnen
              </button>
            )}
          </div>
          <div className="phone">
            {clip.preview_path && !previewJob ? (
              <video key={clip.preview_version} src={urls.previewFile(vodId, clipId, String(clip.preview_version))} controls preload="metadata" />
            ) : (
              <div className="phone-empty">
                {previewJob ? (
                  <>
                    {previewJob.status === "queued" ? "Vorschau wartet" : STEP_LABELS.preview}
                    {activeJob === previewJob && previewJob.status !== "queued" && (
                      <div className="progress">
                        <div style={{ width: `${previewJob.progress * 100}%` }} />
                      </div>
                    )}
                  </>
                ) : (
                  "Noch keine aktuelle Vorschau"
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
