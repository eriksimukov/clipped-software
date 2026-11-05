import { useEffect, useState } from "react";
import { api, Job } from "./api";
import { ClipEditor } from "./pages/ClipEditor";
import { ExportsPage } from "./pages/ExportsPage";
import { ImportPage } from "./pages/ImportPage";
import { SettingsPage } from "./pages/SettingsPage";
import { VodPage } from "./pages/VodPage";
import { STEP_LABELS, useHashRoute, usePoll } from "./util";

const JOB_LABELS: Record<Job["kind"], string> = {
  process_vod: "VOD",
  render: "Clip",
  preview: "Vorschau",
};

function JobBar({ jobs, onCancel }: { jobs: Job[]; onCancel: (id: number) => void }) {
  const running = jobs.find((job) => job.status !== "queued");
  const queued = jobs.filter((job) => job.status === "queued").length;
  if (!running && queued === 0) return null;
  return (
    <div className="jobbar" role="status">
      {running ? (
        <>
          <span className="jobbar-label">
            {running.status === "cancelling"
              ? "Wird abgebrochen…"
              : (STEP_LABELS[running.step] ?? `${JOB_LABELS[running.kind]} wird vorbereitet`)}
          </span>
          <div className="progress" aria-label="Fortschritt">
            <div style={{ width: `${Math.round(running.progress * 100)}%` }} />
          </div>
          <span className="jobbar-percent">{Math.round(running.progress * 100)} %</span>
          {running.status !== "cancelling" && (
            <button className="link" onClick={() => onCancel(running.id)}>
              Abbrechen
            </button>
          )}
        </>
      ) : (
        <span className="jobbar-label">Warteschlange startet…</span>
      )}
      {queued > 0 && <span className="muted">{queued} wartend</span>}
    </div>
  );
}

export function App() {
  const route = useHashRoute();
  const jobs = usePoll(api.jobs, 1500);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!message) return;
    const timer = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timer);
  }, [message]);

  const activeJobs = jobs.data ?? [];
  const section = route[0] ?? "";

  let page;
  if (section === "vod" && route[1] && route[2] === "clip" && route[3]) {
    page = <ClipEditor key={`${route[1]}/${route[3]}`} vodId={route[1]} clipId={route[3]} jobs={activeJobs} onError={setMessage} />;
  } else if (section === "vod" && route[1]) {
    page = <VodPage key={route[1]} vodId={route[1]} jobs={activeJobs} onError={setMessage} />;
  } else if (section === "exports") {
    page = <ExportsPage onError={setMessage} />;
  } else if (section === "settings") {
    page = <SettingsPage onError={setMessage} />;
  } else {
    page = <ImportPage jobs={activeJobs} onError={setMessage} />;
  }

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">
          CLIPPED
        </a>
        <nav>
          <a href="#/" aria-current={section === "" || section === "vod" ? "page" : undefined}>
            VODs
          </a>
          <a href="#/exports" aria-current={section === "exports" ? "page" : undefined}>
            Fertige Clips
          </a>
          <a href="#/settings" aria-current={section === "settings" ? "page" : undefined}>
            Einstellungen
          </a>
        </nav>
        <JobBar
          jobs={activeJobs}
          onCancel={(id) =>
            void api
              .cancelJob(id)
              .then(() => jobs.refresh())
              .catch((e: Error) => setMessage(e.message))
          }
        />
      </header>
      {jobs.error && !jobs.data && (
        <div className="banner error">Keine Verbindung zum lokalen Server: {jobs.error}</div>
      )}
      {message && (
        <div className="banner error" role="alert">
          {message}
          <button className="link" onClick={() => setMessage(null)}>
            Schließen
          </button>
        </div>
      )}
      <main>{page}</main>
    </div>
  );
}
