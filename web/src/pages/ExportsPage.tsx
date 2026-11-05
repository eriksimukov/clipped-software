import { useState } from "react";
import { api, ExportItem, urls } from "../api";
import { usePoll } from "../util";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="secondary"
      disabled={!text}
      onClick={() =>
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        })
      }
    >
      {copied ? "Kopiert" : label}
    </button>
  );
}

/** Zerlegt die Exportdatei in die Abschnitte, die auf den Plattformen getrennt eingegeben werden. */
function sections(text: string): Record<string, string> {
  const result: Record<string, string> = {};
  let current = "";
  for (const line of text.split("\n")) {
    if (/^(TITEL|BESCHREIBUNG|HASHTAGS)$/.test(line)) {
      current = line;
      result[current] = "";
    } else if (current) {
      result[current] += `${line}\n`;
    }
  }
  return Object.fromEntries(Object.entries(result).map(([key, value]) => [key, value.trim()]));
}

function ExportCard({ item, onError }: { item: ExportItem; onError: (message: string) => void }) {
  const parts = sections(item.text);
  return (
    <li className="card export-card">
      <video src={urls.outputFile(item.vod_id, item.id)} controls preload="metadata" />
      <div className="export-body">
        <strong>{parts.TITEL || item.spec.hook_text || item.id}</strong>
        <p className="muted small">{item.vod_title}</p>
        <p className="small pre">{parts.BESCHREIBUNG}</p>
        <p className="small hashtags">{parts.HASHTAGS}</p>
        <div className="clip-actions">
          <CopyButton text={parts.TITEL ?? ""} label="Titel kopieren" />
          <CopyButton text={parts.BESCHREIBUNG ?? ""} label="Beschreibung kopieren" />
          <CopyButton text={parts.HASHTAGS ?? ""} label="Hashtags kopieren" />
        </div>
        <div className="clip-actions">
          <a className="link" href={`#/vod/${item.vod_id}/clip/${item.id}`}>
            Bearbeiten
          </a>
          <button className="link" onClick={() => void api.openFolder(item.vod_id).catch((e: Error) => onError(e.message))}>
            Ordner öffnen
          </button>
        </div>
      </div>
    </li>
  );
}

export function ExportsPage({ onError }: { onError: (message: string) => void }) {
  const exports = usePoll(api.exports, 5000);
  const items = exports.data ?? [];
  return (
    <div className="page">
      <div className="page-head">
        <h1>Fertige Clips</h1>
        <button className="secondary" onClick={() => void api.openFolder().catch((e: Error) => onError(e.message))}>
          Export-Ordner öffnen
        </button>
      </div>
      {exports.data && items.length === 0 && (
        <p className="muted">Noch keine fertigen Clips. Gib Vorschläge frei, dann erscheinen sie hier.</p>
      )}
      <ul className="export-grid">
        {items.map((item) => (
          <ExportCard key={`${item.vod_id}/${item.id}`} item={item} onError={onError} />
        ))}
      </ul>
    </div>
  );
}
