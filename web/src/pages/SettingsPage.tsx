import { FormEvent, useEffect, useState } from "react";
import { api, Channel, Settings } from "../api";

const BROWSERS = ["firefox", "chrome", "edge", "brave", "opera", "vivaldi", "chromium"];
const PLATFORMS = ["tiktok", "youtube", "instagram", "twitch", "kick", "x", "facebook", "discord"];

export function SettingsPage({ onError }: { onError: (message: string) => void }) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [saved, setSaved] = useState(false);
  const [restart, setRestart] = useState(false);

  useEffect(() => {
    api.settings().then(setSettings).catch((e: Error) => onError(e.message));
  }, [onError]);

  if (!settings) return <div className="page muted">Lade…</div>;

  function change(update: (old: Settings) => Settings) {
    setSettings((old) => (old ? update(old) : old));
    setSaved(false);
  }

  const setBranding = (values: Partial<Settings["branding"]>) =>
    change((old) => ({ ...old, branding: { ...old.branding, ...values } }));
  const setClips = (values: Partial<Settings["clips"]>) =>
    change((old) => ({ ...old, clips: { ...old.clips, ...values } }));
  const setStyle = (values: Partial<Settings["style"]>) =>
    change((old) => ({ ...old, style: { ...old.style, ...values } }));
  const setDownload = (values: Partial<Settings["download"]>) =>
    change((old) => ({ ...old, download: { ...old.download, ...values } }));

  function setChannel(index: number, values: Partial<Channel>) {
    setBranding({
      channels: settings!.branding.channels.map((channel, i) => (i === index ? { ...channel, ...values } : channel)),
    });
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    try {
      const result = await api.saveSettings(settings!);
      setSettings(result.settings);
      setRestart(result.restart_required);
      setSaved(true);
    } catch (reason) {
      onError((reason as Error).message);
    }
  }

  return (
    <form className="page narrow settings" onSubmit={(event) => void save(event)}>
      <h1>Einstellungen</h1>

      <section className="card form">
        <h2>Deine Kanäle</h2>
        <p className="hint">Stehen auf der Endkarte am Ende jedes Clips.</p>
        {settings.branding.channels.map((channel, index) => (
          <div className="channel-row" key={index}>
            <select
              aria-label="Plattform"
              value={channel.platform}
              onChange={(event) => setChannel(index, { platform: event.target.value })}
            >
              {PLATFORMS.includes(channel.platform) ? null : <option>{channel.platform}</option>}
              {PLATFORMS.map((platform) => (
                <option key={platform}>{platform}</option>
              ))}
            </select>
            <input
              aria-label="Handle"
              value={channel.handle}
              onChange={(event) => setChannel(index, { handle: event.target.value })}
              placeholder="@name"
            />
            <button
              type="button"
              className="link danger"
              onClick={() => setBranding({ channels: settings.branding.channels.filter((_, i) => i !== index) })}
            >
              Entfernen
            </button>
          </div>
        ))}
        <button
          type="button"
          className="secondary fit"
          onClick={() => setBranding({ channels: [...settings.branding.channels, { platform: "tiktok", handle: "" }] })}
        >
          Kanal hinzufügen
        </button>
        <label>
          Wasserzeichen
          <input value={settings.branding.watermark} onChange={(event) => setBranding({ watermark: event.target.value })} />
        </label>
        <div className="form-row">
          <label>
            Titel der Endkarte
            <input value={settings.branding.endcard_title} onChange={(event) => setBranding({ endcard_title: event.target.value })} />
          </label>
          <label>
            Dauer in Sekunden
            <input
              type="number"
              min={1}
              max={6}
              step={0.5}
              value={settings.branding.endcard_seconds}
              onChange={(event) => setBranding({ endcard_seconds: Number(event.target.value) })}
            />
          </label>
        </div>
      </section>

      <section className="card form">
        <h2>Stil</h2>
        <div className="form-row">
          <label>
            Farbe des aktiven Worts
            <input type="color" value={settings.style.highlight_color} onChange={(event) => setStyle({ highlight_color: event.target.value.toUpperCase() })} />
          </label>
          <label>
            Schriftgröße der Untertitel
            <input
              type="number"
              min={40}
              max={220}
              value={settings.style.subtitle_size}
              onChange={(event) => setStyle({ subtitle_size: Number(event.target.value) })}
            />
          </label>
        </div>
      </section>

      <section className="card form">
        <h2>Clip-Auswahl</h2>
        <label>
          In Untertiteln, Hooks und Texten maskieren
          <select
            value={settings.style.censor}
            onChange={(event) => setStyle({ censor: event.target.value as Settings["style"]["censor"] })}
          >
            <option value="slurs">Beleidigungen</option>
            <option value="all">Beleidigungen und Flüche</option>
            <option value="none">Nichts</option>
          </select>
        </label>
        <p className="hint">Der Ton bleibt unverändert. Clips mit Beleidigungen im Ton werden markiert und abgewertet.</p>
        <div className="form-row">
          <label>
            Mindestlänge (s)
            <input type="number" min={5} value={settings.clips.min_seconds} onChange={(event) => setClips({ min_seconds: Number(event.target.value) })} />
          </label>
          <label>
            Höchstlänge (s)
            <input type="number" min={10} value={settings.clips.max_seconds} onChange={(event) => setClips({ max_seconds: Number(event.target.value) })} />
          </label>
          <label>
            Vorschläge pro VOD
            <input
              type="number"
              min={1}
              max={100}
              value={settings.clips.candidates_per_vod}
              onChange={(event) => setClips({ candidates_per_vod: Number(event.target.value) })}
            />
          </label>
        </div>
        <p className="hint">Gilt für neue VODs. Ein Vorschlag braucht auf deinem PC nur wenige Sekunden.</p>
      </section>

      <section className="card form">
        <h2>Anmeldung für altersbeschränkte Videos</h2>
        <p className="hint">
          Nur nötig, wenn YouTube ein Video ohne Anmeldung nicht herausgibt. Laut yt-dlp kann ein Konto dabei gesperrt werden,
          vorübergehend oder dauerhaft. Nimm am besten ein Zweitkonto. Die Cookie-Datei ist der empfohlene Weg, die Schritte stehen
          in der README. Wenn du beides leer lässt, wird nichts verwendet.
        </p>
        <label>
          Cookie-Datei
          <input
            value={settings.download.cookies_file}
            onChange={(event) => setDownload({ cookies_file: event.target.value })}
            placeholder="C:\Pfad\zu\youtube-cookies.txt"
          />
        </label>
        <label>
          Oder Cookies direkt aus dem Browser lesen
          <select
            value={settings.download.cookies_from_browser}
            onChange={(event) => setDownload({ cookies_from_browser: event.target.value })}
          >
            <option value="">Nicht verwenden</option>
            {BROWSERS.map((browser) => (
              <option key={browser}>{browser}</option>
            ))}
          </select>
        </label>
        <p className="hint">Die Cookie-Datei hat Vorrang. Chrome und Edge sperren ihre Cookies unter Windows häufig, Firefox klappt eher.</p>
      </section>

      <section className="card form">
        <h2>System</h2>
        <label>
          Datenordner
          <input
            value={settings.paths.data_dir}
            onChange={(event) => change((old) => ({ ...old, paths: { data_dir: event.target.value } }))}
          />
        </label>
        <label>
          Sprachmodell (Ollama)
          <input
            value={settings.llm.model}
            onChange={(event) => change((old) => ({ ...old, llm: { ...old.llm, model: event.target.value } }))}
          />
        </label>
      </section>

      <div className="form-actions sticky">
        {saved && <span className="muted">Gespeichert</span>}
        {restart && <span className="error-text">Der neue Datenordner gilt nach einem Neustart. Bestehende Daten werden nicht verschoben.</span>}
        <button className="primary">Speichern</button>
      </div>
    </form>
  );
}
