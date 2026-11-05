# Clipped

Macht aus langen Streamer-VODs fertige GTA-6-Clips im Hochformat: mit Untertiteln, Hook, Teaser, Zooms, Sounds, Wasserzeichen, Credit und einer Endkarte mit deinen Kanälen. Alles läuft lokal auf deinem PC. Die Software schlägt vor, du entscheidest.

## Starten

Doppelklick auf `start.cmd`. Die Oberfläche öffnet sich unter <http://127.0.0.1:8765>.

Ollama startet die Software bei Bedarf selbst. Das Fenster mit dem Server muss offen bleiben, solange du arbeitest. Schließt du es mitten in einem Job, setzt die Arbeit beim nächsten Start dort wieder an.

## So arbeitest du damit

1. **Einstellungen:** Trage einmal deine Kanäle und dein Wasserzeichen ein. Bis dahin steht `@deinhandle` in den Clips.
2. **VOD importieren:** Gib einen Link (YouTube, Twitch, Kick) oder einen Dateipfad an, oder ziehe eine Datei hinein. Den Streamer-Namen brauchst du nur bei lokalen Dateien.
3. **Warten:** Die Software transkribiert, sucht Höhepunkte und bewertet sie.
4. **Vorschläge prüfen:** Die Liste ist nach Score sortiert, mit Begründung des Modells.
5. **Bearbeiten:** Im Editor stellst du Schnittpunkte, Hook, Untertitel, Layout und Effekte ein. „Vorschau rendern" zeigt dir das Ergebnis in kleiner Qualität.
6. **Freigeben:** Der fertige Clip erscheint unter „Fertige Clips", dort kopierst du Titel, Beschreibung und Hashtags für den Upload.

Die Dateien liegen danach in `G:\clipped-data\export\`, je Clip eine MP4 und eine TXT.

### Was der Score bedeutet

Der Score (0 bis 100) ordnet die Clips eines VODs untereinander: 70 % Einschätzung des Sprachmodells, 30 % Signale aus Lautstärke, Chat und Sprache. Er sagt keine Aufrufe voraus. Deine Prüfung bleibt deshalb fester Teil des Ablaufs.

### GTA-6-Filter

Berücksichtigt werden nur Teile des VODs mit GTA-6-Bezug, in dieser Reihenfolge:

1. Dein manueller Zeitbereich, falls angegeben
2. Kapitel mit GTA 6 im Namen, zum Beispiel die Spielkategorie bei Twitch
3. GTA 6 im Titel des VODs: dann das ganze VOD
4. Erwähnungen im Gesprochenen: jeweils 10 Minuten davor und danach

GTA 6 und GTA V lassen sich am Bild allein nicht zuverlässig unterscheiden. Läuft im VOD beides, gib den Zeitbereich von Hand an.

### Facecam

Die Facecam wird automatisch erkannt. Stimmt der Rahmen nicht, ziehe ihn im Editor unter „Layout" zurecht. Die Korrektur gilt dann für alle offenen Clips des VODs und wird für künftige VODs des Streamers gespeichert.

## Eigene Sounds

Lege Dateien in `assets/sfx/whoosh`, `impact`, `riser` oder `pop`. Sie werden zusammen mit dem eingebauten Starter-Pack verwendet. Details und Lizenzen stehen in [assets/README.md](assets/README.md).

## Fremde Inhalte

Der Credit ersetzt keine Erlaubnis des Streamers. Beim Veröffentlichen sind Claims oder Sperren durch Streamer, Rechteinhaber oder Plattformen möglich, und das Herunterladen von VODs kann gegen Plattformregeln verstoßen. Das Tool prüft das nicht.

## Einrichtung auf einem neuen PC

Voraussetzungen: Windows, eine NVIDIA-Grafikkarte mit mindestens 8 GB, [scoop](https://scoop.sh) und Node.js.

```bash
scoop install uv ffmpeg ollama
```

```bash
uv sync
```

```bash
ollama pull gemma4:e4b
```

Prüfe danach die Umgebung. Das lädt beim ersten Mal ein kleines Whisper-Modell:

```bash
uv run python scripts/doctor.py
```

Beim ersten Transkribieren lädt die Software zusätzlich `large-v3`, rund 3 GB, in den Datenordner.

## Probleme

| Meldung | Lösung |
|---|---|
| `Modell fehlt` | `ollama pull gemma4:e4b` ausführen |
| `Ollama meldet Fehler 500` | Grafikspeicher voll oder GPU belegt. Spiele und andere GPU-Programme schließen, dann „Erneut versuchen" |
| `cublas64_12.dll` fehlt | `uv sync` ausführen, die CUDA-Bibliotheken kommen als Python-Pakete |
| `Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert` | Windows Smart App Control prüft neue DLLs online. Auf diesem PC hat es einmal eine numpy-DLL blockiert und beim nächsten Versuch zugelassen. Erst noch einmal versuchen |
| Kein Chat-Signal | Nicht jedes VOD hat ein abrufbares Chat-Replay. Die Auswahl nutzt dann Audio und Sprache |

## Für die Entwicklung

```bash
uv run pytest
```

```bash
uv run ruff check .
```

Die Oberfläche liegt in `web/`. Nach Änderungen dort neu bauen, mit `npm run build` im Ordner `web`.

```
clipped/
  api/        FastAPI, nur lokal erreichbar
  db/         SQLite: VODs, Streamer, Clips, Jobs
  jobs/       Warteschlange, ein Job nach dem anderen
  pipeline/   Import, Transkript, Signale, Auswahl, Facecam, Rendern, Export
web/          Prüf-Oberfläche (React, Vite, TypeScript)
assets/       Schriften, Facecam-Modell, Sounds
config/       settings.example.toml (settings.toml bleibt lokal)
scripts/      doctor.py, Modellvergleich, Testvideo, Layout-Prüfung
tests/
```

Arbeitsdaten eines VODs liegen in `G:\clipped-data\vods\<id>\`. Jeder Schritt speichert dort sein Ergebnis und überspringt sich beim nächsten Lauf, wenn es schon vorhanden ist.
