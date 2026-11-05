# Assets und ihre Lizenzen

| Datei | Herkunft | Lizenz |
|---|---|---|
| `fonts/Anton-Regular.ttf` | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/anton) | SIL Open Font License, siehe `fonts/Anton-OFL.txt` |
| `fonts/fa-brands-400.ttf` | [Font Awesome Free](https://github.com/FortAwesome/Font-Awesome) | Icons CC BY 4.0, Schrift SIL OFL, siehe `fonts/FontAwesome-LICENSE.txt` |
| `facecam/face_detection_yunet_2023mar.onnx` | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT, siehe `facecam/YuNet-LICENSE.txt` |

Die Icons zeigen Marken der jeweiligen Plattformen. Sie sind nur dafür gedacht, auf die eigenen Kanäle zu verweisen.

## Soundeffekte

`sfx/generated/` wird beim ersten Rendern von `clipped/pipeline/sfx.py` per Code erzeugt und liegt nicht im Repo. Diese Sounds sind synthetisch, es gibt also keine fremden Rechte daran.

Eigene Sounds kommen in einen Ordner je Kategorie, die Software wählt dann unter allen Dateien aus:

```
assets/sfx/whoosh/   Übergang vom Teaser zum Clip
assets/sfx/impact/   Höhepunkt
assets/sfx/riser/    Spannungsaufbau vor dem Höhepunkt
assets/sfx/pop/      Einblendung von Hook und Endkarte
```

Unterstützt werden `.wav`, `.mp3`, `.ogg`, `.flac` und `.m4a`. Eigene Dateien landen nicht im Repo. Achte bei ihnen selbst auf die Lizenz, bevor du Clips veröffentlichst.
