@echo off
rem Startet Clipped: baut die Oberflaeche bei Bedarf und oeffnet sie im Browser.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv wurde nicht gefunden. Siehe README.md, Abschnitt Einrichtung.
    pause
    exit /b 1
)

if not exist "web\dist\index.html" (
    echo Oberflaeche wird gebaut...
    pushd web
    call npm install --no-audit --no-fund
    if errorlevel 1 goto :build_failed
    call npm run build
    if errorlevel 1 goto :build_failed
    popd
)

if not defined CLIPPED_NO_BROWSER start "" "http://127.0.0.1:8765"
uv run clipped serve --port 8765
if errorlevel 1 pause
exit /b 0

:build_failed
popd
echo Die Oberflaeche konnte nicht gebaut werden.
pause
exit /b 1
