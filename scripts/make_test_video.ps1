# Erzeugt ein kurzes Testvideo mit Sprache (Windows-Sprachausgabe) und Testbild.
# Aufruf: powershell -File scripts/make_test_video.ps1 -Out G:/clipped-data/test/test.mp4
param([string]$Out = "G:/clipped-data/test/test.mp4")

Add-Type -AssemblyName System.Speech

$text = @"
Okay chat, we are finally playing GTA 6. Look at this map, this is Vice City and it is huge.
Wait, wait, wait. Why is the police already here? I did nothing wrong.
Oh my god, no way! He just jumped off the bridge with the car. That is the craziest thing I have ever seen in GTA 6.
Chat, did you see that? I can't believe that actually worked. Let's do it again.
Anyway, let's go and check the next mission. This one is going to be a little bit slower.
"@

$dir = Split-Path $Out -Parent
New-Item -ItemType Directory -Force $dir | Out-Null
$wav = Join-Path $dir "speech.wav"

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile($wav)
$synth.Speak($text)
$synth.Dispose()

ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=s=1920x1080:r=30" -i $wav `
    -shortest -c:v h264_nvenc -c:a aac -pix_fmt yuv420p $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item $wav
Write-Output "Testvideo: $Out"
