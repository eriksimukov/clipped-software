# Erzeugt ein Testvideo mit Sprache (Windows-Sprachausgabe), wechselndem Bild und optionaler Facecam.
# Aufruf: scripts/make_test_video.ps1 -Out G:/clipped-data/test/test_cam.mp4 -CamImage bild.png
param(
    [string]$Out = "G:/clipped-data/test/test.mp4",
    [string]$CamImage = ""
)

Add-Type -AssemblyName System.Speech

$text = @"
Okay chat, we are finally playing GTA 6. Look at this map, this is Vice City and it is huge.
Wait, wait, wait. Why is the police already here? I did nothing wrong.
Oh my god, no way! He just jumped off the bridge with the car. That is the craziest thing I have ever seen in GTA 6.
Chat, did you see that? I can't believe that actually worked. Let's do it again.
Anyway, let's go and check the next mission. This one is going to be a little bit slower.
So the mission marker is on the other side of the city, which means we have a pretty long drive ahead of us.
I'm going to change the radio station real quick, and then we are going to check out the beach, because everybody in chat keeps talking about it.
Thanks for the follow by the way, and thanks for the five gifted subs, I really appreciate you all.
Hold on. Is that a helicopter? No way. Are you serious? They are chasing me with a helicopter and I only stole one bicycle!
Let's go! Let's go! I'm going under the bridge. I made it! I actually made it! That was insane, clip that right now!
Okay, okay, I need a second. My heart is racing. This game is unbelievable. GTA 6 really is the greatest game ever made.
Now let's see what is inside this store. It looks like you can buy clothes here, and also some food for the character.
"@

$dir = Split-Path $Out -Parent
New-Item -ItemType Directory -Force $dir | Out-Null
$wav = Join-Path $dir "speech.wav"

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile($wav)
$synth.Speak($text)
$synth.Dispose()

$game = "mandelbrot=s=1920x1080:r=30:start_scale=2.5:end_scale=0.0005:end_pts=6000,hue=h=t*40"

if ($CamImage) {
    # Kopf bewegt sich leicht; die Facecam sitzt unten links wie bei vielen Streams.
    $graph = "[1:v]crop=400:225:x='56+12*sin(t)':y='4+4*cos(t*1.3)',scale=480:270,setsar=1[cam];" +
             "[0:v][cam]overlay=40:770:shortest=1[v]"
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i $game -loop 1 -framerate 30 -i $CamImage -i $wav `
        -filter_complex $graph -map "[v]" -map 2:a -shortest -c:v h264_nvenc -cq 23 -c:a aac -pix_fmt yuv420p $Out
} else {
    ffmpeg -hide_banner -loglevel error -y -f lavfi -i $game -i $wav `
        -map 0:v -map 1:a -shortest -c:v h264_nvenc -cq 23 -c:a aac -pix_fmt yuv420p $Out
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item $wav
Write-Output "Testvideo: $Out"
