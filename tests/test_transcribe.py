import wave

import numpy as np
import pytest

from clipped.pipeline.transcribe import load_wav


def _write_wav(path, rate=16000, channels=1, samples=(0, 16384, -32768)):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.array(samples, dtype=np.int16).tobytes())


def test_load_wav_scales_to_float(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path)
    assert load_wav(path).tolist() == [0.0, 0.5, -1.0]


def test_load_wav_rejects_wrong_format(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, rate=44100)
    with pytest.raises(ValueError):
        load_wav(path)
