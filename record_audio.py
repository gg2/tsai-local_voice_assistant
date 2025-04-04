import os
import sounddevice as sd
import wave
import numpy as np


sampling_rate = 16000  # set sample rate to 16 kHz for compatibility with whisper.cpp

# Record audio using sounddevice
def record_audio(duration, data_path):

    recorded_audio = sd.rec(
        int(duration * sampling_rate),
        samplerate=sampling_rate,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()  # Wait until recording is finished

    # Save audio to WAV file
    audio_file = os.path.join(data_path, 'recorded_audio.wav')
    with wave.open(audio_file, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sampling_rate)
        wf.writeframes(recorded_audio.tobytes())

    return audio_file
