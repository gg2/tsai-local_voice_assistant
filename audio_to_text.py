import os
import subprocess


whisper_install_path = os.getcwd()
WHISPER_BINARY_PATH = os.path.join(whisper_install_path, 'whisper.cpp', 'main')
MODEL_PATH = os.path.join(whisper_install_path, 'whisper.cpp', 'models', 'ggml-base.en.bin')


def audio_to_text(audio_file):

    try:
        result = subprocess.run(
            [
                WHISPER_BINARY_PATH,
                '-m',
                MODEL_PATH,
                '-f',
                audio_file,
                '-l',
                'en',
                '-otxt',
            ],
            capture_output=True,
            text=True,
        )
        # Display the transcription
        transcription = result.stdout.strip()
    except FileNotFoundError as e:
        print("Whisper.cpp binary not found. Make sure the path to the binary is correct.")
        raise

    return transcription
