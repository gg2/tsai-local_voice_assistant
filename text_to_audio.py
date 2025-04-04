from io import BytesIO
import nemo_tts
import torchaudio


# Integrate NVIDIA NeMo TTS to read the answer from ollama
def text_to_audio(answer):

    if answer:
        try:
            # Load the FastPitch and HiFi-GAN models from NeMo
            fastpitch_model = nemo_tts.models.FastPitchModel.from_pretrained(
                model_name='tts_en_fastpitch'
            )
            hifigan_model = nemo_tts.models.HifiGanModel.from_pretrained(
                model_name='tts_en_lj_hifigan_ft_mixerttsx'
            )

            # Set the FastPitch model to evaluation mode
            fastpitch_model.eval()
            parsed_text = fastpitch_model.parse(answer)
            spectrogram = fastpitch_model.generate_spectrogram(tokens=parsed_text)

            # Convert the spectrogram into an audio waveform using HiFi-GAN vocoder
            hifigan_model.eval()
            audio = hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)

            # Save the audio to a byte stream
            audio_buffer = BytesIO()
            torchaudio.save(audio_buffer, audio.cpu(), sample_rate=22050, format='wav')
            audio_buffer.seek(0)

        except Exception as e:
            print(f"An error occurred during speech synthesis:\n{e}")
            raise
