#!/usr/bin/env python

import os
from record_audio import record_audio
from audio_to_text import audio_to_text
from text_to_llm import text_to_llm
from text_to_audio import text_to_audio



def main():

    data_path = os.path.join(os.getcwd(), 'data')

    duration = 1
    audio_file = record_audio(duration, data_path)

    transcription = audio_to_text(audio_file)

    placeholder = '<PLACEHOLDER>'
    prompt_template = f"""
    Please ignore the text [BLANK_AUDIO]. Given this question: "{placeholder}", please answer it in less than 15 words.
    """
    llm_response = text_to_llm(transcription, prompt_template, placeholder)

    text_to_audio(llm_response)

    return 0


if __name__ == "__main__":
    main()
