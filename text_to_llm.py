import subprocess
import re


def run_ollama_command(model, prompt):
    try:
        # Execute the ollama command using subprocess
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )

        # Output the result from Ollama
        print("Response from Ollama:")
        print(result.stdout)
        return result.stdout

    # Handle errors in case of a problem with the command
    except subprocess.CalledProcessError as e:
        print(f"Error executing Ollama command:\n{e.stderr}")
        raise


# Parse the transcription text
def text_to_llm(transcription, prompt_template, placeholder):

    # Use regex to find all text after timestamps
    matches = re.findall(r'] *(.*)', transcription)

    # Concatenate all extracted text
    concatenated_text = ' '.join(matches)

    # Call ollama to get an answer
    prompt = prompt_template.replace(placeholder, concatenated_text)
    llm_response = run_ollama_command(model='qwen:0.5b', prompt=prompt)

    return llm_response
