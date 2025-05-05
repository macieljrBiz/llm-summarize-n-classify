import os
import sys
import argparse

from dotenv import load_dotenv
from openai import AzureOpenAI
from PyPDF2 import PdfReader

# Just to make it easier to test ;)
# You need to comment to run from the command line, otherwise parameters will be ignored
sys.argv = ['main.py', '--file', './peticoes/previdencia/01.pdf', '--debug']	

def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Sets the script to use DEBUG.env environment vars')
    parser.add_argument('--file', type=str,
                        help='The file to process')
    return parser.parse_args()

# Helper function to load the environment variables
def load_env_vars(_args):
    if _args.debug:
        load_dotenv('DEBUG.env', override=True)
    else:
        load_dotenv()

def get_openai_client():
    return AzureOpenAI(
        api_version="2024-12-01-preview",
        api_key=os.getenv("AOAI_KEY"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT")
    )

def load_prompt_from_file():
    # Load the prompt from a file
    with open('prompt.txt', 'r', encoding='utf-8') as prompt_file:
        prompt_content = prompt_file.read()
    return prompt_content

def generate_prompt(pdf_content):
    # Prepare the messages for the chat completion
    messages = [
        {"role": "system", "content": load_prompt_from_file()},
        {"role": "user", "content": pdf_content}
    ]
    return messages

def load_pdf_text(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def main():
    args = set_argparse()
    load_env_vars(args)
    aoai_client = get_openai_client()
    pdf_content = load_pdf_text(args.file)
    # Call the Azure OpenAI API
    response = aoai_client.chat.completions.create(
        messages=generate_prompt(pdf_content),
        # max_tokens=4096,
        temperature=0.1,
        top_p=0.1,
        model=os.getenv("AOAI_MODEL")
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    main()
