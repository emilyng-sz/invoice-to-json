# Invoice to json

## About this repository
- Converts multi-page invoice PDF to image format
- Each image is input to an LLM with a prompt to output a raw json dump
- Intermediate function cleans and combines the multipage json output with Python logic
- Final output is a parsed json file
- Note: the prompt is tailored to the project's specific downstream use case. This repository is not a one-size fits all invoice parser. Further customisations may be necessary for your use case.

## Prerequisites
1. Initialise and activate python venv
```
python -m venv venv
source venv/bin/activate
```
2. Download required libraries from `requirements.txt`: `pip install -r requirements.txt`
3. Ensure `config.yml` and `config.env` file is in place with the sample keys below:
```config.yml
log_file_path: <log_file_path>
purpose: invoice
LOCAL_PDF_SOURCE_DIR: data/raw_pdf
LOCAL_IMG_DEST_DIR: data/images
LOCAL_JSON_DEST_DIR: data/json
GPT_ENV_FILE: config.env
prompt_file_path: prompts.json
input_pdf: <input_pdf>
```

```config.env
API_KEY_GPT4=<PLACEHOLDER>
AZURE_OPENAI_ENDPOINT=<PLACEHOLDER>
API_VERSION_GPT4=<PLACEHOLDER>
DEPLOYMENT_NAME_GPT4=<PLACEHOLDER>
MAX_COMPLETION_TOKENS=2000
```
4. Based on the `LOCAL_PDF_SOURCE_DIR` and `input_pdf` value specified in the `config.yml`, place the `input_pdf` file in PDF format in `LOCAL_PDF_SOURCE_DIR`.

## How to run pipeline
1. Ensure `purpose` in `config.yml` is "invoice", this controls which prompt is being read. There is currently only "invoice".
2. Ensure raw pdf files (invoices) are placed in the folder specified in `LOCAL_PDF_SOURCE_DIR` variable of the `config.yml`
3. Run `invoice_to_json.py`.

## Main Functions to understand the pipeline:
- `invoice_to_json`: 
    - Converts invoice in PDF format to json. 
    - For input pdf named "invoice_abc.pdf", each page will first be converted to an image, which is saved in `LOCAL_IMG_DEST_DIR` folder (specified in `config.yml`). These images will be named "invoice_abc_i_of_X.png" for range(1, X+1) of a X-paged pdf.
    - A process will then encode each image and return a parsed json using OpenAI LLM. 
    - All image jsons are compiled into one json file and saved as "invoice_abc_X_of_X.json"

- `parse_pdf_json` takes in a compiled pdf json, e.g. "invoice_abc_X_of_X.json" and returns "invoice_abc_X_of_X_parsed.json" with Python logic