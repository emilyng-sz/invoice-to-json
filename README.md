# Invoice to JSON Pipeline

This repository contains code that converts an invoice PDF to images and returns two JSON files: the first file containing raw data, and the latter with parsed formats using Python logic.

**Note:** the code is tailored to the project's specific use case and downstream processing (excluded in this repository). This pipeline is therefore NOT a one-size-fits-all invoice parser. Further customisations may be necessary for your use case.

## Core Functions and Processes:
- `invoice_to_json`: 
    - Converts invoice in PDF format to JSON. 
    - For input pdf named "invoice_abc.pdf", each page will first be converted to an image, which is saved in `LOCAL_IMG_DEST_DIR` folder (specified in `config.yml`). These images will be named "invoice_abc_i_of_X.png" for range(1, X+1) of a X-paged pdf.
    - A process will encode each image and return a parsed JSON using OpenAI LLM. 
    - All image JSONs are compiled into one JSON file and saved as "invoice_abc_X_of_X.json"

- `parse_pdf_json` takes in a compiled pdf JSON, e.g. "invoice_abc_X_of_X.json" and returns "invoice_abc_X_of_X_parsed.json" with Python logic

## Prequisites
1. Ensure `config.yml` and `config.env` file is in place with the sample keys below:
- Please setup your own OpenAI API key accordingly
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
2. Based on the `LOCAL_PDF_SOURCE_DIR` and `input_pdf` value specified in the `config.yml`, place the `input_pdf` file in PDF format in `LOCAL_PDF_SOURCE_DIR`.
3. Ensure `purpose` in `config.yml` is "invoice", this controls which prompt is being read. There is currently only "invoice".

## Run pipeline
1. Initialise and activate python venv
```
python -m venv venv
source venv/bin/activate
```
2. Navigate to the root of this repository
3. Download required libraries from `requirements.txt`: `pip install -r requirements.txt`
4. Run `python invoice_to_json.py`