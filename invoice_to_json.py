import base64
import io
import json
import os

from datetime import datetime
from dotenv import load_dotenv
import fitz  # PyMuPDF
from openai import AzureOpenAI
from pathlib import Path
from PIL import Image
import structlog
from typing import (
    Any,
    List,
    Union
)
import yaml

import logging
from logging.handlers import TimedRotatingFileHandler

# Setup the config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Setup the logger
log_file_path = config.get("log_file_path")
log_file = Path(log_file_path)
trf_handler = TimedRotatingFileHandler(
    log_file,
    when="midnight",
    encoding="utf-8",
    backupCount=int(os.getenv("LOG_FILE_BACKUP_COUNT", "30")))
trf_handler.suffix = "%Y%m%d"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[trf_handler]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure structlog to use stdlib logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Create a structlog logger for this module
logger = structlog.get_logger()


def _check_if_negative(s: str) -> bool:
    if s != "-" and (s.endswith("-") or s.startswith("-")):
        return True
    return False


def _check_if_zero(
        s: str,
        zero_strings: List[str] = ["-", "0.00", "Free"]) -> bool:
    for string in zero_strings:
        if s == string:
            return True
    return False


def _create_prompt_for_image_to_json(
        encoded_image: str,
        purpose: str,
        json_file_path: Path = None) -> str:
    """
    Create prompt to convert image to JSON.
    Args:
        encoded_image: the base64 encoded image string.
        purpose: the use case of the json

    Returns:
        str: The constructed prompt
    """
    prompt_dict = _read_json_file(
        json_file_path=json_file_path).get(purpose, {})

    if not prompt_dict:
        logger.info(
            event_type="error",
            event="Purpose not found in prompt dict",
            file_path=json_file_path,
            purpose=purpose
        )

    # Read the model-specific prompt, and construct the PromptBuilder
    prompt_dict = prompt_dict["prompt"]
    prompt_structure = prompt_dict["prompt_structure"]

    # Replace placeholders in the prompt structure
    prompt_structure = prompt_structure.replace(
        "{system_context}", "\n".join(prompt_dict["system_context"]))
    prompt_structure = prompt_structure.replace(
        "{core_instructions}", "\n".join(prompt_dict["core_instructions"]))
    prompt_structure = prompt_structure.replace(
        "{extraction_rules}", "\n".join(prompt_dict["extraction_rules"]))
    prompt_structure = prompt_structure.replace(
        "{data_requirements}", "\n".join(prompt_dict["data_requirements"]))
    prompt_structure = prompt_structure.replace(
        "{output_format}", "\n".join(prompt_dict["output_format"]))
    prompt_structure = prompt_structure.replace(
        "{validation_rules}", "\n".join(prompt_dict["validation_rules"]))
    prompt_structure = prompt_structure.replace(
        "{examples}", "\n".join(prompt_dict["examples"]))

    # Construct the image_input message
    image_input = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}"
        }
    }

    messages = [
        {"role": "system", "content": prompt_structure},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract the details from this invoice."},
            image_input
        ]}
    ]
    return messages


def _dict_to_json(raw_data: dict, json_file_path: Path) -> None:
    """
    Write raw_dict to json file path
    """
    try:
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(raw_data, file, indent=4)

        logger.info(
            "Wrote to file",
            file_name=json_file_path,
            event_type="info")
    except Exception as e:
        logger.info(
            event_type="error",
            event="Error encountered writing to file",
            file_name=json_file_path,
            exception=e)


def _encoded_image_to_json(
        encoded_image: str, purpose: str, image_name: str) -> dict:
    """
    Convert an image to JSON using OpenAI API.
    """
    logger.info(
        f"Converting image to JSON using OpenAI API for {purpose}",
        event_type="info")

    # Construct the prompt using the service
    constructed_prompt = _create_prompt_for_image_to_json(
        encoded_image=encoded_image,
        purpose=purpose,
        json_file_path=config.get("prompt_file_path"))

    client = AzureOpenAI(
        api_version=os.getenv("API_VERSION_GPT4"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("API_KEY_GPT4"),
    )

    try:
        response = client.chat.completions.parse(
            model=os.getenv("DEPLOYMENT_NAME_GPT4"),
            messages=constructed_prompt,
            max_completion_tokens=int(os.getenv("MAX_COMPLETION_TOKENS")),
            response_format={"type": "json_object"}
        )
        raw_api_resp_as_str = response.choices[0].message.content

        # Log token usage in structured format
        logger.info(
            event="llm_api_call",
            event_type="token_usage",
            image_name=image_name,
            encoded_image_length=len(encoded_image),
            model=os.getenv("DEPLOYMENT_NAME_GPT4"),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
            response_length=len(raw_api_resp_as_str),
        )

    except Exception as e:
        logger.info(
            event_type="error",
            event="Error during OpenAI API call",
            exception=e)
        return {}

    # Parse OpenAI response as json
    try:
        response_dict = json.loads(raw_api_resp_as_str)
        if response_dict:
            logger.info(
                "Information extracted from invoice as json",
                event_type="info")
            return response_dict
    except Exception as e:
        logger.info(
            event_type="error",
            event="Further parsing of LLM response required",
            exception=e)
        return {}


def _get_key(key_list: List[str], s: str) -> Union[str, None]:
    """Return item from list of string, if the string contains s"""
    for key in key_list:
        if s in key:
            return key
    return None



def _get_total_page_numbers_from_list(list_nums: List[str]) -> int:
    """
    Parse a list of integer strings, and get the largest integer
    """
    try:
        total_pg_count = int(max(list_nums))
        logger.info(
            f"Total page count is {total_pg_count}",
            event_type="info")
        return total_pg_count
    except Exception as e:
        logger.info(
            event_type="error",
            event="Unexpected key value pair of parsed pdf json",
            exception=e)
        return None


def _parse_line_item_dict(charge_key: str, charge_dict: dict) -> dict:
    item = charge_dict.get("item", "")
    item_description = f"{charge_key}-{item}"
    quantity = charge_dict.get("units", None)
    code = charge_dict.get("code", None)
    # Return 0 if Free?? or what should it be??
    amount = _str_to_float(charge_dict.get("amount", None))

    return {
        "item_description": item_description,
        "quantity": quantity,
        "tax_rate": None,
        "tax_amount": None,
        "amount": amount,
        "code": code
    }


def _parse_page_1(page_json: dict) -> dict:
    """
    Parse page one of invoice and return Dictionary with first level of details
    """
    acc_no = page_json.get("Account no.", "")
    bill_id = page_json.get("Bill ID", "")
    invoice_date = page_json.get("Bill Date", "")
    bill_to_entity = page_json.get("bill_to_entity", "")
    bill_to_address = page_json.get("bill_to_address", "")
    bill_from_entity = page_json.get("bill_from_entity", "")
    bill_from_address = page_json.get("bill_from_address", "")
    return {
        "invoice_id": f"{acc_no}-{bill_id}",
        "invoice_number": acc_no,
        "invoice_date": invoice_date,
        "bill_from_entity": bill_from_entity,
        "bill_from_address": bill_from_address,
        "bill_to_entity": bill_to_entity,
        "bill_to_address": bill_to_address,
        "pages": None
    }


def _pdf_to_bytes(
        pdf_file_path: Path):
    if not pdf_file_path.exists():
        logger.info(
            event_type="error",
            event="File does not exists",
            file_name=pdf_file_path,
            exception=FileNotFoundError(pdf_file_path))
        raise FileNotFoundError(pdf_file_path)
    else:
        return pdf_file_path.read_bytes()


def invoice_to_json(
        pdf_file_path: Path,
        purpose: str,
        local_imag_dest_dir: Path,
        local_json_dest_dir: str) -> tuple[Any]:
    """
    Takes in a path to pdf
    Returns (raw_data: dict, json_file_path: str)
    """
    pdf_file_contents = _pdf_to_bytes(pdf_file_path=pdf_file_path)

    image_metadata, pg_count = _split_pdf_to_images_and_save_to_folder(
        pdf_file_name=pdf_file_path,
        pdf_file_contents=pdf_file_contents,
        destination_folder=local_imag_dest_dir
    )

    raw_data = {}
    image_name = pdf_file_path.stem
    for pg in range(pg_count):
        image_name = image_metadata[pg]["file_name"].split('.')[0]
        image_file_path = \
            Path(local_imag_dest_dir) / image_metadata[pg]['file_name']
        logger.info(
            "Start image processing",
            file_name=image_file_path,
            event_type="info")

        # Analyse image
        encoded_image = _read_image_convert_base64(
            image_file_path=image_file_path)

        # Get json
        response_dict = _encoded_image_to_json(
            encoded_image=encoded_image,
            purpose=purpose,
            image_name=image_file_path.stem)

        # Append to raw_data, start at page 1
        raw_data[pg+1] = response_dict

    # Dump raw_data to json file
    json_file_path = Path(local_json_dest_dir) / f"{image_name}.json"
    _dict_to_json(
        raw_data=raw_data,
        json_file_path=json_file_path)

    return raw_data, json_file_path


def _split_pdf_to_images_and_save_to_folder(
        pdf_file_name: str,
        pdf_file_contents: bytes,
        destination_folder: Path,
        dpi=108) -> tuple[Any]:
    """
    Convert PDF pages to images using_pymupdf
    and upload them to local folder

    Returns (image_metadata: List[dict], pg_count: int)
    """

    # Calculate the zoom factor based on DPI
    zoom = dpi / 72  # 72 is the default PDF DPI
    image_metadata = []

    try:
        # Create a temporary file-like object for the PDF
        with io.BytesIO(pdf_file_contents) as pdf_stream:
            # Open the PDF
            pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            pg_count = pdf_document.page_count

            # Iterate through pages
            logger.info(
                f"{pdf_file_name} contains {pg_count} pages",
                event_type="info")
            for page in pdf_document:  # iterate the document pages
                page_no = page.number + 1

                # Create a transformation matrix with rotation
                mat = fitz.Matrix(zoom, zoom)

                # Render the page
                pix = page.get_pixmap(matrix=mat, alpha=True)

                # Convert to PIL Image
                img = Image.frombytes(
                    "RGBA", [pix.width, pix.height], pix.samples)

                # Create a white background for transparency
                img_w_white_bg =\
                    Image.new("RGB", img.size, (255, 255, 255))
                # Use alpha channel as mask
                img_w_white_bg.paste(img, mask=img.split()[3])

                # Generate the image file name
                image_file_name =\
                    (f"{Path(pdf_file_name).stem}_{page_no}_of_{pg_count}"
                        ".png")

                # Upload image to local folder

                # Ensure we only use the filename (no path traversal)
                # Enforce .png extension
                safe_stem = Path(image_file_name).stem
                processed_dt = int(datetime.now().strftime("%Y%m%d%H%M%S"))

                # Can append timestamp here to avoid overwrite
                final_name = f"{safe_stem}.png"
                dest_path = destination_folder / final_name

                # Save the image locally
                img_w_white_bg.save(dest_path, format="PNG")

                # Build a local file URL
                image_local_url = dest_path.resolve().as_uri()

                # Add image metadata
                image_metadata.append({
                    "file_name": final_name,
                    "processed_dt": processed_dt,
                    "url": image_local_url,   # local file URL
                })

            pdf_document.close()
            return image_metadata, pg_count

    except Exception as e:
        logger.info(
            event_type="error",
            event="Error converting PDF to images",
            exception=e,
            pdf_file_name=pdf_file_name)
        return [], 0


def _str_to_float(s: str) -> float:
    """
    Returns a float value from string
    """
    if _check_if_zero(s):
        return 0

    is_negative = _check_if_negative(s)
    if is_negative:
        s = s.replace("-", "")

    # Handle strings with ","
    if s.count(",") and not s.replace(",", "").replace(".", "").isdigit():
        raise ValueError("Invalid numeric format")
    else:
        s = s.replace(",", "")

    try:
        value = float(s)
        if is_negative:
            return -value
        return value

    except Exception as e:
        logger.info(
            event_type="error",
            event="Unexpected error in converting {s} to integer",
            exception=e)


def _read_image_convert_base64(image_file_path: Path) -> str:
    """
    Read image from file path and convert to base64
    """
    image_data = image_file_path.read_bytes()
    encoded_image = base64.b64encode(image_data).decode("ascii")
    logger.info(
        "Image read and converted to base64",
        file_name=image_file_path,
        event_type="info")
    return encoded_image


def _read_json_file(json_file_path: str = "prompts.json") -> dict:
    logger.info(
        "Reading json file",
        event_type="info",
        file_name=json_file_path)

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        logger.info(
            event_type="error",
            event="File not found",
            file_name=json_file_path,
            exception="FileNotFoundError")
        raise
    except json.JSONDecodeError:
        logger.info(
            event_type="error",
            event="Error decoding JSON File",
            file_name=json_file_path,
            exception=e)
        raise
    except Exception as e:
        logger.info(
            event_type="error",
            event="Unexpected error reading JSON File",
            file_name=json_file_path,
            exception=e)
        raise


def parse_pdf_json(
        pdf_json_file_path: Path,
        pdf_json: dict = None) -> dict:
    """Takes in either a pdf file path or the dictionary"""

    if not pdf_json:
        pdf_json = _read_json_file(pdf_json_file_path)

    keys = list(pdf_json.keys())
    total_pg_count = _get_total_page_numbers_from_list(list_nums=keys)
    if not total_pg_count:
        logger.info(
            event_type="error",
            event="Error encountered in parsing file. Exiting",
            file_name=pdf_json_file_path,
            exception="No page count")
        return

    # Initialise pdf_json_parsed and line_items
    pdf_json_parsed, line_items, gst_items = {}, [], []

    # Parse first page
    if total_pg_count >= 1:
        pdf_json_parsed = _parse_page_1(page_json=pdf_json.get("1"))

    # Parse second page onwards
    for num in range(2, total_pg_count+1):
        # Convert num to corresponding json page number key
        page_json = pdf_json.get(str(num), None)
        if not page_json:
            logger.info(
                event_type="warning",
                event=f"Page {num} not found in file. Skipping",
                file_name=pdf_json_file_path)
            continue

        accounts_list = page_json.get("Accounts", [])
        for account in accounts_list:
            charges = account.get("Charges", {})
            for charge_key, charge_value_list in charges.items():
                if charge_key == "Payments & Other Transactions":
                    continue
                for charge_dict in charge_value_list:
                    # Check if charge within an account has non-zero amount
                    amount = charge_dict.get("amount")
                    if not amount:
                        logger.warning("Amount not found in charge_dict")
                    amount_float = _str_to_float(amount)
                    # If amount is non-zero, parse and append to line_items
                    if amount_float != 0:
                        charge_dict_parsed = _parse_line_item_dict(
                            charge_key=charge_key,
                            charge_dict=charge_dict)
                        line_items.append(charge_dict_parsed)

            gst_key = _get_key(key_list=list(account.keys()), s="GST")
            if gst_key and account.get(gst_key):
                gst_items.append(account.get(gst_key))

        gst_key = _get_key(key_list=list(page_json.keys()), s="GST")
        if gst_key:
            gst_items_value = page_json.get(gst_key)
            if isinstance(gst_items_value, list):
                for item in gst_items_value:
                    gst_items.append(item)
            if isinstance(gst_items_value, dict):
                gst_items.append(gst_items_value)

    # Update pdf_json_parsed
    pdf_json_parsed["pages"] = {
        "line_items": line_items,
        "other_notes": "",
        "handwritten_note": "",
        "total_zero_rated_supplies": "",
        "total_standard_rated_supplies": "",
        "tax_rate": "9%",
        "tax_amount": None,
        "gst_items": gst_items
    }

    file_path = pdf_json_file_path.with_stem(
        f"{pdf_json_file_path.stem}_parsed")

    _dict_to_json(
        raw_data=pdf_json_parsed,
        json_file_path=file_path)

    return pdf_json_parsed


# -- Main Code --
if __name__ == "__main__":
    load_dotenv(config.get("GPT_ENV_FILE"))

    # Retrieve config variables
    LOCAL_PDF_SOURCE_DIR = Path(config.get("LOCAL_PDF_SOURCE_DIR"))
    LOCAL_IMG_DEST_DIR = Path(config.get("LOCAL_IMG_DEST_DIR"))
    LOCAL_JSON_DEST_DIR = Path(config.get("LOCAL_JSON_DEST_DIR"))
    purpose = config.get("purpose")
    file = config.get("input_pdf")

    # Initialise directories
    for directory in [
        LOCAL_PDF_SOURCE_DIR,
        LOCAL_IMG_DEST_DIR,
        LOCAL_JSON_DEST_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Parse files
    pdf_file_path = LOCAL_PDF_SOURCE_DIR / file

    pdf_json, pdf_json_file_path = invoice_to_json(
        pdf_file_path=pdf_file_path,
        purpose=purpose,
        local_imag_dest_dir=LOCAL_IMG_DEST_DIR,
        local_json_dest_dir=LOCAL_JSON_DEST_DIR)

    parsed_pdf_json = parse_pdf_json(
        pdf_json_file_path=Path(pdf_json_file_path))
