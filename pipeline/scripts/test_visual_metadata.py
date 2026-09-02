from pathlib import Path
import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


PROMPT_PATH = Path("/app/pipeline/prompts/visual_metadata_prompt.txt")
IMAGE_PATH = Path("/app/data/processed_images/89119263/08.jpg")

MODEL = "gpt-4.1-mini"


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def analyze_one_image(image_path: Path) -> dict:
    load_dotenv("/app/.env")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    image_base64 = encode_image_base64(image_path)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }
        ],
    )

    return json.loads(response.output_text)


if __name__ == "__main__":
    result = analyze_one_image(IMAGE_PATH)
    print(json.dumps(result, indent=2, ensure_ascii=False))