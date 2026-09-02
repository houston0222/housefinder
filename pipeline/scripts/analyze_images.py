from pathlib import Path
import base64
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROMPT_PATH = Path("/app/pipeline/prompts/visual_metadata_prompt.txt")

PROCESSED_IMAGES_DIR = Path("/app/data/processed_images")
IMAGE_ANALYSIS_DIR = Path("/app/data/image_analysis")

MODEL = "gpt-4.1-mini"

MAX_IMAGES = 20
SLEEP_SECONDS = 1


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    return PROMPT_PATH.read_text(encoding="utf-8")


def get_image_paths(
    processed_images_dir: Path = PROCESSED_IMAGES_DIR,
) -> list[Path]:
    if not processed_images_dir.exists():
        raise FileNotFoundError(f"Processed images directory not found: {processed_images_dir}")

    image_paths = []

    property_dirs = sorted(
        path for path in processed_images_dir.iterdir()
        if path.is_dir()
    )

    for property_dir in property_dirs:
        property_image_paths = sorted(
            path for path in property_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

        image_paths.extend(property_image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No processed images found in: {processed_images_dir}")

    return image_paths


def get_output_path(image_path: Path) -> Path:
    property_id = image_path.parent.name
    image_id = image_path.stem

    return IMAGE_ANALYSIS_DIR / property_id / f"{image_id}.json"


def analyze_one_image(
    client: OpenAI,
    prompt: str,
    image_path: Path,
) -> dict[str, Any]:
    image_base64 = encode_image_base64(image_path)

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


def save_analysis(output_path: Path, analysis: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def analyze_images(
    max_images: int = MAX_IMAGES,
    sleep_seconds: int = SLEEP_SECONDS,
) -> None:
    load_dotenv("/app/.env")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = load_prompt()
    image_paths = get_image_paths()

    analyzed_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Found processed images: {len(image_paths)}")
    print(f"Max new images to analyze: {max_images}")
    print()

    for image_path in image_paths:
        if analyzed_count >= max_images:
            print(f"Reached max_images: {max_images}")
            break

        output_path = get_output_path(image_path)

        if output_path.exists():
            skipped_count += 1
            print(f"Skipped existing: {output_path}")
            continue

        print(f"Analyzing: {image_path}")

        try:
            analysis = analyze_one_image(
                client=client,
                prompt=prompt,
                image_path=image_path,
            )

            save_analysis(output_path, analysis)

            analyzed_count += 1

            print(f"Saved: {output_path}")
            print()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        except Exception as error:
            failed_count += 1
            print(f"Failed: {image_path}")
            print(error)
            print()

    print("Done.")
    print(f"Analyzed: {analyzed_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    analyze_images(
        max_images=6000,
        sleep_seconds=0,
    )