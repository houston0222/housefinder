from pathlib import Path
import base64
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROMPT_PATH = Path("/app/pipeline/prompts/visual_metadata_prompt.txt")

NORMALIZED_IMAGES_DIR = Path("/app/data/normalized_images")
IMAGE_METADATA_DIR = Path("/app/data/image_metadata")

MODEL = "gpt-4.1-mini"

MAX_IMAGES = 20
SLEEP_SECONDS = 1

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    return PROMPT_PATH.read_text(encoding="utf-8")


def get_property_image_paths(
    normalized_images_dir: Path = NORMALIZED_IMAGES_DIR,
) -> list[Path]:
    if not normalized_images_dir.exists():
        raise FileNotFoundError(
            f"Normalized images directory not found: {normalized_images_dir}"
        )

    image_paths = []

    property_dirs = sorted(
        path
        for path in normalized_images_dir.iterdir()
        if path.is_dir()
    )

    for property_dir in property_dirs:
        property_image_paths = sorted(
            path
            for path in property_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        image_paths.extend(property_image_paths)

    if not image_paths:
        raise FileNotFoundError(
            f"No normalized images found in: {normalized_images_dir}"
        )

    return image_paths


def get_image_metadata_output_path(image_path: Path) -> Path:
    property_id = image_path.parent.name
    image_id = image_path.stem

    return IMAGE_METADATA_DIR / property_id / f"{image_id}.json"


def generate_image_metadata(
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


def save_image_metadata(
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_property_image_metadata(
    max_images: int = MAX_IMAGES,
    sleep_seconds: int = SLEEP_SECONDS,
) -> None:
    load_dotenv("/app/.env")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = load_prompt()
    image_paths = get_property_image_paths()

    generated_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Found normalized images: {len(image_paths)}")
    print(f"Max new metadata files to generate: {max_images}")
    print()

    for image_path in image_paths:
        if generated_count >= max_images:
            print(f"Reached max_images: {max_images}")
            break

        output_path = get_image_metadata_output_path(image_path)

        if output_path.exists():
            skipped_count += 1
            print(f"Skipped existing: {output_path}")
            continue

        print(f"Generating metadata: {image_path}")

        try:
            metadata = generate_image_metadata(
                client=client,
                prompt=prompt,
                image_path=image_path,
            )

            save_image_metadata(
                output_path=output_path,
                metadata=metadata,
            )

            generated_count += 1

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
    print(f"Generated: {generated_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    generate_property_image_metadata(
        max_images=6000,
        sleep_seconds=0,
    )