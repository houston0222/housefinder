from pathlib import Path
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROPERTY_TEXT_DIR = Path("/app/data/embedding_text/properties")
IMAGE_TEXT_DIR = Path("/app/data/embedding_text/images")

PROPERTY_EMBEDDINGS_DIR = Path("/app/data/embeddings/properties")
IMAGE_EMBEDDINGS_DIR = Path("/app/data/embeddings/images")

MODEL = "text-embedding-3-small"

MAX_ITEMS = 20
SLEEP_SECONDS = 0


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def generate_text_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=MODEL,
        input=text,
    )

    return response.data[0].embedding


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_property_embedding(
    client: OpenAI,
    text_path: Path,
) -> bool:
    property_id = text_path.stem
    output_path = PROPERTY_EMBEDDINGS_DIR / f"{property_id}.json"

    if output_path.exists():
        print(f"Skipped existing property embedding: {property_id}")
        return False

    text = load_text(text_path)

    if not text:
        print(f"Skipped empty property text: {text_path}")
        return False

    embedding = generate_text_embedding(client, text)

    save_json(
        output_path,
        {
            "level": "property",
            "property_id": property_id,
            "source_text_path": str(text_path),
            "embedding_model": MODEL,
            "embedding": embedding,
        },
    )

    print(f"Saved property embedding: {property_id}")
    return True


def generate_image_embedding(
    client: OpenAI,
    text_path: Path,
) -> bool:
    property_id = text_path.parent.name
    image_id = text_path.stem

    output_path = IMAGE_EMBEDDINGS_DIR / property_id / f"{image_id}.json"

    if output_path.exists():
        print(f"Skipped existing image embedding: {property_id}/{image_id}")
        return False

    text = load_text(text_path)

    if not text:
        print(f"Skipped empty image text: {text_path}")
        return False

    embedding = generate_text_embedding(client, text)

    save_json(
        output_path,
        {
            "level": "image",
            "property_id": property_id,
            "image_id": image_id,
            "source_text_path": str(text_path),
            "embedding_model": MODEL,
            "embedding": embedding,
        },
    )

    print(f"Saved image embedding: {property_id}/{image_id}")
    return True


def get_text_paths() -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []

    property_text_paths = sorted(PROPERTY_TEXT_DIR.glob("*.txt"))

    for path in property_text_paths:
        paths.append(("property", path))

    image_text_paths = sorted(
        path
        for path in IMAGE_TEXT_DIR.rglob("*.txt")
        if path.is_file()
    )

    for path in image_text_paths:
        paths.append(("image", path))

    return paths


def generate_text_embeddings(
    max_items: int | None = MAX_ITEMS,
    sleep_seconds: int = SLEEP_SECONDS,
) -> None:
    load_dotenv("/app/.env")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    text_paths = get_text_paths()

    created_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Found text files: {len(text_paths)}")
    print(f"Max new embeddings to create: {max_items}")
    print()

    for level, text_path in text_paths:
        if max_items is not None and created_count >= max_items:
            print(f"Reached max_items: {max_items}")
            break

        try:
            if level == "property":
                was_created = generate_property_embedding(client, text_path)
            else:
                was_created = generate_image_embedding(client, text_path)

            if was_created:
                created_count += 1

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            else:
                skipped_count += 1

        except Exception as error:
            failed_count += 1
            print(f"Failed: {text_path}")
            print(error)
            print()

    print()
    print("Done.")
    print(f"Created: {created_count}")
    print(f"Skipped existing/empty: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    generate_text_embeddings(
        max_items=None,
        sleep_seconds=0,
    )