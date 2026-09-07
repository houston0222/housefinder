from pathlib import Path
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.app.models.property import Base, Property
from backend.app.models.property_image import PropertyImage

DATA_DIR = Path("/property_data")

PARSED_DIR = DATA_DIR / "parsed"

PROPERTY_TEXT_DIR = DATA_DIR / "embedding_text/properties"
PROPERTY_EMBEDDINGS_DIR = DATA_DIR / "embeddings/properties"

IMAGE_METADATA_DIR = DATA_DIR / "image_metadata"
IMAGE_TEXT_DIR = DATA_DIR / "embedding_text/images"
IMAGE_EMBEDDINGS_DIR = DATA_DIR / "embeddings/images"
NORMALIZED_IMAGES_DIR = DATA_DIR / "normalized_images"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8").strip()


def parse_price_amount(price_text: str | None) -> int | None:
    if not price_text:
        return None

    digits = re.sub(r"[^0-9]", "", price_text)

    if not digits:
        return None

    return int(digits)


def load_embedding_if_exists(
    path: Path,
) -> tuple[str | None, list[float] | None]:
    if not path.exists():
        return None, None

    data = load_json(path)

    return data.get("embedding_model"), data.get("embedding")


def get_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is missing")

    return database_url


def get_or_create_property(
    session: Session,
    property_id: str,
) -> tuple[Property, bool]:
    property_row = session.scalar(
        select(Property).where(Property.property_id == property_id)
    )

    if property_row:
        return property_row, False

    property_row = Property(property_id=property_id)
    session.add(property_row)

    return property_row, True


def get_or_create_property_image(
    session: Session,
    property_id: str,
    image_id: str,
) -> tuple[PropertyImage, bool]:
    image_row = session.scalar(
        select(PropertyImage).where(
            PropertyImage.property_id == property_id,
            PropertyImage.image_id == image_id,
        )
    )

    if image_row:
        return image_row, False

    image_row = PropertyImage(
        property_id=property_id,
        image_id=image_id,
    )
    session.add(image_row)

    return image_row, True


def import_property_images(
    session: Session,
    property_id: str,
    image_urls: list[str],
) -> tuple[int, int]:
    property_image_metadata_dir = IMAGE_METADATA_DIR / property_id

    if not property_image_metadata_dir.exists():
        return 0, 0

    image_metadata_paths = sorted(property_image_metadata_dir.glob("*.json"))

    created_count = 0
    updated_count = 0

    for index, image_metadata_path in enumerate(image_metadata_paths):
        image_id = image_metadata_path.stem
        image_metadata = load_json(image_metadata_path)

        image_row, was_created = get_or_create_property_image(
            session=session,
            property_id=property_id,
            image_id=image_id,
        )

        embedding_text = load_text_if_exists(
            IMAGE_TEXT_DIR / property_id / f"{image_id}.txt"
        )

        embedding_model, embedding = load_embedding_if_exists(
            IMAGE_EMBEDDINGS_DIR / property_id / f"{image_id}.json"
        )

        normalized_image_path = (
            NORMALIZED_IMAGES_DIR / property_id / f"{image_id}.jpg"
        )

        image_row.source_image_url = (
            image_urls[index] if index < len(image_urls) else None
        )
        image_row.raw_image_path = None
        image_row.processed_image_path = (
            str(normalized_image_path)
            if normalized_image_path.exists()
            else None
        )

        image_row.image_type = image_metadata.get("image_type")
        image_row.room_or_area = image_metadata.get("room_or_area")
        image_row.caption = image_metadata.get("caption")

        image_row.search_phrases = image_metadata.get("search_phrases")
        image_row.visual_observations = image_metadata.get("visual_observations")
        image_row.capacity_estimates = image_metadata.get("capacity_estimates")
        image_row.overall_confidence = image_metadata.get("overall_confidence")

        image_row.embedding_text = embedding_text
        image_row.embedding_model = embedding_model
        image_row.embedding = embedding

        if was_created:
            created_count += 1
        else:
            updated_count += 1

    return created_count, updated_count


def import_property(
    session: Session,
    parsed_path: Path,
) -> tuple[bool, bool, int, int]:
    parsed_json = load_json(parsed_path)

    property_id = str(parsed_json.get("property_id", "")).strip()

    if not property_id:
        print(f"Skipped missing property_id: {parsed_path}")
        return False, False, 0, 0

    property_row, property_was_created = get_or_create_property(
        session,
        property_id,
    )

    embedding_text = load_text_if_exists(
        PROPERTY_TEXT_DIR / f"{property_id}.txt"
    )

    embedding_model, embedding = load_embedding_if_exists(
        PROPERTY_EMBEDDINGS_DIR / f"{property_id}.json"
    )

    price_text = parsed_json.get("price")

    property_row.title = parsed_json.get("title")
    property_row.canonical_url = parsed_json.get("canonical_url")

    property_row.description = parsed_json.get("description")
    property_row.og_title = parsed_json.get("og_title")
    property_row.og_description = parsed_json.get("og_description")

    property_row.listing_type = parsed_json.get("listing_type")

    property_row.price_text = price_text
    property_row.price_amount = parse_price_amount(price_text)

    property_row.location = parsed_json.get("location")
    property_row.bedrooms = parsed_json.get("bedrooms")
    property_row.property_type = parsed_json.get("property_type")
    property_row.agent = parsed_json.get("agent")

    property_row.main_image_url = parsed_json.get("main_image_url")
    property_row.image_count = parsed_json.get("image_count")

    property_row.key_features = parsed_json.get("key_features")
    property_row.listing_description = parsed_json.get("listing_description")
    property_row.room_measurements = parsed_json.get("room_measurements")

    property_row.embedding_text = embedding_text
    property_row.embedding_model = embedding_model
    property_row.embedding = embedding

    images_created, images_updated = import_property_images(
        session=session,
        property_id=property_id,
        image_urls=parsed_json.get("image_urls") or [],
    )

    action = "Created" if property_was_created else "Updated"

    print(
        f"{action} property: {property_id} "
        f"| images created: {images_created} "
        f"| images updated: {images_updated}"
    )

    return True, property_was_created, images_created, images_updated


def import_pipeline_data_to_db() -> None:
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)

    parsed_paths = sorted(PARSED_DIR.glob("*.json"))

    properties_created = 0
    properties_updated = 0
    images_created = 0
    images_updated = 0

    with Session(engine) as session:
        for parsed_path in parsed_paths:
            (
                imported_property,
                property_was_created,
                created_images,
                updated_images,
            ) = import_property(session, parsed_path)

            if not imported_property:
                continue

            if property_was_created:
                properties_created += 1
            else:
                properties_updated += 1

            images_created += created_images
            images_updated += updated_images

        session.commit()

    print()
    print("Done.")
    print(f"Properties created: {properties_created}")
    print(f"Properties updated: {properties_updated}")
    print(f"Images created: {images_created}")
    print(f"Images updated: {images_updated}")


if __name__ == "__main__":
    import_pipeline_data_to_db()