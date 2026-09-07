from pathlib import Path
import json
from typing import Any


PARSED_DIR = Path("/property_data/parsed")
IMAGE_METADATA_DIR = Path("/property_data/image_metadata")

PROPERTY_TEXT_DIR = Path("/property_data/embedding_text/properties")
IMAGE_TEXT_DIR = Path("/property_data/embedding_text/images")

MIN_OVERALL_CONFIDENCE = 0.6
MIN_VISUAL_OBSERVATION_CONFIDENCE = 0.75
MIN_CAPACITY_ESTIMATE_CONFIDENCE = 0.65


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "; ".join(normalize_text_value(item) for item in value if item)

    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {normalize_text_value(val)}"
            for key, val in value.items()
            if val not in [None, "", []]
        )

    return str(value).strip()


def get_confidence_value(item: dict[str, Any]) -> float:
    value = item.get("confidence")

    if isinstance(value, int | float):
        return float(value)

    return 0.0


def filter_by_confidence(
    items: list[dict[str, Any]],
    min_confidence: float,
) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if get_confidence_value(item) >= min_confidence
    ]


def format_metadata_items(items: list[dict[str, Any]]) -> str:
    parts = []

    for item in items:
        label = normalize_text_value(item.get("label"))
        description = normalize_text_value(item.get("description"))

        if label and description:
            parts.append(f"{label}: {description}")
        elif description:
            parts.append(description)
        elif label:
            parts.append(label)

    return "; ".join(parts)


def build_image_embedding_text(image_json: dict[str, Any]) -> str:
    overall_confidence = image_json.get("overall_confidence")

    if isinstance(overall_confidence, int | float):
        if overall_confidence < MIN_OVERALL_CONFIDENCE:
            return ""

    visual_observations = filter_by_confidence(
        image_json.get("visual_observations", []),
        MIN_VISUAL_OBSERVATION_CONFIDENCE,
    )

    capacity_estimates = filter_by_confidence(
        image_json.get("capacity_estimates", []),
        MIN_CAPACITY_ESTIMATE_CONFIDENCE,
    )

    image_type = normalize_text_value(image_json.get("image_type"))
    room_or_area = normalize_text_value(image_json.get("room_or_area"))
    caption = normalize_text_value(image_json.get("caption"))
    search_phrases = normalize_text_value(image_json.get("search_phrases"))

    visual_observation_text = format_metadata_items(visual_observations)
    capacity_estimate_text = format_metadata_items(capacity_estimates)

    lines = []

    if image_type:
        lines.append(f"Image type: {image_type}")

    if room_or_area:
        lines.append(f"Room or area: {room_or_area}")

    if caption:
        lines.append(f"Caption: {caption}")

    if search_phrases:
        lines.append(f"Search phrases: {search_phrases}")

    if visual_observation_text:
        lines.append(f"Visual observations: {visual_observation_text}")

    if capacity_estimate_text:
        lines.append(f"Capacity estimates: {capacity_estimate_text}")

    return "\n".join(lines)


def build_property_embedding_text(parsed_json: dict[str, Any]) -> str:
    lines = []

    fields = [
        ("Property ID", parsed_json.get("property_id")),
        ("Title", parsed_json.get("title")),
        ("Location", parsed_json.get("location")),
        ("Price", parsed_json.get("price")),
        ("Bedrooms", parsed_json.get("bedrooms")),
        ("Property type", parsed_json.get("property_type")),
        ("Listing type", parsed_json.get("listing_type")),
        ("Agent", parsed_json.get("agent")),
        ("Key features", parsed_json.get("key_features")),
        ("Description", parsed_json.get("listing_description")),
        ("Room measurements", parsed_json.get("room_measurements")),
    ]

    for label, value in fields:
        text = normalize_text_value(value)

        if text:
            lines.append(f"{label}: {text}")

    return "\n\n".join(lines)


def get_image_metadata_paths(property_id: str) -> list[Path]:
    property_dir = IMAGE_METADATA_DIR / property_id

    if not property_dir.exists():
        return []

    return sorted(property_dir.glob("*.json"))


def build_embedding_text(max_properties: int | None = None) -> None:
    PROPERTY_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    parsed_paths = sorted(PARSED_DIR.glob("*.json"))

    if max_properties is not None:
        parsed_paths = parsed_paths[:max_properties]

    property_count = 0
    image_count = 0
    skipped_image_count = 0

    for parsed_path in parsed_paths:
        parsed_json = load_json(parsed_path)

        property_id = normalize_text_value(parsed_json.get("property_id"))

        if not property_id:
            print(f"Skipped missing property_id: {parsed_path}")
            continue

        property_text = build_property_embedding_text(parsed_json)

        property_output_path = PROPERTY_TEXT_DIR / f"{property_id}.txt"
        property_output_path.write_text(property_text, encoding="utf-8")

        property_count += 1

        image_text_count = 0

        for image_metadata_path in get_image_metadata_paths(property_id):
            image_json = load_json(image_metadata_path)
            image_text = build_image_embedding_text(image_json)

            if not image_text:
                skipped_image_count += 1
                print(f"Skipped low-confidence image metadata: {image_metadata_path}")
                continue

            image_output_dir = IMAGE_TEXT_DIR / property_id
            image_output_dir.mkdir(parents=True, exist_ok=True)

            image_output_path = image_output_dir / f"{image_metadata_path.stem}.txt"
            image_output_path.write_text(image_text, encoding="utf-8")

            image_count += 1
            image_text_count += 1

        print(
            f"Created embedding text for property {property_id} "
            f"and {image_text_count} image texts"
        )

    print()
    print("Done.")
    print(f"Properties: {property_count}")
    print(f"Images included: {image_count}")
    print(f"Images skipped: {skipped_image_count}")


if __name__ == "__main__":
    build_embedding_text()