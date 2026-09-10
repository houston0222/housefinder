from pathlib import Path
import json

import pytest

from scripts.build_embedding_text import (
    build_embedding_text,
    build_image_embedding_text,
    build_property_embedding_text,
    filter_by_confidence,
    format_metadata_items,
    get_confidence_value,
    get_image_metadata_paths,
    load_json,
    normalize_text_value,
)


def test_loads_json_file(
    tmp_path: Path,
) -> None:
    """Load JSON content from a file."""

    json_path = tmp_path / "property.json"

    expected = {
        "property_id": "12345678",
        "title": "Detached house",
    }

    json_path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    result = load_json(json_path)

    assert result == expected


def test_normalizes_scalar_text_values() -> None:
    """Normalize scalar values into stripped text."""

    assert normalize_text_value("  detached house  ") == "detached house"
    assert normalize_text_value(4) == "4"
    assert normalize_text_value(None) == ""


def test_normalizes_list_values() -> None:
    """Normalize list values into semicolon-separated text."""

    value = [
        "garden",
        "garage",
        None,
        "",
    ]

    assert normalize_text_value(value) == "garden; garage"


def test_normalizes_dictionary_values() -> None:
    """Normalize dictionary values while excluding empty values."""

    value = {
        "width": "4.2m",
        "length": "5.1m",
        "notes": None,
    }

    assert normalize_text_value(value) == "width: 4.2m; length: 5.1m"


def test_returns_numeric_confidence_value() -> None:
    """Return numeric confidence values as floats."""

    assert get_confidence_value({"confidence": 0.8}) == 0.8
    assert get_confidence_value({"confidence": 1}) == 1.0


def test_returns_zero_for_invalid_confidence() -> None:
    """Return zero when confidence is missing or non-numeric."""

    assert get_confidence_value({}) == 0.0
    assert get_confidence_value({"confidence": "high"}) == 0.0
    assert get_confidence_value({"confidence": None}) == 0.0


def test_filters_items_by_minimum_confidence() -> None:
    """Keep items whose confidence meets the minimum threshold."""

    high_confidence = {
        "label": "large window",
        "confidence": 0.9,
    }

    threshold_confidence = {
        "label": "island",
        "confidence": 0.75,
    }

    low_confidence = {
        "label": "fireplace",
        "confidence": 0.5,
    }

    items = [
        high_confidence,
        threshold_confidence,
        low_confidence,
    ]

    result = filter_by_confidence(
        items,
        min_confidence=0.75,
    )

    assert result == [
        high_confidence,
        threshold_confidence,
    ]


def test_formats_metadata_items() -> None:
    """Format metadata labels and descriptions into searchable text."""

    items = [
        {
            "label": "windows",
            "description": "large floor-to-ceiling windows",
        },
        {
            "label": "",
            "description": "bright natural light",
        },
        {
            "label": "wooden flooring",
            "description": "",
        },
    ]

    result = format_metadata_items(items)

    assert result == (
        "windows: large floor-to-ceiling windows; "
        "bright natural light; "
        "wooden flooring"
    )


def test_builds_image_embedding_text() -> None:
    """Build searchable image text from sufficiently confident metadata."""

    image_json = {
        "overall_confidence": 0.9,
        "image_type": "interior",
        "room_or_area": "kitchen",
        "caption": "Modern kitchen with a central island",
        "search_phrases": [
            "modern kitchen",
            "kitchen island",
        ],
        "visual_observations": [
            {
                "label": "windows",
                "description": "large windows",
                "confidence": 0.9,
            },
            {
                "label": "possible fireplace",
                "description": "partially visible",
                "confidence": 0.4,
            },
        ],
        "capacity_estimates": [
            {
                "label": "seating",
                "description": "approximately four people",
                "confidence": 0.8,
            },
            {
                "label": "storage",
                "description": "unknown capacity",
                "confidence": 0.5,
            },
        ],
    }

    result = build_image_embedding_text(image_json)

    assert result == (
        "Image type: interior\n"
        "Room or area: kitchen\n"
        "Caption: Modern kitchen with a central island\n"
        "Search phrases: modern kitchen; kitchen island\n"
        "Visual observations: windows: large windows\n"
        "Capacity estimates: seating: approximately four people"
    )


def test_skips_image_with_low_overall_confidence() -> None:
    """Return empty text when overall image confidence is below the threshold."""

    image_json = {
        "overall_confidence": 0.5,
        "image_type": "interior",
        "room_or_area": "kitchen",
        "caption": "Kitchen",
    }

    result = build_image_embedding_text(image_json)

    assert result == ""


def test_builds_property_embedding_text() -> None:
    """Build searchable property text from available parsed fields."""

    parsed_json = {
        "property_id": "12345678",
        "title": "Four bedroom detached house",
        "location": "Newbury",
        "price": "£650,000",
        "bedrooms": 4,
        "property_type": "Detached",
        "listing_type": "For sale",
        "agent": "Example Estate Agents",
        "key_features": [
            "Large garden",
            "Garage",
        ],
        "listing_description": "A spacious family home.",
        "room_measurements": {
            "Kitchen": "5.0m x 4.0m",
            "Living room": "6.0m x 4.5m",
        },
    }

    result = build_property_embedding_text(parsed_json)

    assert result == (
        "Property ID: 12345678\n\n"
        "Title: Four bedroom detached house\n\n"
        "Location: Newbury\n\n"
        "Price: £650,000\n\n"
        "Bedrooms: 4\n\n"
        "Property type: Detached\n\n"
        "Listing type: For sale\n\n"
        "Agent: Example Estate Agents\n\n"
        "Key features: Large garden; Garage\n\n"
        "Description: A spacious family home.\n\n"
        "Room measurements: "
        "Kitchen: 5.0m x 4.0m; Living room: 6.0m x 4.5m"
    )


def test_omits_empty_property_fields() -> None:
    """Exclude empty property fields from embedding text."""

    parsed_json = {
        "property_id": "12345678",
        "title": "Detached house",
        "location": None,
        "agent": "",
        "key_features": [],
    }

    result = build_property_embedding_text(parsed_json)

    assert result == (
        "Property ID: 12345678\n\n"
        "Title: Detached house"
    )


def test_returns_sorted_image_metadata_paths(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return image metadata JSON files in sorted order."""

    image_metadata_dir = temp_property_data / "image_metadata"

    property_dir = image_metadata_dir / "12345678"
    property_dir.mkdir()

    second_path = property_dir / "02.json"
    first_path = property_dir / "01.json"

    second_path.write_text("{}", encoding="utf-8")
    first_path.write_text("{}", encoding="utf-8")
    (property_dir / "notes.txt").write_text(
        "not metadata",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    result = get_image_metadata_paths("12345678")

    assert result == [
        first_path,
        second_path,
    ]


def test_returns_no_image_metadata_paths_when_property_directory_is_missing(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an empty list when a property has no image metadata directory."""

    image_metadata_dir = temp_property_data / "image_metadata"

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    result = get_image_metadata_paths("12345678")

    assert result == []


def test_builds_property_and_image_embedding_text_files(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create property and image embedding text files from pipeline data."""

    parsed_dir = temp_property_data / "parsed"
    image_metadata_dir = temp_property_data / "image_metadata"

    property_text_dir = (
        temp_property_data
        / "embedding_text"
        / "properties"
    )

    image_text_dir = (
        temp_property_data
        / "embedding_text"
        / "images"
    )

    property_id = "12345678"

    parsed_path = parsed_dir / f"{property_id}.json"
    parsed_path.write_text(
        json.dumps(
            {
                "property_id": property_id,
                "title": "Detached house",
                "bedrooms": 4,
            }
        ),
        encoding="utf-8",
    )

    property_image_metadata_dir = image_metadata_dir / property_id
    property_image_metadata_dir.mkdir()

    image_metadata_path = property_image_metadata_dir / "01.json"
    image_metadata_path.write_text(
        json.dumps(
            {
                "overall_confidence": 0.9,
                "image_type": "interior",
                "room_or_area": "kitchen",
                "caption": "Modern kitchen",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PARSED_DIR",
        parsed_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PROPERTY_TEXT_DIR",
        property_text_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_TEXT_DIR",
        image_text_dir,
    )

    build_embedding_text()

    property_output_path = property_text_dir / f"{property_id}.txt"
    image_output_path = image_text_dir / property_id / "01.txt"

    assert property_output_path.exists()
    assert image_output_path.exists()

    assert property_output_path.read_text(
        encoding="utf-8"
    ) == (
        "Property ID: 12345678\n\n"
        "Title: Detached house\n\n"
        "Bedrooms: 4"
    )

    assert image_output_path.read_text(
        encoding="utf-8"
    ) == (
        "Image type: interior\n"
        "Room or area: kitchen\n"
        "Caption: Modern kitchen"
    )


def test_skips_low_confidence_image_when_building_embedding_text(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not create image text for low-confidence image metadata."""

    parsed_dir = temp_property_data / "parsed"
    image_metadata_dir = temp_property_data / "image_metadata"

    property_text_dir = (
        temp_property_data
        / "embedding_text"
        / "properties"
    )

    image_text_dir = (
        temp_property_data
        / "embedding_text"
        / "images"
    )

    property_id = "12345678"

    parsed_path = parsed_dir / f"{property_id}.json"
    parsed_path.write_text(
        json.dumps(
            {
                "property_id": property_id,
                "title": "Detached house",
            }
        ),
        encoding="utf-8",
    )

    property_image_metadata_dir = image_metadata_dir / property_id
    property_image_metadata_dir.mkdir()

    image_metadata_path = property_image_metadata_dir / "01.json"
    image_metadata_path.write_text(
        json.dumps(
            {
                "overall_confidence": 0.5,
                "image_type": "interior",
                "caption": "Uncertain room",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PARSED_DIR",
        parsed_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PROPERTY_TEXT_DIR",
        property_text_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_TEXT_DIR",
        image_text_dir,
    )

    build_embedding_text()

    property_output_path = property_text_dir / f"{property_id}.txt"
    image_output_path = image_text_dir / property_id / "01.txt"

    assert property_output_path.exists()
    assert not image_output_path.exists()


def test_skips_property_without_property_id(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not create embedding text for parsed data without a property ID."""

    parsed_dir = temp_property_data / "parsed"

    property_text_dir = (
        temp_property_data
        / "embedding_text"
        / "properties"
    )

    image_text_dir = (
        temp_property_data
        / "embedding_text"
        / "images"
    )

    parsed_path = parsed_dir / "missing-id.json"
    parsed_path.write_text(
        json.dumps(
            {
                "title": "Detached house",
                "bedrooms": 4,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PARSED_DIR",
        parsed_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PROPERTY_TEXT_DIR",
        property_text_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_TEXT_DIR",
        image_text_dir,
    )

    build_embedding_text()

    assert list(property_text_dir.glob("*.txt")) == []
    assert list(image_text_dir.rglob("*.txt")) == []


def test_limits_number_of_properties(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process only the requested maximum number of properties."""

    parsed_dir = temp_property_data / "parsed"

    property_text_dir = (
        temp_property_data
        / "embedding_text"
        / "properties"
    )

    image_text_dir = (
        temp_property_data
        / "embedding_text"
        / "images"
    )

    first_property = {
        "property_id": "11111111",
        "title": "First property",
    }

    second_property = {
        "property_id": "22222222",
        "title": "Second property",
    }

    (parsed_dir / "01.json").write_text(
        json.dumps(first_property),
        encoding="utf-8",
    )

    (parsed_dir / "02.json").write_text(
        json.dumps(second_property),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PARSED_DIR",
        parsed_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.PROPERTY_TEXT_DIR",
        property_text_dir,
    )

    monkeypatch.setattr(
        "scripts.build_embedding_text.IMAGE_TEXT_DIR",
        image_text_dir,
    )

    build_embedding_text(
        max_properties=1,
    )

    assert (
        property_text_dir / "11111111.txt"
    ).exists()

    assert not (
        property_text_dir / "22222222.txt"
    ).exists()