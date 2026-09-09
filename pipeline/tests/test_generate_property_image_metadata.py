from pathlib import Path
import base64
import json
from typing import Any

import pytest

from scripts.generate_property_image_metadata import (
    encode_image_base64,
    generate_image_metadata,
    generate_property_image_metadata,
    get_image_metadata_output_path,
    get_property_image_paths,
    save_image_metadata,
)


def test_encodes_image_as_base64(
    tmp_path: Path,
) -> None:
    """Encode image file bytes as a UTF-8 base64 string."""

    image_path = tmp_path / "01.jpg"
    image_bytes = b"test-image-content"

    image_path.write_bytes(image_bytes)

    encoded_image = encode_image_base64(image_path)

    expected = base64.b64encode(image_bytes).decode("utf-8")

    assert encoded_image == expected


def test_returns_supported_property_image_paths(
    temp_property_data: Path,
) -> None:
    """Return supported images from property directories in sorted order."""

    normalized_images_dir = temp_property_data / "normalized_images"

    first_property_dir = normalized_images_dir / "12345678"
    second_property_dir = normalized_images_dir / "87654321"

    first_property_dir.mkdir()
    second_property_dir.mkdir()

    (first_property_dir / "02.png").write_bytes(b"image")
    (first_property_dir / "01.jpg").write_bytes(b"image")
    (first_property_dir / "notes.txt").write_text("not an image")

    (second_property_dir / "01.webp").write_bytes(b"image")

    image_paths = get_property_image_paths(
        normalized_images_dir=normalized_images_dir,
    )

    assert image_paths == [
        first_property_dir / "01.jpg",
        first_property_dir / "02.png",
        second_property_dir / "01.webp",
    ]


def test_raises_when_normalized_images_directory_is_missing(
    tmp_path: Path,
) -> None:
    """Raise FileNotFoundError when the normalized images directory is missing."""

    missing_dir = tmp_path / "normalized_images"

    with pytest.raises(
        FileNotFoundError,
        match="Normalized images directory not found",
    ):
        get_property_image_paths(
            normalized_images_dir=missing_dir,
        )


def test_raises_when_no_normalized_images_exist(
    temp_property_data: Path,
) -> None:
    """Raise FileNotFoundError when no supported normalized images exist."""

    normalized_images_dir = temp_property_data / "normalized_images"

    property_dir = normalized_images_dir / "12345678"
    property_dir.mkdir()

    (property_dir / "notes.txt").write_text("not an image")

    with pytest.raises(
        FileNotFoundError,
        match="No normalized images found",
    ):
        get_property_image_paths(
            normalized_images_dir=normalized_images_dir,
        )


def test_builds_metadata_output_path(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the metadata path from the property ID and image ID."""

    image_metadata_dir = temp_property_data / "image_metadata"

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    image_path = (
        temp_property_data
        / "normalized_images"
        / "12345678"
        / "01.jpg"
    )

    output_path = get_image_metadata_output_path(image_path)

    assert output_path == image_metadata_dir / "12345678" / "01.json"


def test_saves_image_metadata_as_json(
    tmp_path: Path,
) -> None:
    """Save metadata as formatted JSON and create missing parent directories."""

    output_path = (
        tmp_path
        / "image_metadata"
        / "12345678"
        / "01.json"
    )

    metadata = {
        "room_type": "kitchen",
        "features": [
            "island",
            "wooden flooring",
        ],
    }

    save_image_metadata(
        output_path=output_path,
        metadata=metadata,
    )

    assert output_path.exists()

    saved_metadata = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_metadata == metadata


def test_generates_image_metadata_from_openai_response(
    tmp_path: Path,
) -> None:
    """Send the expected request and parse metadata from the OpenAI response."""

    image_path = tmp_path / "01.jpg"
    image_bytes = b"test-image-content"
    image_path.write_bytes(image_bytes)

    prompt = "Describe this property image."

    expected_metadata = {
        "room_type": "kitchen",
        "features": [
            "island",
            "wooden flooring",
        ],
    }

    captured_request: dict[str, Any] = {}

    class FakeResponse:
        output_text = json.dumps(expected_metadata)

    class FakeResponses:
        def create(self, **kwargs: Any) -> FakeResponse:
            captured_request.update(kwargs)
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    metadata = generate_image_metadata(
        client=FakeClient(),
        prompt=prompt,
        image_path=image_path,
    )

    expected_base64 = base64.b64encode(image_bytes).decode("utf-8")

    assert metadata == expected_metadata
    assert captured_request["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{expected_base64}",
                },
            ],
        }
    ]


def test_skips_existing_metadata(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip images that already have metadata without calling OpenAI."""

    normalized_images_dir = temp_property_data / "normalized_images"
    image_metadata_dir = temp_property_data / "image_metadata"

    property_dir = normalized_images_dir / "12345678"
    property_dir.mkdir()

    image_path = property_dir / "01.jpg"
    image_path.write_bytes(b"test-image-content")

    output_dir = image_metadata_dir / "12345678"
    output_dir.mkdir()

    output_path = output_dir / "01.json"
    output_path.write_text(
        json.dumps({"existing": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.get_property_image_paths",
        lambda: [image_path],
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.load_prompt",
        lambda: "test prompt",
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("generate_image_metadata should not be called")

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.generate_image_metadata",
        fail_if_called,
    )

    generate_property_image_metadata(
        sleep_seconds=0,
    )

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == {
        "existing": True,
    }


def test_generates_and_saves_metadata(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate metadata for an image and save it to the expected output path."""

    normalized_images_dir = temp_property_data / "normalized_images"
    image_metadata_dir = temp_property_data / "image_metadata"

    property_dir = normalized_images_dir / "12345678"
    property_dir.mkdir()

    image_path = property_dir / "01.jpg"
    image_path.write_bytes(b"test-image-content")

    expected_metadata = {
        "room_type": "kitchen",
        "features": [
            "island",
            "wooden flooring",
        ],
    }

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.get_property_image_paths",
        lambda: [image_path],
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.load_prompt",
        lambda: "test prompt",
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.generate_image_metadata",
        lambda **kwargs: expected_metadata,
    )

    generate_property_image_metadata(
        sleep_seconds=0,
    )

    output_path = image_metadata_dir / "12345678" / "01.json"

    assert output_path.exists()

    saved_metadata = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_metadata == expected_metadata


def test_continues_after_metadata_generation_failure(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue processing remaining images when one metadata generation fails."""

    normalized_images_dir = temp_property_data / "normalized_images"
    image_metadata_dir = temp_property_data / "image_metadata"

    property_dir = normalized_images_dir / "12345678"
    property_dir.mkdir()

    first_image_path = property_dir / "01.jpg"
    second_image_path = property_dir / "02.jpg"

    first_image_path.write_bytes(b"first-image")
    second_image_path.write_bytes(b"second-image")

    expected_metadata = {
        "room_type": "bedroom",
        "features": [
            "large window",
        ],
    }

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.IMAGE_METADATA_DIR",
        image_metadata_dir,
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.get_property_image_paths",
        lambda: [
            first_image_path,
            second_image_path,
        ],
    )

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.load_prompt",
        lambda: "test prompt",
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_generate_image_metadata(
        client: Any,
        prompt: str,
        image_path: Path,
    ) -> dict[str, Any]:
        if image_path == first_image_path:
            raise RuntimeError("metadata generation failed")

        return expected_metadata

    monkeypatch.setattr(
        "scripts.generate_property_image_metadata.generate_image_metadata",
        fake_generate_image_metadata,
    )

    generate_property_image_metadata(
        sleep_seconds=0,
    )

    first_output_path = (
        image_metadata_dir
        / "12345678"
        / "01.json"
    )

    second_output_path = (
        image_metadata_dir
        / "12345678"
        / "02.json"
    )

    assert not first_output_path.exists()
    assert second_output_path.exists()

    saved_metadata = json.loads(
        second_output_path.read_text(encoding="utf-8")
    )

    assert saved_metadata == expected_metadata