from pathlib import Path
import json
from typing import Any

import pytest

from scripts.generate_text_embeddings import (
    generate_image_embedding,
    generate_property_embedding,
    generate_text_embedding,
    generate_text_embeddings,
    get_text_paths,
    load_text,
    save_json,
)


def test_loads_and_strips_text(
    tmp_path: Path,
) -> None:
    """Load UTF-8 text and remove surrounding whitespace."""

    text_path = tmp_path / "property.txt"
    text_path.write_text(
        "\n  Detached house with large garden  \n",
        encoding="utf-8",
    )

    result = load_text(text_path)

    assert result == "Detached house with large garden"


def test_saves_json_and_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """Save formatted JSON while creating missing parent directories."""

    output_path = (
        tmp_path
        / "embeddings"
        / "properties"
        / "12345678.json"
    )

    data = {
        "property_id": "12345678",
        "embedding": [0.1, 0.2, 0.3],
    }

    save_json(
        output_path,
        data,
    )

    assert output_path.exists()

    saved_data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_data == data


def test_generates_text_embedding_from_openai_response() -> None:
    """Return the embedding vector from the OpenAI response."""

    expected_embedding = [
        0.1,
        0.2,
        0.3,
    ]

    captured_request: dict[str, Any] = {}

    class FakeEmbeddingData:
        embedding = expected_embedding

    class FakeResponse:
        data = [FakeEmbeddingData()]

    class FakeEmbeddings:
        def create(self, **kwargs: Any) -> FakeResponse:
            captured_request.update(kwargs)
            return FakeResponse()

    class FakeClient:
        embeddings = FakeEmbeddings()

    result = generate_text_embedding(
        client=FakeClient(),
        text="Detached house with large garden",
    )

    assert result == expected_embedding
    assert captured_request["input"] == (
        "Detached house with large garden"
    )


def test_generates_property_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate and save an embedding for property text."""

    property_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "properties"
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.PROPERTY_EMBEDDINGS_DIR",
        property_embeddings_dir,
    )

    text_path = tmp_path / "12345678.txt"
    text_path.write_text(
        "Detached house with large garden",
        encoding="utf-8",
    )

    expected_embedding = [
        0.1,
        0.2,
        0.3,
    ]

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        lambda client, text: expected_embedding,
    )

    was_created = generate_property_embedding(
        client=object(),
        text_path=text_path,
    )

    output_path = property_embeddings_dir / "12345678.json"

    assert was_created is True
    assert output_path.exists()

    saved_data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_data["level"] == "property"
    assert saved_data["property_id"] == "12345678"
    assert saved_data["source_text_path"] == str(text_path)
    assert saved_data["embedding"] == expected_embedding


def test_skips_existing_property_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip property text when its embedding already exists."""

    property_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "properties"
    )

    property_embeddings_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.PROPERTY_EMBEDDINGS_DIR",
        property_embeddings_dir,
    )

    text_path = tmp_path / "12345678.txt"
    text_path.write_text(
        "Detached house",
        encoding="utf-8",
    )

    output_path = property_embeddings_dir / "12345678.json"
    output_path.write_text(
        json.dumps({"existing": True}),
        encoding="utf-8",
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("generate_text_embedding should not be called")

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        fail_if_called,
    )

    was_created = generate_property_embedding(
        client=object(),
        text_path=text_path,
    )

    assert was_created is False

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == {
        "existing": True,
    }


def test_skips_empty_property_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip property text containing only whitespace."""

    property_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "properties"
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.PROPERTY_EMBEDDINGS_DIR",
        property_embeddings_dir,
    )

    text_path = tmp_path / "12345678.txt"
    text_path.write_text(
        "   \n",
        encoding="utf-8",
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("generate_text_embedding should not be called")

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        fail_if_called,
    )

    was_created = generate_property_embedding(
        client=object(),
        text_path=text_path,
    )

    assert was_created is False

    assert not (
        property_embeddings_dir / "12345678.json"
    ).exists()


def test_generates_image_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate and save an embedding for image text."""

    image_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "images"
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.IMAGE_EMBEDDINGS_DIR",
        image_embeddings_dir,
    )

    property_dir = tmp_path / "12345678"
    property_dir.mkdir()

    text_path = property_dir / "01.txt"
    text_path.write_text(
        "Modern kitchen with large island",
        encoding="utf-8",
    )

    expected_embedding = [
        0.4,
        0.5,
        0.6,
    ]

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        lambda client, text: expected_embedding,
    )

    was_created = generate_image_embedding(
        client=object(),
        text_path=text_path,
    )

    output_path = (
        image_embeddings_dir
        / "12345678"
        / "01.json"
    )

    assert was_created is True
    assert output_path.exists()

    saved_data = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_data["level"] == "image"
    assert saved_data["property_id"] == "12345678"
    assert saved_data["image_id"] == "01"
    assert saved_data["source_text_path"] == str(text_path)
    assert saved_data["embedding"] == expected_embedding


def test_skips_existing_image_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip image text when its embedding already exists."""

    image_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "images"
    )

    output_dir = image_embeddings_dir / "12345678"
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.IMAGE_EMBEDDINGS_DIR",
        image_embeddings_dir,
    )

    property_dir = tmp_path / "12345678"
    property_dir.mkdir()

    text_path = property_dir / "01.txt"
    text_path.write_text(
        "Modern kitchen",
        encoding="utf-8",
    )

    output_path = output_dir / "01.json"
    output_path.write_text(
        json.dumps({"existing": True}),
        encoding="utf-8",
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("generate_text_embedding should not be called")

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        fail_if_called,
    )

    was_created = generate_image_embedding(
        client=object(),
        text_path=text_path,
    )

    assert was_created is False

    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == {
        "existing": True,
    }


def test_skips_empty_image_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip image text containing only whitespace."""

    image_embeddings_dir = (
        tmp_path
        / "embeddings"
        / "images"
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.IMAGE_EMBEDDINGS_DIR",
        image_embeddings_dir,
    )

    property_dir = tmp_path / "12345678"
    property_dir.mkdir()

    text_path = property_dir / "01.txt"
    text_path.write_text(
        "\n   ",
        encoding="utf-8",
    )

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        pytest.fail("generate_text_embedding should not be called")

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_text_embedding",
        fail_if_called,
    )

    was_created = generate_image_embedding(
        client=object(),
        text_path=text_path,
    )

    assert was_created is False

    assert not (
        image_embeddings_dir
        / "12345678"
        / "01.json"
    ).exists()


def test_returns_property_and_image_text_paths(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return property text first followed by image text in sorted order."""

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

    second_property_path = property_text_dir / "02.txt"
    first_property_path = property_text_dir / "01.txt"

    second_property_path.write_text("property 2", encoding="utf-8")
    first_property_path.write_text("property 1", encoding="utf-8")

    first_image_dir = image_text_dir / "12345678"
    second_image_dir = image_text_dir / "87654321"

    first_image_dir.mkdir()
    second_image_dir.mkdir()

    first_image_path = first_image_dir / "01.txt"
    second_image_path = second_image_dir / "02.txt"

    first_image_path.write_text("image 1", encoding="utf-8")
    second_image_path.write_text("image 2", encoding="utf-8")

    (first_image_dir / "metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.PROPERTY_TEXT_DIR",
        property_text_dir,
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.IMAGE_TEXT_DIR",
        image_text_dir,
    )

    result = get_text_paths()

    assert result == [
        ("property", first_property_path),
        ("property", second_property_path),
        ("image", first_image_path),
        ("image", second_image_path),
    ]


def test_generates_property_and_image_embeddings(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate embeddings for property and image text during orchestration."""

    property_text_path = (
        temp_property_data
        / "embedding_text"
        / "properties"
        / "12345678.txt"
    )

    image_text_path = (
        temp_property_data
        / "embedding_text"
        / "images"
        / "12345678"
        / "01.txt"
    )

    property_text_path.write_text(
        "Detached house",
        encoding="utf-8",
    )

    image_text_path.parent.mkdir(parents=True)
    image_text_path.write_text(
        "Modern kitchen",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.get_text_paths",
        lambda: [
            ("property", property_text_path),
            ("image", image_text_path),
        ],
    )

    created_paths: list[Path] = []

    def fake_generate_property_embedding(
        client: Any,
        text_path: Path,
    ) -> bool:
        created_paths.append(text_path)
        return True

    def fake_generate_image_embedding(
        client: Any,
        text_path: Path,
    ) -> bool:
        created_paths.append(text_path)
        return True

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_property_embedding",
        fake_generate_property_embedding,
    )

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_image_embedding",
        fake_generate_image_embedding,
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    generate_text_embeddings(
        sleep_seconds=0,
    )

    assert created_paths == [
        property_text_path,
        image_text_path,
    ]


def test_continues_after_embedding_generation_failure(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue processing remaining text after one embedding fails."""

    first_text_path = (
        temp_property_data
        / "embedding_text"
        / "properties"
        / "11111111.txt"
    )

    second_text_path = (
        temp_property_data
        / "embedding_text"
        / "properties"
        / "22222222.txt"
    )

    first_text_path.write_text("First property", encoding="utf-8")
    second_text_path.write_text("Second property", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.get_text_paths",
        lambda: [
            ("property", first_text_path),
            ("property", second_text_path),
        ],
    )

    processed_paths: list[Path] = []

    def fake_generate_property_embedding(
        client: Any,
        text_path: Path,
    ) -> bool:
        processed_paths.append(text_path)

        if text_path == first_text_path:
            raise RuntimeError("embedding generation failed")

        return True

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_property_embedding",
        fake_generate_property_embedding,
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    generate_text_embeddings(
        sleep_seconds=0,
    )

    assert processed_paths == [
        first_text_path,
        second_text_path,
    ]


def test_limits_number_of_created_embeddings(
    temp_property_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop after creating the requested maximum number of embeddings."""

    text_paths = [
        (
            "property",
            temp_property_data
            / "embedding_text"
            / "properties"
            / f"{property_id}.txt",
        )
        for property_id in [
            "11111111",
            "22222222",
            "33333333",
        ]
    ]

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.get_text_paths",
        lambda: text_paths,
    )

    processed_paths: list[Path] = []

    def fake_generate_property_embedding(
        client: Any,
        text_path: Path,
    ) -> bool:
        processed_paths.append(text_path)
        return True

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.generate_property_embedding",
        fake_generate_property_embedding,
    )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(
        "scripts.generate_text_embeddings.OpenAI",
        FakeOpenAI,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    generate_text_embeddings(
        max_items=2,
        sleep_seconds=0,
    )

    assert processed_paths == [
        text_paths[0][1],
        text_paths[1][1],
    ]