from pathlib import Path

import pytest


@pytest.fixture
def temp_property_data(tmp_path: Path) -> Path:
    property_data = tmp_path / "property_data"

    (property_data / "raw_images").mkdir(parents=True)
    (property_data / "parsed").mkdir()
    (property_data / "normalized_images").mkdir()
    (property_data / "image_metadata").mkdir()

    (property_data / "embedding_text" / "properties").mkdir(parents=True)
    (property_data / "embedding_text" / "images").mkdir(parents=True)

    (property_data / "embeddings" / "properties").mkdir(parents=True)
    (property_data / "embeddings" / "images").mkdir(parents=True)

    return property_data