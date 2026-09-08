from pathlib import Path

from PIL import Image

from scripts.normalize_property_images import normalize_property_images


def test_normalizes_oversized_png_to_rgb_jpeg(
    temp_property_data: Path,
) -> None:
    """Normalize an oversized PNG while preserving aspect ratio and file structure."""

    raw_images_dir = temp_property_data / "raw_images"
    normalized_images_dir = temp_property_data / "normalized_images"

    property_dir = raw_images_dir / "12345678"
    property_dir.mkdir()

    input_path = property_dir / "01.png"

    with Image.new("RGB", (2000, 1000)) as image:
        image.save(input_path)

    normalize_property_images(
        raw_images_dir=raw_images_dir,
        normalized_images_dir=normalized_images_dir,
    )

    output_path = normalized_images_dir / "12345678" / "01.jpg"

    assert output_path.exists()

    with Image.open(output_path) as normalized_image:
        assert normalized_image.format == "JPEG"
        assert normalized_image.mode == "RGB"
        assert normalized_image.size == (1024, 512)


def test_preserves_small_image_dimensions(
    temp_property_data: Path,
) -> None:
    """Keep image dimensions unchanged when already within the size limit."""

    raw_images_dir = temp_property_data / "raw_images"
    normalized_images_dir = temp_property_data / "normalized_images"

    property_dir = raw_images_dir / "12345678"
    property_dir.mkdir()

    input_path = property_dir / "01.png"

    with Image.new("RGB", (800, 600)) as image:
        image.save(input_path)

    normalize_property_images(
        raw_images_dir=raw_images_dir,
        normalized_images_dir=normalized_images_dir,
    )

    output_path = normalized_images_dir / "12345678" / "01.jpg"

    assert output_path.exists()

    with Image.open(output_path) as normalized_image:
        assert normalized_image.size == (800, 600)


def test_skips_existing_normalized_image(
    temp_property_data: Path,
) -> None:
    """Do not overwrite an image that has already been normalized."""

    raw_images_dir = temp_property_data / "raw_images"
    normalized_images_dir = temp_property_data / "normalized_images"

    property_dir = raw_images_dir / "12345678"
    property_dir.mkdir()

    input_path = property_dir / "01.png"

    with Image.new("RGB", (2000, 1000)) as image:
        image.save(input_path)

    output_dir = normalized_images_dir / "12345678"
    output_dir.mkdir()

    output_path = output_dir / "01.jpg"

    with Image.new("RGB", (100, 100)) as existing_image:
        existing_image.save(output_path, format="JPEG")

    normalize_property_images(
        raw_images_dir=raw_images_dir,
        normalized_images_dir=normalized_images_dir,
    )

    with Image.open(output_path) as normalized_image:
        assert normalized_image.size == (100, 100)


def test_ignores_unsupported_file_extension(
    temp_property_data: Path,
) -> None:
    """Ignore files whose extensions are not supported image formats."""

    raw_images_dir = temp_property_data / "raw_images"
    normalized_images_dir = temp_property_data / "normalized_images"

    property_dir = raw_images_dir / "12345678"
    property_dir.mkdir()

    unsupported_file = property_dir / "notes.txt"
    unsupported_file.write_text("not an image")

    normalize_property_images(
        raw_images_dir=raw_images_dir,
        normalized_images_dir=normalized_images_dir,
    )

    output_path = normalized_images_dir / "12345678" / "notes.jpg"

    assert not output_path.exists()