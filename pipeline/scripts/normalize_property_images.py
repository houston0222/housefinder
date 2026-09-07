from pathlib import Path

from PIL import Image, UnidentifiedImageError


RAW_IMAGES_DIR = Path("/app/data/raw_images")
NORMALIZED_IMAGES_DIR = Path("/app/data/normalized_images")

MAX_SIZE = 1024
JPEG_QUALITY = 80
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_image(input_path: Path, output_path: Path) -> bool:
    if output_path.exists():
        print(f"Skipped existing: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image.thumbnail((MAX_SIZE, MAX_SIZE))

        image.save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )

    print(f"Normalized: {input_path} -> {output_path}")
    return True


def normalize_property_images(
    raw_images_dir: Path = RAW_IMAGES_DIR,
    normalized_images_dir: Path = NORMALIZED_IMAGES_DIR,
    max_properties: int | None = None,
) -> None:
    if not raw_images_dir.exists():
        raise FileNotFoundError(
            f"Raw images directory not found: {raw_images_dir}"
        )

    property_dirs = sorted(
        path
        for path in raw_images_dir.iterdir()
        if path.is_dir()
    )

    if max_properties is not None:
        property_dirs = property_dirs[:max_properties]

    normalized_count = 0
    skipped_count = 0

    for property_dir in property_dirs:
        property_id = property_dir.name

        image_paths = sorted(
            path
            for path in property_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            output_path = (
                normalized_images_dir
                / property_id
                / f"{image_path.stem}.jpg"
            )

            try:
                was_normalized = normalize_image(
                    input_path=image_path,
                    output_path=output_path,
                )
            except (UnidentifiedImageError, OSError) as exc:
                print(f"Failed to normalize {image_path}: {exc}")
                continue

            if was_normalized:
                normalized_count += 1
            else:
                skipped_count += 1

    print("Done.")
    print(f"Normalized: {normalized_count}")
    print(f"Skipped existing: {skipped_count}")


if __name__ == "__main__":
    normalize_property_images()