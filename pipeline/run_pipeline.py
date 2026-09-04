from scripts.normalize_property_images import normalize_property_images
from scripts.generate_property_image_metadata import generate_property_image_metadata
from scripts.build_embedding_text import build_embedding_text
from scripts.generate_text_embeddings import generate_text_embeddings
from scripts.import_pipeline_data_to_db import import_pipeline_data_to_db


def run_pipeline() -> None:
    print("Starting HouseFinder pipeline.")

    print("\n[1/5] Normalizing property images...")
    normalize_property_images()

    print("\n[2/5] Generating property image metadata...")
    generate_property_image_metadata()

    print("\n[3/5] Building embedding text...")
    build_embedding_text()

    print("\n[4/5] Generating text embeddings...")
    generate_text_embeddings()

    print("\n[5/5] Importing pipeline data into database...")
    import_pipeline_data_to_db()

    print("\nHouseFinder pipeline complete.")


if __name__ == "__main__":
    run_pipeline()