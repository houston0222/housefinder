# HouseFinder Pipeline

A data processing pipeline that prepares property images and metadata for use by the HouseFinder application.

The pipeline starts with **raw property images** stored in `raw_images/` and grouped by property ID. Each image is converted to **RGB**, resized to fit within **1024 × 1024 pixels while preserving its aspect ratio**, and saved as an optimized **JPEG with quality 80** in `normalized_images/`.

Each normalized image is then analyzed using **GPT-4.1 mini** with the visual metadata prompt. The model returns structured JSON metadata describing the visual content of the property image, which is saved in `image_metadata/` using the same property and image identifiers.

Existing files are skipped at both stages, allowing the pipeline to resume without repeating completed work.

**`raw_images/` → RGB conversion → Resize → JPEG compression → `normalized_images/` → GPT-4.1 mini visual analysis → `image_metadata/`**