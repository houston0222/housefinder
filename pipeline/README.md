# HouseFinder Pipeline

A data processing pipeline that prepares property images, metadata, and embedding-ready text for use by the HouseFinder application.

The pipeline starts with **raw property images** stored in `raw_images/` and grouped by property ID. Each image is converted to **RGB**, resized to fit within **1024 × 1024 pixels while preserving its aspect ratio**, and saved as an optimized **JPEG with quality 80** in `normalized_images/`.

Each normalized image is then analyzed using **GPT-4.1 mini** with the visual metadata prompt. The model returns structured JSON metadata describing the visual content of the property image, which is saved in `image_metadata/` using the same property and image identifiers.

Property data from `parsed/` and image metadata from `image_metadata/` are then converted into separate text representations for embedding. Property text includes fields such as location, price, bedrooms, property type, key features, description, and room measurements. Image text includes visual metadata such as image type, room or area, caption, search phrases, visual observations, and capacity estimates. The resulting text files are saved in `embedding_text/properties/` and `embedding_text/images/`.

**`raw_images/` → Image normalization → `normalized_images/` → GPT-4.1 mini visual analysis → `image_metadata/` → Embedding text construction → `embedding_text/`**