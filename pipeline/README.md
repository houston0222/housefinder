# HouseFinder Pipeline

A data processing pipeline that prepares property images for use by the HouseFinder application.

The pipeline starts with **raw property images** grouped by property ID. Each image is converted to **RGB**, resized to fit within **1024 × 1024 pixels while preserving its aspect ratio**, and saved as an optimized **JPEG with quality 80**.

Images that have already been normalized are skipped, allowing the pipeline to resume without repeating completed work.

**Raw property images → RGB conversion → Resize → JPEG compression → Normalized property images**