import { useState } from "react";
import type React from "react";
import "./App.css";

type PropertyImage = {
  id: number;
  property_id: string;
  image_id: string;
  source_image_url: string;
  image_type: string | null;
  room_or_area: string | null;
  caption: string | null;
  is_best_match: boolean;
};

type SearchResult = {
  id: number;
  property_id: string;
  title: string;
  canonical_url: string | null;
  price_text: string | null;
  price_amount: number | null;
  location: string | null;
  bedrooms: number | null;
  property_type: string | null;
  agent: string | null;
  main_image_url: string | null;
  image_count: number | null;

  property_similarity: number | null;
  image_similarity: number | null;
  score: number;
  match_type: "text_and_image" | "image_only" | "text_only" | "no_match";

  best_image_id: string | null;
  best_image_url: string | null;
  best_image_caption: string | null;
  best_image_room_or_area: string | null;

  images: PropertyImage[];
};

type SearchResponse = {
  query: string;
  count: number;
  results: SearchResult[];
};

const API_BASE_URL = "http://localhost:8000";

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return value.toFixed(3);
}

function getMatchLabel(matchType: SearchResult["match_type"]) {
  if (matchType === "text_and_image") {
    return "Text + Image Match";
  }

  if (matchType === "image_only") {
    return "Image Match";
  }

  if (matchType === "text_only") {
    return "Text Match";
  }

  return "No Match";
}

function App() {
  const [query, setQuery] = useState("large garden");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const [selectedImages, setSelectedImages] = useState<PropertyImage[]>([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(
    null
  );

  async function handleSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const cleanQuery = query.trim();

    if (!cleanQuery) {
      setErrorMessage("Please enter a search query.");
      return;
    }

    setIsLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/search?query=${encodeURIComponent(cleanQuery)}`
      );

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const result = (await response.json()) as SearchResponse;
      setData(result);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong.";

      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  }

  function openImageViewer(images: PropertyImage[], index: number) {
    setSelectedImages(images);
    setSelectedImageIndex(index);
  }

  function closeImageViewer() {
    setSelectedImageIndex(null);
    setSelectedImages([]);
  }

  function showPreviousImage() {
    if (selectedImageIndex === null || selectedImages.length === 0) {
      return;
    }

    setSelectedImageIndex(
      (selectedImageIndex - 1 + selectedImages.length) %
      selectedImages.length
    );
  }

  function showNextImage() {
    if (selectedImageIndex === null || selectedImages.length === 0) {
      return;
    }

    setSelectedImageIndex(
      (selectedImageIndex + 1) % selectedImages.length
    );
  }

  const selectedImage =
    selectedImageIndex !== null
      ? selectedImages[selectedImageIndex]
      : null;

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">HouseFinder</p>

        <h1>Find properties by text and visual features</h1>

        <p className="heroText">
          Search natural language queries like “large garden”, “solar panel”,
          “modern kitchen”, or “driveway”. The backend searches both property
          text embeddings and AI-generated image metadata embeddings.
        </p>

        <form className="searchForm" onSubmit={handleSearch}>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Try: solar panel, large garden, modern kitchen"
          />

          <button type="submit" disabled={isLoading}>
            {isLoading ? "Searching..." : "Search"}
          </button>
        </form>

        {errorMessage && <p className="error">{errorMessage}</p>}
      </section>

      {data && (
        <section className="resultsHeader">
          <h2>Results for “{data.query}”</h2>
          <p>{data.count} properties found</p>
        </section>
      )}

      <section className="resultsGrid">
        {data?.results.map((property) => (
          <article
            className="propertyCard"
            key={property.property_id}
          >
            <div className="imageWrapper">
              <img
                src={
                  property.best_image_url ||
                  property.main_image_url ||
                  ""
                }
                alt={property.title}
              />

              <span
                className={`matchBadge ${property.match_type}`}
              >
                {getMatchLabel(property.match_type)}
              </span>
            </div>

            <div className="propertyBody">
              <h3>{property.title}</h3>

              <p className="price">
                {property.price_text || "Price unavailable"}
              </p>

              <p className="meta">
                {property.bedrooms
                  ? `${property.bedrooms} bedrooms`
                  : "Bedrooms N/A"}

                {property.property_type
                  ? ` · ${property.property_type}`
                  : ""}
              </p>

              {property.location && (
                <p className="location">
                  {property.location}
                </p>
              )}

              <div className="scores">
                <span>
                  Score: {formatScore(property.score)}
                </span>

                <span>
                  Text: {formatScore(property.property_similarity)}
                </span>

                <span>
                  Image: {formatScore(property.image_similarity)}
                </span>
              </div>

              {property.best_image_caption && (
                <p className="matchReason">
                  <strong>Best visual match:</strong>{" "}
                  {property.best_image_caption}
                </p>
              )}

              <div className="gallery">
                {property.images.map((image, index) => (
                  <button
                    type="button"
                    className={`thumb ${image.is_best_match ? "bestThumb" : ""
                      }`}
                    key={image.id}
                    onClick={() =>
                      openImageViewer(property.images, index)
                    }
                  >
                    <img
                      src={image.source_image_url}
                      alt={image.caption || property.title}
                    />

                    {image.is_best_match && (
                      <span className="thumbBadge">
                        Matched
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </article>
        ))}
      </section>

      {selectedImage && selectedImageIndex !== null && (
        <div
          className="imageModal"
          onClick={closeImageViewer}
        >
          <button
            type="button"
            className="modalClose"
            onClick={closeImageViewer}
            aria-label="Close image viewer"
          >
            ×
          </button>

          {selectedImages.length > 1 && (
            <button
              type="button"
              className="modalArrow modalArrowLeft"
              onClick={(event) => {
                event.stopPropagation();
                showPreviousImage();
              }}
              aria-label="Previous image"
            />
          )}

          <div
            className="modalContent"
            onClick={(event) => event.stopPropagation()}
          >
            <img
              src={selectedImage.source_image_url}
              alt={selectedImage.caption || "Property"}
            />

            <div className="modalInfo">
              <span>
                {selectedImageIndex + 1} / {selectedImages.length}
              </span>

              {selectedImage.caption && (
                <p>{selectedImage.caption}</p>
              )}
            </div>
          </div>

          {selectedImages.length > 1 && (
            <button
              type="button"
              className="modalArrow modalArrowRight"
              onClick={(event) => {
                event.stopPropagation();
                showNextImage();
              }}
              aria-label="Next image"
            />
          )}
        </div>
      )}
    </main>
  );
}

export default App;