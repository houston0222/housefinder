# Search Flow: Text + Image Semantic Search

HouseFinder allows users to search for properties using natural-language queries across both property information and visual features.

For example:

> "Show me properties with solar panels"

Rather than relying only on exact keyword matching or traditional filters, HouseFinder converts the query into a vector embedding and compares it against both property-level text embeddings and image-level visual embeddings.

This enables the system to find relevant properties even when a requested feature is visible in the property images but is not explicitly mentioned in the listing text.

---

## 1. User Search Query

The user enters a natural-language query in the frontend search interface.

For example:

```text
Show me properties with solar panels
```

The frontend sends the query to the FastAPI backend, where the semantic search process begins.

---

## 2. Query Embedding

The backend converts the user's query into a vector embedding using the same embedding model used to prepare the searchable property data.

Conceptually:

```text
"Show me properties with solar panels"
                    ↓
        [0.021, -0.014, 0.083, ...]
```

An embedding is a numerical representation of the semantic meaning of the text.

Using embeddings allows HouseFinder to compare concepts rather than relying only on exact words. Queries and property information with similar meanings should therefore have vectors that are closer together in the embedding space.

---

## 3. Property-Level Semantic Search

Each property contains searchable textual information such as:

- Location
- Property type
- Number of bedrooms
- Price
- Listing description
- Other property details

During the data preparation pipeline, this information is converted into a property-level embedding and stored alongside the property record in PostgreSQL.

At search time, pgvector compares the query embedding against these property embeddings.

For example:

```text
User query:
"Family home near a school"

Property information:
"Spacious three-bedroom house close to local schools"
```

Although the wording is different, semantic similarity allows the system to identify that the property is relevant to the user's request.

---

## 4. Image-Level Semantic Search

Property listings often contain useful information that exists only in their photographs.

To make these visual features searchable, each property image is analysed by a vision model during the offline processing pipeline.

The vision model generates structured visual metadata describing features visible in the image.

For example:

```text
Exterior view of a house with solar panels on the roof,
a driveway, and a front garden.
```

This visual description is then converted into an embedding and stored with the corresponding image record.

At search time, the same user query embedding is compared against these image embeddings using pgvector similarity search.

For example:

```text
User query:
"Show me properties with solar panels"

Listing text:
No mention of solar panels

Image metadata:
"Solar panels visible on the roof"
```

The property can therefore be discovered through its images even though the feature is absent from the listing text.

This is the main advantage of combining text and visual semantic search.

---

## 5. Properties with Multiple Images

A property can contain many images, and each image has its own visual metadata and embedding.

For example:

```text
Property A
├── Image 1: Front exterior
├── Image 2: Kitchen
├── Image 3: Living room
├── Image 4: Garden
└── Image 5: Roof with solar panels
```

The system evaluates image similarity at the individual image level.

A property does not require every image to match the query. Instead, the strongest matching image is used as the property's visual similarity score.

Conceptually:

```text
best_image_score = max(similarity score of each property image)
```

For a query such as:

```text
Show me properties with solar panels
```

the images might produce:

```text
Image 1: Front exterior       → weak match
Image 2: Kitchen              → weak match
Image 3: Living room          → weak match
Image 4: Garden               → weak match
Image 5: Solar panels on roof → strong match
```

Because Image 5 strongly matches the query, the property can be considered a visual match.

This approach reflects the way property search works in practice: a requested feature may only appear in one photograph, and requiring every image to match would incorrectly exclude relevant properties.

---

## 6. Combining Property and Image Matches

The backend performs semantic search across two sources:

```text
                    User Query
                        │
                        ▼
                  Query Embedding
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
     Property Embeddings      Image Embeddings
             │                     │
             ▼                     ▼
       Text Similarity       Visual Similarity
             │                     │
             └──────────┬──────────┘
                        ▼
                 Property Ranking
```

Results from both searches are combined at the property level.

A property can therefore match in three useful ways:

1. **Text match + image match**
2. **Image match only**
3. **Text match only**

Properties that do not sufficiently match either source are excluded.

---

## 7. Ranking Strategy

The ranking strategy is designed to reward evidence from both property information and visual content while preserving the value of image-only discovery.

### Text Match + Image Match

This represents the strongest result.

For example:

```text
Listing:
"Energy-efficient three-bedroom home"

Image metadata:
"Solar panels visible across the roof"
```

Both sources provide evidence that the property is relevant to the query.

---

### Image Match Only

The property text does not contain the requested feature, but one or more images provide strong visual evidence.

For example:

```text
Listing:
No mention of solar panels

Image metadata:
"Solar panels visible on the roof"
```

This is particularly important to HouseFinder because it allows users to discover property features that conventional text-based search could miss.

---

### Text Match Only

The property information matches the query, but the feature is not identified in the available image metadata.

For example:

```text
Listing:
"The property includes roof-mounted solar panels"

Image metadata:
No solar panels identified
```

The result remains relevant because the property-level semantic search provides evidence for the match.

---

### No Match

If neither the property information nor the image metadata reaches the required similarity threshold, the property is excluded from the results.

---

## 8. Final Ranking

Conceptually, the result priority is:

```text
1. Text match + image match
2. Image match only
3. Text match only
4. No match → excluded
```

Within these categories, semantic similarity scores are used to rank the strongest results.

The result is a multimodal property search system that combines:

```text
Property information
        +
AI-generated visual metadata
        +
Vector similarity search
        =
Multimodal semantic property search
```

This allows HouseFinder to search not only what a property listing **says**, but also what its images **show**.