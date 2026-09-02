from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import check_database_connection, get_db

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import Depends
from app.services.embeddings import create_query_embedding

app = FastAPI(
    title="HouseFinder API",
    description="API for natural language property search.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/db-health")
def db_health_check() -> dict[str, str]:
    check_database_connection()
    return {"database": "ok"}

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)) -> dict[str, int]:
    property_count = db.execute(text("SELECT COUNT(*) FROM properties")).scalar_one()
    image_count = db.execute(text("SELECT COUNT(*) FROM property_images")).scalar_one()

    return {
        "properties": property_count,
        "property_images": image_count,
    }

@app.get("/properties/sample")
def get_sample_properties(db: Session = Depends(get_db)) -> dict[str, list[dict]]:
    rows = db.execute(
        text(
            """
            SELECT
                id,
                property_id,
                title,
                price_text,
                price_amount,
                location,
                bedrooms,
                property_type,
                main_image_url,
                image_count
            FROM properties
            ORDER BY id
            LIMIT 5
            """
        )
    ).mappings().all()

    return {
        "properties": [dict(row) for row in rows]
    }

@app.get("/properties/{property_id}")
def get_property_detail(
    property_id: str,
    db: Session = Depends(get_db),
) -> dict:
    property_row = db.execute(
        text(
            """
            SELECT
                id,
                property_id,
                title,
                canonical_url,
                description,
                listing_type,
                price_text,
                price_amount,
                location,
                bedrooms,
                property_type,
                agent,
                main_image_url,
                image_count,
                key_features,
                listing_description,
                room_measurements
            FROM properties
            WHERE property_id = :property_id
            """
        ),
        {"property_id": property_id},
    ).mappings().first()

    if property_row is None:
        return {"error": "Property not found"}

    image_rows = db.execute(
        text(
            """
            SELECT
                id,
                property_id,
                image_id,
                source_image_url,
                raw_image_path,
                processed_image_path,
                image_type,
                room_or_area,
                caption,
                search_phrases,
                visual_observations,
                capacity_estimates,
                overall_confidence
            FROM property_images
            WHERE property_id = :property_id
            ORDER BY id
            """
        ),
        {"property_id": property_id},
    ).mappings().all()

    return {
        "property": dict(property_row),
        "images": [dict(row) for row in image_rows],
    }

@app.get("/search-basic")
def search_basic(
    query: str,
    db: Session = Depends(get_db),
) -> dict:
    search_text = f"%{query}%"

    rows = db.execute(
        text(
            """
            SELECT
                id,
                property_id,
                title,
                canonical_url,
                price_text,
                price_amount,
                location,
                bedrooms,
                property_type,
                agent,
                main_image_url,
                image_count
            FROM properties
            WHERE
                title ILIKE :search_text
                OR location ILIKE :search_text
                OR description ILIKE :search_text
                OR listing_description ILIKE :search_text
                OR property_type ILIKE :search_text
            ORDER BY id
            LIMIT 20
            """
        ),
        {"search_text": search_text},
    ).mappings().all()

    return {
        "query": query,
        "count": len(rows),
        "results": [dict(row) for row in rows],
    }



@app.get("/embedding-test")
def embedding_test(query: str) -> dict[str, int | str]:
    embedding = create_query_embedding(query)

    return {
        "query": query,
        "dimensions": len(embedding),
    }


@app.get("/search-properties-vector")
def search_properties_vector(
    query: str,
    db: Session = Depends(get_db),
) -> dict:
    query_embedding = create_query_embedding(query)

    rows = db.execute(
        text(
            """
            SELECT
                id,
                property_id,
                title,
                canonical_url,
                price_text,
                price_amount,
                location,
                bedrooms,
                property_type,
                agent,
                main_image_url,
                image_count,
                1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM properties
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT 10
            """
        ),
        {"query_embedding": query_embedding},
    ).mappings().all()

    return {
        "query": query,
        "count": len(rows),
        "results": [dict(row) for row in rows],
    }


@app.get("/search-images-vector")
def search_images_vector(
    query: str,
    db: Session = Depends(get_db),
) -> dict:
    query_embedding = create_query_embedding(query)

    rows = db.execute(
        text(
            """
            SELECT
                pi.id,
                pi.property_id,
                pi.image_id,
                pi.source_image_url,
                pi.image_type,
                pi.room_or_area,
                pi.caption,
                pi.search_phrases,
                pi.visual_observations,
                pi.overall_confidence,
                p.title,
                p.price_text,
                p.price_amount,
                p.location,
                p.bedrooms,
                p.property_type,
                p.main_image_url,
                1 - (pi.embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM property_images pi
            JOIN properties p
                ON p.property_id = pi.property_id
            WHERE pi.embedding IS NOT NULL
            ORDER BY pi.embedding <=> CAST(:query_embedding AS vector)
            LIMIT 20
            """
        ),
        {"query_embedding": query_embedding},
    ).mappings().all()

    return {
        "query": query,
        "count": len(rows),
        "results": [dict(row) for row in rows],
    }

@app.get("/search-images-properties")
def search_images_properties(
    query: str,
    db: Session = Depends(get_db),
) -> dict:
    query_embedding = create_query_embedding(query)

    rows = db.execute(
        text(
            """
            WITH ranked_images AS (
                SELECT
                    pi.id AS image_db_id,
                    pi.property_id,
                    pi.image_id,
                    pi.source_image_url,
                    pi.image_type,
                    pi.room_or_area,
                    pi.caption,
                    pi.search_phrases,
                    pi.visual_observations,
                    pi.overall_confidence,
                    1 - (pi.embedding <=> CAST(:query_embedding AS vector)) AS image_similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY pi.property_id
                        ORDER BY pi.embedding <=> CAST(:query_embedding AS vector)
                    ) AS image_rank
                FROM property_images pi
                WHERE pi.embedding IS NOT NULL
            )
            SELECT
                p.id,
                p.property_id,
                p.title,
                p.canonical_url,
                p.price_text,
                p.price_amount,
                p.location,
                p.bedrooms,
                p.property_type,
                p.agent,
                p.main_image_url,
                p.image_count,
                ri.image_db_id AS best_image_db_id,
                ri.image_id AS best_image_id,
                ri.source_image_url AS best_image_url,
                ri.image_type AS best_image_type,
                ri.room_or_area AS best_image_room_or_area,
                ri.caption AS best_image_caption,
                ri.search_phrases AS best_image_search_phrases,
                ri.visual_observations AS best_image_visual_observations,
                ri.overall_confidence AS best_image_overall_confidence,
                ri.image_similarity
            FROM ranked_images ri
            JOIN properties p
                ON p.property_id = ri.property_id
            WHERE ri.image_rank = 1
            ORDER BY ri.image_similarity DESC
            LIMIT 10
            """
        ),
        {"query_embedding": query_embedding},
    ).mappings().all()

    return {
        "query": query,
        "count": len(rows),
        "results": [dict(row) for row in rows],
    }

@app.get("/search")
def search(
    query: str,
    db: Session = Depends(get_db),
) -> dict:
    query_embedding = create_query_embedding(query)

    rows = db.execute(
        text(
            """
            WITH property_matches AS (
                SELECT
                    p.property_id,
                    1 - (p.embedding <=> CAST(:query_embedding AS vector)) AS property_similarity
                FROM properties p
                WHERE p.embedding IS NOT NULL
            ),
            best_image_matches AS (
                SELECT
                    ranked.property_id,
                    ranked.image_db_id,
                    ranked.image_id,
                    ranked.source_image_url,
                    ranked.image_type,
                    ranked.room_or_area,
                    ranked.caption,
                    ranked.search_phrases,
                    ranked.visual_observations,
                    ranked.overall_confidence,
                    ranked.image_similarity
                FROM (
                    SELECT
                        pi.id AS image_db_id,
                        pi.property_id,
                        pi.image_id,
                        pi.source_image_url,
                        pi.image_type,
                        pi.room_or_area,
                        pi.caption,
                        pi.search_phrases,
                        pi.visual_observations,
                        pi.overall_confidence,
                        1 - (pi.embedding <=> CAST(:query_embedding AS vector)) AS image_similarity,
                        ROW_NUMBER() OVER (
                            PARTITION BY pi.property_id
                            ORDER BY pi.embedding <=> CAST(:query_embedding AS vector)
                        ) AS image_rank
                    FROM property_images pi
                    WHERE pi.embedding IS NOT NULL
                ) ranked
                WHERE ranked.image_rank = 1
            )
            SELECT
                p.id,
                p.property_id,
                p.title,
                p.canonical_url,
                p.price_text,
                p.price_amount,
                p.location,
                p.bedrooms,
                p.property_type,
                p.agent,
                p.main_image_url,
                p.image_count,

                pm.property_similarity,

                bim.image_db_id AS best_image_db_id,
                bim.image_id AS best_image_id,
                bim.source_image_url AS best_image_url,
                bim.image_type AS best_image_type,
                bim.room_or_area AS best_image_room_or_area,
                bim.caption AS best_image_caption,
                bim.search_phrases AS best_image_search_phrases,
                bim.visual_observations AS best_image_visual_observations,
                bim.overall_confidence AS best_image_overall_confidence,
                bim.image_similarity,

                CASE
                    WHEN pm.property_similarity >= 0.35
                         AND bim.image_similarity >= 0.35
                    THEN 'text_and_image'

                    WHEN bim.image_similarity >= 0.35
                    THEN 'image_only'

                    WHEN pm.property_similarity >= 0.35
                    THEN 'text_only'

                    ELSE 'no_match'
                END AS match_type,

                CASE
                    WHEN pm.property_similarity >= 0.35
                         AND bim.image_similarity >= 0.35
                    THEN (bim.image_similarity * 0.65) + (pm.property_similarity * 0.35) + 0.10

                    WHEN bim.image_similarity >= 0.35
                    THEN bim.image_similarity * 0.65

                    WHEN pm.property_similarity >= 0.35
                    THEN pm.property_similarity * 0.35

                    ELSE 0
                END AS score

            FROM properties p
            LEFT JOIN property_matches pm
                ON pm.property_id = p.property_id
            LEFT JOIN best_image_matches bim
                ON bim.property_id = p.property_id
            WHERE
                pm.property_similarity >= 0.35
                OR bim.image_similarity >= 0.35
            ORDER BY score DESC
            LIMIT 20
            """
        ),
        {"query_embedding": query_embedding},
    ).mappings().all()

    results = [dict(row) for row in rows]

    property_ids = [result["property_id"] for result in results]

    if property_ids:
        image_rows = db.execute(
            text(
                """
                SELECT
                    id,
                    property_id,
                    image_id,
                    source_image_url,
                    image_type,
                    room_or_area,
                    caption
                FROM property_images
                WHERE property_id = ANY(:property_ids)
                ORDER BY property_id, id
                """
            ),
            {"property_ids": property_ids},
        ).mappings().all()

        images_by_property: dict[str, list[dict]] = {}

        for image_row in image_rows:
            image = dict(image_row)
            property_id = image["property_id"]

            images_by_property.setdefault(property_id, []).append(image)

        for result in results:
            best_image_db_id = result.get("best_image_db_id")
            images = images_by_property.get(result["property_id"], [])

            for image in images:
                image["is_best_match"] = image["id"] == best_image_db_id

            result["images"] = images

    else:
        for result in results:
            result["images"] = []

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }