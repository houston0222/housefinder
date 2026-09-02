# HouseFinder

An AI-powered property search application that lets users find homes using natural-language queries across both property information and visual features.

Instead of relying only on traditional listing filters such as **price, location, property type, and number of bedrooms**, HouseFinder also searches visual features extracted from property images — allowing users to search for things like **"large garden"**, **"modern kitchen"**, or **"solar panels"**.

## How It Works

HouseFinder uses an offline AI pipeline to prepare property data for semantic search:

1. A vision model analyses property images and generates structured descriptions of visible features.
2. Property text and image descriptions are converted into vector embeddings.
3. Embeddings are stored in PostgreSQL using pgvector.
4. At search time, the user's query is embedded and compared against both property and image vectors.
5. Text and image similarity are combined to rank the most relevant properties.

This allows properties to be discovered based on features that may only be visible in their photos and not explicitly mentioned in the listing text.

## Tech Stack

**Frontend:** React, TypeScript  
**Backend:** Python, FastAPI, Pydantic  
**Database:** PostgreSQL, pgvector  
**AI / Search:** Vision model, text embeddings, hybrid vector search  
**Infrastructure:** Docker, Docker Compose

## Architecture

`React → FastAPI → PostgreSQL + pgvector`

The AI enrichment pipeline runs separately from the search API, generating the image descriptions and embeddings used during search.

## Running Locally

1. Copy `.env.example` to `.env` and configure the required environment variables.

2. Start the application:

```bash
docker compose up --build
```

3. Open the application:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`