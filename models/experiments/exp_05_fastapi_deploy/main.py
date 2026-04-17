# main.py
# FastAPI app for the Lunch Buddy recommendation endpoint.
# Initialized once at startup: Vertex AI client + env vars.
# POST /recommend is the only inference endpoint -- stateless by design.
# The caller holds the preference vector between requests and sends it back each time.

import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pathlib import Path
from pydantic import BaseModel, Field

from recommend_utils import (
    call_gemini,
    deduplicate,
    fetch_dish_embedding,
    filter_allergens,
    filter_placeholders,
    get_embedding,
    retrieve_dishes,
    summarize_preferences,
    update_preference_vector,
)

# load env vars from fastapi/.env -- parents[3] = project root, then into fastapi/
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

# use IPv4 pooler on campus networks (IPv6 direct hostname won't resolve)
# falls back to DATABASE_URL if DATABASE_URL_IPV4 is not set
DATABASE_URL = os.getenv("DATABASE_URL_IPV4") or os.getenv("DATABASE_URL")
PROJECT_ID = os.getenv("PROJECT_ID")       # GCP project for Vertex AI
LOCATION = "us-central1"                   # Vertex AI region

# module-level client -- initialized once in lifespan, reused for every request
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts -- initialize the Vertex AI client
    global _client
    _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    yield
    # runs once when the server shuts down -- nothing to clean up
    _client = None


app = FastAPI(title="Lunch Buddy Recommend API", lifespan=lifespan)


# --- request / response schemas ---

class RecommendRequest(BaseModel):
    user_query: str = Field(..., description="Natural language food preference description")
    preference_vector: Optional[list[float]] = Field(
        None,
        description="768-dim preference vector from the previous response. "
                    "Pass null on the first request -- the server will embed user_query."
    )
    allergens: list[str] = Field(
        default_factory=list,
        description="List of allergens to hard-filter (e.g. ['gluten', 'dairy'])"
    )
    date: Optional[str] = Field(
        None,
        description="Date to search menus for, in YYYY-MM-DD format. Defaults to today."
    )
    chosen_dishes: list[str] = Field(
        default_factory=list,
        description="All dish names the user has picked in previous rounds. "
                    "Used to update the preference summary. Empty on first request."
    )


class Recommendation(BaseModel):
    dish_name: str
    dining_hall: str
    reason: str


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    updated_preference_vector: list[float]  # caller stores this and sends it back next time
    preference_summary: str                 # Gemini's fresh 1-sentence summary of user tastes


# --- routes ---

@app.get("/")
def root():
    return {"message": "Lunch Buddy Recommend API"}


@app.get("/health")
def health():
    # Vertex AI endpoint health check -- confirms the client initialized successfully
    if _client is None:
        return {"status": "unhealthy", "detail": "Vertex AI client not initialized"}
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> RecommendResponse:
    if _client is None:
        raise HTTPException(status_code=503, detail="Vertex AI client not initialized")

    # resolve the target date -- default to today if caller didn't provide one
    target_date = body.date or str(date.today())

    # step 1: get the preference vector
    # if the caller sent one, use it directly
    # if not (first request), embed the user_query to create the initial vector
    if body.preference_vector is not None:
        pref_vec = np.array(body.preference_vector, dtype=float)
    else:
        pref_vec = get_embedding(body.user_query, _client)

    # step 2: cosine search daily_menu for today's top 20 dishes
    candidates = retrieve_dishes(pref_vec, target_date, DATABASE_URL, limit=20)

    # step 3: filter out station placeholders, allergen conflicts, and duplicates
    candidates = filter_placeholders(candidates)
    candidates = filter_allergens(candidates, body.allergens)
    candidates = deduplicate(candidates)[:5]   # top 5 go to Gemini

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No safe dishes found for date {target_date} after filtering."
        )

    # step 4: ask Gemini to re-rank the 5 candidates and return top 3
    # on the first request, chosen_dishes is empty so preference_summary = user_query
    preference_summary = summarize_preferences(body.user_query, body.chosen_dishes, _client)
    recs = call_gemini(preference_summary, candidates, _client)

    if not recs:
        raise HTTPException(
            status_code=502,
            detail="Gemini failed to return recommendations after 3 attempts."
        )

    # step 5: update the preference vector toward the top recommendation
    # we use the first recommendation (Gemini's best pick) as the learning signal
    top_dish_name = recs[0]["dish_name"]
    dish_vec = fetch_dish_embedding(top_dish_name, target_date, DATABASE_URL)

    if dish_vec is not None:
        # nudge the preference vector 30% toward the top dish and re-normalize
        updated_pref_vec = update_preference_vector(pref_vec, dish_vec)
    else:
        # dish embedding not found (shouldn't happen) -- return vector unchanged
        updated_pref_vec = pref_vec

    return RecommendResponse(
        recommendations=[Recommendation(**r) for r in recs],
        updated_preference_vector=updated_pref_vec.tolist(),   # numpy -> plain list for JSON
        preference_summary=preference_summary,
    )
