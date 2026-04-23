"""
Location-aware recommendation wrapper for Lunch Buddy.

This module adds a proximity filter on top of the existing
gemini_flash_rag pipeline without modifying any of that code.

The only new piece is retrieve_dishes_with_halls() -- the same cosine
search as the original retrieve_dishes(), but with an extra
AND dining_hall = ANY(...) clause to restrict the pool to nearby halls.

Everything else (embedding, filtering, Gemini call) is imported directly
from gemini_flash_rag.recommend and reused unchanged.

Public entry point:
    recommend_with_location(pref_vec, preference_summary, user_allergens,
                            date_str, daily_mood, user_lat, user_lon,
                            radius_m, restrict_to_nearby, table)
"""

import sys
import psycopg2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# location_filter/ lives one level below the project root (lunch_buddy/).
# We add the project root to sys.path so that Python can find the
# models.gemini_flash_rag package when we import from it below.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # lunch_buddy/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from the existing pipeline
# ---------------------------------------------------------------------------

# We import only the pieces we reuse -- nothing is copied or duplicated.
# If Lynn updates any of these, we automatically pick up her changes.
from models.gemini_flash_rag.recommend import (   # noqa: E402
    DATABASE_URL,             # Supabase connection string (from models/.env)
    get_embedding,            # text -> 768-dim vector via Vertex AI
    blend_mood,               # (pref_vec, mood_vec, beta) -> blended query vec
    filter_placeholders,      # remove generic station labels
    filter_allergens,         # hard-block dishes with user's allergens
    deduplicate,              # keep highest-ranked copy of each dish name
    call_gemini,              # send candidates to Gemini, get top 3 + 2 alts
    BETA_WITH_MOOD,           # 0.5 -- how strongly mood shifts the query vec
)

from location_filter.proximity import filter_nearby_halls   # our proximity layer


# ---------------------------------------------------------------------------
# Location-filtered retrieval
# ---------------------------------------------------------------------------

def retrieve_dishes_with_halls(
    query_vec: np.ndarray,
    date_str: str,
    halls: list[str],
    table: str = "daily_menu",
    limit: int = 40,
) -> list[dict]:
    """
    Cosine search for today's dishes, restricted to a specific set of halls.

    This is the only meaningful addition over the original retrieve_dishes().
    The SQL is identical except for the extra AND dining_hall = ANY(%s) clause
    which limits results to halls the user can actually walk to.

    Args:
        query_vec:  768-dim preference/mood vector used for cosine ranking
        date_str:   date to pull dishes for, "YYYY-MM-DD"
        halls:      list of dining_hall strings to include (from filter_nearby_halls)
        table:      Supabase table to search ("daily_menu" or "backfill_menu")
        limit:      max rows to return before downstream filtering (default 40)

    Returns:
        List of dish dicts with keys: dish_name, dining_hall, meal_time,
        allergens, ingredients.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM {table}
        WHERE date_served = %s
          AND dining_hall = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (
        date_str,
        halls,                  # psycopg2 automatically formats a Python list
        query_vec.tolist(),     # pgvector expects a plain JSON array, not ndarray
        limit,
    ))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # pack each row into a named dict so downstream code uses field names,
    # not fragile integer indices
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],    # NULL in DB -> empty list
            "ingredients": row[4] or "",    # NULL in DB -> empty string
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Location-aware recommend() wrapper
# ---------------------------------------------------------------------------

def recommend_with_location(
    pref_vec: np.ndarray,
    preference_summary: str,         # one-sentence description of user's taste
    user_allergens: list[str],        # hard exclusions (e.g. ["gluten", "milk"])
    date_str: str,                    # "YYYY-MM-DD"
    daily_mood: str | None = None,    # optional free-text mood for today
    user_lat: float | None = None,    # user's latitude in decimal degrees
    user_lon: float | None = None,    # user's longitude in decimal degrees
    radius_m: float = 800.0,          # walking radius in meters (default ~10 min)
    restrict_to_nearby: bool = True,  # set False to skip location filter entirely
    table: str = "daily_menu",        # Supabase table to search
) -> tuple[list[dict], list[dict], np.ndarray, list[str]]:
    """
    Run the full recommendation pipeline with an optional location filter.

    Location filter behavior:
      - If restrict_to_nearby=False: skip location entirely, use full dish pool.
      - If restrict_to_nearby=True and user_lat/lon provided:
            compute nearby halls -> filter dish pool to those halls only.
      - If restrict_to_nearby=True but no halls are within radius (e.g. off-campus):
            log a warning and fall back to the full pool gracefully.
      - If restrict_to_nearby=True but no coordinates provided:
            skip location filter (same as restrict_to_nearby=False).

    Args:
        pref_vec:           user's stored 768-dim preference vector
        preference_summary: human-readable taste summary for Gemini's prompt
        user_allergens:     list of allergen strings to hard-block
        date_str:           date string "YYYY-MM-DD" to pull menus for
        daily_mood:         optional mood string (e.g. "something light")
        user_lat:           user's latitude (decimal degrees)
        user_lon:           user's longitude (decimal degrees)
        radius_m:           walking radius cutoff in meters
        restrict_to_nearby: master toggle for the location filter
        table:              which Supabase table to query

    Returns:
        (recs, alts, query_vec, halls_used)
        recs:       list of up to 3 recommendation dicts from Gemini
        alts:       list of up to 2 diverse alternative dicts from Gemini
        query_vec:  the vector used for retrieval (blended if mood given)
        halls_used: the list of halls that were actually searched
                    (useful for display and debugging)
    """

    # ------------------------------------------------------------------
    # Step 1 -- build query vector (same logic as original recommend())
    # ------------------------------------------------------------------

    if daily_mood:
        # embed the mood string and blend 50/50 with the stored profile
        # result is ephemeral -- it shifts today's search but never gets saved
        mood_vec = get_embedding(daily_mood)
        query_vec = blend_mood(pref_vec, mood_vec, BETA_WITH_MOOD)
    else:
        # no mood -- use the stored preference vector directly as the query
        query_vec = pref_vec

    # ------------------------------------------------------------------
    # Step 2 -- determine which halls to search
    # ------------------------------------------------------------------

    halls_used = None   # None means "all halls" (no filter applied)

    if restrict_to_nearby and user_lat is not None and user_lon is not None:
        # compute halls within radius
        nearby = filter_nearby_halls(user_lat, user_lon, radius_m)

        if nearby:
            # at least one hall is reachable -- use the filtered pool
            halls_used = nearby
        else:
            # user is off-campus or between halls -- fall back gracefully
            print(
                f"  [location] No halls within {radius_m:.0f}m "
                f"of ({user_lat:.4f}, {user_lon:.4f}). "
                "Falling back to full pool."
            )

    # ------------------------------------------------------------------
    # Step 3 -- retrieve dishes (filtered or full pool)
    # ------------------------------------------------------------------

    if halls_used:
        # location filter is active -- restrict the cosine search to nearby halls
        candidates = retrieve_dishes_with_halls(
            query_vec, date_str, halls_used, table=table, limit=40
        )
    else:
        # no location filter -- import and use the original retrieve_dishes()
        # so we stay consistent with the baseline pipeline
        from models.gemini_flash_rag.recommend import retrieve_dishes
        candidates = retrieve_dishes(query_vec, date_str, table=table, limit=40)

    # ------------------------------------------------------------------
    # Step 4 -- clean the candidate pool (identical to original pipeline)
    # ------------------------------------------------------------------

    candidates = filter_placeholders(candidates)          # drop station labels
    candidates = filter_allergens(candidates, user_allergens)   # hard allergen block
    candidates = deduplicate(candidates)[:10]             # top 10 unique dishes -> Gemini

    # ------------------------------------------------------------------
    # Step 5 -- ask Gemini to pick top 3 + 2 alternatives
    # ------------------------------------------------------------------

    recs, alts = call_gemini(preference_summary, candidates, daily_mood=daily_mood)

    # return halls_used so the demo can print "Searching 3 halls near you"
    return recs, alts, query_vec, (halls_used or ["all halls"])
