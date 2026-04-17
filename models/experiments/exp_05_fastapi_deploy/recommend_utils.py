# recommend_utils.py
# Pure functions extracted from exp_04_RAG_learning/interactive_demo.py.
# No state -- every function receives what it needs as arguments.
# The caller (main.py) initializes the Vertex AI client and DB URL once
# at startup and passes them in here.
#
# Changes vs interactive_demo.py:
#   - Table changed: backfill_menu -> daily_menu (production data)
#   - genai.Client is passed in, not module-level (cleaner for FastAPI)
#   - No print() calls -- logging is the caller's job

import json
import time

import numpy as np
import psycopg2


# station labels that are not real dishes -- same set used across all experiments
PLACEHOLDER_DISHES = {
    "Grilled Vegan/Vegetarian",
    "Assorted Allergy-Friendly Items",
    "Chef's Choice Breakfast Special",
    "Chef's Choice Egg Special",
    "Chef's Choice Vegan Breakfast",
    "Chefs Dinner Special",
    "Burger Bar",
    "Omelet Station",
    "Panini Station",
    "Fried Rice Bar",
    "Performance Bar",
    "Soup of the Day",
}

EMBED_MODEL = "text-embedding-004"   # must match the model used to embed dishes in daily_menu
GEN_MODEL = "gemini-2.5-flash"       # generator model for recommendations and summarization
ALPHA = 0.3                          # EMA weight -- how strongly each pick pulls the pref vector


def get_embedding(text: str, client) -> np.ndarray:
    # embed a single string via Vertex AI and return as a numpy array
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


def normalize(vec: np.ndarray) -> np.ndarray:
    # scale a vector to unit length -- keeps cosine distances well-behaved
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def update_preference_vector(pref_vec: np.ndarray, dish_vec: np.ndarray) -> np.ndarray:
    # exponential moving average: nudge pref_vec ALPHA% toward the chosen dish embedding
    # then re-normalize so magnitude stays consistent across days
    updated = (1 - ALPHA) * pref_vec + ALPHA * dish_vec
    return normalize(updated)


def retrieve_dishes(query_vector: np.ndarray, date_str: str, db_url: str, limit: int = 20) -> list[dict]:
    # cosine search daily_menu for the given date, ranked by similarity to the query vector
    # uses pgvector's <=> operator (cosine distance -- lower = more similar)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM daily_menu
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (date_str, query_vector.tolist(), limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],
            "ingredients": row[4] or "",
        }
        for row in rows
    ]


def fetch_dish_embedding(dish_name: str, date_str: str, db_url: str) -> np.ndarray | None:
    # look up the stored embedding for a dish by name and date
    # uses ILIKE for case-insensitive match -- Gemini sometimes tweaks capitalization
    # returns None if the dish is not found (e.g. Gemini hallucinated a name)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding
        FROM daily_menu
        WHERE dish_name ILIKE %s AND date_served = %s
        LIMIT 1;
    """, (dish_name, date_str))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    # pgvector returns the embedding as a string "[0.1, 0.2, ...]" -- parse to float array
    return np.array(json.loads(row[0]), dtype=float)


def filter_placeholders(dishes: list[dict]) -> list[dict]:
    # remove generic station labels that are not real dishes
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    # hard filter: drop any dish that shares an allergen with the user
    if not user_allergens:
        return dishes
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def deduplicate(dishes: list[dict]) -> list[dict]:
    # keep only the first occurrence of each dish name -- first = highest cosine rank
    seen = set()
    unique = []
    for d in dishes:
        if d["dish_name"] not in seen:
            seen.add(d["dish_name"])
            unique.append(d)
    return unique


def call_gemini(preference_summary: str, candidates: list[dict], client) -> list[dict]:
    # ask Gemini to re-rank the 5 candidate dishes and return the top 3
    # preference_summary evolves each day -- starts as signup text, then Gemini's running summary
    dish_lines = "\n".join([
        f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
        f" -- Ingredients: {d['ingredients']}"
        for i, d in enumerate(candidates)
    ])

    prompt = f"""You are a Stanford dining hall recommender.

User preference: {preference_summary}

Available dishes today:
{dish_lines}

Recommend the top 3 dishes from the list above that best match the user's preference.
For each dish include: dish name, dining hall, and one concise reason why it fits.
Only recommend dishes from the list above -- do not invent dishes.
Respond in this exact JSON format:
{{
  "recommendations": [
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}}
  ]
}}"""

    # retry up to 3 times -- covers malformed JSON and transient network errors
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            result = json.loads(response.text)
            # drop any "N/A" entries Gemini returns when it cannot find a good match
            recs = [r for r in result["recommendations"] if r.get(
                "dish_name") and r["dish_name"].strip().upper() != "N/A"]
            return recs
        except Exception as e:
            if attempt == 2:
                return []
            time.sleep(2)


def summarize_preferences(signup_text: str, chosen_dishes: list[str], client) -> str:
    # ask Gemini to write a fresh 1-sentence preference summary
    # based on what the user originally said + every dish they have picked so far
    # this summary becomes the natural language context for the next recommendation call
    history = ", ".join(chosen_dishes) if chosen_dishes else "none yet"
    prompt = f"""A user signed up for a dining recommender with this description:
"{signup_text}"

So far they have selected these dishes:
{history}

Based on both their stated preferences and their actual selections,
write a single concise sentence (max 20 words) that captures their current food preferences.
Only output the sentence -- no preamble, no quotes."""

    # retry up to 3 times on any error -- network blips or rate limits
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                # fallback: return the original signup text so the caller can continue
                return signup_text
            time.sleep(2)
