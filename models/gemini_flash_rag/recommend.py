# models/gemini_flash_rag/recommend.py
# Production recommendation pipeline for Lunch Buddy.
# Retrieves dishes from Supabase, blends an optional daily mood into the query
# vector, asks Gemini to pick the top 3, and updates the user's preference vector
# after they choose.
#
# Flow:
#   1. Load user's stored preference vector (pref_vec) from Supabase
#   2. If user provided a daily mood string, embed it and blend into a query_vec
#       (beta=0.5 so mood meaningfully shifts retrieval)
#   3. Cosine search backfill_menu / daily_menu using query_vec
#   4. Filter placeholders, allergens, duplicates -> top 5 candidates
#   5. Call Gemini with candidates + preference summary + mood string
#   6. User picks one; fetch that dish's embedding and EMA-update pref_vec
#   7. Save updated pref_vec back to Supabase
#
# Key changes from exp_06 benchmark (Apr 12 2026):
#   - thinking_budget=0: identical picks to default thinking, 7x faster (~2s vs ~13s),
#     100% JSON reliability vs 33-67% with default thinking
#   - mood-primary prompt (v2): when mood given, it leads as the primary constraint
#     and profile becomes a tiebreaker. Produces meaningfully different (mood-aligned)
#     picks vs the old profile-dominant prompt (1/3 overlap in benchmark)
#   - dynamic beta: 0.5 when mood present (was fixed 0.3) so retrieval and prompt
#     are aligned in prioritizing the user's explicit request

import os
import json
import time
import socket                      # used to DNS-check the Supabase hostname
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# load env vars from models/.env (two levels up from gemini_flash_rag/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _resolve_database_url() -> str:
    """
    Auto-select the right Supabase connection string for the current network.

    Some networks (e.g. USF campus) block IPv6 DNS resolution for the
    Supabase hostname, causing a 'nodename nor servname provided' error.
    DATABASE_URL_IPV4 uses a numeric IP to bypass that.

    Strategy: try a quick DNS lookup on the hostname extracted from
    DATABASE_URL. If it resolves, we're on a network that can reach
    Supabase normally. If it fails, fall back to DATABASE_URL_IPV4.
    """
    default_url = os.getenv("DATABASE_URL")
    ipv4_url = os.getenv("DATABASE_URL_IPV4")

    if not default_url:
        # no primary URL configured -- nothing to resolve
        return ipv4_url or ""

    # extract just the hostname from the postgres:// connection string
    # e.g. "postgresql://user:pass@db.xxx.supabase.co:5432/postgres"
    #   -> "db.xxx.supabase.co"
    try:
        # grab token after @ and before :
        host = default_url.split("@")[1].split(":")[0]
        # DNS lookup -- raises on failure
        socket.getaddrinfo(host, None)
        # resolved OK -- use the normal URL
        return default_url
    except (IndexError, socket.gaierror):
        # DNS failed (gaierror) or URL couldn't be parsed (IndexError)
        if ipv4_url:
            print("  [db] Hostname not reachable -- using DATABASE_URL_IPV4")
            return ipv4_url
        return default_url   # no IPv4 fallback configured -- return original and let psycopg2 raise


DATABASE_URL = _resolve_database_url()
PROJECT_ID = os.getenv("PROJECT_ID")                # GCP project for Vertex AI
LOCATION = os.getenv("LOCATION", "us-central1")   # Vertex AI region

EMBED_MODEL = "text-embedding-004"   # must match the model used to embed dishes
GEN_MODEL = "gemini-2.5-flash"     # generator LLM

ALPHA = 0.3    # how strongly each dish pick nudges the stored preference vector
# how strongly a daily mood nudges the query vector (when mood given)
BETA_WITH_MOOD = 0.5
BETA_NO_MOOD = 0.0   # no mood means query_vec = pref_vec directly
# alpha and beta are independent:
#   alpha updates pref_vec permanently after a pick
#   beta shifts query_vec temporarily for today's search only
# beta=0.5 when mood is given so the user's explicit request meaningfully
# shifts what surfaces -- validated in exp_06 benchmark

# initialize the Vertex AI client once -- reused for all API calls in this session
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# generic station labels that appear in the menu but aren't real dishes
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


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity between two vectors -- 1.0 = identical, 0.0 = unrelated
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(vec: np.ndarray) -> np.ndarray:
    # scale a vector to unit length so cosine similarity stays well-behaved
    # if the vector is all zeros (edge case), return it unchanged
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def blend_mood(pref_vec: np.ndarray, mood_vec: np.ndarray, beta: float = BETA_WITH_MOOD) -> np.ndarray:
    # blend the user's stored preference vector with a daily mood embedding
    # result is a temporary query vector used only for today's retrieval --
    # the stored pref_vec is never modified here
    #
    # beta=0.5 means: 50% learned profile, 50% today's mood
    # when the user explicitly says what they want, that signal should
    # meaningfully shift what surfaces in the search
    blended = (1 - beta) * pref_vec + beta * mood_vec
    # normalize so cosine search treats both users equally
    return normalize(blended)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    # send a string to Vertex AI text-embedding-004 and get back a 768-dim vector
    # this is used for: signup text, daily mood strings, and the final reveal score
    # dish embeddings are already stored in Supabase -- we never re-embed them here
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


# ---------------------------------------------------------------------------
# Supabase retrieval
# ---------------------------------------------------------------------------

def get_available_dates(table: str = "backfill_menu") -> list[str]:
    # pull all distinct dates from the given table, sorted oldest to newest
    # used by the multi-day demo loop to know which dates to iterate over
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT DISTINCT date_served
        FROM {table}
        ORDER BY date_served ASC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # convert date objects to "YYYY-MM-DD" strings
    return [str(row[0]) for row in rows]


def retrieve_dishes(query_vec: np.ndarray, date_str: str,
                    table: str = "daily_menu", limit: int = 20) -> list[dict]:
    # cosine search the given table for today's dishes ranked by similarity to query_vec
    # table defaults to "daily_menu" (production) but can be set to "backfill_menu"
    # (experiment data) -- same schema, so the query works for both
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM {table}
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (date_str, query_vec.tolist(), limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # pack each row into a dict so the rest of the pipeline can use field names
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],   # NULL in DB becomes empty list
            "ingredients": row[4] or "",   # NULL in DB becomes empty string
        }
        for row in rows
    ]


def fetch_dish_embedding(dish_name: str, date_str: str,
                         table: str = "daily_menu") -> np.ndarray | None:
    # look up the stored embedding for a specific dish by name and date
    # returns None if not found (e.g. Gemini hallucinated a dish name)
    # uses ILIKE so a minor capitalization difference doesn't cause a miss
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT embedding
        FROM {table}
        WHERE dish_name ILIKE %s AND date_served = %s
        LIMIT 1;
    """, (dish_name, date_str))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    # pgvector returns the embedding as a string "[0.1, 0.2, ...]" -- parse to floats
    return np.array(json.loads(row[0]), dtype=float)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_placeholders(dishes: list[dict]) -> list[dict]:
    # remove generic station labels that appear in the HTML but aren't real dishes
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    # hard block -- drop any dish that shares an allergen with the user
    # this runs before scoring; no similarity score can override an allergen conflict
    if not user_allergens:
        return dishes                          # skip the check entirely if no allergens
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def deduplicate(dishes: list[dict]) -> list[dict]:
    # keep only the first occurrence of each dish name
    # first = highest cosine rank, so later duplicates are strictly worse
    seen = set()
    unique = []
    for d in dishes:
        if d["dish_name"] not in seen:
            seen.add(d["dish_name"])
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# Gemini recommendation call
# ---------------------------------------------------------------------------

def call_gemini(preference_summary: str, candidates: list[dict],
                daily_mood: str | None = None) -> list[dict]:
    # ask Gemini to re-rank the candidates and return the top 3 with reasons
    #
    # preference_summary -- one-sentence description of the user's learned taste
    # candidates         -- top 10 dishes from the cosine search (already filtered)
    # daily_mood         -- optional free-text mood string from the user today
    #                       e.g. "I want something light" or "craving sushi"
    #
    # Prompt strategy (validated in exp_06):
    #   - when daily_mood is provided: mood leads as the primary constraint,
    #     taste profile is framed as a tiebreaker. This prevents the profile
    #     from overriding what the user explicitly asked for today.
    #   - when no mood: profile-driven prompt (mood section omitted entirely).
    #
    # Model config (validated in exp_06):
    #   - thinking_budget=0 disables the extended reasoning step in 2.5 Flash.
    #     Benchmark showed identical dish picks to default thinking at 7x lower
    #     latency (~2s vs ~13s) with higher JSON reliability (100% vs 33-67%).

    # build the numbered dish list Gemini reads
    dish_lines = "\n".join([
        f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
        f" -- Ingredients: {d['ingredients']}"
        for i, d in enumerate(candidates)
    ])

    # shared output format instructions
    format_rules = (
        "Only use dishes from the list above -- do not invent dishes.\n"
        "No dish should appear in both recommendations and alternatives.\n"
        "For dish_name use ONLY the dish name "
        '(e.g. "Gochujang Spiced Chicken"), not the dining hall or meal time.'
    )

    json_schema = """\
Respond in this exact JSON format:
{{
  "recommendations": [
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}}
  ],
  "alternatives": [
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}}
  ]
}}"""

    if daily_mood:
        # mood-primary prompt: mood leads, profile is tiebreaker
        prompt = f"""You are a Stanford dining hall recommender.
 
The user is specifically craving: "{daily_mood}"
Prioritize this above their general taste profile.
 
General taste profile (use for tie-breaking only): {preference_summary}
 
Available dishes today:
{dish_lines}
 
From the list above, return two things:
 
1. "recommendations" -- the top 3 dishes that best match today's craving. \
Use the general taste profile to break ties between equally good matches, \
but do not let the profile override what the user asked for today. \
For each include: dish name, dining hall, and one concise reason it fits today's craving.
 
2. "alternatives" -- 2 dishes from the remaining list that are meaningfully different \
in cuisine or style from the top 3 (to broaden the user's options). \
For each include: dish name, dining hall, and one concise reason it's worth trying.
 
{format_rules}
{json_schema}"""
    else:
        # profile-driven prompt: no mood to prioritize
        prompt = f"""You are a Stanford dining hall recommender.
 
User preference: {preference_summary}
 
Available dishes today:
{dish_lines}
 
From the list above, return two things:
 
1. "recommendations" -- the top 3 dishes that best match the user's preference. \
For each include: dish name, dining hall, and one concise reason why it fits.
 
2. "alternatives" -- 2 dishes from the remaining list that are meaningfully different \
in cuisine or style from the top 3 (to broaden the user's options). \
For each include: dish name, dining hall, and one concise reason it's worth trying.
 
{format_rules}
{json_schema}"""

    # retry up to 3 times -- covers transient network errors and malformed JSON
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    # disable extended thinking -- re-ranking 10 dishes is pattern
                    # matching, not multi-step reasoning. exp_06 confirmed identical
                    # picks at 7x lower latency with budget=0.
                    "thinking_config": genai_types.ThinkingConfig(
                        thinking_budget=0
                    ),
                },
            )
            result = json.loads(response.text)

            def clean(lst):
                cleaned = []
                for r in lst:
                    name = (r.get("dish_name") or "").strip()
                    if not name or name.upper() == "N/A":
                        continue
                    # strip trailing " at Dining Hall (Meal Time)" if the model
                    # copied the full line from the candidate list into dish_name
                    if " at " in name:
                        name = name.split(" at ")[0].strip()
                    r["dish_name"] = name
                    cleaned.append(r)
                return cleaned

            recs = clean(result.get("recommendations", []))
            alts = clean(result.get("alternatives", []))
            return recs, alts
        except Exception as e:
            print(f"  [gemini attempt {attempt + 1} failed: {e}]", flush=True)
            if attempt == 2:
                return [], []   # give up after 3 failures
            time.sleep(2)


# ---------------------------------------------------------------------------
# Preference vector update
# ---------------------------------------------------------------------------

def summarize_preferences(signup_text: str, chosen_dishes: list[str]) -> str:
    # ask Gemini to write a fresh one-sentence preference summary
    # based on the original signup text + every dish picked so far
    # this becomes the natural language prompt for the next day's recommendation
    history = ", ".join(chosen_dishes) if chosen_dishes else "none yet"
    prompt = f"""A user signed up for a dining recommender with this description:
"{signup_text}"

So far they have selected these dishes:
{history}

Based on both their stated preferences and their actual selections,
write a single concise sentence (max 20 words) that captures their current food preferences.
Only output the sentence -- no preamble, no quotes."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config={
                    "thinking_config": genai_types.ThinkingConfig(
                        thinking_budget=0
                    ),
                },
            )
            return response.text.strip()
        except Exception as e:
            print(
                f"  [summarize attempt {attempt + 1} failed: {e}]", flush=True)
            if attempt == 2:
                return signup_text   # fall back to original signup text
            time.sleep(2)


def update_preference_vector(pref_vec: np.ndarray, dish_vec: np.ndarray,
                             alpha: float = ALPHA) -> np.ndarray:
    # EMA update: nudge pref_vec alpha% toward the chosen dish, then re-normalize
    # this is the only place pref_vec changes -- mood never touches it
    updated = (1 - alpha) * pref_vec + alpha * dish_vec
    return normalize(updated)


# ---------------------------------------------------------------------------
# Top-level recommend() -- the single entry point for the full pipeline
# ---------------------------------------------------------------------------

def recommend(
    # user's current preference vector (768-dim)
    pref_vec: np.ndarray,
    preference_summary: str,        # one-sentence description of user's taste
    # hard allergen exclusions (e.g. ["gluten"])
    user_allergens: list[str],
    date_str: str,                  # date to pull dishes for, "YYYY-MM-DD"
    daily_mood: str | None = None,  # optional free-text mood for today
    table: str = "daily_menu",      # which Supabase table to search
) -> tuple[list[dict], np.ndarray]:
    # runs the full pipeline and returns:
    #   - recs: list of up to 3 recommendation dicts from Gemini
    #   - query_vec: the vector that was used for retrieval (blended if mood given)
    #     caller uses query_vec for nothing -- it's returned only for logging/debug
    #
    # NOTE: pref_vec is NOT updated here -- updating requires the user's pick,
    # which happens outside this function. See update_preference_vector() above.

    # step 1 -- if mood was provided, embed it and blend into a temporary query vector
    if daily_mood:
        mood_vec = get_embedding(daily_mood)   # embed the mood string
        # 50/50 profile/mood (BETA_WITH_MOOD)
        query_vec = blend_mood(pref_vec, mood_vec)
    else:
        query_vec = pref_vec                    # no mood -- use profile directly

    # step 2 -- cosine search: find today's top 40 dishes closest to query_vec
    # 40 gives the filter steps enough to work with on sparse days (weekends,
    # limited halls) without pulling in dishes that are too far from the preference
    candidates = retrieve_dishes(query_vec, date_str, table=table, limit=40)

    # step 3 -- clean up: remove station labels, allergen conflicts, duplicates
    candidates = filter_placeholders(candidates)
    candidates = filter_allergens(candidates, user_allergens)
    candidates = deduplicate(candidates)[:10]   # top 10 go to Gemini
    # (was 5 -- extra 5 give Gemini room
    #  to pick 2 diverse alternatives)

    # step 4 -- ask Gemini to pick top 3 recommendations + 2 diverse alternatives
    recs, alts = call_gemini(
        preference_summary, candidates, daily_mood=daily_mood)

    return recs, alts, query_vec
