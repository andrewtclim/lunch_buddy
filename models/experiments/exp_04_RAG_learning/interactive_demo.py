# models/experiments/exp_04_RAG_learning/interactive_demo.py
# Interactive learning demo: user describes their tastes, picks dishes over 10 days,
# and their preference vector evolves with each pick (EMA update, alpha=0.3).
# Gemini re-summarizes preferences after each pick so its prompt stays in sync.
# At the end, user reveals true hidden preferences -- we score how well we converged.
#
# WHICH SCRIPT TO USE:
#   - At home / IPv6 network  --> use this file (interactive_demo.py)
#                                  uses DATABASE_URL in fastapi/.env
#   - On campus / IPv4 network --> use CAMPUS_interactive_demo.py instead
#                                  uses DATABASE_URL_IPV4 in fastapi/.env
#
# WHY: Campus networks (USF, Stanford, etc.) are IPv4-only. Supabase's direct
# connection hostname (db.wthzvpjnxpldewhzaqzn.supabase.co) is IPv6-only and
# won't resolve on campus. CAMPUS_interactive_demo.py points to the Supabase
# shared pooler (aws-1-us-east-2.pooler.supabase.com) which is IPv4-compatible.
#
# STEPS:
#   1. ONBOARDING - User types their food preferences in plain English. That text
#      is sent to Vertex AI and embedded into a 768-dimension vector using
#      text-embedding-004. This becomes the starting preference vector.
#
#   2. DB CONNECTION - psycopg2 connects directly to Supabase (PostgreSQL + pgvector)
#      using DATABASE_URL from fastapi/.env.
#
#   3. RETRIEVAL (each day, 10 days total) - The preference vector is used to run
#      a cosine similarity search against the backfill_menu table in Supabase,
#      which holds pre-embedded dishes from 10 days of Stanford menus. The top 20
#      closest dishes for that day are returned.
#
#   4. FILTERING - Placeholder station labels (e.g. "Soup of the Day") and any
#      dishes that conflict with the user's allergens are removed. The top 5
#      remaining candidates are passed to Gemini.
#
#   5. RECOMMENDATION - Gemini 2.5 Flash receives the 5 candidates and the user's
#      current preference summary, and picks the top 3 with reasons. The user picks one.
#
#   6. VECTOR UPDATE - The chosen dish's stored embedding is fetched from Supabase.
#      The preference vector is nudged 30% toward it via an exponential moving average
#      (EMA), then re-normalized. This is the "learning" step.
#
#   7. SUMMARY UPDATE - Gemini re-reads the original signup text plus every dish
#      picked so far, and writes a fresh 1-sentence preference summary. This summary
#      is what Gemini uses as context on the next day, keeping its prompt in sync
#      with how tastes have evolved.
#
#   8. FINAL SCORE - After 10 days, the user reveals their true preferences. That
#      text is embedded and compared (cosine similarity) against both the initial
#      and final preference vectors to measure how much the vector converged.

import os
import json
import time
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# load env vars -- parents[2] reaches models/ from experiments/exp_04_RAG_learning/
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")          # Supabase connection string
PROJECT_ID   = os.getenv("PROJECT_ID")            # GCP project for Vertex AI
LOCATION     = os.getenv("LOCATION", "us-central1")  # Vertex AI region
EMBED_MODEL = "text-embedding-004"       # must match model used to embed dishes
# model for recommendations and summarization
GEN_MODEL = "gemini-2.5-flash"
ALPHA = 0.3                        # how strongly each pick pulls the preference vector

MOCK_USERS_PATH = Path(__file__).parent.parent / \
    "exp_01_single_day" / "mock_users.json"

# generic station labels to exclude -- same set used across all experiments
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

# initialize Vertex AI client once -- reused for all API calls in this session
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def get_embedding(text: str) -> np.ndarray:
    # embed a single string and return as a numpy array
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


def normalize(vec: np.ndarray) -> np.ndarray:
    # scale a vector to unit length -- keeps cosine math well-behaved
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def update_preference_vector(pref_vec: np.ndarray, dish_vec: np.ndarray) -> np.ndarray:
    # exponential moving average: nudge pref_vec 30% toward the chosen dish embedding
    # then re-normalize so magnitude stays consistent across days
    updated = (1 - ALPHA) * pref_vec + ALPHA * dish_vec
    return normalize(updated)


def get_available_dates() -> list[str]:
    # pull all distinct dates from backfill_menu, sorted oldest to newest
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT date_served
        FROM backfill_menu
        ORDER BY date_served ASC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # convert date objects to "YYYY-MM-DD" strings
    return [str(row[0]) for row in rows]


def retrieve_dishes(query_vector: np.ndarray, date_str: str, limit: int = 20) -> list[dict]:
    # cosine search backfill_menu for the given date, ranked by similarity to the query vector
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM backfill_menu
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


def fetch_dish_embedding(dish_name: str, date_str: str) -> np.ndarray | None:
    # look up the stored embedding for a dish by name and date
    # uses ILIKE for case-insensitive match -- Gemini sometimes tweaks capitalization
    # returns None if the dish isn't found (e.g. Gemini hallucinated a name)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding
        FROM backfill_menu
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
    # remove generic station labels that aren't real dishes
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity between two unit-ish vectors -- 1.0 = identical direction
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def summarize_preferences(signup_text: str, chosen_dishes: list[str]) -> str:
    # ask Gemini to write a fresh 1-2 sentence preference summary
    # based on what the user originally said + every dish they have picked so far
    # this summary becomes the natural language prompt for tomorrow's recommendation call
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
            return response.text.strip()   # plain text, no JSON needed here
        except Exception as e:
            print(
                f"  [summarize attempt {attempt + 1} failed: {e}]", flush=True)
            if attempt == 2:
                # fallback: just return the original signup text so the loop can continue
                return signup_text
            time.sleep(2)


def call_gemini(preference_summary: str, candidates: list[dict]) -> list[dict]:
    # ask Gemini to re-rank the candidate dishes and pick the top 3
    # preference_summary evolves each day -- day 1 uses signup text, later days use the running summary
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
            # drop any "N/A" entries Gemini returns when it can't find a good match
            recs = [r for r in result["recommendations"] if r.get(
                "dish_name") and r["dish_name"].strip().upper() != "N/A"]
            return recs
        except Exception as e:
            print(f"  [gemini attempt {attempt + 1} failed: {e}]", flush=True)
            if attempt == 2:
                print(
                    f"  Warning: giving up after 3 attempts, skipping day.", flush=True)
                return []
            time.sleep(2)


def main():
    print("\n=== Lunch Buddy -- Learning Demo ===\n", flush=True)

    # --- onboarding ---
    signup_text = input(
        "What kind of food do you enjoy? Describe your tastes:\n> ").strip()
    allergens_raw = input(
        "\nAny allergens to avoid? (comma-separated, or press Enter to skip):\n> ").strip()
    # parse allergens into a list -- empty string becomes an empty list
    allergens = [a.strip() for a in allergens_raw.split(",") if a.strip()]

    # embed the signup text -- this is the starting preference vector
    print("\nEmbedding your preferences...", flush=True)
    initial_pref_vec = get_embedding(
        signup_text)   # saved for final comparison
    pref_vec = initial_pref_vec.copy()       # this one evolves each day

    # the natural language summary Gemini uses as its prompt -- starts as the raw signup text
    preference_summary = signup_text

    # pull the 10 available dates from the DB
    dates = get_available_dates()
    print(
        f"\nDates available: {dates[0]} -> {dates[-1]} ({len(dates)} days)\n", flush=True)

    # log structure -- saved to JSON at the end
    log = {
        "signup_text": signup_text,
        "allergens":   allergens,
        "days":        [],           # one entry per day
    }

    chosen_dishes = []   # running list of dish names the user has picked -- fed to summarizer

    # --- day loop ---
    for day_num, date_str in enumerate(dates):
        print(f"--- Day {day_num + 1} ({date_str}) ---", flush=True)
        print(
            f"Current preference summary: \"{preference_summary}\"\n", flush=True)

        # retrieve top 20 dishes by cosine distance to the evolving pref_vec
        candidates = retrieve_dishes(pref_vec, date_str, limit=20)

        # clean up candidates -- remove placeholders, allergen conflicts, and duplicates
        candidates = filter_placeholders(candidates)
        candidates = filter_allergens(candidates, allergens)
        candidates = deduplicate(candidates)[:5]   # top 5 go to Gemini

        if not candidates:
            print("  No safe dishes found for today, skipping.\n", flush=True)
            continue

        # call Gemini to re-rank and pick top 3 -- uses today's evolving summary
        recs = call_gemini(preference_summary, candidates)

        if not recs:
            print("  Gemini couldn't make recommendations today, skipping.\n", flush=True)
            continue

        # display Gemini's picks to the user
        print("Today's top recommendations:\n", flush=True)
        for i, rec in enumerate(recs):
            print(
                f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}", flush=True)
            print(f"     Why: {rec['reason']}\n", flush=True)

        # get the user's pick -- validate that input is 1, 2, or 3
        while True:
            choice_raw = input(
                "Which dish do you pick? (1 / 2 / 3):\n> ").strip()
            if choice_raw in ("1", "2", "3") and int(choice_raw) <= len(recs):
                break
            print(
                f"  Please enter a number between 1 and {len(recs)}.", flush=True)

        # the dict Gemini returned for this pick
        chosen_rec = recs[int(choice_raw) - 1]
        chosen_name = chosen_rec["dish_name"]
        chosen_dishes.append(chosen_name)

        print(f"\nYou picked: {chosen_name}\n", flush=True)

        # fetch the chosen dish's embedding from Supabase -- no extra API call needed
        dish_vec = fetch_dish_embedding(chosen_name, date_str)

        if dish_vec is not None:
            # update the preference vector toward the chosen dish
            pref_vec = update_preference_vector(pref_vec, dish_vec)
        else:
            # dish not found in DB (shouldn't happen) -- skip vector update for this day
            print(
                f"  Warning: embedding not found for '{chosen_name}', vector unchanged.", flush=True)

        # ask Gemini to re-summarize preferences based on all picks so far
        print("Updating your preference summary...", flush=True)
        preference_summary = summarize_preferences(signup_text, chosen_dishes)
        print(f"New summary: \"{preference_summary}\"\n", flush=True)

        # log this day
        log["days"].append({
            "day":              day_num + 1,
            "date":             date_str,
            "recommendations":  recs,
            "chosen":           chosen_name,
            "preference_summary_after": preference_summary,
        })

    # --- final reveal ---
    print("=== All 10 days complete! ===\n", flush=True)
    print("Now reveal your true food preferences -- be as specific as you like:\n", flush=True)
    true_preferences = input("> ").strip()

    print("\nEmbedding your true preferences...", flush=True)
    true_vec = get_embedding(true_preferences)

    # compare how close initial vs final preference vector is to the true preferences
    initial_score = cosine_similarity(initial_pref_vec, true_vec)
    final_score = cosine_similarity(pref_vec, true_vec)
    delta = final_score - initial_score

    print(f"\n--- Results ---", flush=True)
    print(
        f"Initial preference vector similarity to true preferences: {initial_score:.4f}", flush=True)
    print(
        f"Final preference vector similarity to true preferences:   {final_score:.4f}", flush=True)
    print(
        f"Improvement after 10 days of learning:                   {delta:+.4f}", flush=True)

    # save everything to JSON
    log["true_preferences"] = true_preferences
    log["initial_score"] = initial_score
    log["final_score"] = final_score
    log["score_improvement"] = delta

    results_path = Path(__file__).parent / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nFull log saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
