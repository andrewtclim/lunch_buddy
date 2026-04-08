# models/experiments/exp_03_RAG_static/rag_simulation.py
# RAG simulation: each mock user sends their signup_text as a natural language query each day.
# Gemini Flash retrieves and re-ranks top dishes; we score Gemini's picks against the hidden profile.
# Compare results against exp_02 (pure vector baseline) to evaluate RAG quality.

import os
import json
import time
import mlflow
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# load env vars -- parents[3] because we are 3 levels deep from project root
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

DATABASE_URL    = os.getenv("DATABASE_URL")    # Supabase connection string
PROJECT_ID      = os.getenv("PROJECT_ID")      # GCP project for Vertex AI
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
LOCATION        = "us-central1"
EMBED_MODEL     = "text-embedding-004"         # must match model used to embed dishes
GEN_MODEL       = "gemini-2.5-flash"           # model for generating recommendations
MOCK_USERS_PATH = Path(__file__).parent.parent / "exp_01_single_day" / "mock_users.json"

# generic station labels to exclude -- same set as exp_01 and exp_02
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

# initialize Vertex AI client once -- reused for all embedding and generation calls
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def get_embedding(text: str) -> np.ndarray:
    # embed a single string and return as a numpy array
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


def get_available_dates() -> list[str]:
    # pull all distinct dates from backfill_menu, sorted oldest to newest
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT DISTINCT date_served
        FROM backfill_menu
        ORDER BY date_served ASC;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [str(row[0]) for row in rows]   # convert date objects to "YYYY-MM-DD" strings


def retrieve_dishes(query_vector: np.ndarray, date_str: str, limit: int = 20) -> list[dict]:
    # cosine search backfill_menu for the given date, ranked by distance to the query vector
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
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
    # look up the stored embedding for a dish by name and date -- avoids re-embedding for scoring
    # uses ILIKE for case-insensitive match -- Gemini sometimes tweaks capitalization
    # returns None if the dish isn't found (e.g. Gemini hallucinated a name)
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
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
    # remove generic station labels that aren't real dishes -- same set as exp_02
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    # hard filter: drop any dish that shares an allergen with the user
    if not user_allergens:
        return dishes
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def deduplicate(dishes: list[dict]) -> list[dict]:
    # keep only the first occurrence of each dish name -- first = highest cosine rank
    seen   = set()
    unique = []
    for d in dishes:
        if d["dish_name"] not in seen:
            seen.add(d["dish_name"])
            unique.append(d)
    return unique


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity between two unit-ish vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def call_gemini(query: str, candidates: list[dict]) -> list[dict]:
    # build a prompt with the user's query and the top candidates, call Gemini Flash
    # returns the parsed list of recommendation dicts (dish_name, dining_hall, reason)
    dish_lines = "\n".join([
        f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
        f" -- Ingredients: {d['ingredients']}"
        for i, d in enumerate(candidates)
    ])

    prompt = f"""You are a Stanford dining hall recommender.

User preference: {query}

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

    # retry up to 3 times -- covers malformed JSON and transient network errors (e.g. httpx.ReadError)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            result = json.loads(response.text)
            # filter out any "N/A" placeholders Gemini returns when it can't find a match
            recs = [r for r in result["recommendations"] if r.get("dish_name") and r["dish_name"].strip().upper() != "N/A"]
            return recs
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                # all 3 attempts failed -- return empty list so the day is skipped gracefully
                print(f"    Warning: giving up after 3 attempts, skipping day.", flush=True)
                return []
            time.sleep(2)   # short pause before retry -- avoids hammering Vertex AI after a blip


def run_simulation_for_user(user: dict, dates: list[str]) -> dict:
    # run the full RAG pipeline for one user across all available dates
    # embed once up front -- signup_text doesn't change day to day, so no need to re-embed
    hidden_vec = get_embedding(user["hidden_profile"])   # ground truth, fixed
    query_vec  = get_embedding(user["signup_text"])      # used as retrieval query every day

    day_scores    = []   # one avg cosine score per day
    chosen_dishes = []   # Gemini's top pick each day (first recommendation)

    for day_num, date_str in enumerate(dates):
        # retrieve top 20 by cosine distance to the signup query
        candidates = retrieve_dishes(query_vec, date_str, limit=20)

        # filter out placeholders, allergens, and duplicates -- then take top 5 for Gemini
        candidates = filter_placeholders(candidates)
        candidates = filter_allergens(candidates, user["allergens"])
        candidates = deduplicate(candidates)[:5]

        if len(candidates) < 1:
            print(f"  No safe dishes for {user['user_id']} on {date_str}, skipping.", flush=True)
            continue

        # call Gemini -- returns top 3 recommendations as parsed dicts
        recs = call_gemini(user["signup_text"], candidates)

        # score each of Gemini's picks against the hidden profile vector
        # use stored embeddings from Supabase -- no extra API calls needed
        scores = []
        for rec in recs:
            dish_vec = fetch_dish_embedding(rec["dish_name"], date_str)
            if dish_vec is None:
                # Gemini occasionally tweaks dish names slightly -- skip if no exact match
                print(f"    Warning: no embedding found for '{rec['dish_name']}' on {date_str}", flush=True)
                continue
            scores.append(cosine_similarity(hidden_vec, dish_vec))

        if not scores:
            print(f"  No scoreable dishes for {user['user_id']} on {date_str}, skipping.", flush=True)
            continue

        avg_score = float(np.mean(scores))
        day_scores.append(avg_score)

        # track Gemini's first pick as the "chosen" dish for the day
        chosen_dishes.append(recs[0]["dish_name"])

        print(f"  {user['user_id']} day {day_num} ({date_str}): score={avg_score:.4f}", flush=True)

    return {
        "user_id":       user["user_id"],
        "day_scores":    day_scores,
        "chosen_dishes": chosen_dishes,
        "score_delta":   day_scores[-1] - day_scores[0] if len(day_scores) > 1 else 0.0,
    }


def main():
    with open(MOCK_USERS_PATH) as f:
        users = json.load(f)

    # pull the 10 available dates from the DB
    dates = get_available_dates()
    print(f"Dates available: {dates[0]} -> {dates[-1]} ({len(dates)} days)", flush=True)
    print(f"Running RAG simulation: {len(dates)} days, {len(users)} users\n", flush=True)

    all_results = []
    for user in users:
        result = run_simulation_for_user(user, dates)
        all_results.append(result)

    # compute per-day averages across all users
    num_days = min(len(r["day_scores"]) for r in all_results)
    avg_scores_per_day = [
        float(np.mean([r["day_scores"][i] for r in all_results]))
        for i in range(num_days)
    ]

    # save results locally -- mlflow.log_artifact fails on Mac with mlflow 3.x
    results_path = Path(__file__).parent / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # log to MLflow -- same experiment as exp_01 and exp_02 for easy comparison
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy")

    with mlflow.start_run(run_name="exp03-rag-simulation"):
        mlflow.log_param("num_users",   len(users))
        mlflow.log_param("num_days",    len(dates))
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("gen_model",   GEN_MODEL)
        mlflow.log_param("date_range",  f"{dates[0]} -> {dates[-1]}")
        mlflow.log_param("experiment",  "exp_03_RAG_static")

        # per-day average cosine score -- step = day index
        for i, score in enumerate(avg_scores_per_day):
            mlflow.log_metric("avg_cosine_score", score, step=i)

        # overall delta from day 0 to final day
        mlflow.log_metric("score_improvement", avg_scores_per_day[-1] - avg_scores_per_day[0])

    print(f"\nResults saved to {results_path}", flush=True)
    print(f"\nAvg cosine score per day:", flush=True)
    for i, (date_str, score) in enumerate(zip(dates[:num_days], avg_scores_per_day)):
        print(f"  Day {i:2d} ({date_str}): {score:.4f}", flush=True)
    print(f"\nOverall improvement: {avg_scores_per_day[-1] - avg_scores_per_day[0]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
