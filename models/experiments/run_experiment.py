# models/experiments/run_experiment.py
# Runs the full recommendation pipeline against 15 mock users.
# Tests Gemini Flash and Pro, logs a single MLflow run with results for both.

import os
import json
import time
import mlflow
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# load env vars from fastapi/.env (DATABASE_URL, PROJECT_ID, MLFLOW_TRACKING_URI)
load_dotenv(Path(__file__).resolve().parents[2] / "fastapi" / ".env")

DATABASE_URL    = os.getenv("DATABASE_URL")
PROJECT_ID      = os.getenv("PROJECT_ID")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
LOCATION        = "us-central1"
EMBED_MODEL     = "text-embedding-004"
MOCK_USERS_PATH = Path(__file__).parent / "mock_users.json"

# authenticate to Vertex AI using local gcloud credentials (no API key needed)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# generic station labels and catch-all names that aren't real dishes
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


def get_embedding(text: str) -> list[float]:
    # convert a text string into a 768-dim vector using Google's embedding model
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return response.embeddings[0].values


def get_top_dishes(user_vector: list[float], today: str, limit: int = 10) -> list[dict]:
    # query Supabase for today's dishes ranked by cosine distance to the user vector
    # <=> is the pgvector cosine distance operator (lower = more similar)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM daily_menu
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (today, user_vector, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "dish_name":  row[0],
            "dining_hall": row[1],
            "meal_time":  row[2],
            "allergens":  row[3] or [],
            "ingredients": row[4]
        }
        for row in rows
    ]


def filter_placeholders(dishes: list[dict]) -> list[dict]:
    # remove generic station labels that aren't real dishes
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    # hard filter: drop any dish whose allergens overlap with the user's allergens
    # this runs before any scoring so allergen dishes never reach the LLM
    if not user_allergens:
        return dishes
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def build_prompt(user: dict, dishes: list[dict]) -> str:
    # build the LLM prompt from the user's signup text and their top candidate dishes
    dish_lines = "\n".join([
        f"- {d['dish_name']} ({d['dining_hall']}, {d['meal_time']}): {d['ingredients']}"
        for d in dishes
    ])
    return f"""You are a dining hall recommendation assistant.

User profile: {user['signup_text']}

Today's top matching dishes:
{dish_lines}

Give a friendly top-3 recommendation with one sentence explaining why each dish fits this user.
Format your response as a numbered list."""


def get_recommendation(prompt: str, model: str) -> tuple[str, float]:
    # call the specified Gemini model and return the response text + latency in seconds
    start = time.time()
    response = client.models.generate_content(model=model, contents=prompt)
    latency = time.time() - start
    return response.text, latency


def score_recommendation(top_dishes: list[dict], hidden_profile: str) -> float:
    # evaluate quality by comparing the hidden profile vector against each recommended dish
    # higher average cosine similarity = recommendations closer to what the user actually likes
    hidden_vec = np.array(get_embedding(hidden_profile))
    scores = []
    for dish in top_dishes:
        dish_text = dish["dish_name"] + ". " + dish["ingredients"]
        dish_vec = np.array(get_embedding(dish_text))
        cosine = np.dot(hidden_vec, dish_vec) / (
            np.linalg.norm(hidden_vec) * np.linalg.norm(dish_vec)
        )
        scores.append(cosine)
    return float(np.mean(scores))


def run_for_model(model_name: str, users: list[dict], today: str) -> list[dict]:
    # run the full pipeline for every mock user with a given model
    # returns a list of per-user result dicts
    results = []
    for user in users:
        print(f"  [{model_name}] Processing {user['user_id']}...")

        # embed the user's signup text to get their preference vector
        user_vec = get_embedding(user["signup_text"])

        # fetch top 10 dishes from Supabase by cosine similarity
        candidates = get_top_dishes(user_vec, today)

        # remove generic station labels before any scoring
        candidates = filter_placeholders(candidates)

        # remove any dishes that conflict with the user's allergens
        safe = filter_allergens(candidates, user["allergens"])[:5]

        if not safe:
            print(f"  No safe dishes found for {user['user_id']}, skipping.")
            continue

        # call the LLM to generate a top-3 recommendation from the safe candidates
        prompt = build_prompt(user, safe)
        recommendation, latency = get_recommendation(prompt, model_name)

        # score the top 3 recommended dishes against the hidden profile
        avg_score = score_recommendation(safe[:3], user["hidden_profile"])

        results.append({
            "user_id":          user["user_id"],
            "recommendation":   recommendation,
            "avg_cosine_score": avg_score,
            "latency_sec":      latency,
            "dishes_shown":     [d["dish_name"] for d in safe[:3]]
        })

    return results


def main():
    from datetime import date
    today = date.today().isoformat()  # e.g. "2026-04-06"

    # load the 15 mock users from disk
    with open(MOCK_USERS_PATH) as f:
        users = json.load(f)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy")

    # models to compare — Flash is fast and cheap, Pro is higher quality
    # using 2.5 series, the latest available on this Vertex AI project
    models_to_test = [
        ("gemini-2.5-flash", "flash"),
        ("gemini-2.5-pro",   "pro"),
    ]

    with mlflow.start_run(run_name="mock-user-experiment"):
        # log experiment-level params once
        mlflow.log_param("num_users",    len(users))
        mlflow.log_param("date",         today)
        mlflow.log_param("embed_model",  EMBED_MODEL)

        for model_name, label in models_to_test:
            print(f"\nRunning experiment with {model_name}...")
            results = run_for_model(model_name, users, today)

            # compute aggregate metrics across all users
            avg_score   = np.mean([r["avg_cosine_score"] for r in results])
            avg_latency = np.mean([r["latency_sec"] for r in results])

            # log aggregate metrics to MLflow (prefixed by model label)
            mlflow.log_metric(f"{label}_avg_cosine_score", avg_score)
            mlflow.log_metric(f"{label}_avg_latency_sec",  avg_latency)
            mlflow.log_metric(f"{label}_users_scored",     len(results))

            # save full per-user results locally (mlflow artifact upload skipped due to mlflow 3.x path issue)
            results_path = Path(__file__).parent / f"{label}_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  results saved to {results_path}")

            print(f"  avg cosine score : {avg_score:.4f}")
            print(f"  avg latency      : {avg_latency:.2f}s")

    print("\nDone. Results logged to MLflow.")


if __name__ == "__main__":
    main()
