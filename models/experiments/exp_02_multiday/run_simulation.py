# models/experiments/exp_02_multiday/run_simulation.py
# Multi-day simulation: runs one round per day across all dates in backfill_menu.
# More realistic than exp_01 — each day has a different real menu from GCS.
# Flow per user: start from signup vector, pick best dish each day, blend vector,
# track cosine score improvement over 10 days.

import os
import json
import mlflow
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# load env vars from fastapi/.env
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

DATABASE_URL    = os.getenv("DATABASE_URL")
PROJECT_ID      = os.getenv("PROJECT_ID")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
LOCATION        = "us-central1"
EMBED_MODEL     = "text-embedding-004"
ALPHA           = 0.85   # how much old preferences persist vs new choice
MOCK_USERS_PATH = Path(__file__).parent.parent / "exp_01_single_day" / "mock_users.json"

# generic station labels to exclude — same set as exp_01
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

# initialize Vertex AI client once — reused for all embedding calls
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


def get_top_dishes(user_vector: np.ndarray, date_str: str, limit: int = 10) -> list[dict]:
    # fetch dishes for a specific date ranked by cosine distance to the user vector
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM backfill_menu
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (date_str, user_vector.tolist(), limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],
            "ingredients": row[4],
        }
        for row in rows
    ]


def filter_placeholders(dishes: list[dict]) -> list[dict]:
    # remove generic station labels that aren't real dishes
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    # hard filter: drop any dish that shares an allergen with the user
    if not user_allergens:
        return dishes
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # cosine similarity between two vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pick_best_dish(candidates: list[dict], hidden_vec: np.ndarray) -> dict:
    # simulate the user picking the dish closest to their hidden profile
    best_dish, best_score = None, -1.0
    for dish in candidates:
        dish_text = dish["dish_name"] + ". " + dish["ingredients"]
        dish_vec  = get_embedding(dish_text)
        score     = cosine_similarity(hidden_vec, dish_vec)
        if score > best_score:
            best_score      = score
            best_dish       = dish
            best_dish["_vec"] = dish_vec   # cache so we don't re-embed later
    return best_dish


def blend_vectors(user_vec: np.ndarray, dish_vec: np.ndarray, alpha: float) -> np.ndarray:
    # weighted average — alpha controls how sticky the old preferences are
    new_vec = alpha * user_vec + (1 - alpha) * dish_vec
    return new_vec / np.linalg.norm(new_vec)   # re-normalize to unit length


def run_simulation_for_user(user: dict, dates: list[str]) -> dict:
    # run one round per day for a single user across all available dates
    hidden_vec = get_embedding(user["hidden_profile"])   # ground truth, fixed
    user_vec   = get_embedding(user["signup_text"])      # starting point

    day_scores    = []   # one score per day
    chosen_dishes = []   # which dish was picked each day

    for day_num, date_str in enumerate(dates):
        # get and filter today's candidates
        candidates = get_top_dishes(user_vec, date_str)
        candidates = filter_placeholders(candidates)
        candidates = filter_allergens(candidates, user["allergens"])[:5]

        if not candidates:
            print(f"  No safe dishes for {user['user_id']} on {date_str}, skipping.")
            continue

        # score top 3 against hidden profile — measures recommendation quality
        top3   = candidates[:3]
        scores = [
            cosine_similarity(hidden_vec, get_embedding(d["dish_name"] + ". " + (d["ingredients"] or "")))
            for d in top3
        ]
        avg_score = float(np.mean(scores))
        day_scores.append(avg_score)

        print(f"  {user['user_id']} day {day_num} ({date_str}): score={avg_score:.4f}")

        # simulate the user picking the dish most similar to their hidden profile
        chosen = pick_best_dish(top3, hidden_vec)
        chosen_dishes.append(chosen["dish_name"])

        # update the user's preference vector toward the chosen dish
        user_vec = blend_vectors(user_vec, chosen["_vec"], ALPHA)

    return {
        "user_id":       user["user_id"],
        "day_scores":    day_scores,
        "chosen_dishes": chosen_dishes,
        "score_delta":   day_scores[-1] - day_scores[0] if len(day_scores) > 1 else 0.0,
    }


def main():
    with open(MOCK_USERS_PATH) as f:
        users = json.load(f)

    # pull available dates from the DB
    dates = get_available_dates()
    print(f"Dates available: {dates[0]} → {dates[-1]} ({len(dates)} days)")
    print(f"Running simulation: {len(dates)} days, {len(users)} users, alpha={ALPHA}\n")

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

    # save results locally — mlflow.log_artifact fails on Mac with mlflow 3.x
    results_path = Path(__file__).parent / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)       # point at the GCP MLflow server
    mlflow.set_experiment("lunch-buddy")       # same experiment as exp_01

    with mlflow.start_run(run_name="exp02-multiday-simulation"):
        # params — describe what this run is
        mlflow.log_param("num_users",   len(users))
        mlflow.log_param("num_days",    len(dates))
        mlflow.log_param("alpha",       ALPHA)
        mlflow.log_param("date_range",  f"{dates[0]} → {dates[-1]}")
        mlflow.log_param("experiment",  "exp_02_multiday")

        # per-day average cosine score — step = day index, shows learning curve
        for i, score in enumerate(avg_scores_per_day):
            mlflow.log_metric("avg_cosine_score", score, step=i)

        # overall improvement from day 0 to final day
        mlflow.log_metric("score_improvement", avg_scores_per_day[-1] - avg_scores_per_day[0])

    print(f"\nResults saved to {results_path}")
    print(f"\nAvg cosine score per day:")
    for i, (date_str, score) in enumerate(zip(dates[:num_days], avg_scores_per_day)):
        print(f"  Day {i:2d} ({date_str}): {score:.4f}")
    print(f"\nOverall improvement: {avg_scores_per_day[-1] - avg_scores_per_day[0]:+.4f}")


if __name__ == "__main__":
    main()
