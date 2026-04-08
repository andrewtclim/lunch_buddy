# models/experiments/run_cold_simulation.py
# Simulates 50 rounds of user choices with NO signup text.
# Each user starts from the average dish embedding (neutral prior).
# Tests whether the vector can converge to the hidden profile through choices alone.
# Generates a learning curve plot saved locally.

import os
import json
import mlflow
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
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
MOCK_USERS_PATH = Path(__file__).parent / "mock_users.json"
NUM_ROUNDS      = 50
ALPHA           = 0.85

# generic station labels to exclude
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

# authenticate to Vertex AI using local gcloud credentials
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def get_embedding(text: str) -> np.ndarray:
    # embed a string and return as a numpy array
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


def get_all_dishes(today: str) -> list[dict]:
    # fetch all of today's dishes with their stored embedding vectors
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients, embedding
        FROM daily_menu
        WHERE date_served = %s;
    """, (today,))
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
            "vector":      np.array([float(x) for x in row[5].strip("[]").split(",")])  # parse string to floats
        }
        for row in rows
    ]


def compute_average_embedding(dishes: list[dict]) -> np.ndarray:
    # average all dish vectors to get a neutral starting point
    # represents "no preference" — the center of the menu space
    vectors = np.array([d["vector"] for d in dishes])
    avg = vectors.mean(axis=0)
    # normalize to unit length so cosine similarity stays well-behaved
    return avg / np.linalg.norm(avg)


def rank_by_similarity(user_vec: np.ndarray, dishes: list[dict]) -> list[dict]:
    # sort dishes by cosine similarity to the user vector, highest first
    for dish in dishes:
        dot = np.dot(user_vec, dish["vector"])
        dish["_score"] = dot / (np.linalg.norm(user_vec) * np.linalg.norm(dish["vector"]))
    return sorted(dishes, key=lambda d: d["_score"], reverse=True)


def filter_placeholders(dishes: list[dict]) -> list[dict]:
    return [d for d in dishes if d["dish_name"] not in PLACEHOLDER_DISHES]


def filter_allergens(dishes: list[dict], user_allergens: list[str]) -> list[dict]:
    if not user_allergens:
        return dishes
    user_set = set(user_allergens)
    return [d for d in dishes if set(d["allergens"]).isdisjoint(user_set)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def blend_vectors(user_vec: np.ndarray, dish_vec: np.ndarray, alpha: float) -> np.ndarray:
    new_vec = alpha * user_vec + (1 - alpha) * dish_vec
    return new_vec / np.linalg.norm(new_vec)


def run_cold_simulation_for_user(
    user: dict,
    dishes: list[dict],
    avg_vec: np.ndarray,
    hidden_vec: np.ndarray
) -> list[float]:
    # run NUM_ROUNDS starting from the average embedding (no signup info)
    # returns a list of cosine scores, one per round

    user_vec = avg_vec.copy()
    round_scores = []

    # filter out placeholders and allergens once — menu is the same every round
    safe_dishes = filter_placeholders(dishes)
    safe_dishes = filter_allergens(safe_dishes, user["allergens"])

    for round_num in range(NUM_ROUNDS + 1):  # +1 to include round 0 as baseline
        # rank safe dishes by similarity to current user vector
        ranked = rank_by_similarity(user_vec, safe_dishes)
        top3 = ranked[:3]

        # score top 3 against the hidden profile
        scores = [cosine_similarity(hidden_vec, d["vector"]) for d in top3]
        avg_score = float(np.mean(scores))
        round_scores.append(avg_score)

        # round 0 is baseline, no choice made
        if round_num == 0:
            continue

        # simulate user picking the dish most similar to their hidden profile
        best = max(top3, key=lambda d: cosine_similarity(hidden_vec, d["vector"]))
        user_vec = blend_vectors(user_vec, best["vector"], ALPHA)

    return round_scores


def plot_learning_curve(avg_scores: list[float], output_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_scores)), avg_scores, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("Round")
    plt.ylabel("Avg Cosine Similarity (vs hidden profile)")
    plt.title("Cold Start Learning Curve — 15 Mock Users, 50 Rounds")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    from datetime import date
    today = date.today().isoformat()

    with open(MOCK_USERS_PATH) as f:
        users = json.load(f)

    print("Fetching all dishes and computing average embedding...")
    all_dishes = get_all_dishes(today)

    # compute the neutral starting vector once and reuse for all users
    avg_vec = compute_average_embedding(all_dishes)
    print(f"Average embedding computed from {len(all_dishes)} dishes.\n")

    print(f"Running cold start simulation: {NUM_ROUNDS} rounds, {len(users)} users, alpha={ALPHA}\n")

    all_round_scores = []
    for user in users:
        print(f"  Processing {user['user_id']}...")
        # embed the hidden profile once per user — ground truth for scoring
        hidden_vec = get_embedding(user["hidden_profile"])
        scores = run_cold_simulation_for_user(user, all_dishes, avg_vec, hidden_vec)
        all_round_scores.append(scores)

    # average scores across all users per round
    num_rounds_completed = min(len(s) for s in all_round_scores)
    avg_per_round = [
        float(np.mean([s[i] for s in all_round_scores]))
        for i in range(num_rounds_completed)
    ]

    # log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy")

    with mlflow.start_run(run_name="cold-start-simulation"):
        mlflow.log_param("num_users",   len(users))
        mlflow.log_param("num_rounds",  NUM_ROUNDS)
        mlflow.log_param("alpha",       ALPHA)
        mlflow.log_param("init_method", "average_dish_embedding")
        mlflow.log_param("date",        today)

        for i, score in enumerate(avg_per_round):
            mlflow.log_metric("avg_cosine_score", score, step=i)

        mlflow.log_metric("score_improvement", avg_per_round[-1] - avg_per_round[0])

    # save results locally
    results_path = Path(__file__).parent / "cold_simulation_results.json"
    with open(results_path, "w") as f:
        json.dump({"avg_per_round": avg_per_round, "all_user_scores": all_round_scores}, f, indent=2)

    # generate and save the learning curve plot
    plot_path = Path(__file__).parent / "cold_start_curve.png"
    plot_learning_curve(avg_per_round, plot_path)

    print(f"\nAvg cosine score per round (every 10):")
    for i in range(0, len(avg_per_round), 10):
        print(f"  Round {i:2d}: {avg_per_round[i]:.4f}")
    print(f"\nOverall improvement: {avg_per_round[-1] - avg_per_round[0]:+.4f}")


if __name__ == "__main__":
    main()
