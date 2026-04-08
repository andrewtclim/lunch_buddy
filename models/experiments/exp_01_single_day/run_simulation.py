# models/experiments/run_simulation.py
# Simulates 10 rounds of user choices for each mock user.
# Each round: recommend dishes, pick the best match to the hidden profile,
# blend that dish's vector into the user's preference vector, record the score.
# No LLM calls — this tests whether the vector math converges over time.

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
MOCK_USERS_PATH = Path(__file__).parent / "mock_users.json"
NUM_ROUNDS      = 10
ALPHA           = 0.85  # controls how much old preferences persist vs new choices

# generic station labels to exclude from recommendations
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
    # embed a string and return as a numpy array for easy math
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return np.array(response.embeddings[0].values)


def get_top_dishes(user_vector: np.ndarray, today: str, limit: int = 10) -> list[dict]:
    # fetch today's dishes ranked by cosine distance to the user vector
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM daily_menu
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (today, user_vector.tolist(), limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],
            "ingredients": row[4]
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
    # compute cosine similarity between two vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pick_best_dish(candidates: list[dict], hidden_vec: np.ndarray) -> dict:
    # simulate the user choosing the dish that best matches their hidden profile
    # in reality a user is noisier — this is an optimistic upper bound on convergence
    best_dish, best_score = None, -1.0
    for dish in candidates:
        dish_text = dish["dish_name"] + ". " + dish["ingredients"]
        dish_vec = get_embedding(dish_text)
        score = cosine_similarity(hidden_vec, dish_vec)
        if score > best_score:
            best_score = score
            best_dish = dish
            best_dish["_vec"] = dish_vec  # cache the vector so we don't re-embed later
    return best_dish


def blend_vectors(user_vec: np.ndarray, dish_vec: np.ndarray, alpha: float) -> np.ndarray:
    # weighted average: alpha controls how sticky the old preferences are
    new_vec = alpha * user_vec + (1 - alpha) * dish_vec
    # re-normalize to unit length so cosine similarity stays well-behaved
    return new_vec / np.linalg.norm(new_vec)


def run_simulation_for_user(user: dict, today: str) -> dict:
    # run NUM_ROUNDS of simulated choices for a single user
    # returns per-round cosine scores and the chosen dish at each round

    # embed the hidden profile once — used as ground truth throughout
    hidden_vec = get_embedding(user["hidden_profile"])

    # start from the signup text embedding (cold start)
    user_vec = get_embedding(user["signup_text"])

    round_scores = []
    chosen_dishes = []

    for round_num in range(NUM_ROUNDS + 1):  # +1 to include round 0 as baseline
        # get top dishes for the current user vector
        candidates = get_top_dishes(user_vec, today)
        candidates = filter_placeholders(candidates)
        candidates = filter_allergens(candidates, user["allergens"])[:5]

        if not candidates:
            print(f"  No safe dishes for {user['user_id']} at round {round_num}, stopping.")
            break

        # score the current top 3 against the hidden profile
        top3 = candidates[:3]
        scores = [
            cosine_similarity(hidden_vec, get_embedding(d["dish_name"] + ". " + d["ingredients"]))
            for d in top3
        ]
        avg_score = float(np.mean(scores))
        round_scores.append(avg_score)

        print(f"  {user['user_id']} round {round_num}: score={avg_score:.4f}")

        # round 0 is baseline only — no choice made yet
        if round_num == 0:
            chosen_dishes.append(None)
            continue

        # simulate the user picking the dish most similar to their hidden profile
        chosen = pick_best_dish(top3, hidden_vec)
        chosen_dishes.append(chosen["dish_name"])

        # update the user's preference vector toward the chosen dish
        user_vec = blend_vectors(user_vec, chosen["_vec"], ALPHA)

    return {
        "user_id":       user["user_id"],
        "round_scores":  round_scores,
        "chosen_dishes": chosen_dishes,
        "score_delta":   round_scores[-1] - round_scores[0] if len(round_scores) > 1 else 0.0
    }


def main():
    from datetime import date
    today = date.today().isoformat()

    with open(MOCK_USERS_PATH) as f:
        users = json.load(f)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy")

    print(f"Running simulation: {NUM_ROUNDS} rounds, {len(users)} users, alpha={ALPHA}\n")

    all_results = []
    for user in users:
        result = run_simulation_for_user(user, today)
        all_results.append(result)

    # compute per-round averages across all users
    num_rounds_completed = min(len(r["round_scores"]) for r in all_results)
    avg_scores_per_round = [
        float(np.mean([r["round_scores"][i] for r in all_results]))
        for i in range(num_rounds_completed)
    ]

    with mlflow.start_run(run_name="simulation-learning-curve"):
        mlflow.log_param("num_users",  len(users))
        mlflow.log_param("num_rounds", NUM_ROUNDS)
        mlflow.log_param("alpha",      ALPHA)
        mlflow.log_param("date",       today)

        # log per-round average score so MLflow shows the learning curve
        for i, score in enumerate(avg_scores_per_round):
            mlflow.log_metric("avg_cosine_score", score, step=i)

        # log overall improvement from round 0 to final round
        mlflow.log_metric("score_improvement", avg_scores_per_round[-1] - avg_scores_per_round[0])

    # save full results locally
    results_path = Path(__file__).parent / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"\nAvg cosine score per round:")
    for i, score in enumerate(avg_scores_per_round):
        print(f"  Round {i:2d}: {score:.4f}")
    print(f"\nOverall improvement: {avg_scores_per_round[-1] - avg_scores_per_round[0]:+.4f}")


if __name__ == "__main__":
    main()
