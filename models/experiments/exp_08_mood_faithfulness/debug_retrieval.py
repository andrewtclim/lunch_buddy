"""
Retrieval diagnostic: for a given (mood, date, profile), fetch the top-40
candidates from retrieve_dishes() and look up each dish in the judge cache.

Answers: is the failure at retrieval (few/no matches in top-40) or at
reranking (matches present but not selected by recommend())?

All judge labels come from the existing cache -- no new API calls.

Usage:
    python debug_retrieval.py
    python debug_retrieval.py --mood "a burger" --date 2026-04-05 \
        --profile "I love East Asian flavors -- stir-fries, ramen, and rice bowls."
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "gemini_flash_rag"))
from recommend import (
    BETA_WITH_MOOD,
    blend_mood,
    get_embedding,
    retrieve_dishes,
)  # noqa: E402
from judge import _load_cache, _key  # noqa: E402

LABEL_MARKER = {"match": "[+]", "weak": "[~]", "miss": "[-]", None: "[?]"}


def run(mood: str, date: str, profile: str, top_k: int, beta: float) -> None:
    print(f"\nMood:    {mood!r}")
    print(f"Date:    {date}")
    print(f"Profile: {profile!r}")
    print(f"Beta:    {beta}  (top_k={top_k})")
    print()

    profile_vec = get_embedding(profile)
    mood_vec = get_embedding(mood)
    query_vec = blend_mood(profile_vec, mood_vec, beta=beta)
    candidates = retrieve_dishes(query_vec, date, table="backfill_menu", limit=top_k)

    cache = _load_cache()
    results = []
    for rank, dish in enumerate(candidates, start=1):
        entry = cache.get(_key(mood, dish["dish_name"]))
        label = entry["label"] if entry else None
        reason = entry["reason"] if entry else "not in cache"
        results.append((rank, dish["dish_name"], label, reason))

    matches = [r for r in results if r[2] == "match"]
    weaks = [r for r in results if r[2] == "weak"]

    print(f"{'Rank':<5} {'Label':<7} {'Dish':<55} Reason")
    print("-" * 100)
    for rank, name, label, reason in results:
        marker = LABEL_MARKER.get(label, "[?]")
        print(f"{rank:<5} {marker:<7} {name:<55} {reason}")

    print()
    print(
        f"Summary: {len(matches)} match / {len(weaks)} weak out of {len(candidates)} candidates"
    )
    if matches:
        print("Matches at ranks:", [r[0] for r in matches])
    else:
        print("NO matches in retrieved set -- retrieval failure.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mood", default="low-carb")
    parser.add_argument("--date", default="2026-04-03")
    parser.add_argument(
        "--profile", default="I love hearty Italian pasta dishes and crusty bread."
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--beta", type=float, default=BETA_WITH_MOOD)
    args = parser.parse_args()
    run(args.mood, args.date, args.profile, args.top_k, args.beta)


if __name__ == "__main__":
    main()
