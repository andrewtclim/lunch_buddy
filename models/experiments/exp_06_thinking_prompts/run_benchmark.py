# models/experiments/exp_06_thinking_benchmark/run_benchmark.py
#
# Benchmarks two independent variables in a 4×2 matrix:
#   Axis 1 — Model/thinking config (4 levels):
#     A: gemini-2.5-flash, default thinking
#     B: gemini-2.5-flash, thinking_budget=1024
#     C: gemini-2.5-flash, thinking_budget=0
#     D: gemini-2.0-flash (non-thinking architecture)
#
#   Axis 2 — Prompt version (2 levels):
#     v1: current prompt (profile-dominant, mood secondary)
#     v2: restructured prompt (mood-primary when given, profile as tiebreaker)
#
# This gives 8 combinations per run. Each combination is repeated N times
# (default 3) to measure variance. All 8 share the same candidate dish set
# so differences are purely from the model/prompt change.
#
# MLflow structure:
#   Parent run: "exp06-thinking-benchmark-{date}"
#     ├── Child: "A_baseline × v1"   (params: model, budget, prompt_version)
#     ├── Child: "A_baseline × v2"
#     ├── Child: "B_reduced × v1"
#     │   ...
#     └── Child: "D_2_0_flash × v2"
#
#   Parent-level metrics for quick comparison:
#     - {config}_{prompt}_avg_latency  (8 values — full matrix)
#     - avg_latency_by_config_{config} (4 values — marginal over prompts)
#     - avg_latency_by_prompt_{prompt} (2 values — marginal over configs)
#     - overlap_{config_a}_vs_{config_b}_{prompt}  (pairwise agreement)
#
# Usage:
#   python run_benchmark.py --date 2026-04-01 --table backfill_menu --mood "something light"
#   python run_benchmark.py --date 2026-04-11 --table daily_menu --runs 5
#   python run_benchmark.py --date 2026-04-01 --table backfill_menu  # no mood (v2 prompt falls back to v1 behavior)
#
# Requires: models/.env with DATABASE_URL, PROJECT_ID, LOCATION, MLFLOW_TRACKING_URI

import os
import sys
import json
import time
import argparse
import mlflow
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# add gemini_flash_rag/ to path so we can import recommend.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "gemini_flash_rag"))

from recommend import (
    get_embedding,
    retrieve_dishes,
    filter_placeholders,
    filter_allergens,
    deduplicate,
    blend_mood,
)

# load env vars from models/.env (two levels up from exp_06_thinking_benchmark/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.232.122.64:5000")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ---------------------------------------------------------------------------
# Axis 1: Model / thinking configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "A_baseline": {
        "model": "gemini-2.5-flash",
        "thinking_budget": None,  # default -- model decides
        "label": "2.5-flash-default",
    },
    "B_reduced": {
        "model": "gemini-2.5-flash",
        "thinking_budget": 1024,  # light reasoning
        "label": "2.5-flash-budget-1024",
    },
    "C_no_thinking": {
        "model": "gemini-2.5-flash",
        "thinking_budget": 0,  # no thinking at all
        "label": "2.5-flash-budget-0",
    },
    "D_2_0_flash": {
        "model": "gemini-2.0-flash",
        "thinking_budget": None,  # 2.0 has no thinking mode
        "label": "2.0-flash",
    },
}


# ---------------------------------------------------------------------------
# Axis 2: Prompt versions
# ---------------------------------------------------------------------------


def build_prompt_v1(
    preference_summary: str, candidates: list[dict], daily_mood: str | None = None
) -> str:
    """Current prompt -- profile is the primary signal, mood is appended."""
    dish_lines = "\n".join(
        [
            f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
            f" -- Ingredients: {d['ingredients']}"
            for i, d in enumerate(candidates)
        ]
    )

    mood_line = f"\nToday's mood: {daily_mood}" if daily_mood else ""

    return f"""You are a Stanford dining hall recommender.

User preference: {preference_summary}{mood_line}

Available dishes today:
{dish_lines}

From the list above, return two things:

1. "recommendations" -- the top 3 dishes that best match the user's preference\
{ " and today's mood" if daily_mood else ""}. \
For each include: dish name, dining hall, and one concise reason why it fits. \
If the user provided a mood, acknowledge it briefly in the reason where relevant.

2. "alternatives" -- 2 dishes from the remaining list that are meaningfully different \
in cuisine or style from the top 3 (to broaden the user's options). \
For each include: dish name, dining hall, and one concise reason it's worth trying.

Only use dishes from the list above -- do not invent dishes.
No dish should appear in both recommendations and alternatives.
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


def build_prompt_v2(
    preference_summary: str, candidates: list[dict], daily_mood: str | None = None
) -> str:
    """Restructured prompt -- mood leads when provided, profile is tiebreaker.
    When no mood is given, falls back to the same profile-driven structure as v1."""
    dish_lines = "\n".join(
        [
            f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
            f" -- Ingredients: {d['ingredients']}"
            for i, d in enumerate(candidates)
        ]
    )

    if daily_mood:
        return f"""You are a Stanford dining hall recommender.

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

Only use dishes from the list above -- do not invent dishes.
No dish should appear in both recommendations and alternatives.
For dish_name use ONLY the dish name (e.g. "Gochujang Spiced Chicken"), not the dining hall or meal time.
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
    else:
        # no mood -- v2 falls back to profile-driven (same as v1)
        return build_prompt_v1(preference_summary, candidates, daily_mood=None)


PROMPT_VERSIONS = {
    "v1": {
        "builder": build_prompt_v1,
        "label": "prompt-v1-profile-dominant",
    },
    "v2": {
        "builder": build_prompt_v2,
        "label": "prompt-v2-mood-primary",
    },
}


# ---------------------------------------------------------------------------
# Single Gemini call with timing
# ---------------------------------------------------------------------------


def call_gemini(prompt: str, model_config: dict) -> tuple[dict | None, float]:
    """Call Gemini with a specific model config. Returns (parsed_result, seconds)."""
    gen_config = {"response_mime_type": "application/json"}

    if model_config["thinking_budget"] is not None:
        gen_config["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=model_config["thinking_budget"]
        )

    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_config["model"],
            contents=prompt,
            config=gen_config,
        )
        elapsed = time.perf_counter() - start
        result = json.loads(response.text)
        return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"    ERROR: {e}")
        return None, elapsed


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    date_str: str,
    table: str,
    preference_summary: str,
    allergens: list[str],
    daily_mood: str | None = None,
    runs_per_config: int = 3,
):
    """Run the full 4×2 matrix, log to MLflow, save results locally."""

    print(f"\n{'='*70}")
    print(f"exp_06 — Thinking Budget × Prompt Version Benchmark")
    print(f"{'='*70}")
    print(f"Date: {date_str}  |  Table: {table}  |  Mood: {daily_mood or '(none)'}")
    print(f'Profile: "{preference_summary}"')
    print(
        f"Matrix: {len(MODEL_CONFIGS)} model configs × {len(PROMPT_VERSIONS)} prompts"
        f" × {runs_per_config} runs = {len(MODEL_CONFIGS) * len(PROMPT_VERSIONS) * runs_per_config} total calls"
    )
    print(f"{'='*70}\n")

    if not daily_mood:
        print("NOTE: no --mood provided. v2 prompt falls back to v1 behavior,")
        print(
            "so prompt differences will only show up in model/thinking comparisons.\n"
        )

    # ---- step 1: prepare candidates (shared across ALL combinations) ----
    print("Preparing candidates...", flush=True)
    pref_vec = get_embedding(preference_summary)

    if daily_mood:
        mood_vec = get_embedding(daily_mood)
        query_vec = blend_mood(pref_vec, mood_vec, beta=0.3)
    else:
        query_vec = pref_vec

    candidates = retrieve_dishes(query_vec, date_str, table=table, limit=40)
    candidates = filter_placeholders(candidates)
    candidates = filter_allergens(candidates, allergens)
    candidates = deduplicate(candidates)[:10]

    candidate_names = [d["dish_name"] for d in candidates]
    print(f"Candidates ({len(candidates)} dishes):")
    for i, d in enumerate(candidates):
        print(f"  {i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})")
    print()

    # ---- step 2: build both prompts (shared across model configs) ----
    prompts = {}
    for pv_name, pv in PROMPT_VERSIONS.items():
        prompts[pv_name] = pv["builder"](preference_summary, candidates, daily_mood)

    # ---- step 3: MLflow setup ----
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy-rag")

    # results[config_name][prompt_version] = {...}
    results = {}

    with mlflow.start_run(run_name=f"exp06-thinking-benchmark-{date_str}"):

        # parent-level params
        mlflow.log_param("experiment", "exp_06_thinking_benchmark")
        mlflow.log_param("date", date_str)
        mlflow.log_param("table", table)
        mlflow.log_param("preference_summary", preference_summary)
        mlflow.log_param("daily_mood", daily_mood or "none")
        mlflow.log_param("runs_per_config", runs_per_config)
        mlflow.log_param("num_candidates", len(candidates))
        mlflow.log_param("candidate_dishes", json.dumps(candidate_names))
        mlflow.log_param(
            "matrix_size",
            f"{len(MODEL_CONFIGS)}×{len(PROMPT_VERSIONS)}×{runs_per_config}",
        )

        # ---- step 4: run the full matrix ----
        for config_name, model_config in MODEL_CONFIGS.items():
            results[config_name] = {}

            for pv_name, pv in PROMPT_VERSIONS.items():
                combo_label = f"{model_config['label']} × {pv_name}"
                print(f"--- {combo_label} ---")

                with mlflow.start_run(run_name=combo_label, nested=True):

                    # child-level params — fully describe this cell of the matrix
                    mlflow.log_param("model", model_config["model"])
                    mlflow.log_param(
                        "thinking_budget",
                        (
                            model_config["thinking_budget"]
                            if model_config["thinking_budget"] is not None
                            else "default"
                        ),
                    )
                    mlflow.log_param("config_name", config_name)
                    mlflow.log_param("prompt_version", pv_name)
                    mlflow.log_param("prompt_label", pv["label"])

                    latencies = []
                    outputs = []

                    for run_idx in range(runs_per_config):
                        result, elapsed = call_gemini(prompts[pv_name], model_config)
                        latencies.append(elapsed)

                        mlflow.log_metric("latency_sec", elapsed, step=run_idx)

                        if result:
                            recs = result.get("recommendations", [])
                            rec_names = [r["dish_name"] for r in recs]
                            print(
                                f"  Run {run_idx+1}: {elapsed:.2f}s  recs={rec_names}"
                            )
                            outputs.append(result)
                            mlflow.log_metric("num_recs", len(recs), step=run_idx)
                        else:
                            print(f"  Run {run_idx+1}: {elapsed:.2f}s  FAILED")
                            mlflow.log_metric("num_recs", 0, step=run_idx)

                        if run_idx < runs_per_config - 1:
                            time.sleep(1)

                    # aggregate metrics for this cell
                    avg_latency = float(np.mean(latencies))
                    std_latency = float(np.std(latencies))
                    min_latency = float(np.min(latencies))
                    max_latency = float(np.max(latencies))
                    success_rate = len(outputs) / runs_per_config

                    mlflow.log_metric("avg_latency_sec", avg_latency)
                    mlflow.log_metric("std_latency_sec", std_latency)
                    mlflow.log_metric("min_latency_sec", min_latency)
                    mlflow.log_metric("max_latency_sec", max_latency)
                    mlflow.log_metric("success_rate", success_rate)

                    if outputs:
                        first_recs = [
                            r["dish_name"]
                            for r in outputs[0].get("recommendations", [])
                        ]
                        first_alts = [
                            r["dish_name"] for r in outputs[0].get("alternatives", [])
                        ]
                        mlflow.log_param("first_run_recs", json.dumps(first_recs))
                        mlflow.log_param("first_run_alts", json.dumps(first_alts))

                    results[config_name][pv_name] = {
                        "label": combo_label,
                        "model": model_config["model"],
                        "thinking_budget": model_config["thinking_budget"],
                        "prompt_version": pv_name,
                        "avg_latency": round(avg_latency, 2),
                        "std_latency": round(std_latency, 2),
                        "min_latency": round(min_latency, 2),
                        "max_latency": round(max_latency, 2),
                        "latencies": [round(l, 2) for l in latencies],
                        "outputs": outputs,
                        "success_rate": success_rate,
                    }

                    print(f"  Avg: {avg_latency:.2f}s (±{std_latency:.2f}s)\n")

        # ---- step 5: parent-level marginal metrics ----
        # these let you isolate each axis's effect from the MLflow UI

        # per-cell metrics (full matrix)
        for cfg_name, pv_results in results.items():
            for pv_name, r in pv_results.items():
                cfg_label = MODEL_CONFIGS[cfg_name]["label"]
                mlflow.log_metric(
                    f"{cfg_label}_{pv_name}_avg_latency", r["avg_latency"]
                )

        # marginal over prompt versions → isolates thinking effect
        for cfg_name in MODEL_CONFIGS:
            cfg_label = MODEL_CONFIGS[cfg_name]["label"]
            cell_latencies = [
                results[cfg_name][pv]["avg_latency"]
                for pv in PROMPT_VERSIONS
                if results[cfg_name].get(pv)
            ]
            if cell_latencies:
                mlflow.log_metric(
                    f"marginal_config_{cfg_label}", float(np.mean(cell_latencies))
                )

        # marginal over model configs → isolates prompt effect
        for pv_name in PROMPT_VERSIONS:
            cell_latencies = [
                results[cfg][pv_name]["avg_latency"]
                for cfg in MODEL_CONFIGS
                if results.get(cfg, {}).get(pv_name)
            ]
            if cell_latencies:
                mlflow.log_metric(
                    f"marginal_prompt_{pv_name}", float(np.mean(cell_latencies))
                )

        # pairwise recommendation overlap within each prompt version
        # (shows whether cheaper configs agree with baseline)
        for pv_name in PROMPT_VERSIONS:
            rec_sets = {}
            for cfg_name in MODEL_CONFIGS:
                cell = results.get(cfg_name, {}).get(pv_name)
                if cell and cell["outputs"]:
                    recs = cell["outputs"][0].get("recommendations", [])
                    rec_sets[cfg_name] = frozenset(d["dish_name"] for d in recs)

            cfg_names = list(rec_sets.keys())
            for i in range(len(cfg_names)):
                for j in range(i + 1, len(cfg_names)):
                    overlap = len(rec_sets[cfg_names[i]] & rec_sets[cfg_names[j]])
                    li = MODEL_CONFIGS[cfg_names[i]]["label"]
                    lj = MODEL_CONFIGS[cfg_names[j]]["label"]
                    key = f"overlap_{li}_vs_{lj}_{pv_name}"
                    mlflow.log_metric(key[:250], overlap)

        # cross-prompt overlap per config
        # (shows whether v2 produces different picks from v1 for the same model)
        for cfg_name in MODEL_CONFIGS:
            sets = {}
            for pv_name in PROMPT_VERSIONS:
                cell = results.get(cfg_name, {}).get(pv_name)
                if cell and cell["outputs"]:
                    recs = cell["outputs"][0].get("recommendations", [])
                    sets[pv_name] = frozenset(d["dish_name"] for d in recs)
            if len(sets) == 2:
                overlap = len(sets["v1"] & sets["v2"])
                cfg_label = MODEL_CONFIGS[cfg_name]["label"]
                mlflow.log_metric(f"v1_vs_v2_overlap_{cfg_label}", overlap)

    # ---- step 6: print results ----

    print(f"\n{'='*70}")
    print("FULL MATRIX RESULTS")
    print(f"{'='*70}")
    print(
        f"{'Config':<25} {'Prompt':<6} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'OK':>6}"
    )
    print("-" * 73)
    for cfg_name, pv_results in results.items():
        for pv_name, r in pv_results.items():
            cfg_label = MODEL_CONFIGS[cfg_name]["label"]
            print(
                f"{cfg_label:<25} {pv_name:<6} {r['avg_latency']:>7.2f}s "
                f"{r['std_latency']:>7.2f}s {r['min_latency']:>7.2f}s "
                f"{r['max_latency']:>7.2f}s {r['success_rate']:>5.0%}"
            )

    # marginal analysis
    print(f"\n--- Marginal: avg latency by model config (averaged over prompts) ---")
    print(f"    (isolates thinking effect)")
    for cfg_name in MODEL_CONFIGS:
        cfg_label = MODEL_CONFIGS[cfg_name]["label"]
        lats = [
            results[cfg_name][pv]["avg_latency"]
            for pv in PROMPT_VERSIONS
            if results[cfg_name].get(pv)
        ]
        if lats:
            print(f"  {cfg_label:<25} {np.mean(lats):>7.2f}s")

    print(f"\n--- Marginal: avg latency by prompt version (averaged over configs) ---")
    print(f"    (isolates prompt effect)")
    for pv_name in PROMPT_VERSIONS:
        lats = [
            results[cfg][pv_name]["avg_latency"]
            for cfg in MODEL_CONFIGS
            if results.get(cfg, {}).get(pv_name)
        ]
        if lats:
            print(f"  {pv_name:<25} {np.mean(lats):>7.2f}s")

    # recommendation comparison
    print(f"\n--- Recommendations: v1 vs v2 per config ---")
    for cfg_name in MODEL_CONFIGS:
        cfg_label = MODEL_CONFIGS[cfg_name]["label"]
        v1_cell = results.get(cfg_name, {}).get("v1")
        v2_cell = results.get(cfg_name, {}).get("v2")
        if v1_cell and v1_cell["outputs"] and v2_cell and v2_cell["outputs"]:
            v1_recs = sorted(
                d["dish_name"] for d in v1_cell["outputs"][0].get("recommendations", [])
            )
            v2_recs = sorted(
                d["dish_name"] for d in v2_cell["outputs"][0].get("recommendations", [])
            )
            overlap = len(set(v1_recs) & set(v2_recs))
            print(f"  {cfg_label}:")
            print(f"    v1: {v1_recs}")
            print(f"    v2: {v2_recs}")
            print(f"    overlap: {overlap}/3")

    # save results locally
    out_path = Path(__file__).parent / "benchmark_results.json"
    results_slim = {}
    for cfg_name, pv_results in results.items():
        results_slim[cfg_name] = {}
        for pv_name, r in pv_results.items():
            results_slim[cfg_name][pv_name] = {
                k: v for k, v in r.items() if k != "outputs"
            }
    with open(out_path, "w") as f:
        json.dump(results_slim, f, indent=2)
    print(f"\nResults saved to {out_path}")

    full_out_path = Path(__file__).parent / "benchmark_full_outputs.json"
    serializable = {}
    for cfg_name, pv_results in results.items():
        serializable[cfg_name] = {}
        for pv_name, r in pv_results.items():
            entry = dict(r)
            entry["latencies"] = [float(l) for l in entry["latencies"]]
            serializable[cfg_name][pv_name] = entry
    with open(full_out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Full outputs saved to {full_out_path}")

    print(f"\nLogged to MLflow experiment 'lunch-buddy-rag'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exp_06: Benchmark thinking budget × prompt version"
    )
    parser.add_argument(
        "--date", default="2026-04-11", help="Date to pull dishes for (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--table", default="daily_menu", choices=["daily_menu", "backfill_menu"]
    )
    parser.add_argument(
        "--mood",
        default=None,
        help="Optional daily mood string (needed to see v2 prompt effect)",
    )
    parser.add_argument(
        "--profile",
        default="I enjoy spicy and asian foods",
        help="Preference summary text",
    )
    parser.add_argument("--allergens", default="", help="Comma-separated allergens")
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per combination (default 3)"
    )
    args = parser.parse_args()

    allergens = [a.strip() for a in args.allergens.split(",") if a.strip()]

    run_benchmark(
        date_str=args.date,
        table=args.table,
        preference_summary=args.profile,
        allergens=allergens,
        daily_mood=args.mood,
        runs_per_config=args.runs,
    )
