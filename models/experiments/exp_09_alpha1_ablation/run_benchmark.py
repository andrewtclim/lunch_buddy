# models/experiments/exp_09_alpha1_ablation/run_benchmark.py
#
# α₁ ablation: find the optimal mood weight (beta) for the 3-vector architecture.
#
# Background: exp_08 showed the production blend (beta=0.5) causes a retrieval
# dead zone when profile and mood are far apart. wider_funnel (top_k=80/20) shipped
# as the near-term fix. This experiment tests whether increasing the mood weight
# (alpha1/beta) further improves faithfulness without hurting adjacent cases.
#
# Key difference from exp_08: each eval user has a simulated 4-day pick history
# before the eval date (no-mood picks, random top-3 choice). This means every
# profile has the same history length so alpha2 is consistent across all eval
# items -- differences between variants are purely from alpha1.
#
# Simulation design (single-eval-day):
#   - Eval dates in the set: Mar 30, Apr 1, Apr 3, Apr 5, Apr 7.
#   - Sim dates = first 4 (Mar 30 → Apr 5). Each profile picks once per sim date.
#   - Eval date = Apr 7 only. All 48 eval items are on this date.
#   - Histories generated once with a fixed seed, shared across all alpha1 variants
#     so the only variable between variants is beta.
#   - On the eval day, recommend() calls three_way_blend(mood, recent, original).
#   - original_profile_vec = same embedding as pref_vec (profile text -> 768-dim vec).
#
# Variants (all use top_k=80/20 -- isolates beta as the only variable):
#   - wider_funnel:  beta=0.50  <- reference baseline (current production)
#   - alpha1_0_6:    beta=0.60
#   - alpha1_0_7:    beta=0.70
#   - alpha1_0_8:    beta=0.80
#
# Decision rule: pick the highest beta where adjacent faith@3 >= wider_funnel baseline.
# That value becomes BETA_WITH_MOOD in recommend.py.
#
# Histories are saved to sim_histories.json after the first run. Use --skip-sim
# on subsequent runs to skip the ~6 min simulation and load from that file instead.
#
# Usage:
#   python run_benchmark.py                              # all 4 variants, full eval
#   python run_benchmark.py --variants wider_funnel alpha1_0_8
#   python run_benchmark.py --runs 1 --limit 5          # smoke test
#   python run_benchmark.py --skip-sim                  # reuse saved sim histories

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import mlflow
import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "gemini_flash_rag"))
from recommend import (  # noqa: E402
    BETA_NO_MOOD,
    BETA_WITH_MOOD,
    _get_pool,
    fetch_dish_embedding,
    get_embedding,
    recommend,
)

from judge import label_menu  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

import recommend as _recommend_module
from google import genai as _genai

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.232.122.64:5000")

EVAL_SET_PATH = (
    Path(__file__).parent.parent / "exp_08_mood_faithfulness" / "eval_set.json"
)
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"
SIM_CACHE_PATH = Path(__file__).parent / "sim_histories.json"

SIM_N_DAYS = 4  # one pick per sim date; 4 sim dates gives every profile 4 picks
SIM_SEED = 42  # fixed seed so histories are reproducible across runs

VARIANTS: dict[str, dict] = {
    "wider_funnel": {"beta": None, "top_k_retrieval": 80, "top_k_gemini": 20},
    "alpha1_0_6": {"beta": 0.6, "top_k_retrieval": 80, "top_k_gemini": 20},
    "alpha1_0_7": {"beta": 0.7, "top_k_retrieval": 80, "top_k_gemini": 20},
    "alpha1_0_8": {"beta": 0.8, "top_k_retrieval": 80, "top_k_gemini": 20},
}


def resolve_beta(beta_arg: float | None, mood_present: bool) -> float:
    if beta_arg is not None:
        return beta_arg
    return BETA_WITH_MOOD if mood_present else BETA_NO_MOOD


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def get_sim_dates(all_eval_dates: list[str]) -> list[str]:
    """Return all eval dates except the last one -- these become the sim (history) dates."""
    sim = all_eval_dates[:-1]
    print(f"  Simulation dates ({len(sim)}): {sim[0]} -> {sim[-1]}", flush=True)
    return sim


def simulate_pick_history(
    profile: str,
    pref_vec: np.ndarray,
    sim_dates: list[str],
    rng: random.Random,
    table: str = "backfill_menu",
) -> list[list[float]]:
    """
    Simulate SIM_N_DAYS of no-mood picks for one profile.
    Returns recent_choices_raw (list of float lists, most-recent first) --
    the same format saved in user_pref.recent_choices.
    """
    recent_raw: list[list[float]] = []
    for date_str in sim_dates:
        try:
            recs, _, _ = recommend(
                pref_vec=pref_vec,
                preference_summary=profile,
                user_allergens=[],
                date_str=date_str,
                daily_mood=None,
                table=table,
                top_k_retrieval=80,
                top_k_gemini=20,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [sim] recommend failed on {date_str}: {e}", flush=True)
            continue

        if not recs:
            print(f"    [sim] no recs on {date_str}, skipping", flush=True)
            continue

        pick = rng.choice(recs[:3])
        dish_vec = fetch_dish_embedding(pick["dish_name"], date_str, table=table)
        if dish_vec is None:
            print(
                f"    [sim] embedding not found for {pick['dish_name']!r} on {date_str}",
                flush=True,
            )
            continue

        recent_raw.insert(0, dish_vec.tolist())  # prepend: most-recent first

    return recent_raw[:SIM_N_DAYS]


def build_pick_histories(
    profiles: list[str],
    profile_vec_cache: dict[str, np.ndarray],
    sim_dates: list[str],
) -> dict[str, list[list[float]]]:
    """
    Simulate pick histories for all unique profiles. Returns a dict mapping
    profile string -> recent_choices_raw (list of float lists, most-recent first).
    """
    rng = random.Random(SIM_SEED)
    histories: dict[str, list[list[float]]] = {}
    print(
        f"\nSimulating {SIM_N_DAYS}-day pick histories for {len(profiles)} profiles...",
        flush=True,
    )
    for i, profile in enumerate(profiles, 1):
        print(f"  [{i}/{len(profiles)}] {profile[:70]}...", flush=True)
        histories[profile] = simulate_pick_history(
            profile=profile,
            pref_vec=profile_vec_cache[profile],
            sim_dates=sim_dates,
            rng=rng,
        )
        n_picks = len(histories[profile])
        print(f"    -> {n_picks} picks simulated", flush=True)
    return histories


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def fetch_menu(date_str: str, table: str = "backfill_menu") -> list[dict]:
    pool = _get_pool()
    last_err: Exception | None = None
    for attempt in range(2):
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT dish_name, ingredients FROM {table} WHERE date_served = %s;",
                (date_str,),
            )
            rows = cur.fetchall()
            cur.close()
            pool.putconn(conn)
            return [{"dish_name": r[0], "ingredients": r[1] or ""} for r in rows]
        except psycopg2.DatabaseError as e:
            last_err = e
            pool.putconn(conn, close=True)
            if attempt == 0:
                print(f"  [db] retrying after stale connection: {e}", flush=True)
    raise RuntimeError(f"fetch_menu failed twice for {date_str}: {last_err}")


def score_recs(
    recs: list[dict], menu_labels: dict[str, str], available_matches: int
) -> dict:
    top3_labels: list[str] = []
    for r in recs[:3]:
        name = r.get("dish_name", "")
        label = menu_labels.get(name)
        if label is None:
            for k, v in menu_labels.items():
                if k.lower() == name.lower():
                    label = v
                    break
        top3_labels.append(label or "miss")

    match_count = sum(1 for L in top3_labels if L == "match")
    weak_count = sum(1 for L in top3_labels if L == "weak")
    faithfulness = match_count / 3.0
    recall = match_count / available_matches if available_matches > 0 else None

    return {
        "top3_labels": top3_labels,
        "match_count": match_count,
        "weak_count": weak_count,
        "faithfulness": faithfulness,
        "recall": recall,
    }


def run_variant(
    variant_name: str,
    variant_kwargs: dict,
    eval_items: list[dict],
    menu_labels_by_key: dict,
    profile_vec_cache: dict[str, np.ndarray],
    pick_history_cache: dict[str, list[list[float]]],
    runs_per: int,
) -> list[dict]:
    rows = []
    print(f"\n--- variant: {variant_name} ({variant_kwargs}) ---", flush=True)

    for item in eval_items:
        key = (item["mood"], item["date"])
        item_labels = menu_labels_by_key[key]
        available_matches = sum(1 for L in item_labels.values() if L == "match")
        pref_vec = profile_vec_cache[item["profile"]]

        recent_raw = pick_history_cache.get(item["profile"], [])
        recent_vecs = [np.array(v, dtype=float) for v in recent_raw]

        for repeat in range(runs_per):
            t0 = time.perf_counter()
            try:
                recs, _alts, _qv = recommend(
                    pref_vec=pref_vec,
                    preference_summary=item["profile"],
                    user_allergens=[],
                    date_str=item["date"],
                    daily_mood=item["mood"],
                    table="backfill_menu",
                    original_profile_vec=pref_vec,  # profile embedding = original signal
                    recent_choices_vecs=recent_vecs,
                    **variant_kwargs,
                )
                elapsed = time.perf_counter() - t0
                json_ok = bool(recs)
            except Exception as e:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                print(
                    f"    [error on {item['mood']!r} @ {item['date']}: {e}]", flush=True
                )
                recs = []
                json_ok = False

            scored = score_recs(recs, item_labels, available_matches)
            rows.append(
                {
                    "variant": variant_name,
                    "mood": item["mood"],
                    "date": item["date"],
                    "tag": item["tag"],
                    "relationship": item["relationship"],
                    "profile": item["profile"],
                    "n_sim_picks": len(recent_vecs),
                    "repeat": repeat,
                    "available_matches": available_matches,
                    "match_count": scored["match_count"],
                    "weak_count": scored["weak_count"],
                    "faithfulness": scored["faithfulness"],
                    "recall": scored["recall"],
                    "latency_sec": elapsed,
                    "json_ok": json_ok,
                    "top3": [r.get("dish_name", "") for r in recs[:3]],
                    "top3_labels": scored["top3_labels"],
                }
            )
            rel_marker = {"adjacent": "≈", "contrast": "✗", "aligned": "✓"}.get(
                item["relationship"], "?"
            )
            print(
                f"  [{rel_marker}] {item['mood']:30s} @ {item['date']} "
                f"(n_picks={len(recent_vecs)}) r{repeat}: "
                f"f={scored['faithfulness']:.2f} m={scored['match_count']}/3 "
                f"avail={available_matches} {elapsed:.1f}s",
                flush=True,
            )

    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    faith = [r["faithfulness"] for r in rows]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    latencies = [r["latency_sec"] for r in rows]
    json_oks = [r["json_ok"] for r in rows]

    out = {
        "n": n,
        "mood_faithfulness_at_3": statistics.mean(faith),
        "mood_recall_at_3": statistics.mean(recalls) if recalls else 0.0,
        "mean_latency_sec": statistics.mean(latencies),
        "json_reliability": sum(json_oks) / n,
    }

    by_tag: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["faithfulness"])
    for tag, vals in by_tag.items():
        out[f"faithfulness_tag_{tag}"] = statistics.mean(vals)

    by_rel: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_rel[r["relationship"]].append(r["faithfulness"])
    for rel, vals in by_rel.items():
        out[f"faithfulness_rel_{rel}"] = statistics.mean(vals)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="alpha1 ablation benchmark")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="repeats per (item, variant)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap eval items (smoke test)"
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help=f"load saved histories from {SIM_CACHE_PATH.name} instead of re-simulating",
    )
    args = parser.parse_args()

    variant_names = list(VARIANTS) if "all" in args.variants else args.variants

    eval_set = json.loads(EVAL_SET_PATH.read_text())

    # Single-eval-day design: sim on first 4 dates, evaluate only on the last.
    all_eval_dates = sorted({item["date"] for item in eval_set["items"]})
    sim_dates = get_sim_dates(all_eval_dates)  # Mar 30, Apr 1, Apr 3, Apr 5
    eval_date = all_eval_dates[-1]  # Apr 7

    eval_items = [i for i in eval_set["items"] if i["date"] == eval_date]
    if args.limit:
        eval_items = eval_items[: args.limit]

    print(f"Sim dates: {sim_dates[0]} -> {sim_dates[-1]} ({len(sim_dates)} days)")
    print(f"Eval date: {eval_date} ({len(eval_items)} items)")
    print(
        f"Variants: {variant_names} | runs/cell: {args.runs} | "
        f"benchmark calls: {len(eval_items) * len(variant_names) * args.runs}\n"
    )

    # ---- step 1: fetch and label eval date menu (all cached -- no API calls) ----
    print("Fetching eval menu...", flush=True)
    eval_menu = fetch_menu(eval_date)
    print(f"  {eval_date}: {len(eval_menu)} dishes", flush=True)

    print("\nLabelling eval menu via judge.py (cached)...", flush=True)
    menu_labels_by_key: dict[tuple, dict[str, str]] = {}
    for item in eval_items:
        key = (item["mood"], eval_date)
        if key in menu_labels_by_key:
            continue
        labels = label_menu(item["mood"], eval_menu, verbose=False)
        menu_labels_by_key[key] = {row["dish_name"]: row["label"] for row in labels}
        n_match = sum(1 for L in menu_labels_by_key[key].values() if L == "match")
        print(
            f"  {item['mood']:30s}: {n_match} matches / {len(labels)} dishes",
            flush=True,
        )

    # ---- step 2: embed profiles ----
    unique_profiles = sorted({item["profile"] for item in eval_items})
    print(f"\nEmbedding {len(unique_profiles)} unique profiles...")
    profile_vec_cache = {p: get_embedding(p) for p in unique_profiles}

    # ---- step 3: simulate pick histories (or load from cache) ----
    if args.skip_sim and SIM_CACHE_PATH.exists():
        print(f"\nLoading saved histories from {SIM_CACHE_PATH.name}...", flush=True)
        pick_history_cache: dict[str, list[list[float]]] = json.loads(
            SIM_CACHE_PATH.read_text()
        )
        missing = [p for p in unique_profiles if p not in pick_history_cache]
        if missing:
            print(f"  {len(missing)} profiles not in cache -- simulating those now")
            extra = build_pick_histories(missing, profile_vec_cache, sim_dates)
            pick_history_cache.update(extra)
    else:
        pick_history_cache = build_pick_histories(
            unique_profiles, profile_vec_cache, sim_dates
        )
        SIM_CACHE_PATH.write_text(json.dumps(pick_history_cache))
        print(f"  Histories saved to {SIM_CACHE_PATH.name}", flush=True)

    avg_picks = statistics.mean(len(v) for v in pick_history_cache.values())
    print(
        f"  Average simulated picks per profile: {avg_picks:.1f}/{SIM_N_DAYS}\n",
        flush=True,
    )

    # fresh Vertex AI client after the labelling phase to avoid HTTP/2 stall
    # (idle Vertex AI connections go cold during the judge phase -- new client
    # gets a fresh connection pool; see exp_08 run_benchmark.py for full explanation)
    _recommend_module.client = _genai.Client(
        vertexai=True,
        project=os.environ["PROJECT_ID"],
        location=os.environ.get("LOCATION", "us-central1"),
    )

    # ---- step 4: run variants, log to MLflow ----
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy-rag")

    today = _date.today().isoformat()
    all_rows: list[dict] = []

    with mlflow.start_run(run_name=f"exp09-alpha1-ablation-{today}"):
        mlflow.log_param("experiment", "exp_09_alpha1_ablation")
        mlflow.log_param("eval_set_size", len(eval_items))
        mlflow.log_param("runs_per_cell", args.runs)
        mlflow.log_param("variants", json.dumps(variant_names))
        mlflow.log_param("sim_n_days", SIM_N_DAYS)
        mlflow.log_param("sim_seed", SIM_SEED)
        mlflow.log_param("avg_sim_picks", round(avg_picks, 2))

        rel_counts: dict[str, int] = defaultdict(int)
        for it in eval_items:
            rel_counts[it["relationship"]] += 1
        for rel, ct in rel_counts.items():
            mlflow.log_param(f"items_rel_{rel}", ct)

        for vname in variant_names:
            variant_kwargs = VARIANTS[vname]
            with mlflow.start_run(run_name=vname, nested=True):
                mlflow.log_param("variant", vname)
                resolved_beta = resolve_beta(variant_kwargs["beta"], mood_present=True)
                mlflow.log_param("beta", resolved_beta)
                mlflow.log_param("top_k_retrieval", variant_kwargs["top_k_retrieval"])
                mlflow.log_param("top_k_gemini", variant_kwargs["top_k_gemini"])

                rows = run_variant(
                    variant_name=vname,
                    variant_kwargs=variant_kwargs,
                    eval_items=eval_items,
                    menu_labels_by_key=menu_labels_by_key,
                    profile_vec_cache=profile_vec_cache,
                    pick_history_cache=pick_history_cache,
                    runs_per=args.runs,
                )
                all_rows.extend(rows)

                metrics = aggregate(rows)
                for k, v in metrics.items():
                    if k == "n":
                        mlflow.log_param("n_calls", v)
                    else:
                        mlflow.log_metric(k, v)
                print(
                    f"  -> {vname} (beta={resolved_beta:.2f}): "
                    f"faith@3={metrics['mood_faithfulness_at_3']:.3f}  "
                    f"recall@3={metrics['mood_recall_at_3']:.3f}  "
                    f"adj={metrics.get('faithfulness_rel_adjacent', float('nan')):.3f}  "
                    f"con={metrics.get('faithfulness_rel_contrast', float('nan')):.3f}  "
                    f"aln={metrics.get('faithfulness_rel_aligned', float('nan')):.3f}  "
                    f"latency={metrics['mean_latency_sec']:.2f}s",
                    flush=True,
                )

    RESULTS_PATH.write_text(json.dumps(all_rows, indent=2))
    print(f"\nRaw rows saved to {RESULTS_PATH}")

    print("\n=== SUMMARY ===")
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_variant[r["variant"]].append(r)
    print(
        f"  {'variant':14s} {'beta':>5s} {'faith@3':>8s} {'recall@3':>9s} "
        f"{'adj':>6s} {'con':>6s} {'aln':>6s} {'lat(s)':>7s}"
    )
    for vname in variant_names:
        rows = by_variant.get(vname, [])
        if not rows:
            continue
        m = aggregate(rows)
        beta_resolved = resolve_beta(VARIANTS[vname]["beta"], mood_present=True)
        print(
            f"  {vname:14s} {beta_resolved:>5.2f} "
            f"{m['mood_faithfulness_at_3']:>8.3f} "
            f"{m['mood_recall_at_3']:>9.3f} "
            f"{m.get('faithfulness_rel_adjacent', float('nan')):>6.3f} "
            f"{m.get('faithfulness_rel_contrast', float('nan')):>6.3f} "
            f"{m.get('faithfulness_rel_aligned', float('nan')):>6.3f} "
            f"{m['mean_latency_sec']:>7.2f}"
        )


if __name__ == "__main__":
    main()
