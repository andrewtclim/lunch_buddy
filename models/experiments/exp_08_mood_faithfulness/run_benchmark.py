# models/experiments/exp_08_mood_faithfulness/run_benchmark.py
#
# Mood-faithfulness benchmark for the recommendation pipeline.
#
# Goal: measure how well the system honors the user's stated mood. The bug we
# observed was a "noodles" mood returning 0/5 noodle dishes despite multiple
# being on the menu. This benchmark quantifies that across a 37-item eval set
# and tests four retrieval-stage variants.
#
# Variants (Step 2 of the plan):
#   - baseline:     beta = production default (BETA_WITH_MOOD = 0.5),
#                   top_k_retrieval=40, top_k_gemini=10
#   - beta_0_8:     beta=0.8 (mood-dominant blend)
#   - beta_1_0:     beta=1.0 (mood-only retrieval; profile ignored at retrieval)
#   - wider_funnel: top_k_retrieval=80, top_k_gemini=20 (cheap insurance: maybe matches
#                   sit at rank 41-80)
#
# We use beta=None in the VARIANTS dict to mean "production default", and
# resolve it to the actual value (BETA_WITH_MOOD) at MLflow logging time so the
# tracked params show the resolved number rather than "default".
#
# Eval set (eval_set.json) is keyed by (mood, date, profile, relationship):
#   - adjacent (30):  profile is near but not at the mood (spicy Korean / noodles).
#                     Realistic user behavior; primary signal.
#   - contrast  (5):  profile is adversarial to the mood. Worst-case stress test.
#   - aligned   (2):  profile reinforces the mood. Sanity check that fixes don't
#                     break the easy case.
#
# Metrics:
#   - mood_faithfulness@3: fraction of top-3 recs labelled "match" by the judge (primary)
#   - mood_recall@3:       matches in top-3 / matches available on menu
#   - mean_latency_sec
#   - json_reliability:    fraction of recommend() calls that returned successfully
# Faithfulness is also sliced by `tag` (cuisine / vague / ...) and `relationship`
# (adjacent / contrast / aligned) so we can see if a fix helps differentially.
#
# DB connections reuse recommend._get_pool() so we have one connection-management
# story across the project and don't pay the ~530ms TLS handshake per fetch.
#
# MLflow structure:
#   Parent: "exp08-mood-faithfulness-{date}"
#     ├── Child: "baseline"
#     ├── Child: "beta_0_8"
#     ├── Child: "beta_1_0"
#     └── Child: "wider_funnel"
#
# Usage:
#   python run_benchmark.py                       # all 4 variants, full eval
#   python run_benchmark.py --variants baseline   # baseline only
#   python run_benchmark.py --runs 1 --limit 5    # smoke test

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import mlflow
import psycopg2
from dotenv import load_dotenv

# wire in production recommend.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "gemini_flash_rag"))
from recommend import (  # noqa: E402
    BETA_NO_MOOD,
    BETA_WITH_MOOD,
    _get_pool,
    get_embedding,
    recommend,
)

# local
from judge import label_menu  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

import recommend as _recommend_module
from google import genai as _genai

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.232.122.64:5000")

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"

# Variant kwargs passed to recommend(). beta=None means "production default";
# resolved to BETA_WITH_MOOD / BETA_NO_MOOD at runtime and logged explicitly.
VARIANTS: dict[str, dict] = {
    "baseline": {"beta": None, "top_k_retrieval": 40, "top_k_gemini": 10},
    "beta_0_8": {"beta": 0.8, "top_k_retrieval": 40, "top_k_gemini": 10},
    "beta_1_0": {"beta": 1.0, "top_k_retrieval": 40, "top_k_gemini": 10},
    "wider_funnel": {"beta": None, "top_k_retrieval": 80, "top_k_gemini": 20},
}


def resolve_beta(beta_arg: float | None, mood_present: bool) -> float:
    """Resolve a None beta to the production default that recommend() would use."""
    if beta_arg is not None:
        return beta_arg
    return BETA_WITH_MOOD if mood_present else BETA_NO_MOOD


def fetch_menu(date_str: str, table: str = "backfill_menu") -> list[dict]:
    """
    All dishes served on a given date. Pool comes from recommend._get_pool().

    Retries once on a stale-connection error: Supabase closes idle pooled
    connections, and the pool doesn't validate before handing them out. The
    first attempt may get a dead connection -- we discard it and try again.
    """
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
            # discard the stale connection from the pool so it's not handed out again
            pool.putconn(conn, close=True)
            if attempt == 0:
                print(f"  [db] retrying after stale connection: {e}", flush=True)
    raise RuntimeError(f"fetch_menu failed twice for {date_str}: {last_err}")


def score_recs(
    recs: list[dict], menu_labels: dict[str, str], available_matches: int
) -> dict:
    """
    Compute mood_faithfulness@3 and mood_recall@3 for one recommendation set.

    Lookup is by exact dish_name; falls back to lowercase match. If Gemini
    returns a dish name not on the menu, it counts as "miss" -- the user can't
    actually eat it.
    """
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
    # If no matches available on the menu, recall is undefined -- skip in aggregation
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
    profile_vec_cache: dict[str, "any"],
    runs_per: int,
) -> list[dict]:
    """Run one variant across the full eval set. Returns per-call result rows."""
    rows = []
    print(f"\n--- variant: {variant_name} ({variant_kwargs}) ---", flush=True)

    for item in eval_items:
        key = (item["mood"], item["date"])
        item_labels = menu_labels_by_key[key]
        available_matches = sum(1 for L in item_labels.values() if L == "match")
        pref_vec = profile_vec_cache[item["profile"]]

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
                    **variant_kwargs,
                )
                elapsed = time.perf_counter() - t0
                json_ok = bool(recs)
            except Exception as e:
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
                f"  [{rel_marker}] {item['mood']:30s} @ {item['date']} r{repeat}: "
                f"f={scored['faithfulness']:.2f} m={scored['match_count']}/3 "
                f"avail={available_matches} {elapsed:.1f}s",
                flush=True,
            )

    return rows


def aggregate(rows: list[dict]) -> dict:
    """Per-variant aggregate metrics, including tag and relationship slices."""
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


def main():
    parser = argparse.ArgumentParser(description="Mood-faithfulness benchmark")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()) + ["all"],
        default=["all"],
        help="which variants to run (default: all 4)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="repeats per (item, variant)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap eval items (smoke-test mode); default = full eval set",
    )
    args = parser.parse_args()

    variant_names = list(VARIANTS) if "all" in args.variants else args.variants

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    eval_items = eval_set["items"]
    if args.limit:
        eval_items = eval_items[: args.limit]

    print(
        f"Eval items: {len(eval_items)} | variants: {variant_names} | runs/cell: {args.runs}"
    )
    print(
        f"Total recommend() calls: {len(eval_items) * len(variant_names) * args.runs}\n"
    )

    # ---- step 1a: fetch all menus upfront in a tight loop ----
    # Doing all DB queries first means the pool's connections don't sit idle
    # during the long Gemini labelling phase below. (We hit a stale-connection
    # bug when fetch_menu was interleaved with label_menu.)
    print("Fetching menus...", flush=True)
    unique_dates = sorted({item["date"] for item in eval_items})
    menus_by_date: dict[str, list[dict]] = {}
    for d in unique_dates:
        menus_by_date[d] = fetch_menu(d)
        print(f"  {d}: {len(menus_by_date[d])} dishes", flush=True)

    # ---- step 1b: now label them via judge.py (cached, so reruns are free) ----
    print("\nLabelling menus via judge.py (cached)...", flush=True)
    menu_labels_by_key: dict[tuple, dict[str, str]] = {}
    for item in eval_items:
        key = (item["mood"], item["date"])
        if key in menu_labels_by_key:
            continue
        labels = label_menu(item["mood"], menus_by_date[item["date"]], verbose=False)
        menu_labels_by_key[key] = {row["dish_name"]: row["label"] for row in labels}
        n_match = sum(1 for L in menu_labels_by_key[key].values() if L == "match")
        print(
            f"  {item['mood']:30s} @ {item['date']}: {n_match} matches / {len(labels)} dishes",
            flush=True,
        )

    # ---- step 2: embed each unique profile once ----
    unique_profiles = sorted({item["profile"] for item in eval_items})
    print(f"\nEmbedding {len(unique_profiles)} unique profiles...")
    profile_vec_cache = {p: get_embedding(p) for p in unique_profiles}

    # Replace the recommend module's Vertex AI client with a fresh instance.
    # Root cause of the 400-600s stalls: the HTTP/2 connection goes cold during
    # the labelling phase (judge uses AI Studio, so Vertex AI is idle for minutes),
    # and the first recommend() call on the idle connection stalls. A new client
    # gets a fresh connection pool; subsequent calls stay warm (~2s each) so the
    # pool never goes cold again for the rest of the benchmark.
    _recommend_module.client = _genai.Client(
        vertexai=True,
        project=os.environ["PROJECT_ID"],
        location=os.environ.get("LOCATION", "us-central1"),
    )

    # ---- step 3: run each variant, log to MLflow ----
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy-rag")

    today = _date.today().isoformat()
    all_rows: list[dict] = []

    with mlflow.start_run(run_name=f"exp08-mood-faithfulness-{today}"):
        mlflow.log_param("experiment", "exp_08_mood_faithfulness")
        mlflow.log_param("eval_set_size", len(eval_items))
        mlflow.log_param("runs_per_cell", args.runs)
        mlflow.log_param("variants", json.dumps(variant_names))

        rel_counts: dict[str, int] = defaultdict(int)
        for it in eval_items:
            rel_counts[it["relationship"]] += 1
        for rel, ct in rel_counts.items():
            mlflow.log_param(f"items_rel_{rel}", ct)

        for vname in variant_names:
            variant_kwargs = VARIANTS[vname]
            with mlflow.start_run(run_name=vname, nested=True):
                mlflow.log_param("variant", vname)
                # log resolved beta so MLflow UI shows the actual number, not "default"
                resolved_beta_with_mood = resolve_beta(
                    variant_kwargs["beta"], mood_present=True
                )
                mlflow.log_param("beta", resolved_beta_with_mood)
                mlflow.log_param("top_k_retrieval", variant_kwargs["top_k_retrieval"])
                mlflow.log_param("top_k_gemini", variant_kwargs["top_k_gemini"])

                rows = run_variant(
                    variant_name=vname,
                    variant_kwargs=variant_kwargs,
                    eval_items=eval_items,
                    menu_labels_by_key=menu_labels_by_key,
                    profile_vec_cache=profile_vec_cache,
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
                    f"  -> {vname} (beta={resolved_beta_with_mood}): "
                    f"faith@3={metrics['mood_faithfulness_at_3']:.3f}  "
                    f"recall@3={metrics['mood_recall_at_3']:.3f}  "
                    f"adj={metrics.get('faithfulness_rel_adjacent', float('nan')):.3f}  "
                    f"con={metrics.get('faithfulness_rel_contrast', float('nan')):.3f}  "
                    f"aln={metrics.get('faithfulness_rel_aligned', float('nan')):.3f}  "
                    f"latency={metrics['mean_latency_sec']:.2f}s",
                    flush=True,
                )

    # ---- step 4: save raw rows for offline inspection ----
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
