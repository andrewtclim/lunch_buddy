# models/experiments/exp_07_connection_pooling/run_benchmark.py
#
# Benchmarks two connection strategies for the Supabase DB layer:
#   no_pool  — fresh psycopg2 connection per call (current behavior in recommend.py)
#   with_pool — ThreadedConnectionPool initialized once, connections reused
#
# Why isolate the DB layer?
#   Gemini calls take 2–14s (exp_06 data). Connection overhead is ~50–150ms.
#   Including Gemini would swamp the signal. We call retrieve_dishes() directly
#   with a pre-fetched embedding so each run only pays the DB round-trip cost.
#
# Requires: models/.env with DATABASE_URL, PROJECT_ID, LOCATION, MLFLOW_TRACKING_URI

import os
import sys
import json
import time
import argparse
import mlflow
import numpy as np
import psycopg2
import psycopg2.pool
from pathlib import Path
from dotenv import load_dotenv

# add gemini_flash_rag/ to path so we can import recommend.py utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "gemini_flash_rag"))

from recommend import get_embedding, DATABASE_URL

# load env vars from models/.env (recommend.py loads it too on import, but this
# ensures MLFLOW_TRACKING_URI is available before we call mlflow.set_tracking_uri)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://35.232.122.64:5000")

# table name is injected via f-string (not a psycopg2 parameter) — safe because
# argparse enforces choices=["daily_menu", "backfill_menu"] before it reaches here
RETRIEVE_QUERY = """
    SELECT dish_name, dining_hall, meal_time, allergens, ingredients
    FROM {table}
    WHERE date_served = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""


# ---------------------------------------------------------------------------
# Two versions of the same query — one fresh connection, one pooled
# ---------------------------------------------------------------------------


def retrieve_no_pool(
    query_vec: np.ndarray, date_str: str, table: str, limit: int = 20
) -> list:
    """Open a new connection, run the query, close. Mirrors current recommend.py behavior."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            RETRIEVE_QUERY.format(table=table), (date_str, query_vec.tolist(), limit)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return rows


def retrieve_with_pool(
    pool: psycopg2.pool.ThreadedConnectionPool,
    query_vec: np.ndarray,
    date_str: str,
    table: str,
    limit: int = 20,
) -> list:
    """Borrow a connection from the pool, run the query, return the connection."""
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(
            RETRIEVE_QUERY.format(table=table), (date_str, query_vec.tolist(), limit)
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        # always return the connection — even on exception — to prevent pool exhaustion
        pool.putconn(conn)
    return rows


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------


def summarize(latencies: list[float]) -> dict:
    arr = np.array(latencies)
    return {
        "avg_ms": round(float(np.mean(arr)) * 1000, 2),
        "std_ms": round(float(np.std(arr)) * 1000, 2),
        "min_ms": round(float(np.min(arr)) * 1000, 2),
        "max_ms": round(float(np.max(arr)) * 1000, 2),
        "p50_ms": round(float(np.percentile(arr, 50)) * 1000, 2),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 2),
        "latencies_ms": [round(l * 1000, 2) for l in latencies],
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(date_str: str, table: str, profile: str, runs: int):
    print(f"\n{'='*60}")
    print("exp_07 — Connection Pooling Latency Benchmark")
    print(f"{'='*60}")
    print(f"Date: {date_str}  |  Table: {table}  |  Runs per condition: {runs}")
    print(f'Profile: "{profile}"')
    print(f"{'='*60}\n")

    # one Gemini embedding call up front — reused for all N benchmark calls
    # this keeps Gemini latency out of the DB measurement
    print("Embedding profile (one Gemini call, reused for all runs)...", flush=True)
    query_vec = get_embedding(profile)
    print("Done.\n")

    # --- condition 1: no pool ---
    print(f"Condition 1/2 — no_pool ({runs} calls)", flush=True)
    no_pool_latencies = []
    for i in range(runs):
        start = time.perf_counter()
        retrieve_no_pool(query_vec, date_str, table)
        elapsed = time.perf_counter() - start
        no_pool_latencies.append(elapsed)
        print(f"  Call {i+1:>2}: {elapsed*1000:6.1f}ms", flush=True)

    # --- condition 2: with pool ---
    # minconn=1: one connection stays alive after the first call
    # maxconn=5: headroom if multiple threads need connections simultaneously
    print(f"\nCondition 2/2 — with_pool ({runs} calls)", flush=True)
    pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
    pool_latencies = []
    try:
        for i in range(runs):
            start = time.perf_counter()
            retrieve_with_pool(pool, query_vec, date_str, table)
            elapsed = time.perf_counter() - start
            pool_latencies.append(elapsed)
            print(f"  Call {i+1:>2}: {elapsed*1000:6.1f}ms", flush=True)
    finally:
        pool.closeall()

    # --- compute stats ---
    no_pool_stats = summarize(no_pool_latencies)
    pool_stats = summarize(pool_latencies)
    speedup = (
        no_pool_stats["avg_ms"] / pool_stats["avg_ms"]
        if pool_stats["avg_ms"] > 0
        else 0
    )
    savings_ms = no_pool_stats["avg_ms"] - pool_stats["avg_ms"]

    # --- print results table ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(
        f"{'Condition':<12} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'P50':>8} {'P95':>8}"
    )
    print("-" * 68)
    for label, s in [("no_pool", no_pool_stats), ("with_pool", pool_stats)]:
        print(
            f"{label:<12} {s['avg_ms']:>7.1f}ms {s['std_ms']:>7.1f}ms "
            f"{s['min_ms']:>7.1f}ms {s['max_ms']:>7.1f}ms "
            f"{s['p50_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms"
        )
    print(f"\nSpeedup: {speedup:.1f}x  |  Avg savings: {savings_ms:.1f}ms per call")
    if runs > 1:
        # show first vs rest for the pool condition to illustrate the warmup effect —
        # call 1 pays connection setup; calls 2+ reuse it
        first_ms = pool_latencies[0] * 1000
        rest_avg_ms = np.mean(pool_latencies[1:]) * 1000
        print(
            f"Pool warmup: call 1 = {first_ms:.1f}ms, calls 2–{runs} avg = {rest_avg_ms:.1f}ms"
        )

    # --- MLflow logging ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("lunch-buddy-rag")

    with mlflow.start_run(run_name=f"exp07-connection-pooling-{date_str}"):
        mlflow.log_param("experiment", "exp_07_connection_pooling")
        mlflow.log_param("date", date_str)
        mlflow.log_param("table", table)
        mlflow.log_param("profile", profile)
        mlflow.log_param("runs_per_condition", runs)
        mlflow.log_param("pool_minconn", 1)
        mlflow.log_param("pool_maxconn", 5)

        # parent-level summary for quick comparison in the MLflow UI
        mlflow.log_metric("no_pool_avg_ms", no_pool_stats["avg_ms"])
        mlflow.log_metric("with_pool_avg_ms", pool_stats["avg_ms"])
        mlflow.log_metric("speedup", round(speedup, 2))
        mlflow.log_metric("avg_savings_ms", round(savings_ms, 1))

        with mlflow.start_run(run_name="no_pool", nested=True):
            mlflow.log_param("condition", "no_pool")
            for i, lat in enumerate(no_pool_latencies):
                mlflow.log_metric("latency_ms", lat * 1000, step=i)
            mlflow.log_metric("avg_latency_ms", no_pool_stats["avg_ms"])
            mlflow.log_metric("p95_latency_ms", no_pool_stats["p95_ms"])

        with mlflow.start_run(run_name="with_pool", nested=True):
            mlflow.log_param("condition", "with_pool")
            for i, lat in enumerate(pool_latencies):
                mlflow.log_metric("latency_ms", lat * 1000, step=i)
            mlflow.log_metric("avg_latency_ms", pool_stats["avg_ms"])
            mlflow.log_metric("p95_latency_ms", pool_stats["p95_ms"])

    # --- save to JSON ---
    results = {
        "no_pool": no_pool_stats,
        "with_pool": pool_stats,
        "speedup": round(speedup, 2),
        "avg_savings_ms": round(savings_ms, 1),
        "meta": {
            "date": date_str,
            "table": table,
            "profile": profile,
            "runs_per_condition": runs,
        },
    }
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Logged to MLflow experiment 'lunch-buddy-rag'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exp_07: Benchmark connection pooling vs fresh connections"
    )
    parser.add_argument(
        "--date", default="2026-04-11", help="Date to pull dishes for (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--table", default="daily_menu", choices=["daily_menu", "backfill_menu"]
    )
    parser.add_argument(
        "--profile",
        default="I enjoy spicy and asian foods",
        help="Preference summary text to embed (one Gemini call, reused for all runs)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Calls per condition (default 10; pass --runs 30 for a more rigorous measurement)",
    )
    args = parser.parse_args()

    run_benchmark(
        date_str=args.date,
        table=args.table,
        profile=args.profile,
        runs=args.runs,
    )
