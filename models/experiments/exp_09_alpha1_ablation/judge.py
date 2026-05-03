# models/experiments/exp_08_mood_faithfulness/judge.py
#
# LLM-as-judge for the mood-faithfulness eval. Labels each (dish, mood) pair
# as match / weak / miss. Caches labels to disk so reruns are free.
#
# We label the *menu* (every dish on a date) rather than just the system's
# recommendations, so we have a denominator for mood_recall@3:
#   mood_recall@3 = #matches in top-3 / #matches available on the menu
# This avoids penalizing the system on days where only one match exists.
#
# Cache key is (mood_lowercased, dish_name_lowercased) so a dish that recurs
# across dates is labelled once. Saved as JSON for human inspection.
#
# Note on dietary constraints: the production system handles vegetarian/vegan/
# gluten-free/dairy-free as hard allergen filters in recommend.filter_allergens(),
# not via the mood path. So this eval intentionally does NOT include pure dietary
# moods (see eval_set.json). The judge prompt only mentions dietary as a soft
# rule for compound moods like "vegan today" -- left in for forward-compat with
# the planned compound-mood parser (see plan: Follow-up priority 3).

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types

# AI Studio client -- intentionally separate from the Vertex AI client in
# recommend.py. The judge makes dense sequential bulk calls (up to 345 per mood)
# which triggered HTTP/2 stalling on Vertex AI under sustained load. AI Studio
# uses a plain REST transport and is stable for this pattern.
# recommend.py's load_dotenv runs before this module is imported, so GEMINI_API_KEY
# is already in the environment by the time this line executes.
GEN_MODEL = "gemini-2.5-flash"
_judge_client: genai.Client | None = None  # initialized lazily on first API call


def _get_judge_client() -> genai.Client:
    global _judge_client
    if _judge_client is None:
        _judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _judge_client


LABELS = ("match", "weak", "miss")
CACHE_PATH = Path(__file__).parent / "judge_cache.json"
SAVE_EVERY = 25  # flush cache to disk every N new labels so a crash doesn't lose work
CALL_TIMEOUT_SEC = 60  # per-call wall clock cap
RPM_SLEEP_SEC = (
    3.0  # ~15 RPM free-tier limit; calls take ~0.5s so 3s keeps us at ~13 RPM
)


def _call_with_timeout(fn, args: tuple, timeout: float):
    """
    Run fn(*args) on a daemon thread; raise TimeoutError if it doesn't return
    within `timeout` seconds. Daemon thread means a stuck thread won't block
    process shutdown -- it dies when main exits.

    We needed this after observing that concurrent Vertex AI calls sometimes
    stall for ~10 minutes per call. Sequential calls also stall rarely; this
    is cheap belt-and-suspenders for the sequential path.
    """
    result: list = []
    err: list = []

    def target():
        try:
            result.append(fn(*args))
        except Exception as e:  # noqa: BLE001 -- we want to surface any error
            err.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"call exceeded {timeout}s")
    if err:
        raise err[0]
    return result[0]


JUDGE_PROMPT = """You are labelling whether a dining-hall dish satisfies a user's mood/craving.

Mood: "{mood}"

Dish name: "{dish_name}"
Ingredients: {ingredients}

Classify the dish for THIS mood. Output exactly one of these labels:
  - "match": clearly satisfies the mood. A noodle dish for "noodles", a soup for "soup",
             a burger for "a burger", a fully Italian dish for "something Italian".
  - "weak":  partially satisfies but isn't an obvious answer. A pasta salad for "noodles"
             (has noodles but framed as a salad). A bowl with rice + grilled veg for "a rice bowl"
             (it's a rice bowl but minimal). Use sparingly -- prefer match or miss when clear.
  - "miss":  does not satisfy the mood.

Guidelines:
  - Cuisine moods judge cuisine fit, not single ingredients. A lone parmesan sprinkle does not
    make a dish "Italian"; lasagna or risotto does. Same for "Mexican", "Mediterranean", etc.
  - Vague moods ("comfort food", "something light", "surprise me", "craving carbs", "easy on the
    stomach") should be labelled the way a typical eater would interpret them. "Surprise me" ->
    "match" only for unusual/less-common dishes; "miss" for obvious staples like plain pasta.
  - If the mood embeds an explicit dietary constraint (e.g. "vegan tonight", "gluten-free pizza"),
    a dish that violates that constraint is "miss" regardless of how well it matches otherwise.
  - When ingredients are missing or sparse, infer from the dish name; if still ambiguous, "weak".

Return ONLY this JSON object, no preamble:
  {{"label": "match" | "weak" | "miss", "reason": "<= 12 words explaining why"}}
"""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


# short hash of the judge prompt -- changes whenever JUDGE_PROMPT is edited so
# stale labels (graded against an old rubric) auto-invalidate without us having
# to remember to delete the cache file
PROMPT_HASH = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:8]


def _key(mood: str, dish_name: str) -> str:
    return f"{PROMPT_HASH}||{mood.strip().lower()}||{dish_name.strip().lower()}"


def _judge_api_call(mood: str, dish_name: str, ingredients: str) -> dict:
    """
    Single API call to the judge. No caching. Returns {"label", "reason"}.
    Raises RuntimeError after 3 failed attempts -- failures should be loud
    per project convention.
    """
    prompt = JUDGE_PROMPT.format(
        mood=mood,
        dish_name=dish_name,
        ingredients=ingredients or "(no ingredients listed)",
    )

    def _do_call():
        return _get_judge_client().models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                # judge is pattern-matching, not multi-step reasoning -- match
                # the recommender's exp_06 finding that budget=0 is fine
                "thinking_config": genai_types.ThinkingConfig(thinking_budget=0),
            },
        )

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = _call_with_timeout(_do_call, (), CALL_TIMEOUT_SEC)
            result = json.loads(response.text)
            label = (result.get("label") or "").strip().lower()
            if label not in LABELS:
                raise ValueError(f"unexpected judge label: {label!r}")
            return {
                "label": label,
                "reason": (result.get("reason") or "").strip(),
            }
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2)

    raise RuntimeError(
        f"judge failed 3x for mood={mood!r}, dish={dish_name!r}: {last_err}"
    )


def label_dish(mood: str, dish_name: str, ingredients: str, cache: dict) -> dict:
    """
    Cached single-dish judge. Mutates `cache` in place.
    Returns {"label": "match"|"weak"|"miss"|"error", "reason": str}.
    "error" means the API failed 3x; it is NOT written to cache so it retries
    fresh on the next run. Callers should treat "error" as "miss" for scoring
    but log a warning so the eval isn't silently corrupted.
    """
    k = _key(mood, dish_name)
    if k in cache:
        return cache[k]
    try:
        entry = _judge_api_call(mood, dish_name, ingredients)
    except RuntimeError as exc:
        print(
            f"  [judge] WARNING: {exc} -- skipping dish, treating as miss",
            file=sys.stderr,
            flush=True,
        )
        return {"label": "error", "reason": str(exc)}
    cache[k] = entry
    return entry


def label_menu(mood: str, dishes: list[dict], verbose: bool = False) -> list[dict]:
    """
    Label every dish on the menu for a given mood. Returns
        [{"dish_name": ..., "label": ..., "reason": ...}, ...]
    in the same order as `dishes`.

    Sequential by design. Earlier we tried ThreadPoolExecutor concurrency, but
    a probe revealed that concurrent Vertex AI calls intermittently stall a
    single call for ~10 minutes (likely an HTTP/2 stream-multiplexing edge
    case). Sequential calls are uniformly fast (~0.3-0.9s each in the probe),
    so 325 dishes finish in 3-5 minutes and 8.4k dishes in ~70 minutes -- both
    acceptable. _judge_api_call has a 60s per-call timeout as a safety net.

    Cache is consulted before each call; misses hit the API and the result is
    written back. Cache is flushed to disk every SAVE_EVERY new labels.
    "error" labels (3x timeout) are returned but NOT cached so they retry fresh.
    """
    cache = _load_cache()
    initial_size = len(cache)
    out: list[dict] = []
    n_errors = 0

    for d in dishes:
        cache_size_before = len(cache)
        entry = label_dish(mood, d["dish_name"], d.get("ingredients", ""), cache)
        if entry["label"] == "error":
            n_errors += 1
        elif len(cache) > cache_size_before:
            # only sleep after a real API call (cache miss); cache hits are free
            time.sleep(RPM_SLEEP_SEC)
        out.append(
            {
                "dish_name": d["dish_name"],
                "label": entry["label"],
                "reason": entry["reason"],
            }
        )
        new_labels = len(cache) - initial_size
        if new_labels and new_labels % SAVE_EVERY == 0:
            _save_cache(cache)
            if verbose:
                print(
                    f"  [judge] checkpoint: {new_labels} new labels "
                    f"(total cache size: {len(cache)})",
                    flush=True,
                )

    _save_cache(cache)
    if n_errors:
        print(
            f"  [judge] WARNING: {n_errors}/{len(dishes)} dishes could not be labelled "
            f"(timeout 3x) for mood={mood!r}; treated as miss in scoring.",
            file=sys.stderr,
            flush=True,
        )
    return out


def summarize_menu_labels(labels: list[dict]) -> dict:
    """Quick aggregate counts for inspection."""
    counts = {"match": 0, "weak": 0, "miss": 0, "error": 0}
    for row in labels:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    counts["total"] = len(labels)
    return counts


# ---------------------------------------------------------------------------
# CLI: spot-check helper. Run this to label a single mood vs. a single date and
# eyeball the results before trusting the judge for the full eval.
#   python judge.py --mood "noodles" --date 2026-03-30 --limit 30
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    parser = argparse.ArgumentParser(description="Spot-check the LLM-as-judge.")
    parser.add_argument("--mood", required=True, help="mood string to label against")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD in backfill_menu")
    parser.add_argument("--limit", type=int, default=30, help="dishes to label")
    parser.add_argument("--table", default="backfill_menu")
    args = parser.parse_args()

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute(
        f"SELECT dish_name, ingredients FROM {args.table} "
        f"WHERE date_served = %s LIMIT %s;",
        (args.date, args.limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    dishes = [{"dish_name": r[0], "ingredients": r[1] or ""} for r in rows]
    print(f"Labelling {len(dishes)} dishes against mood={args.mood!r}...\n")
    labels = label_menu(args.mood, dishes, verbose=True)

    for d, L in zip(dishes, labels):
        marker = {"match": "[+]", "weak": "[~]", "miss": "[-]"}[L["label"]]
        print(f"{marker} {d['dish_name']:<55s}  -- {L['reason']}")

    print()
    print("summary:", summarize_menu_labels(labels))
