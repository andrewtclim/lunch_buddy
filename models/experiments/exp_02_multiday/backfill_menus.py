# models/experiments/backfill_menus.py
# Self-contained backfill script: pulls Stanford dining menu JSONs from GCS
# for the last N days and upserts each day's dishes (with embeddings) into
# Supabase.  Does not touch any shared scraper/ingest modules.
#
# Usage:
#   python backfill_menus.py            # backfills all available days in GCS
#   python backfill_menus.py --days 7   # backfills last 7 available days

import argparse
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.cloud import storage
import psycopg2

# load DATABASE_URL and PROJECT_ID from fastapi/.env
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")   # Supabase postgres connection string
PROJECT_ID   = os.getenv("PROJECT_ID")    # GCP project for Vertex AI

BUCKET_NAME  = "stanford-dining-menus"
EMBED_MODEL  = "text-embedding-004"
LOCATION     = "us-central1"


# ── GCS fetch ─────────────────────────────────────────────────────────────────

def list_available_dates():
    """
    Lists all menu JSON files in the GCS bucket and returns their dates as
    sorted date objects (oldest first).
    File naming convention: menu_YYYY-MM-DD.json
    """
    client  = storage.Client()
    bucket  = client.bucket(BUCKET_NAME)
    blobs   = bucket.list_blobs()                   # list all objects in the bucket

    dates = []
    for blob in blobs:
        # extract the date portion from e.g. "menu_2026-03-29.json"
        name = blob.name
        if name.startswith("menu_") and name.endswith(".json"):
            date_str = name[len("menu_"):-len(".json")]     # "2026-03-29"
            try:
                dates.append(date.fromisoformat(date_str))
            except ValueError:
                pass    # skip any files with unexpected naming

    return sorted(dates)    # oldest → newest


def fetch_from_gcs(target_date):
    """
    Downloads and parses the menu JSON for target_date from GCS.
    Returns the parsed dict, or None if the file doesn't exist.
    """
    import json
    client    = storage.Client()
    bucket    = client.bucket(BUCKET_NAME)
    blob_name = f"menu_{target_date.isoformat()}.json"
    blob      = bucket.blob(blob_name)

    if not blob.exists():
        print(f"  [warn] {blob_name} not found in GCS, skipping.")
        return None

    return json.loads(blob.download_as_text())   # parse JSON string into dict


# ── Flattening ────────────────────────────────────────────────────────────────

def flatten_menu(menu_data, date_str):
    """
    Converts the nested menu JSON into a flat list of dicts, one per
    (dish, dining_hall, meal_time) combination.
    """
    flattened  = {}
    daily_menu = menu_data.get(date_str, {})

    for dining_hall, meals in daily_menu.items():
        for meal_time, dishes in meals.items():
            for dish_name, details in dishes.items():
                # text fed to the embedding model — dish name + ingredients
                search_text = f"Dish: {dish_name}. Ingredients: {details.get('ingredients', '')}"

                unique_key = (dish_name, dining_hall, meal_time)
                flattened[unique_key] = {
                    "dish_name":   dish_name,
                    "dining_hall": dining_hall,
                    "meal_time":   meal_time,
                    "search_text": search_text,
                    "tags":      [tag for tag in ["vegan", "vegetarian", "halal"] if details.get(tag)],
                    "allergens": [a.strip().lower() for a in details["allergens"].split(",")] if details.get("allergens") else [],
                    "ingredients": details.get("ingredients", ""),
                }

    return list(flattened.values())


# ── Embedding ─────────────────────────────────────────────────────────────────

def add_embeddings(items, client, batch_size=50):
    """
    Calls the Vertex AI embedding API in batches of 50 (API limit).
    Adds a 'vector' key to each item in-place.
    Returns the same list with vectors attached.
    """
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [item["search_text"] for item in batch]

        print(f"    Embedding batch {i // batch_size + 1} ({len(batch)} items)...")
        response = client.models.embed_content(model=EMBED_MODEL, contents=texts)

        for j, item in enumerate(batch):
            item["vector"] = response.embeddings[j].values   # 768-dim float list

    return items


# ── Supabase upsert ───────────────────────────────────────────────────────────

def upsert_to_supabase(items, date_str):
    """
    Inserts rows into backfill_menu.  ON CONFLICT DO UPDATE means
    re-running the script for the same date is safe — embeddings get refreshed.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    upsert_query = """
        INSERT INTO backfill_menu
            (dish_name, dining_hall, meal_time, tags, allergens, ingredients, date_served, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS vector))
        ON CONFLICT (dish_name, dining_hall, meal_time, date_served)
        DO UPDATE SET embedding = EXCLUDED.embedding;
    """

    # insert one row at a time — avoids execute_values byte-joining issues on Python 3.14
    for item in items:
        try:
            cur.execute(upsert_query, (
                item["dish_name"],
                item["dining_hall"],
                item["meal_time"],
                item["tags"],
                item["allergens"],
                item["ingredients"],
                date_str,
                "[" + ",".join(str(x) for x in item["vector"]) + "]",
            ))
        except Exception as e:
            print(f"    [error] Failed on: {item['dish_name']} | {type(e).__name__}: {e}")
            raise
    conn.commit()
    cur.close()
    conn.close()
    print(f"    Upserted {len(items)} rows into backfill_menu for {date_str}.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill Supabase daily_menu from GCS menu JSONs."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of most-recent available days to backfill (default: all)",
    )
    args = parser.parse_args()

    # get all dates available in GCS, then optionally trim to the last N
    all_dates = list_available_dates()
    if not all_dates:
        print("No menu files found in GCS bucket. Exiting.")
        return

    dates = all_dates[-args.days:] if args.days else all_dates  # slice from end = most recent N

    print(f"Found {len(all_dates)} days in GCS. Backfilling {len(dates)} "
          f"({dates[0].isoformat()} → {dates[-1].isoformat()})...")

    # initialize Vertex AI client once — reused for all embedding batches
    ai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    for target_date in dates:
        date_str = target_date.isoformat()
        print(f"\n[{date_str}] Fetching from GCS...")

        # 1. Fetch from GCS
        menu_data = fetch_from_gcs(target_date)
        if menu_data is None:
            continue

        total_dishes = sum(
            len(dishes)
            for halls in menu_data[date_str].values()
            for dishes in halls.values()
        )
        print(f"  {total_dishes} dishes across {len(menu_data[date_str])} halls.")

        # 2. Flatten
        items = flatten_menu(menu_data, date_str)
        if not items:
            print(f"  No items after flatten, skipping.")
            continue

        # 3. Embed
        items = add_embeddings(items, ai_client)

        # 4. Upsert to Supabase
        upsert_to_supabase(items, date_str)

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
