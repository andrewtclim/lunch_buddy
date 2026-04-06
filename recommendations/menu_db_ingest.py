# Setup

import os
import json
from datetime import datetime
from google import genai
from google.cloud import storage
import psycopg2
from psycopg2.extras import execute_values

# Functions


def get_menu_from_gcs():
    """Downloads today's menu JSON from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(os.getenv("BUCKET_NAME"))
    today_str = datetime.now().strftime('%Y-%m-%d')
    blob = bucket.blob(f"menu_{today_str}.json")
    return json.loads(blob.download_as_text()), today_str


def flatten_menu_data(menu_data, target_date):
    """Converts nested JSON into a flat list of dicts."""
    flattened = []
    daily_menu = menu_data.get(target_date, {})

    for dining_hall, meals in daily_menu.items():
        for meal_time, dishes in meals.items():
            for dish_name, details in dishes.items():
                # Construct the pure culinary text for the embedding model
                search_text = f"Dish: {dish_name}. Ingredients: {details.get('ingredients', '')}"

                flattened.append(
                    {
                        "dish_name": dish_name,
                        "dining_hall": dining_hall,
                        "meal_time": meal_time,
                        "search_text": search_text,
                        "tags": [
                            tag for tag in [
                                "vegan",
                                "vegetarian",
                                "halal"] if details.get(tag)],
                        "allergens": [
                            a.strip().lower() for a in details.get(
                                "allergens",
                                "").split(",")] if details.get("allergens") else [],
                        "ingredients": details.get(
                            "ingredients",
                            "")})
    return flattened


def get_embeddings_in_batches(items, client, batch_size=50):
    """Adds 'vector' key to each item using batch API calls."""
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        texts = [item["search_text"] for item in batch]

        print(f"Embedding batch {i//batch_size + 1}...")
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=texts
        )

        for j, item in enumerate(batch):
            item["vector"] = response.embeddings[j].values

    return items


def upload_to_supabase(data_rows):
    """Performs an 'Upsert' so we don't get duplicate rows if the script runs twice."""
    DB_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    upsert_query = """
        INSERT INTO daily_menu
        (dish_name, dining_hall, meal_time, tags, allergens, ingredients, date_served, embedding)
        VALUES %s
        ON CONFLICT (dish_name, dining_hall, date_served)
        DO UPDATE SET embedding = EXCLUDED.embedding;
    """

    execute_values(cur, upsert_query, data_rows)
    conn.commit()
    cur.close()
    conn.close()


def prepare_for_upsert(items, date_str):
    """Converts a list of dicts into a list of tuples for psycopg2."""
    return [
        (
            item["dish_name"],
            item["dining_hall"],
            item["meal_time"],
            item["tags"],
            item["allergens"],
            item["ingredients"],
            date_str,
            item["vector"]
        ) for item in items
    ]


def main():

    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID"),
        location="us-central1")
    # 1. Extract
    raw_data, today_str = get_menu_from_gcs()

    # 2. Transform (Flatten first!)
    flattened_items = flatten_menu_data(raw_data, today_str)

    # 3. Enrich (Get embeddings second!)
    items_with_embeddings = get_embeddings_in_batches(flattened_items, client)

    # 4. Load
    data_as_tuples = prepare_for_upsert(items_with_embeddings, today_str)
    upload_to_supabase(data_as_tuples)


if __name__ == "__main__":
    main()
