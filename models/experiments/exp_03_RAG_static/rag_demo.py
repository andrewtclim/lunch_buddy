# models/experiments/exp_03_RAG_static/rag_demo.py
# Interactive RAG demo: type a food preference query, retrieve matching dishes
# from backfill_menu via cosine similarity, and get a Gemini recommendation.
# Run this to verify the full RAG pipeline before building the simulation.

import os
import json
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# load env vars -- parents[3] because we are 3 levels deep from project root
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")   # Supabase connection string
PROJECT_ID   = os.getenv("PROJECT_ID")    # GCP project for Vertex AI
LOCATION     = "us-central1"
EMBED_MODEL  = "text-embedding-004"        # must match the model used to embed dishes
GEN_MODEL    = "gemini-2.5-flash"          # model for generating the recommendation

# initialize Vertex AI client once -- reused for embedding and generation
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def embed_query(text: str) -> list[float]:
    # embed the user's query using the same model used to embed dishes in Supabase
    # critical: must be the same model or the vector spaces won't align
    response = client.models.embed_content(model=EMBED_MODEL, contents=[text])
    return response.embeddings[0].values   # 768-dim float list


def retrieve_dishes(query_vector: list[float], date_str: str, limit: int = 10) -> list[dict]:
    # cosine search backfill_menu for dishes on date_str closest to the query vector
    # the <=> operator is pgvector's cosine distance -- lower = more similar
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT dish_name, dining_hall, meal_time, allergens, ingredients
        FROM backfill_menu
        WHERE date_served = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (date_str, query_vector, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "dish_name":   row[0],
            "dining_hall": row[1],
            "meal_time":   row[2],
            "allergens":   row[3] or [],
            "ingredients": row[4] or "",
        }
        for row in rows
    ]


def recommend(query: str, candidates: list[dict]) -> str:
    # build a prompt that gives Gemini the user's query and the retrieved dishes
    # candidates is the filtered top-5 list -- Gemini never sees the full 300-dish menu
    dish_lines = "\n".join([
        f"{i+1}. {d['dish_name']} at {d['dining_hall']} ({d['meal_time']})"
        f" -- Ingredients: {d['ingredients']}"
        for i, d in enumerate(candidates)
    ])

    prompt = f"""You are a Stanford dining hall recommender.

User preference: {query}

Available dishes today:
{dish_lines}

Recommend the top 3 dishes from the list above that best match the user's preference.
For each dish include: dish name, dining hall, and one concise reason why it fits.
Only recommend dishes from the list above -- do not invent dishes.
Respond in this exact JSON format:
{{
  "recommendations": [
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}},
    {{"dish_name": "...", "dining_hall": "...", "reason": "..."}}
  ]
}}"""

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},   # force valid JSON -- no markdown fences
    )
    return response.text   # raw string -- we parse it in main()


def deduplicate(dishes: list[dict]) -> list[dict]:
    # keep only the first occurrence of each dish name -- removes duplicates across halls
    # the first occurrence is the highest-ranked by cosine similarity so we preserve ordering
    seen = set()
    unique = []
    for d in dishes:
        if d["dish_name"] not in seen:
            seen.add(d["dish_name"])
            unique.append(d)
    return unique


def main():
    # let the user type their own query interactively
    query    = input("What are you in the mood for today? ").strip()
    date_str = input("Date to search (YYYY-MM-DD, default 2026-04-07): ").strip() or "2026-04-07"

    print(f"\nQuery: {query}")
    print(f"Date:  {date_str}\n")

    # step 1: embed the query
    print("Embedding query...")
    query_vector = embed_query(query)

    # step 2: retrieve top 20 to give deduplication enough to work with
    print("Retrieving dishes from Supabase...")
    dishes = retrieve_dishes(query_vector, date_str, limit=20)

    # step 3: deduplicate by dish name, then take top 10 unique
    dishes = deduplicate(dishes)[:10]
    print(f"  {len(dishes)} unique dishes retrieved\n")

    # step 4: print retrieved dishes so retrieval quality is visible
    print("Retrieved dishes (before Gemini):")
    for i, d in enumerate(dishes):
        print(f"  {i+1}. {d['dish_name']} -- {d['dining_hall']} ({d['meal_time']})")
    print()

    # step 5: pass top 5 unique dishes to Gemini
    candidates = dishes[:5]

    print("Calling Gemini...")
    raw = recommend(query, candidates)

    # step 6: parse JSON -- strip markdown fences Gemini sometimes adds
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    result  = json.loads(cleaned)

    print("\nGemini's top 3 recommendations:")
    for i, rec in enumerate(result["recommendations"]):
        print(f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}")
        print(f"     {rec['reason']}")


if __name__ == "__main__":
    main()
