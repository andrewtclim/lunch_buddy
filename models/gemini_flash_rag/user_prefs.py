# models/gemini_flash_rag/user_prefs.py
# Persistence helpers for the user_pref table in Supabase.
# Handles loading and saving a user's preference vector, preference summary,
# and allergen list. Nothing else -- no embedding, no Gemini calls here.
#
# user_id is a UUID string that comes from Supabase Auth (the `sub` claim in
# the JWT that Patrick's get_current_user() extracts in fastapi/main.py).
# It matches the primary key in auth.users and in our user_pref table.

import os
import json
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# load env vars from models/.env (one level up from gemini_flash_rag/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")   # Supabase direct connection string


def load_user_pref(user_id: str) -> dict | None:
    # fetch a user's full preference profile from user_pref
    # returns a dict with keys: preference_vector, preference_summary, allergens
    # returns None if the user has no row yet (e.g. just signed up)
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT preference_vector, preference_summary, allergens
        FROM user_pref
        WHERE user_id = %s
        LIMIT 1;
    """, (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None   # caller should handle cold start (no profile yet)

    # pgvector returns the vector as a string "[0.1, 0.2, ...]" -- parse to numpy
    pref_vec = np.array(json.loads(row[0]), dtype=float)

    return {
        "preference_vector":  pref_vec,
        "preference_summary": row[1],       # plain text string
        "allergens":          row[2] or [],  # postgres text[] -> python list
    }


def save_user_pref(user_id: str, pref_vec: np.ndarray,
                   preference_summary: str, allergens: list[str]) -> None:
    # upsert the user's preference profile into user_pref
    # upsert = insert if no row exists, update if one does
    # this means the same function works for first-time signup and for
    # updating after each dish pick -- no separate "create" vs "update" needed
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO user_pref (user_id, preference_vector, preference_summary,
                               allergens, last_updated)
        VALUES (%s, %s::vector, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE
            SET preference_vector  = EXCLUDED.preference_vector,
                preference_summary = EXCLUDED.preference_summary,
                allergens          = EXCLUDED.allergens,
                last_updated       = now();
    """, (
        user_id,
        pref_vec.tolist(),          # numpy array -> plain python list for psycopg2
        preference_summary,
        allergens,                  # python list -> postgres text[]
    ))
    conn.commit()    # write is not persisted until commit() is called
    cur.close()
    conn.close()
