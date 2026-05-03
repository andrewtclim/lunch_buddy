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

DATABASE_URL = os.getenv("DATABASE_URL_IPV4")  # Supabase direct connection string


def load_user_pref(user_id: str) -> dict | None:
    # fetch a user's full preference profile from user_pref
    # returns None if the user has no row yet (e.g. just signed up)
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT preference_vector, preference_summary, allergens,
               original_profile_vector, recent_choices
        FROM user_pref
        WHERE user_id = %s
        LIMIT 1;
    """,
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        return None  # caller should handle cold start (no profile yet)

    # pgvector returns vectors as strings "[0.1, 0.2, ...]" -- parse to numpy
    pref_vec = np.array(json.loads(row[0]), dtype=float)

    # original_profile_vector: NULL for pre-migration users; fall back to pref_vec
    # so callers always get a usable ndarray and three_way_blend() is always reachable
    orig_raw = row[3]
    original_profile_vector = (
        np.array(json.loads(orig_raw), dtype=float)
        if orig_raw is not None
        else pref_vec
    )

    # recent_choices: JSONB list of float lists, most-recent first
    # keep raw form (for /pick to prepend+trim) and decoded ndarrays (for blending)
    recent_choices_raw: list[list[float]] = row[4] or []
    recent_choices_vecs: list[np.ndarray] = [
        np.array(v, dtype=float) for v in recent_choices_raw
    ]

    return {
        "preference_vector": pref_vec,
        "preference_summary": row[1],
        "allergens": row[2] or [],
        "original_profile_vector": original_profile_vector,
        "recent_choices_raw": recent_choices_raw,  # list[list[float]], for /pick
        "recent_choices_vecs": recent_choices_vecs,  # list[ndarray], for blending
    }


def save_user_pref(
    user_id: str,
    pref_vec: np.ndarray,
    preference_summary: str,
    allergens: list[str],
    recent_choices: list[list[float]] | None = None,
    original_profile_vector: np.ndarray | None = None,
) -> None:
    # upsert the user's preference profile into user_pref
    # upsert = insert if no row exists, update if one does
    #
    # original_profile_vector: written once via COALESCE — if the row already has a
    #   non-NULL value, the existing value is kept. Pass it only on first save.
    #
    # recent_choices: always overwritten when provided; left unchanged when None.
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_pref (user_id, preference_vector, preference_summary,
                               allergens, last_updated,
                               original_profile_vector, recent_choices)
        VALUES (%s, %s::vector, %s, %s, now(),
                %s::vector, COALESCE(%s::jsonb, '[]'::jsonb))
        ON CONFLICT (user_id) DO UPDATE
            SET preference_vector  = EXCLUDED.preference_vector,
                preference_summary = EXCLUDED.preference_summary,
                allergens          = EXCLUDED.allergens,
                last_updated       = now(),
                original_profile_vector = COALESCE(user_pref.original_profile_vector,
                                                    EXCLUDED.original_profile_vector),
                recent_choices = CASE
                    WHEN EXCLUDED.recent_choices <> '[]'::jsonb
                    THEN EXCLUDED.recent_choices
                    ELSE user_pref.recent_choices
                END;
    """,
        (
            user_id,
            pref_vec.tolist(),
            preference_summary,
            allergens,
            (
                original_profile_vector.tolist()
                if original_profile_vector is not None
                else None
            ),
            json.dumps(recent_choices) if recent_choices is not None else None,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
