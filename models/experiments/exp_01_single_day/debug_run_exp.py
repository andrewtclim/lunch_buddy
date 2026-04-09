# models/experiments/exp_01_single_day/debug_run_exp.py
# Stripped-down copy of run_simulation.py for debugging the psycopg2 failure.
# Does NOT run the full simulation or log to MLflow.
# Goal: figure out why psycopg2 can't see tables after genai is initialized.

import os
import json
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# load env vars — same path as run_simulation.py
load_dotenv(Path(__file__).resolve().parents[3] / "fastapi" / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
PROJECT_ID   = os.getenv("PROJECT_ID")
LOCATION     = "us-central1"
EMBED_MODEL  = "text-embedding-004"

print(f"DATABASE_URL loaded: {'YES' if DATABASE_URL else 'NO'}")
print(f"DATABASE_URL prefix: {DATABASE_URL[:40] if DATABASE_URL else 'None'}")

# ── Step 1: test psycopg2 BEFORE genai is imported ────────────────────────────
print("\n[Step 1] psycopg2 connect BEFORE genai import...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_menu")
    print(f"  OK — daily_menu row count: {cur.fetchone()}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

# ── Step 2: import genai (but don't call anything yet) ────────────────────────
print("\n[Step 2] importing google.genai...")
from google import genai
print("  imported.")

print("\n[Step 3] psycopg2 connect AFTER genai import (no API calls yet)...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_menu")
    print(f"  OK — daily_menu row count: {cur.fetchone()}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

# ── Step 3: initialize genai client (sets up auth, connections) ───────────────
print("\n[Step 4] initializing genai.Client...")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
print("  initialized.")

print("\n[Step 5] psycopg2 connect AFTER genai.Client() init (no API calls yet)...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_menu")
    print(f"  OK — daily_menu row count: {cur.fetchone()}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

# ── Step 4: make one real embedding API call ──────────────────────────────────
print("\n[Step 6] making one embedding API call...")
response = client.models.embed_content(model=EMBED_MODEL, contents=["test dish"])
print(f"  got embedding of length {len(response.embeddings[0].values)}")

print("\n[Step 7] psycopg2 connect AFTER real API call...")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_menu")
    print(f"  OK — daily_menu row count: {cur.fetchone()}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"  FAIL — {type(e).__name__}: {e}")

print("\nDone. Check which step first shows FAIL.")
