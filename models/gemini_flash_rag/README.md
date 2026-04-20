# gemini_flash_rag

Production-grade RAG recommendation module for Lunch Buddy.

Built from the lessons of exp_01 through exp_04. See `models/models_log.md` for full experiment history.

## Modules

- `recommend.py` — core pipeline: embed, retrieve, filter, generate, EMA update
- `user_prefs.py` — load/save preference vectors to Supabase `user_pref` table
- `register.py` — MLflow model registration (run once to publish a new version)
- `demo.py` — interactive terminal demo (see below)

---

## Running the demo

```bash
cd models/gemini_flash_rag
conda activate lunch_buddy
```

**Test the learning loop (10 days, EMA updates, final convergence score):**
```bash
python demo.py --table backfill_menu
```

**Test today's live Stanford menu (one round):**
```bash
python demo.py
```

**Test mood blending (live menu + mood):**
```bash
python demo.py --mood "something light today"
```

**Run as a returning user (loads stored profile, saves after each pick):**
```bash
python demo.py --table backfill_menu --user_id <supabase-uuid>
```
Get your UUID from the Supabase dashboard: Authentication > Users.
First run creates the profile. Subsequent runs load it and continue where you left off.

**Combined -- returning user on live menu with mood:**
```bash
python demo.py --user_id <supabase-uuid> --mood "craving something spicy"
```

---

## MLflow

Model registered as `gemini_flash_rag_v1` in the MLflow registry.
- Version 1: recs only
- Version 2: recs + alternatives (current)

UI: http://35.232.122.64:5000
Load: `mlflow.pyfunc.load_model("models:/gemini_flash_rag_v1/2")`
