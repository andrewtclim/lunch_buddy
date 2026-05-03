# Lunch Buddy — Models Log

---

## What we're building

An AI-powered dining hall recommender for Stanford students. Each day the scraper
pulls fresh menus from all Stanford dining halls. When a user opens the app, the
system finds dishes that match their personal taste profile and asks Gemini to write
a friendly top-3 recommendation with reasons. As the user accepts recommendations
over time, their taste profile quietly drifts toward what they actually choose.

---

## Current architecture 

1. **Taste profile (preference vector).** Each user is represented as a 768-number
   vector in the same space as the dish embeddings. At signup, the user types a
   free-text description of their food preferences; that text is embedded via
   `text-embedding-004` (Vertex AI) to produce the starting vector.

2. **Retrieval.** When the user wants a recommendation, we run a cosine similarity
   search against that day's dish embeddings in Supabase/pgvector. We pull the top
   80 closest dishes, drop allergen conflicts and placeholder station labels, and
   pass the top 20 to Gemini.

3. **Generation.** Gemini 2.5 Flash receives the user's current preference summary
   (plain English, one sentence) and the 5 candidate dishes. It picks the top 3 and
   writes a short reason for each.

4. **Learning loop.** When the user picks a dish, we fetch that dish's stored
   embedding from Supabase and blend it into the preference vector via exponential
   moving average (EMA): `new_vec = normalize((1 - alpha) * pref_vec + alpha * dish_vec)`.
   The vector nudges toward what the user actually chose. Gemini also rewrites the
   preference summary to stay in sync.

5. **Allergen filter.** Hard block applied before scoring. Any dish sharing an
   allergen with the user is dropped before the vector search results are used.
   No similarity score can override it.

Stack: Supabase (PostgreSQL + pgvector), Vertex AI (`text-embedding-004`,
Gemini 2.5 Flash), FastAPI, Python, Docker, MLflow (experiment tracking).

---

## Experiments

### exp_01 — Single-day baseline — Apr 6 2026 — done

- **What we tried.** 15 mock users, 1 day of Stanford menu data (246 dishes),
  10-round simulation. Each round: cosine retrieve top 3, user picks the one
  closest to their hidden profile, vector updates via EMA (alpha=0.85).
  Also ran a 50-round cold-start variant with no signup text.
- **What we learned.** Scores were flat across all rounds (avg 0.4702 → 0.4649
  over 10 rounds). Cycling the same single day's menu meant the same dishes
  surfaced every round regardless of vector drift -- no learning signal possible.
  Cold start was equally flat for the same reason.
- **Decision.** Pipeline is correct. Experiment needs multi-day data to show
  anything meaningful. Move to exp_02.
- **Files.** `experiments/exp_01_single_day/`

### exp_01 — Model selection — Apr 6 2026 — done

- **What we tried.** Evaluated Gemini 2.5 Flash, Gemini 2.5 Pro, Qwen2.5-72B
  (Together AI), and Ollama as the LLM layer.
- **Results.** Flash and Pro scored identically (0.4609 avg cosine similarity,
  15 mock users). Pro was 2x slower (15.31s vs 8.28s per user). Both had zero
  allergen violations.
- **Decision.** Gemini 2.5 Flash. Same auth as embeddings (already on Vertex AI),
  no quality penalty, half the latency. Qwen adds a separate API key with no
  advantage. Ollama requires GPU hardware we don't have.
- **Files.** `experiments/models_exp_comparison.md`

### exp_02 — Multi-day simulation — Apr 7 2026 — done

- **What we tried.** Backfilled 10 days of real Stanford menus (Mar 29 - Apr 7 2026)
  from GCS into a `backfill_menu` table in Supabase (~3,091 dishes). 15 mock users,
  alpha=0.85, 1 round per day. Logged to MLflow (run: `exp02-multiday-simulation`).
- **What we learned.**
  - 9 of 15 users got locked onto a dining hall staple by day 2 ("Flavor Forward
    Legumes," "Craveable Grains," "Gluten-Free Waffles") because alpha=0.85 locks
    the vector in immediately after a single pick.
  - Weekend menus have far fewer halls and no niche cuisine options (spicy, Korean,
    Ethiopian). A user who gets Grilled Chicken as the best available on a bad day
    drifts toward generic protein and never recovers in 10 days.
  - Overall result: avg score went from 0.4676 → 0.4510 (-0.0166). Worse, not better.
- **Decision.** alpha=0.85 is too aggressive. Need retrieval diversity to avoid ruts.
  Explore lower alpha (0.1-0.3) and LLM-in-the-loop for richer user simulation.
- **Files.** `experiments/exp_02_multiday/`

### exp_03 — RAG static — Apr 8 2026 — shelved (blocked)

- **What we tried.** Added Gemini to the retrieval loop: embed signup text (fixed,
  not evolving), cosine retrieve top 20, Gemini re-ranks to top 3 each day.
  Scores from a partial run (14/15 users, 10 days) looked above exp_02 baseline
  (~0.42-0.58 vs ~0.467).
- **Why it's blocked.** `rag_simulation.py` hangs silently on the first
  `get_embedding()` call. No timeout on the Vertex AI `embed_content` call --
  connection stalls indefinitely. Fix: wrap in `concurrent.futures` with a deadline.
- **Decision.** Shelve until bandwidth allows. exp_04 is more interesting (learning
  loop) and is the active focus.
- **Files.** `experiments/exp_03_RAG_static/`

### exp_04 — RAG learning (interactive demo) — Apr 8 2026 — active

- **What we tried.** Full learning loop: preference vector AND a Gemini-written
  preference summary both evolve after each pick. Built an interactive terminal
  demo. Ran one live session (Andrew, 10 days, alpha=0.3).
- **Results.**
  - Signup: "I enjoy spicy and asian foods"
  - True preference: "enjoy korean and chinese foods especially, I love spicy
    foods and my favorite protein is chicken"
  - Initial score: 0.7913 / Final score: 0.5926 / Delta: -0.199
  - Score went down. Signup text was already close to true preferences (0.79 is
    high), leaving little room to improve. Picks drifted toward Indian food
    (Butter Chicken 3x, Tandoori, Biryani) which pulled the vector away from
    Korean/Chinese specifically.
- **Key findings.**
  1. Ceiling problem: if signup is already a strong proxy for true preferences,
     learning introduces drift instead of improvement. Experiment design needs
     intentionally mismatched signup vs hidden profile to show gains.
  2. Retrieval rut: once the vector drifted toward Indian food, it kept pulling
     similar dishes. Low diversity compounds the problem.
  3. Alpha sensitivity: 0.3 may be too aggressive. Try 0.1-0.15.
- **What's next.** Build `batch_simulation.py` -- run all 15 mock users
  non-interactively with mismatched signup vs hidden profiles, test alpha=0.1/0.15/0.3,
  log results to MLflow.
- **Files.** `experiments/exp_04_RAG_learning/`

### exp_05 — FastAPI deploy scaffold — Apr 10 2026 — scaffold only

- **What we tried.** Set up a FastAPI + Docker scaffold for serving recommendations
  as a real API endpoint. `POST /predict` skeleton exists. Patrick added JWT auth
  via Supabase JWKS.
- **Status.** Scaffold only -- `/predict` is not yet wired to Supabase + Gemini.
  Docker build confirmed working; `/health` and `/` endpoints respond correctly.
  `python-jose` missing from `requirements.txt` (needed for JWT verify -- Patrick fix).
- **Files.** `experiments/exp_05_fastapi_deploy/`, `fastapi/main.py`

### exp_06 — thinking_budget benchmark + prompt v2 — Apr 12 2026 — done

- **What we tried.** Benchmarked `thinking_budget=0` (disables extended reasoning in
  Gemini 2.5 Flash) against default thinking for the re-ranking task. Also tested a
  revised "mood-primary" prompt where the user's daily mood leads as the primary
  constraint and the taste profile is demoted to a tiebreaker.
- **Results.**
  - `thinking_budget=0` produced identical dish picks to default thinking at 7x lower
    latency (~2s vs ~13s) and 100% JSON reliability vs 33-67% with default thinking.
  - Mood-primary prompt (v2): 1/3 overlap with the old profile-dominant prompt on the
    same candidates. The v2 picks were meaningfully more mood-aligned.
  - Dynamic beta: raising beta from 0.3 to 0.5 when mood is given aligns retrieval with
    the new prompt priority. When no mood, beta=0.0 so query_vec = pref_vec directly.
- **Decisions.**
  - `thinking_budget=0` adopted for all Gemini calls (re-ranking + summary generation).
    Re-ranking 10 dishes is pattern matching, not multi-step reasoning.
  - Prompt v2 (mood-primary) adopted. Profile-only prompt retained for the no-mood path.
  - `beta_with_mood=0.5`, `beta_no_mood=0.0` (was fixed 0.3 for both cases).
  - `top_k_gemini` raised from 5 to 10 -- gives Gemini more room, especially on days
    with aggressive allergen filtering.
  - Defensive `dish_name` cleanup added: strips `" at DiningHall (MealTime)"` suffix
    if Gemini copies the full candidate line into the dish_name field.
- **Registered.** `gemini_flash_rag_v2` in MLflow Model Registry.
- **Files.** `gemini_flash_rag/recommend.py`, `gemini_flash_rag/register.py`

### exp_07 — connection pooling benchmark — Apr 25 2026 — done

- **What we tried.** Benchmarked `psycopg2.pool.ThreadedConnectionPool` vs fresh-connection-per-call on `retrieve_dishes()`. 30 calls each, Gemini excluded from timing.
- **Results.** avg 761ms → 228ms (3.3x); p95 1163ms → 273ms (4.3x). Each fresh connection pays ~530ms TCP+TLS+auth overhead; pooled calls 2+ reuse it.
- **Decision.** Pooling applied to all three DB functions in `recommend.py` (min=1, max=5). `try/finally` on `putconn()` prevents pool exhaustion on exceptions.
- **Files.** `experiments/exp_07_connection_pooling/`

### exp_08 — mood faithfulness eval — Apr 29 2026 — done

- **What we tried.** LLM-as-judge eval for mood fidelity. Judge labels each (mood, dish) pair as match/weak/miss; faith@3 = matches-in-top-3 / 3. 37 profiles × 3 repeats across 3 categories: *adjacent* (profile opposes mood — the hard case), *aligned* (profile supports mood), *contrast* (profile strongly opposes in a different dimension). Ran baseline and three variants.
- **Results.** Baseline adj faith@3=0.633; wider_funnel (top_k_retrieval=80, top_k_gemini=20): adj=0.717, aligned=0.833 (no regression). Wider pool helps but adj still trails — root cause: at beta=0.5 the blended query falls between profile and mood vectors in a retrieval dead zone (mood dilution).
- **Decision.** wider_funnel defaults shipped (`top_k_retrieval=80`, `top_k_gemini=20`). Beta raise + signal separation queued as exp_09.
- **Files.** `experiments/exp_08_mood_faithfulness/`

### exp_09 — alpha1 (mood weight) ablation — May 2 2026 — done

- **What we tried.** Swept `BETA_WITH_MOOD` (α₁) over {0.5, 0.6, 0.7, 0.8} with realistic pick history. Simulation: 4 no-mood picks on Mar 30/Apr 1/Apr 3/Apr 5 (profile-driven, random top-3 pick each day), eval on Apr 7 with mood. Single-eval-day design keeps history length identical across all profiles. Eval set expanded to 48 Apr 7 profiles (33 adj, 13 contrast, 2 aligned; SE=0.050 on adj slice) — new profiles reused existing judge labels, zero extra API calls.
- **Results.** adj faith@3 improved monotonically: 0.774 → 0.808 → 0.872 → **0.889** (α₁=0.5→0.6→0.7→0.8). No aligned regression at any value.
- **Decision.** `BETA_WITH_MOOD` updated 0.5 → 0.8 in `recommend.py`.
- **Files.** `experiments/exp_09_alpha1_ablation/`

---

## Active working directory

`models/gemini_flash_rag/` — production-grade RAG module being built from
the lessons of exp_01 through exp_04. See `gemini_flash_rag/README.md`.

---

## Session log — Apr 11 2026

### Housekeeping

- Created `models/.env` with the four keys the models layer needs: `DATABASE_URL`,
  `DATABASE_URL_IPV4`, `PROJECT_ID`, `LOCATION`. Models no longer depend on
  `fastapi/.env`, which Patrick owns and modifies independently.
- Fixed `load_dotenv` paths in `interactive_demo.py` and `CAMPUS_interactive_demo.py`:
  `parents[3] / "fastapi" / ".env"` -> `parents[2] / ".env"`.
- Fixed a corrupted line in `frontend/.env` where a Slack timestamp had been pasted
  in, merging `GOOGLE_APPLICATION_CREDENTIALS` and `VITE_API_URL` onto one line.
- Moved `WIP_models_log.md`, `RAG_model_log.md`, `models_exp_comparison.md` into
  `experiments/` to keep `models/` root clean.
- Confirmed Docker build and all three FastAPI endpoints working (`/`, `/health`,
  `/predict`). Noted `python-jose` missing from `fastapi/requirements.txt` -- Patrick fix.

### gemini_flash_rag -- production module (new)

Built `models/gemini_flash_rag/` from scratch as the production-grade replacement
for the experiment scripts. Three files:

**`recommend.py`** -- the full recommendation pipeline in one importable module.

- `get_embedding(text)` -- calls Vertex AI text-embedding-004, returns 768-dim numpy array.
- `normalize(vec)` -- scales to unit length; required after any vector blend.
- `blend_mood(pref_vec, mood_vec, beta=0.3)` -- blends a daily mood embedding into
  the preference vector at query time only. The stored `pref_vec` is never modified
  by mood. beta=0.3 means 70% learned profile, 30% today's mood.
- `retrieve_dishes(query_vec, date_str, table, limit)` -- cosine search via pgvector.
  `table` param supports both `daily_menu` (production) and `backfill_menu` (experiments).
- `fetch_dish_embedding(dish_name, date_str, table)` -- fetches a dish's stored
  embedding from Supabase for the EMA update step.
- `filter_placeholders / filter_allergens / deduplicate` -- same filters as exp_04,
  now standalone and individually testable.
- `call_gemini(preference_summary, candidates, daily_mood)` -- builds the prompt with
  an optional mood line, calls Gemini 2.5 Flash, returns top 3 as structured JSON.
  Mood appears in both the context and the instruction so Gemini actually uses it.
- `update_preference_vector(pref_vec, dish_vec, alpha=0.3)` -- EMA update. Separated
  from `recommend()` because updating requires a user pick, which is external to the
  recommendation step.
- `recommend(pref_vec, preference_summary, allergens, date_str, daily_mood, table)`
  -- single entry point. Handles mood blending, retrieval, filtering, and Gemini call.
  Returns `(recs, alts, query_vec)` -- top 3 recommendations, 2 diverse alternatives, query vector used.

**`user_prefs.py`** -- persistence layer for the `user_pref` Supabase table.

- `load_user_pref(user_id)` -- SELECT by UUID, returns dict or None (cold start).
- `save_user_pref(user_id, pref_vec, preference_summary, allergens)` -- upsert.
  Same function handles first-time signup and every subsequent vector update.
- `user_id` is the Supabase Auth UUID from the JWT `sub` claim -- same value
  Patrick's `get_current_user()` in `main.py` extracts.

**`register.py`** -- MLflow registration script.

- `GeminiFlashRAG` pyfunc wrapper: `predict(context, model_input)` accepts a dict
  with `pref_vec`, `preference_summary`, `allergens`, `date_str`, `daily_mood`, `table`.
  Imports from `recommend.py` inside `predict()` (not at module top) so MLflow can
  serialize the class without needing all dependencies at registration time.
- Logs 7 parameters: `embed_model`, `gen_model`, `alpha`, `beta`, `similarity_metric`,
  `top_k_retrieval`, `top_k_gemini`.
- Logs `sample_io.json` as an artifact -- documents the exact input/output schema.
- Note: `code_path` removed from `save_model()` (dropped in mlflow 3.x); `recommend.py`
  and `user_prefs.py` must be importable from the repo at load time.
- Registers as `gemini_flash_rag_v1` in the MLflow Model Registry. Version 2 is current (recs + alternatives).

### Design decisions made

- **alpha and beta are separate knobs.** alpha=0.3 governs how fast the stored vector
  learns from picks. beta=0.3 governs how strongly mood shifts today's query vector.
  They're independent -- tuning one doesn't affect the other.
- **Mood never touches pref_vec.** Only dish picks update the stored profile.
  Mood is a retrieval lens for today, not a learning signal.
- **`recommend()` does not update pref_vec.** The update requires a user pick, which
  is external. Keeping them separate makes the pipeline testable and stateless.
- **Upsert for user_pref.** No separate create/update logic -- one `save_user_pref()`
  call works for signup and for every subsequent pick.

### MLflow registration

- Restarted MLflow server on GCP VM with `--default-artifact-root gs://mlflow-model-artifacts`
  (was previously using local VM disk; experiment artifact location is baked in at creation
  so the old `lunch-buddy` experiment was deleted and recreated as `lunch-buddy-rag`).
- `gemini_flash_rag_v1` registered. Version 2 is current (recs + alternatives). Run ID:
  `a56631660fdd4405ae20c9f4548be4a0`. Artifacts (model + sample_io.json) stored in GCS.
- Compatibility note: mlflow 3.x client + 2.x server -- `log_model()` uses a new
  `/logged-models` endpoint not present in 2.x. Workaround: `save_model()` locally,
  `log_artifacts()` to upload, `register_model()` to register.
- Load the model: `mlflow.pyfunc.load_model("models:/gemini_flash_rag_v1/2")`

### Supabase

- `user_pref` table created (user_id UUID PK, preference_vector vector(768),
  preference_summary TEXT, allergens TEXT[], last_updated TIMESTAMPTZ).
- Table is live -- profiles are written via `--user_id` flag in `demo.py`. Deleting a
  row resets that user to cold start (fresh signup on next run).

### Pipeline design notes

- **mood_vec and query_vec are ephemeral.** Both are created inside a single `recommend()`
  call and never stored. mood_vec = embedding of today's mood string. query_vec = blend
  of pref_vec + mood_vec. Only pref_vec (updated by picks) is persisted to Supabase.
- **Retrieval limit is a known tradeoff.** Currently fetches top 40 from pgvector before
  filtering. 40 is arbitrary -- filtering (allergens + placeholders + dedup) can starve
  the pool on sparse days. Cleaner design: fetch 100-150, filter, then take top 10 by
  cosine rank. Not yet changed; noted for future tuning.
- **Judge LLM [WIP].** Architecture calls for a second Gemini call (Pro) to evaluate
  recommendation quality and alignment before results are returned. Not yet implemented.
  Generator = Gemini 2.5 Flash. Judge = Gemini 2.5 Pro (planned).

### Pending

- FastAPI `/predict` not yet wired to `recommend()` -- next milestone (Patrick coordinates).
- `python-jose` missing from `fastapi/requirements.txt` -- Patrick fix needed.
- Retrieval pool size -- consider raising limit or restructuring to filter-then-rank.
- Judge LLM -- wire Gemini Pro as a second evaluation pass after Flash generates recs.

---

## Session log — Apr 12 2026

### gemini_flash_rag v2 (teammate changes, pulled from `atl_model_exploration`)

Applied benchmark findings from exp_06. Four files changed:

**`recommend.py`**
- `thinking_budget=0` added to both `call_gemini()` and `generate_preference_summary()`.
  Disables extended reasoning in Gemini 2.5 Flash -- confirmed identical picks at 7x
  lower latency with higher JSON reliability.
- Dynamic beta: `BETA_WITH_MOOD=0.5` / `BETA_NO_MOOD=0.0` replacing fixed `BETA=0.3`.
  When no mood is given, `query_vec = pref_vec` directly (blend_mood not called).
- Prompt v2: two separate prompt branches. When `daily_mood` is provided, mood leads
  as the primary constraint and the profile is a tiebreaker. When no mood, profile-only
  prompt (mood section omitted entirely).
- `top_k_gemini` slice raised from `[:5]` to `[:10]` (line 444).
- Defensive `dish_name` cleanup: strips `" at DiningHall (MealTime)"` suffix in `clean()`.
- `format_rules` and `json_schema` extracted as shared strings to avoid duplication
  across the two prompt branches.
- Imported `google.genai.types` as `genai_types` (needed for `ThinkingConfig`).

**`register.py`**
- Model name changed to `gemini_flash_rag_v2`.
- PARAMS updated: `thinking_budget=0`, `prompt_version="v2"`, `beta_with_mood=0.5`,
  `beta_no_mood=0.0`, `top_k_gemini=10`.
- Sample I/O updated to illustrate mood-primary behavior (light dishes surface over
  spicy Korean when mood = "something light today").

**`sample_io.json`**
- Regenerated to match v2 sample output.

**`user_prefs.py`**
- `DATABASE_URL` switched to `DATABASE_URL_IPV4` for Supabase connection.
  Confirm `DATABASE_URL_IPV4` is set in `models/.env` -- missing key silently returns
  None and causes runtime failure in `load_user_pref` / `save_user_pref`.

### MLflow registration

- `gemini_flash_rag_v2` registered in MLflow Model Registry by teammate.
- Load: `mlflow.pyfunc.load_model("models:/gemini_flash_rag_v2/1")`

### Pending

- FastAPI `/predict` still not wired -- primary next milestone (Patrick coordinates).
- `python-jose` still missing from `fastapi/requirements.txt` -- Patrick fix needed.
- Confirm `DATABASE_URL_IPV4` is present in `models/.env` after `user_prefs.py` change.
- Judge LLM -- Gemini Pro evaluation pass still not wired [WIP].
- Retrieval pool size -- fetch 100-150 then filter-then-rank still a future fix.

---

## Session log — May 2 2026

### 3-vector architecture — feature/3-vector-architecture branch

Implemented the signal separation designed to fix mood dilution (diagnosed in exp_08, quantified in exp_09). Previously, signup preference and pick history were permanently fused into one EMA vector (`preference_vector`); mood was blended on top at query time. The new design splits them into three independent vectors so each can be weighted appropriately.

**New blend formula (mood present):**
```
query_vec = α₁·mood_vec + α₂·recent_vec + α₃·original_vec
α₁ = 0.8 (fixed), α₂ = 0.2·(n/5), α₃ = 0.2 - α₂   [n = min(picks, 5)]
```
At n=0: `0.8·mood + 0.2·original` (same as the exp_09 winner). At n≥5: `0.8·mood + 0.2·recent` (original fully replaced). No mood: `query_vec = pref_vec` unchanged.

**`supabase/migrations/20260502000000_add_3vector_columns.sql`**
- `original_profile_vector vector(768)` — nullable; NULL = pre-migration user
- `recent_choices jsonb NOT NULL DEFAULT '[]'::jsonb` — last ≤5 pick embeddings, most-recent first

**`gemini_flash_rag/user_prefs.py`**
- `load_user_pref()` extended: returns `original_profile_vector` (falls back to `pref_vec` when NULL), `recent_choices_raw`, `recent_choices_vecs`
- `save_user_pref()` extended: new optional params `recent_choices` and `original_profile_vector`; `COALESCE` in SQL makes `original_profile_vector` write-once (never overwritten by picks); `recent_choices` conditional-overwrite (only updates when a non-empty array is passed)

**`gemini_flash_rag/recommend.py`**
- New `three_way_blend(mood_vec, recent_vecs, original_vec, beta)` — implements the ramp formula
- `recommend()`: two new optional params (`original_profile_vec`, `recent_choices_vecs`); uses `three_way_blend()` when `original_profile_vec` is not None, falls back to `blend_mood()` for pre-migration users
- `BETA_WITH_MOOD` updated 0.5 → 0.8 (exp_09 result)

### Pending

- Apply Supabase migration to live database
- Wire new fields through `fastapi/main.py`: `/predict` passes `original_profile_vec` and `recent_choices_vecs`; `/pick` prepends new dish embedding to `recent_choices` and trims to 5
