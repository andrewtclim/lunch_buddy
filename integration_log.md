# Integration Log -- feat/wire-predict

## Note on model serving
/predict does NOT load a model from MLflow. It runs `recommend.py` live from the repo -- cosine search, filtering, and Gemini API calls happen in real-time on every request. The MLflow-registered models (gemini_flash_rag_v1, v2) are snapshots for tracking/comparison only. To update the model, update the code in `models/gemini_flash_rag/` and the next request uses it immediately.

## Step 1: Audit (read only)
- Read `fastapi/main.py`, `recommend.py`, `user_prefs.py`, and all `frontend/src/` files
- Identified 11 mismatches between the current skeleton API and the real pipeline
- Key gaps: wrong request/response schemas, `_model.predict()` instead of `recommend()`, no sys.path bridge, new user cold start unhandled, frontend types and rendering don't match pipeline output

## Step 2: Rewrite PredictRequest
- Replaced `{user_id, preferences, constraints}` with `{mood, date}`
- `mood`: optional free-text string passed to recommend() as `daily_mood`
- `date`: optional "YYYY-MM-DD", defaults to today server-side
- Removed `user_id` from body -- comes from JWT instead
- Removed `preferences`/`constraints` -- come from Supabase via load_user_pref()

## Step 3: Rewrite PredictResponse
- Added `DishCard` model: `{dish_name, dining_hall, reason}`
- Replaced `{suggestions: string[], rationale}` with `{recommendations: DishCard[], alternatives: DishCard[], preference_summary}`
- `preference_summary` included so frontend can show what the model "thinks" about the user
- `preference_vector` intentionally NOT returned -- 768-dim array, server-side only

## Step 4: Rewrite predict()
- Added `sys.path` insert so `recommend` and `user_prefs` are importable from `fastapi/`
- `predict()` now: loads user profile -> guards new users with 422 -> calls `recommend()` directly -> returns DishCards
- Removed `_model.predict()` and the 503 model-loaded check -- no longer going through MLflow at request time
- No EMA update -- deferred to future `/pick` endpoint
- `date` defaults to `date.today().isoformat()` if client omits it

## Step 5: Smoke test
- Wrote curl command with JWT + mood payload
- Server started successfully (uvicorn on port 8000)
- Token handling via browser console confirmed working
- Full end-to-end test deferred -- frontend will handle tokens automatically

## Step 6: Frontend recommendation card UI
- `api.ts`: new `DishCard` type, request is `{mood?, date?}`, response is `{recommendations, alternatives, preference_summary}`
- `App.tsx`: form sends `{mood}`, displays 5 dish cards (3 recs + 2 alts), each with dish_name/dining_hall/reason and "Pick this" button
- `App.css`: added `.dish-cards`, `.dish-card`, `.dish-name`, `.dish-hall`, `.dish-reason`, `.preference-hint` styles matching existing design
- "Pick this" logs to console for now -- will wire to `/pick` endpoint later
- Removed unused `useMemo` import and `constraints` computation

## Step 7: Wire into router
- No change needed -- App.tsx already uses view state, not a router
- Recommendation cards are the default authenticated home screen
- Flow: login -> home (mood form + cards) -> profile (via nav link)

## What's left
- **`/pick` endpoint** -- EMA update + save_user_pref() when user clicks "Pick this" (preference vector doesn't evolve until this is wired)
- **Taste profile onboarding** -- new users get 422 because no user_pref row exists; need a signup flow to initialize preference vector
- **Allergens sync** -- ProfilePage saves allergens to localStorage only, not Supabase; load_user_pref() won't see them
- **CI tests** -- test_api.py tests need updating for new PredictRequest/PredictResponse schemas
