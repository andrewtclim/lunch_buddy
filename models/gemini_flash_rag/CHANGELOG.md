# Changelog — gemini_flash_rag

## [2026-05-02] — 3-vector architecture + BETA_WITH_MOOD 0.5 → 0.8

**What:** Separated mood, recent picks, and original signup preference into three independent retrieval signals. Added `three_way_blend()` and updated `recommend()` to accept `original_profile_vec` and `recent_choices_vecs`. `BETA_WITH_MOOD` raised from 0.5 to 0.8.
**Why:** exp_09 ablation (48 profiles, 4-pick simulation, SE=0.05) showed adjacent faith@3 improving monotonically 0.774 → 0.889 as β goes 0.5 → 0.8, with no regression on aligned cases. The single blended `pref_vec` fused signup preferences and pick history irrecoverably; 3-vector design lets recent choices grow in weight (up to 20% at 5 picks) while the original profile fades, without disrupting mood dominance.
**Files:**
- `recommend.py` — new `three_way_blend()`, updated `recommend()` signature (2 new optional params), `BETA_WITH_MOOD=0.8`
- `user_prefs.py` — extended `load_user_pref()` and `save_user_pref()` for `original_profile_vector` and `recent_choices`
- `CHANGELOG.md` — this entry
**Notes:**
- Backward compat: if `original_profile_vec=None` (pre-migration users), falls back to existing `blend_mood()`. No breakage for current users.
- `original_profile_vector` uses `COALESCE` in the upsert SQL — written once at first save, never overwritten by picks.
- Supabase migration `20260502000000_add_3vector_columns.sql` must be applied before the new fields are usable in production.
- `fastapi/main.py` wiring pending (Step 6).

## [2026-04-29] — wider_funnel retrieval defaults (exp_08)

**What:** `top_k_retrieval` raised 40 → 80; `top_k_gemini` raised 10 → 20. These are now the defaults in `recommend()`.
**Why:** exp_08 mood faithfulness eval showed wider pool improves adjacent faith@3 from 0.633 → 0.717 (+0.084) and contrast from 0.652 → 0.844 (+0.192) with no regression on aligned cases. Larger retrieval pool gives the mood-blended query vector more candidate dishes to find mood matches before the allergen + placeholder filters reduce the pool.
**Files:**
- `recommend.py` — updated `top_k_retrieval` default 40 → 80, `top_k_gemini` default 10 → 20
- `CHANGELOG.md` — this entry
**Notes:**
- Gemini latency is insensitive to candidate count in this range (pattern-matching, not generation). No meaningful latency regression expected.
- See `models/experiments/exp_08_mood_faithfulness/` for full results.

## [2026-04-25] — Apply connection pooling to Supabase calls

**What:** All three DB functions (`get_available_dates`, `retrieve_dishes`, `fetch_dish_embedding`) now use a module-level `psycopg2.pool.ThreadedConnectionPool` (minconn=1, maxconn=5) instead of opening a fresh connection per call.
**Why:** exp_07 measured 3.3x speedup on `retrieve_dishes()` (avg 761ms → 228ms) and 4x lower p95 latency (1163ms → 273ms). Each fresh connection to Supabase pays ~530ms of TCP+TLS+auth handshake; the pool keeps connections warm and reuses them. For a full `backfill_menu` demo run (~30 DB calls), this saves ~16 seconds.
**Files:**
- `recommend.py` — added `psycopg2.pool` import, module-level `_pool` init, refactored 3 DB functions to `_pool.getconn()` / `_pool.putconn()` in `try/finally`
- `CHANGELOG.md` — new
**Notes:**
- `try/finally` is critical — without it, an exception during a query would leak the connection and eventually exhaust the pool.
- Pool is process-scoped: it's initialized once when `recommend.py` is imported, persists for the lifetime of the process. Long-running processes (FastAPI server, demo loops) get the most benefit.
- No API change to any of the three functions — call sites in `demo.py`, exp folders, and FastAPI deploy don't need updates.
- See `models/experiments/exp_07_connection_pooling/` for the benchmark and full results.
