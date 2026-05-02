# Changelog — gemini_flash_rag

## [2026-04-29] — Wider retrieval funnel (exp_08 mood-faithfulness benchmark)

**What:** Increased `top_k_retrieval` default 40 → 80 and `top_k_gemini` default 10 → 20 in `recommend()`.
**Why:** exp_08 mood-faithfulness benchmark showed the wider funnel adds +12pp faith@3 (0.652 → 0.775) with ~90ms median latency increase. Root cause: `backfill_menu` stores ~10 rows per unique dish (one per station), so 40 raw candidates collapse to ~5 unique dishes after `deduplicate()`. Fetching 80 gives Gemini a richer and more mood-relevant slate without changing blend ratio or profile signal.
**Files:**
- `recommend.py` — `top_k_retrieval` default 40→80, `top_k_gemini` default 10→20, header comment updated
**Notes:**
- `BETA_WITH_MOOD` intentionally unchanged at 0.5. `beta_1_0` scored higher (0.802) but removes all profile signal at retrieval and regresses one case. The right fix is the planned 3-vector architecture (separate mood / recent-history / original-profile signals), not a beta band-aid.
- `benchmark_results.json` in `exp_08_mood_faithfulness/` has corrected relationship labels (10 items reclassified adjacent→contrast after manual review).
- See `experiments/exp_08_mood_faithfulness/` for full methodology and results.

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
