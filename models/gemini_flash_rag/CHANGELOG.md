# Changelog — gemini_flash_rag

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
