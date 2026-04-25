# Changelog — exp_07 connection pooling

## [2026-04-25] — Add connection pooling benchmark

**What:** New experiment that compares fresh-connection-per-call vs `psycopg2.pool.ThreadedConnectionPool` for the `retrieve_dishes()` Supabase query.
**Why:** `recommend.py` opens and closes a new connection on every DB call. Each new connection pays TCP + TLS + Postgres auth overhead (~50–150ms). Before refactoring `recommend.py` to use a pool, validate empirically that the savings are real and meaningful in this codebase.
**Files:**
- `run_benchmark.py` — new
- `CHANGELOG.md` — new
**Notes:**
- Benchmark isolates DB layer only — no Gemini calls in the timed loop (Gemini latency would swamp the ~100ms signal). One `get_embedding()` call up front, vector reused for all benchmark runs.
- Pool config: `minconn=1, maxconn=5`. First pooled call still pays connection setup; calls 2+ reuse it (warmup effect printed separately).
- Logs to MLflow experiment `lunch-buddy-rag` and writes `benchmark_results.json` locally.
- Default `--runs 10` (20 total DB calls). Use `--runs 30` for a more rigorous final measurement.
- Phase 2 (applying pooling to `recommend.py`) is gated on reviewing these results together.
