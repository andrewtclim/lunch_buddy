# Lunch Buddy — RAG Model Log

---

## Overview

A conversational RAG pipeline where a user sends a natural language query and the system retrieves semantically matching dishes from Supabase, then passes them to Gemini to generate a top-3 recommendation with explanation.

**Current focus: exp_04 (RAG + learning).** Exp_03 (static RAG) is on hold due to a hanging Vertex AI embedding call. Exp_04 is the more interesting direction -- it adds an evolving preference vector and an LLM-generated preference summary that both update after each user pick.

---

## Background

From exp_01 and exp_02 we have:

- **Supabase + pgvector** -- `daily_menu` with 768-dim dish embeddings, cosine distance operator ready. `backfill_menu` has 10 days of data (2026-03-29 to 2026-04-07) for experiment use.
- **Embeddings** -- `text-embedding-004` via Vertex AI. Same model for dishes and users, so everything lives in the same vector space.
- **Gemini 2.5 Flash** -- already authenticated via Vertex AI, no extra setup.
- **FastAPI skeleton** -- `POST /predict` exists, currently a stub.
- **Allergen filter** -- implemented and tested across both experiments.

---

## Experiment Summary

| | Exp_02 (vector-only) | Exp_03 (RAG static) | Exp_04 (RAG learning) |
|---|---|---|---|
| Retrieval query | User preference vector (fixed) | Signup text embedding (fixed) | Preference vector (evolves each day) |
| Final ranking | Cosine similarity | Cosine similarity + LLM | Cosine similarity + LLM |
| Prompt | None | Raw signup text | Gemini-generated running summary |
| Learning loop | Yes (vector updates) | No | Yes (vector + summary both update) |
| Evaluation | Hidden profile cosine | Hidden profile cosine | Initial vs. final vector vs. true preference |
| Demo-ready | No | No | Yes (interactive terminal) |
| Status | Done | Blocked (hanging embed call) | Active |

---

## Exp_03 (RAG Static) -- On Hold

**Folder:** `models/experiments/exp_03_RAG_static/`

Same 15 mock users and 10 days of `backfill_menu` as exp_02. Each day: embed signup text (fixed) → cosine retrieve top 20 → filter → Gemini re-ranks to top 3 → score against hidden profile.

**Status (Apr 8 2026):** Simulation hangs silently on the first `get_embedding()` call. Near-zero CPU for 30+ min, no output. Root cause: no timeout on the Vertex AI `embed_content` call -- connection stalls indefinitely.

**Fix needed:** Add a timeout to the embedding call (e.g. wrap in `concurrent.futures` with a deadline) or set a request timeout on the `genai.Client`. Then re-run and log to MLflow.

Partial results from an earlier run (14/15 users, 10 days): scores ~0.42-0.58 per user per day, visually above exp_02 baseline (~0.467). Full comparison pending a clean run.

---

## Exp_04 (RAG Learning) -- Active

**Folder:** `models/experiments/exp_04_RAG_learning/`

**Key idea:** The preference vector and the natural language summary both evolve after each user pick. Retrieval improves each day because the query vector itself shifts toward what the user has been choosing.

### Pipeline

```
user describes food preferences (signup text)
              |
              v
embed signup text --> initial pref_vec
              |
              v
[for each of 10 days]
              |
              v
cosine search backfill_menu using current pref_vec --> top 20 dishes
              |
              v
filter placeholders, allergens, duplicates --> top 5
              |
              v
Gemini Flash: given preference_summary + 5 dishes, recommend top 3
              |
              v
user picks 1, 2, or 3
              |
              v
fetch chosen dish embedding from Supabase
pref_vec = normalize(0.7 * pref_vec + 0.3 * dish_vec)   [alpha=0.3 EMA]
preference_summary = Gemini summarizes signup_text + all picks so far
              |
              v
[repeat for next day]
              |
              v
user reveals true hidden preferences
score: cosine(final_pref_vec, true_vec) vs cosine(initial_pref_vec, true_vec)
```

### Design Decisions

**Evolving preference vector (alpha=0.3 EMA).** After each pick, `pref_vec` is nudged 30% toward the chosen dish embedding and re-normalized. This shifts tomorrow's retrieval toward what the user actually chose today, not just what they described at signup.

**Gemini-generated running summary (Option B).** After each pick, Gemini writes a fresh 1-sentence description based on the original signup text + all dishes chosen so far. This summary is the natural language prompt for the next day's recommendation call -- both the vector and the words evolve in parallel.

**Evaluation at the end.** User reveals their true hidden preferences as free text. We embed it and compare cosine similarity with both the initial and final preference vectors to measure how much the system converged toward their actual tastes.

### Demo Run Results (Apr 8 2026 -- Andrew, 1 user)

- **Signup text:** "I enjoy spicy and asian foods"
- **True preference:** "enjoy korean and chinese foods especially, I love spicy foods and my favorite protein is chicken"
- **Initial score:** 0.7913
- **Final score:** 0.5926
- **Delta:** -0.199

**What happened:** Score went down. The signup text was already a close proxy for the true preference (0.791 is high), leaving little room to improve. Picks drifted toward Indian/curry food (Butter Chicken appeared 3 times, Tandoori, Biriyani) which pulled the vector away from Korean/Chinese specifically. The system converged, but toward the wrong part of "spicy Asian."

**Key findings:**

1. **Ceiling problem** -- if signup text is already a strong match for true preferences, the learning signal has nowhere to go but introduce drift. The experiment works best when signup is vague or intentionally mismatched.
2. **Retrieval rut** -- once the vector shifted toward Indian food (after a Butter Chicken pick on day 6), it kept pulling similar dishes. Low retrieval diversity compounds the drift.
3. **Alpha sensitivity** -- 0.3 may be too aggressive. A smaller alpha (0.1-0.15) would let the vector shift more gradually and resist one-off picks pulling it off course.

### Open Questions / Next Steps

- Try alpha=0.1 and alpha=0.15 to see if slower learning reduces drift
- Add retrieval diversity -- e.g. penalize dishes from the same dining hall or cuisine cluster as yesterday's pick
- Run mock user batch version (15 users with intentionally mismatched signup text vs. hidden profile) -- this is the case where learning should shine
- Does the running summary actually track the vector? Compare summary text evolution to vector cosine trajectory day by day
- Wire exp_04 learning loop into FastAPI for a live demo
