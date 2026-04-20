# Lunch Buddy Modeling Notes

---

## User Preference System Design

### Core Idea
Each user has a single preference vector (768-dim, same space as dish embeddings). It starts from signup and drifts toward their actual taste over time as they accept recommendations.

### Vector Update Formula
```
new_vector = (α × old_user_vector) + ((1 - α) × chosen_dish_vector)
```
- `α = 0.85` -each accepted rec nudges the vector slightly without overreacting
- No special decay logic needed so signup influence fades naturally as real choices accumulate

### Trigger
Vector updates when the user **explicitly taps/accepts a recommendation**. This fires a call to the FastAPI layer which pulls the dish's embedding from `daily_menu`, blends it into the user's current vector, and writes back to `users`.

### Schema

**`users` table**
```
user_id           -- primary key
preference_text   -- signup answers (human-readable, for display/debug)
preference_vector -- current embedding, updates on each accepted rec
allergens         -- text[], set at signup (e.g. ["peanuts", "gluten"])
created_at
```

**`meal_selections` table**
```
selection_id
user_id       -- FK → users
dish_name
dining_hall
meal_time
date_served
selected_at   -- timestamp
```
`meal_selections` is the audit trail and lets us reconstruct history or change the update formula later without losing data.

### Recommendation Flow
```
1.  User opens app → POST /predict with user_id
2.  Fetch user's preference_vector and allergens from users table
3.  Cosine similarity query against today's daily_menu embeddings (top 10)
4.  Allergen filter removes hard exclusions
5.  LLM #1 (Gemini Flash) generates top-3 recommendation with rationale
6.  LLM #2 (Gemini Flash/Pro) verifies no allergen slippage or hallucinations
7.  Return top 3 with rationale to user
8.  User taps a choice → POST /select
9.  Fetch chosen dish's embedding from daily_menu
10. Blend: new_vector = 0.85 × old + 0.15 × dish_vector
11. Write new_vector back to users table
12. Log row to meal_selections
```

---

## User Preference Embeddings

At signup the user describes their taste in plain text (e.g. "I like spicy food, Korean BBQ, and noodles. I am vegetarian."). That string gets passed through the same `text-embedding-004` model used for dish embeddings, producing a 768-dim vector stored in the `users` table.

`text-embedding-004` is Google's text embedding model, accessible via Vertex AI. It maps text of any length to a fixed 768-dimensional vector and is strong at semantic similarity tasks, meaning it understands that "spicy Korean food" and "mapo tofu with chili bean paste" are related concepts even though the words don't overlap. We use it for both dishes and users because keeping everything in the same model means keeping everything in the same vector space, which is required for cosine similarity to be meaningful.

As the user accepts recommendations, their vector drifts via the weighted average formula toward dishes they actually choose, no manual updates needed.

---

## Allergen Handling

Allergies are a hard filter, not a preference. Any dish with a matching allergen is dropped before scoring even starts. No similarity score overrides it.

Implemented as a standalone function (separate from scoring) so it stays auditable and testable on its own. Uses set intersection: if dish allergens and user allergens overlap at all, the dish is out.

`allergens` stored as a text array on the `users` table, set at signup. Fetched in the same query as the preference vector -no extra round trip.

---

## LLM Pipeline Design

### Division of Labor: Vector Math vs LLM

**Vector similarity (no LLM)**
Cosine similarity search runs purely in Supabase/pgvector (only math). Fast, cheap, runs on every request. Output: a ranked list of candidate dishes numerically close to the user's taste vector.

**LLM #1 -Generator**
Takes the top candidates from vector search and writes the actual recommendation in natural language which explains *why* each dish fits this user.

```
[vector search] → top 5 candidate dishes
      ↓
[LLM prompt]:
  "The user likes spicy, vegetarian food.
   Here are today's top 5 matching dishes with ingredients: ...
   Write a friendly top-3 recommendation with a one-line reason for each."
      ↓
[LLM output]:
  "1. Mapo Tofu -bold, spicy, and matches your vegetarian preference
   2. ..."
```

The LLM adds the reasoning layer: reading user profile + dish details and explaining the match in plain English. Also enables soft logic in the prompt (e.g. "don't recommend the same dish they had yesterday") without hardcoding rules.

**LLM #2 -Verifier**
After LLM #1 picks top 3, a second call acts as a critic: *"Given this user's allergens and preferences, does this recommendation make sense? Flag anything suspicious."* Catches hallucinations or allergy slippage before it hits the user.

### Full Pipeline
```
pgvector (math)    → candidate dishes
LLM #1 (generator) → top-3 recommendation + rationale
LLM #2 (verifier)  → sanity check / flag issues
```

---

## Cosine Similarity

Measures the angle between two vectors, not their length. Score of 1.0 means identical direction, 0.0 means unrelated, -1.0 means opposite.

Chosen over euclidean distance and dot product because it normalizes for vector length -a new user with 2 choices and a returning user with 100 are compared on equal footing.

### Worked Example
```
user_vector = [0.9, 0.1, 0.2]   # savory, not sweet, a little spicy
dish_A      = [0.8, 0.2, 0.3]   # savory, slightly sweet, mild spice
dish_B      = [0.1, 0.9, 0.1]   # very sweet, not savory

similarity(user, dish_A) = 0.984   <- strong match
similarity(user, dish_B) = 0.238   <- poor match
```

### In Supabase
pgvector handles the math natively. One query ranks all of today's dishes by cosine distance to the user's preference vector, returns top 5 candidates for the allergen filter and LLM to process.

---

## Mock Users + Hidden Preferences Experiment

Since there are no real users yet, we will use 15 synthetic users to evaluate recommendation quality. Each user has a vague signup description (what the model sees) and a hidden profile (ground truth, never shown to the model). After generating recommendations, we embed the hidden profile and compute cosine similarity against the suggested dishes to score how well the model did.

The 15 users cover a range of dietary styles, cuisine preferences, and allergen combinations to stress-test the full pipeline. Defined in `models/experiments/mock_users.py`, saved to `mock_users.json` for use by evaluation scripts.

---

## Model Selection

We evaluated three candidate LLMs: Gemini (Flash and Pro), Qwen2.5 via Together AI, and Ollama. The full cost and capability comparison is in `models/models_exp_comparison.md`. Performance results against mock users will be added there once experiments are run.

**Decision: Gemini 2.5 Flash.**

We are already on GCP and using Vertex AI for dish embeddings, so Gemini requires no additional auth or API keys. Flash and Pro scored identically (0.4609 avg cosine similarity across 15 mock users) but Flash was 2x faster (8.28s vs 15.31s). Flash is the clear choice for the generator. Full results in `models/models_exp_comparison.md`.

Qwen adds a separate API key and third-party dependency with no clear quality advantage for this use case. Ollama is useful for local development but is not practical in a containerized GCP deployment without dedicated GPU hardware.

Exp_01 simulations (10-round and 50-round cold start) were near-flat — expected, since the same menu surfaced every round regardless of vector drift. Exp_01 validated the pipeline; exp_02 is where meaningful evaluation begins.

---

## Exp_02 — Multi-Day Simulation (Apr 7, 2026)

**Setup.** Backfilled 10 days of real Stanford dining data (2026-03-29 → 2026-04-07) from GCS into `backfill_menu`. ~300-350 dishes per day, 6-8 halls depending on the day. 15 mock users, alpha=0.85, 1 round per day. MLflow run: `exp02-multiday-simulation`.

**How a mock user decides.** Each day: query `backfill_menu` ranked by the user's current preference vector, filter placeholders and allergens, present top 3. The user picks whichever of those 3 is closest to their hidden profile. That dish gets blended into the vector. The user always picks optimally from what is shown — the constraint is what the system surfaces, not the user's judgment.

**User 04 — Mediterranean (signup: "light meals, mostly Mediterranean", hidden: hummus, falafel, grilled fish, tabbouleh).**

| Day | Chosen | Score |
|-----|--------|-------|
| 0 | Antipasto Salad | 0.482 |
| 1 | Roasted Asparagus | 0.486 |
| 2 | Mediterranean Salad | 0.525 |
| 3 | Beef Gyro | 0.522 |
| 4-9 | Grilled Chicken (every day) | ~0.511 |

Day 4's menu had no strong Mediterranean options in the top 3. Grilled Chicken was the best available, the vector shifted toward generic grilled protein, and Grilled Chicken dominated every subsequent day. Score plateaued at 0.51 — not a collapse, but no further improvement either.

**Bad menu days are a problem — User 05 (signup: "I like spicy food", hidden: thai curry, spicy ramen, hot wings).**

Weekdays surfaced Chicken Guajillo, Butter Chicken, Gochujang — reasonable spicy matches. Apr 4-5 (weekend, 6 halls) had none of those. Grilled Chicken was the closest available. Two weekend picks pulled the vector toward generic chicken; spicy dishes stopped surfacing even after weekdays returned. Score fell from 0.46 to 0.35. Score delta: **-0.112**, worst of all 15 users.

**Results.**

| Metric | Value |
|--------|-------|
| Avg score day 0 | 0.4676 |
| Avg score day 9 | 0.4510 |
| Overall improvement | -0.0166 |
| Flat users (no change across 10 days) | 9 / 15 |

9 users chose the same dish every single day ("Flavor Forward Legumes," "Craveable Grains," "Gluten-Free Waffles" — dining hall staples present on every menu). Once chosen, alpha=0.85 locks the vector in and nothing else surfaces.

**Takeaways.**
- Bad menu days (weekends, limited halls) corrupt the vector and recovery is slow at alpha=0.85.
- Repeat staples cause immediate lock-in. Diversity in what is presented matters as much as ranking quality.
- 10 days is insufficient for niche profiles (spicy, Korean, Ethiopian). More menu history needed.
- Exp_01's apparent improvement was an artifact of cycling the same menu repeatedly.

**Open question — LLM-as-user.** Current mock users choose by pure cosine similarity to a short hidden profile string. A richer approach: give an LLM a detailed persona (dietary history, flavor preferences, mood, context) and have it decide which dish to pick from the presented top 3. This would produce more realistic, noisy choices and stress-test the learning loop more honestly. To be evaluated in a future experiment.
