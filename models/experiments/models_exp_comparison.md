# Lunch Buddy -Model Comparison

## Model Candidates

| | Gemini 2.5 Flash | Gemini 2.5 Pro | Qwen2.5-72B (Together AI) | Ollama |
|---|---|---|---|---|
| Input (per 1M tokens) | $0.30 | $1.25 | $1.20 | Free |
| Output (per 1M tokens) | $2.50 | $10.00 | $1.20 | Free |
| Latency | Fast | Moderate | Depends on host | Slow without GPU |
| GCP integration | Native, same auth as embeddings | Native | Separate API key needed | No cloud integration |
| Dual-LLM fit | Clean | Clean | Works but adds friction | Complicated in containers |
| Best for | Our production path | Higher quality when needed | Open-source self-hosted stack | Local dev and testing |

Notes:
- Gemini 2.5 Flash and Pro are the current available models on this Vertex AI project
- Ollama is free but requires hardware -no GPU on our current GCP setup makes it impractical for production
- For a typical dining recommendation (short prompt, short response) we estimate under 1,000 tokens per call -cost is negligible with Flash

**Starting model: Gemini 2.5 Flash**

---

## Performance Comparison

*Run: Apr 6 2026, 15 mock users, single day of menu data (246 dishes)*

| | Gemini 2.5 Flash | Gemini 2.5 Pro |
|---|---|---|
| Avg cosine similarity (top 3 vs hidden profile) | 0.4609 | 0.4609 |
| Avg latency per user | 8.28s | 15.31s |
| Allergen violations | 0 | 0 |

Both models scored identically because cosine similarity is evaluated against the dish vectors, not the LLM output text. Pro is ~2x slower with no quality gain on this metric. Flash is the clear choice.

## Simulation Results

*10 rounds, 15 mock users, alpha=0.85*

| Round | Avg cosine score |
|---|---|
| 0 (baseline) | 0.4702 |
| 5 | 0.4703 |
| 10 | 0.4649 |
| Overall change | -0.0053 |

Scores were flat across rounds. Only one day of menu data means the same dishes surface every round regardless of vector drift. The menu also lacks coverage for several user profiles (no Korean dishes at all). The math is correct — the experiment needs multi-day data to show convergence. Pinned for later.

**Cold start simulation (side note):** Ran a separate 50-round experiment with no signup text, initializing each user from the average dish embedding. Also flat for the same reasons. `run_cold_simulation.py`, results in `cold_simulation_results.json`.
