# 🥗 Lunch Buddy

> AI-powered lunch recommendations for Stanford students — right place, right meal, right now.

**Team:** Lynn Tong · Andrew Lim · Patrick Crouch  
**Course:** MLOps — Spring 2026

---

## The Problem

Meal information across Stanford's campus is scattered. Students waste time bouncing between dining hall websites before deciding where and what to eat. Lunch Buddy fixes that.

## What It Does

Lunch Buddy is a web app that recommends the best dining hall and dish for you based on:
- Your **location** (closest cafeterias within a set radius)
- Your **preference profile** (tastes, dietary restrictions, allergies)
- **Today's menus** (scraped fresh each morning from Stanford's dining site)

An LLM agent surfaces a ranked top-3 recommendation, personalized to you and updated daily.

---

## System Overview

| Layer | Stack |
|---|---|
| Data ingestion | Web scraper → runs daily at 8am |
| Storage | PostgreSQL (menus, dishes, cafeterias) + vector store (RAG) |
| User profiles | Preferences, allergies, feedback history |
| Recommendation | LLM + RAG pipeline, geolocation-aware ranking |
| Deployment | Docker · GCP · Vertex AI |
| ML Pipeline | MLFlow (experiment tracking) · Metaflow (CD) |
| Monitoring | Evidently (data + prediction drift) |
| CI/CD | GitHub Actions · Pytest |

---

## Datasets

- **Stanford dining hall menus** — scraped daily, stored progressively as training data
- **Geospatial data** — user location + dining hall coordinates for proximity ranking
- **User preference surveys** — manually collected onboarding data

---

## Roadmap

| Week | Dates | Milestone |
|---|---|---|
| 1 | Mar 27 – Apr 2 | Scraper MVP, PostgreSQL schema, MLFlow tracking, FastAPI wrapper |
| 2 | Apr 3 – Apr 9 | Dockerize, GCP Artifact Registry, system design, Vertex AI deploy |
| 3 | Apr 10 – Apr 17 | 9 Pytest test cases, CI via GitHub Actions |
| 4 | Apr 18 – Apr 25 | Full CI/CD pipeline, Metaflow, product pitch video, app demo video |
| 5 | Apr 26 – May 1 | Evidently monitoring, Ruff cleanup, final technical slides |

---

## Getting Started

*Setup instructions coming soon.*

---

## License

For academic use only — Stanford MLOps Spring 2026.