# Lunch Buddy

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

## Model serving API (FastAPI + MLflow)

The milestone API loads a **pyfunc** model from the **MLflow Model Registry** at startup and exposes JSON endpoints for health checks and predictions.

### Where the endpoints are defined

All HTTP routes (`/`, `/health`, `/predict`) live in:

- **`fastapi/main.py`**

### Prerequisites

- **Python 3.11+** recommended (matches the Docker image).
- **MLflow tracking server** reachable from your machine (URL goes in `fastapi/.env` — see below).
- **Model artifacts in GCS**: loading the real model requires Google credentials that can read the artifact bucket, plus a **GCP project ID** for the client libraries.

---

### Run the API locally (without Docker)

From the repo root:

```bash
cd fastapi
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Environment file (`fastapi/.env`).** The real `.env` is **not** in Git (it is listed in `.gitignore`), so you create it locally.

   From `fastapi/`:

   ```bash
   touch .env
   ```

   Then open **`fastapi/.env`** and paste the following, or merge these lines with your own values:

   ```env
   MLFLOW_TRACKING_URI=http://35.232.122.64:5000
   MLFLOW_MODEL_URI=models:/dummy_model/1
   USE_STUB_MODEL=0
   ```

   Change `MLFLOW_MODEL_URI` if your registry name, version, or stage differs (e.g. `models:/dummy_model/Production`). Use `USE_STUB_MODEL=1` only for quick tests without MLflow/GCS.

   `main.py` loads this file automatically via `python-dotenv`.

2. Authenticate for GCS (artifacts) and set the project the libraries should use:

   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project lunch-buddy-491800
   export GOOGLE_CLOUD_PROJECT=lunch-buddy-491800
   ```

   Your Google account also needs **Storage Object Viewer** (or equivalent) on the MLflow artifact bucket.

3. Start the server:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Build and run with Docker

1. **Start Docker and stay signed in**  
   - **Open Docker Desktop** (macOS / Windows) or ensure the Docker daemon is running (Linux). Wait until the UI shows the engine is **running** — `docker build` and `docker run` need this.  
   - **Sign in to Docker Hub** from a terminal (required for **`docker push`**; optional if you only build and run locally):

     ```bash
     docker login
     ```

   - Use your existing Docker Hub username and password (or access token). You should see “Login Succeeded”.

2. **Build the image** from the **`fastapi/`** directory (where the `Dockerfile` lives):

```bash
cd fastapi
docker build -t lunch-buddy-api:latest .
```

3. **Tag and push** to Docker Hub using **your** Docker Hub username (must match the image name you submit and screenshot for grading). You must be logged in (`docker login`) for `docker push`:

```bash
docker tag lunch-buddy-api:latest YOUR_DOCKERHUB_USERNAME/lunch-buddy-api:latest
docker push YOUR_DOCKERHUB_USERNAME/lunch-buddy-api:latest
```

4. **Run the container.** Keep Docker running. Values below match **`fastapi/.env`** / local setup: MLflow at `http://35.232.122.64:5000`, model `models:/dummy_model/1`, and GCP project `lunch-buddy-491800` (same as `GOOGLE_CLOUD_PROJECT` in the local steps). Replace **`YOUR_DOCKERHUB_USERNAME`** with your Docker Hub name. This mounts **Application Default Credentials** and maps host **8000** → container **8080**:

```bash
docker run --rm -p 8000:8080 \
  -e GOOGLE_CLOUD_PROJECT=lunch-buddy-491800 \
  -e MLFLOW_TRACKING_URI=http://35.232.122.64:5000 \
  -e MLFLOW_MODEL_URI=models:/dummy_model/1 \
  -e USE_STUB_MODEL=0 \
  -e CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcloud/adc.json \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcloud/adc.json:ro" \
  YOUR_DOCKERHUB_USERNAME/lunch-buddy-api:latest
```

Then visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

**Docker image name for the rubric:** use the same name in README, `docker run`, and Docker Hub (e.g. `YOUR_DOCKERHUB_USERNAME/lunch-buddy-api:latest`). Replace `YOUR_DOCKERHUB_USERNAME` with your account name after `docker login`.

---

### Example: `POST /predict`

**Request (JSON body)** — fields match `PredictRequest` in `fastapi/main.py`:

| Field | Type | Required |
|-------|------|----------|
| `preferences` | array of strings | yes |
| `constraints` | array of strings | no (default `[]`) |
| `user_id` | string | no |

Example:

```json
{
  "preferences": ["vegetarian", "quick"],
  "constraints": ["no nuts"],
  "user_id": "student-01"
}
```

**Response (200)** — shape matches `PredictResponse`:

```json
{
  "suggestions": ["sandwich", "salad"],
  "rationale": "dummy"
}
```

(Exact strings depend on the registered model; the dummy model returns fixed suggestions.)

**curl example:**

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"preferences": ["vegetarian"], "constraints": []}'
```

---

## Getting Started (full app)

*Broader Lunch Buddy app setup (database, scraper, LLM) — TBD.*

---
# Phase 1: Automated Data Ingestion Pipeline

**Overview:** An automated daily ETL (Extract, Transform, Load) pipeline that pulls scraped university dining menus from Google Cloud Storage, generates semantic vector embeddings using Vertex AI, and syncs the data to a Supabase PostgreSQL database for hybrid search.

## Architectural Decisions

* **Database Connection Pooling (Supabase):** * Configured via a **Transaction Pooler (Port 6543)** rather than a direct connection. This protects the database from connection exhaustion during spiky API traffic.
  * Enabled **IPv4 Shared Pooling** to bypass silent IPv6 networking failures common in ephemeral CI/CD environments.
* **Batch API Processing:** We hit the Vertex AI `text-embedding-004` model in batches of 50 to optimize network latency and strictly adhere to rate limits.
* **Idempotent Database Uploads:**
  * Data is pushed via `psycopg2.extras.execute_values` for high-performance bulk inserts (one database trip instead of hundreds).
  * The database enforces a strict `UNIQUE (dish_name, dining_hall, meal_time, date_served)` constraint. 
  * By using an `ON CONFLICT DO UPDATE` clause, the pipeline is fully **idempotent**. If the GitHub Action fails and restarts, or runs twice in one day, it will simply update existing records rather than duplicating data.
* **CI/CD Automation (GitHub Actions):**
  * Runs automatically on a daily cron schedule, with a manual `workflow_dispatch` trigger for testing.
  * Utilizes Google's official `auth@v2` action for secure Service Account authentication (Vertex AI User role), keeping GCP keys safely in GitHub Secrets.

---

## The ETL Script (`menu_db_ingest.py`) Step-by-Step

The Python ingestion script acts as the bridge between raw cloud storage and our vectorized database. Here is the exact lifecycle of the data during a single run:

### 1. Extract (Fetch Raw Data)
The script authenticates with Google Cloud Storage and downloads the scraped JSON file for the current day. It reads this raw, highly nested JSON directly into memory.

### 2. Transform & Deduplicate (Flattening)
Before touching any external APIs, the script processes the JSON into a flat structure.
* **Search Text Generation:** It combines the dish name and ingredients into a single, clean `search_text` string. This is the exact text the AI will interpret.
* **Application-Layer Deduplication:** Scraped menus frequently contain duplicate entries (e.g., the same pizza served for both Lunch and Dinner). The script creates a unique fingerprint for every item: `(dish_name, dining_hall, meal_time)`. If it detects duplicates within the daily JSON, it safely overwrites them, ensuring only one clean record remains.

### 3. Enrich (Vector Embeddings)
The script iterates through the flattened menu items in batches of 50. It sends the `search_text` for each batch to Google's Vertex AI (`text-embedding-004` model). Vertex AI returns a 768-dimensional mathematical array (a vector) for each item, which the script attaches to the respective Python dictionary.

### 4. Format (Tuple Conversion)
To maximize database insert speeds, the script uses the `psycopg2` library's `execute_values` function. This requires converting our list of Python dictionaries into a strict list of tuples that map perfectly to our Postgres database columns.

### 5. Load (Database Upsert)
The script opens a connection to the Supabase Transaction Pooler and executes a bulk `INSERT` query. It utilizes an `ON CONFLICT` clause tied to our database's unique constraint. If a menu item already exists for that day, meal, and hall, the script simply updates the vector embedding instead of crashing, guaranteeing data integrity.

## License

For academic use only — Stanford MLOps Spring 2026.
