"""
test_api.py -- integration tests for the FastAPI endpoints.

Runs with USE_STUB_MODEL=true so no MLflow or model registry is needed.
The stub path sets _model=None and _load_error="stub mode: no MLflow model",
which means /health reports unhealthy and /predict returns 503 -- both are
the correct and expected behaviors for a skeleton API with no model loaded.
"""

import os
import pytest
from fastapi.testclient import TestClient

# set stub flag before main.py is imported so the lifespan picks it up
os.environ["USE_STUB_MODEL"] = "true"

from main import app   # importable via sys.path set in conftest.py

# TestClient handles startup/shutdown (lifespan) automatically
client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_root_returns_200():
    # the welcome endpoint should always respond, regardless of model state
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_welcome_message():
    # response body must contain the welcome key
    response = client.get("/")
    assert "message" in response.json()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    # /health should always respond with 200 (status is in the body, not HTTP code)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_model_not_loaded_in_stub_mode():
    # stub mode sets _model=None, so model_loaded must be False
    response = client.get("/health")
    assert response.json()["model_loaded"] is False


def test_health_includes_status_field():
    # status field must be present so callers can gate on it without parsing detail
    response = client.get("/health")
    assert "status" in response.json()


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

def test_predict_returns_detail_on_503():
    # the 503 body should explain why so the caller can surface a useful error
    payload = {
        "preferences": ["vegetarian"],
        "constraints": ["no gluten"],
    }
    response = client.post("/predict", json=payload)
    body = response.json()
    assert "detail" in body   # FastAPI HTTPException always puts reason in "detail"
