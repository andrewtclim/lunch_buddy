"""Health check for the ML model via GET /health."""

from pathlib import Path
import sys

# App lives in fastapi/; add it so `import main` works when pytest runs from repo root.
_FASTAPI = Path(__file__).resolve().parents[1] / "fastapi"
if str(_FASTAPI) not in sys.path:
    sys.path.insert(0, str(_FASTAPI))

from fastapi.testclient import TestClient

import main


def test_health_reports_healthy_when_model_is_loaded():
    """Health endpoint should report healthy when a model is present."""
    original_model = main._model
    original_load_error = main._load_error
    try:
        main._model = object()
        main._load_error = None

        client = TestClient(main.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_uri" in data
    finally:
        main._model = original_model
        main._load_error = original_load_error


def test_health_reports_unhealthy_when_model_is_not_loaded():
    """Health endpoint should report unhealthy if model load failed."""
    original_model = main._model
    original_load_error = main._load_error
    try:
        main._model = None
        main._load_error = "model failed to load"

        client = TestClient(main.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "detail" in data
    finally:
        main._model = original_model
        main._load_error = original_load_error
