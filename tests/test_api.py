"""
test_api.py -- integration tests for the FastAPI endpoints.

Runs with USE_STUB_MODEL=true so no MLflow or model registry is needed.
The stub path sets _model=None and _load_error="stub mode: no MLflow model",
which means /health reports unhealthy and /predict returns 503 -- both are
the correct and expected behaviors for a skeleton API with no model loaded.
"""

import os
import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# set stub flag before main.py is imported so the lifespan picks it up
os.environ["USE_STUB_MODEL"] = "true"

from main import app, get_current_user, AuthenticatedUser   # importable via sys.path set in conftest.py

# TestClient handles startup/shutdown (lifespan) automatically
client = TestClient(app)


# helper: override FastAPI's auth dependency for tests that need a logged-in user
_test_user = AuthenticatedUser(user_id="test-user-id")

def _override_auth():
    return _test_user


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


# ---------------------------------------------------------------------------
# POST /pick
# ---------------------------------------------------------------------------

# a fake 768-dim vector for test stubs
_fake_vec = np.random.default_rng(42).random(768).astype(float)

# fake user profile returned by load_user_pref
_fake_profile = {
    "preference_vector": _fake_vec.copy(),
    "preference_summary": "Likes spicy food",
    "allergens": ["shellfish"],
}


def _auth_header():
    # bypass JWT validation by patching get_current_user in the tests below
    return {"Authorization": "Bearer fake-token"}


def test_pick_no_auth():
    # /pick without a bearer token should return 401
    payload = {"dish_name": "Gochujang Chicken", "dining_hall": "Arrillaga"}
    response = client.post("/pick", json=payload)
    assert response.status_code == 401


@patch("main.save_user_pref")
@patch("main.summarize_preferences", return_value="Enjoys spicy noodles")
@patch("main.update_preference_vector", return_value=_fake_vec)
@patch("main.fetch_dish_embedding", return_value=_fake_vec)
@patch("main.load_user_pref", return_value=_fake_profile)
def test_pick_success(mock_load, mock_fetch, mock_update, mock_summarize, mock_save):
    # happy path: user picks a dish, EMA updates, profile saved
    app.dependency_overrides[get_current_user] = _override_auth
    try:
        payload = {"dish_name": "Gochujang Chicken", "dining_hall": "Arrillaga"}
        response = client.post("/pick", json=payload, headers=_auth_header())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        # verify save_user_pref was called once with the updated vector
        mock_save.assert_called_once()
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@patch("main.fetch_dish_embedding", return_value=None)
@patch("main.load_user_pref", return_value=_fake_profile)
def test_pick_dish_not_found(mock_load, mock_fetch):
    # dish name not in today's menu should return 404
    app.dependency_overrides[get_current_user] = _override_auth
    try:
        payload = {"dish_name": "Nonexistent Dish", "dining_hall": "Arrillaga"}
        response = client.post("/pick", json=payload, headers=_auth_header())
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@patch("main.load_user_pref", return_value=None)
def test_pick_no_profile(mock_load):
    # user with no preference profile should get 422
    app.dependency_overrides[get_current_user] = _override_auth
    try:
        payload = {"dish_name": "Gochujang Chicken", "dining_hall": "Arrillaga"}
        response = client.post("/pick", json=payload, headers=_auth_header())
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_current_user, None)
