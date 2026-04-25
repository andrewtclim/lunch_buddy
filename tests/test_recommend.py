"""
test_recommend.py -- unit tests for pure math and filter functions in recommend.py.

All tests are offline -- no Supabase, no Vertex AI, no network calls.
conftest.py pre-mocks google/vertex modules so the import succeeds cleanly.
"""

import numpy as np
import pytest
import recommend  # importable via sys.path set in conftest.py


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    # two identical unit vectors should score exactly 1.0
    v = np.array([1.0, 0.0, 0.0])
    assert recommend.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    # perpendicular vectors share no direction -- similarity should be 0.0
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert recommend.cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_returns_unit_vector():
    # any non-zero vector should become length 1.0 after normalizing
    v = np.array([3.0, 4.0])  # length 5 -- classic 3-4-5 triangle
    result = recommend.normalize(v)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_normalize_zero_vector_unchanged():
    # all-zero vector has no direction -- normalize should return it as-is
    v = np.zeros(4)
    result = recommend.normalize(v)
    np.testing.assert_array_equal(result, v)


# ---------------------------------------------------------------------------
# blend_mood
# ---------------------------------------------------------------------------


def test_blend_mood_result_is_unit_vector():
    # blended query vector must be normalized so cosine search works correctly
    pref = recommend.normalize(np.array([1.0, 0.0, 0.0]))
    mood = recommend.normalize(np.array([0.0, 1.0, 0.0]))
    result = recommend.blend_mood(pref, mood, beta=0.5)
    assert np.linalg.norm(result) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# filter_placeholders
# ---------------------------------------------------------------------------


def test_filter_placeholders_removes_station_labels():
    # "Burger Bar" and "Omelet Station" are in PLACEHOLDER_DISHES -- should be dropped
    dishes = [
        {"dish_name": "Burger Bar", "allergens": [], "ingredients": ""},
        {"dish_name": "Grilled Salmon", "allergens": [], "ingredients": ""},
        {"dish_name": "Omelet Station", "allergens": [], "ingredients": ""},
    ]
    result = recommend.filter_placeholders(dishes)
    names = [d["dish_name"] for d in result]
    assert names == ["Grilled Salmon"]


# ---------------------------------------------------------------------------
# filter_allergens
# ---------------------------------------------------------------------------


def test_filter_allergens_blocks_matching_dish():
    # a dish that shares any allergen with the user must be excluded entirely
    dishes = [
        {"dish_name": "Pasta", "allergens": ["gluten", "dairy"], "ingredients": ""},
        {"dish_name": "Rice Bowl", "allergens": [], "ingredients": ""},
        {"dish_name": "Cheese Plate", "allergens": ["dairy"], "ingredients": ""},
    ]
    result = recommend.filter_allergens(dishes, user_allergens=["dairy"])
    names = [d["dish_name"] for d in result]
    # Pasta and Cheese Plate both contain dairy -- only Rice Bowl should survive
    assert names == ["Rice Bowl"]


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_first_occurrence():
    # duplicate dish names are removed -- the first (highest cosine rank) is kept
    dishes = [
        {"dish_name": "Pad Thai", "allergens": []},
        {"dish_name": "Grilled Tofu", "allergens": []},
        {"dish_name": "Pad Thai", "allergens": []},  # repeat -- should be dropped
    ]
    result = recommend.deduplicate(dishes)
    names = [d["dish_name"] for d in result]
    assert names == ["Pad Thai", "Grilled Tofu"]  # only 2 unique dishes


# ---------------------------------------------------------------------------
# Connection pool behavior
#
# These tests patch psycopg2.pool.ThreadedConnectionPool so no real DB is
# touched. They verify the pool contract:
#   1. The pool is constructed lazily, only once, even across many DB calls
#   2. Every successful query returns the connection to the pool
#   3. An exception during a query still returns the connection (try/finally)
# ---------------------------------------------------------------------------

from unittest.mock import patch, MagicMock


@pytest.fixture
def reset_pool():
    # the pool is module-global; reset before and after each test so tests
    # don't leak state into each other
    recommend._pool = None
    yield
    recommend._pool = None


def test_pool_is_constructed_only_once(reset_pool):
    # calling _get_pool repeatedly should return the same instance --
    # this is the entire point of pooling
    with patch("recommend.psycopg2.pool.ThreadedConnectionPool") as mock_pool_cls:
        mock_pool_cls.return_value = MagicMock()

        first = recommend._get_pool()
        second = recommend._get_pool()
        third = recommend._get_pool()

        assert first is second is third
        # constructor invoked exactly once -- subsequent calls return the cached pool
        assert mock_pool_cls.call_count == 1


def test_retrieve_dishes_returns_connection_to_pool(reset_pool):
    # on a successful query, putconn must be called so the connection
    # can be reused by the next caller
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    with patch(
        "recommend.psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool
    ):
        recommend.retrieve_dishes(np.zeros(768), "2026-04-11", table="daily_menu")

    mock_pool.getconn.assert_called_once()
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_putconn_called_even_when_query_raises(reset_pool):
    # if the cursor raises mid-query, the try/finally must still return
    # the connection -- otherwise repeated failures would exhaust the pool
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("simulated DB error")
    mock_conn.cursor.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    with patch(
        "recommend.psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool
    ):
        with pytest.raises(RuntimeError, match="simulated DB error"):
            recommend.retrieve_dishes(np.zeros(768), "2026-04-11", table="daily_menu")

    # the exception propagated, but the connection was still returned to the pool
    mock_pool.putconn.assert_called_once_with(mock_conn)
