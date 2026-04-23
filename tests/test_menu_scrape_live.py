"""Live scrape: confirms Stanford dining HTML still parses into non-empty menu JSON."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pytest
import requests

_SCRAPER_DIR = Path(__file__).resolve().parents[1] / "scraper"
if str(_SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRAPER_DIR))

import menu_scraper as ms  # noqa: E402


pytestmark = pytest.mark.integration


def test_scrape_all_menus_returns_todays_nested_structure_with_dishes():
    """
    Calls the real scraper (network). Skips if the dining site is unreachable
    so local/offline runs do not fail the whole suite.
    """
    try:
        data = ms.scrape_all_menus()
    except requests.RequestException as exc:
        pytest.skip(f"Dining site unreachable: {exc}")

    today_str = date.today().isoformat()
    assert today_str in data, f"Expected top-level key {today_str!r}"

    day = data[today_str]
    assert isinstance(day, dict), "Day payload should be a dict of halls"

    total_dishes = 0
    for _hall, meals in day.items():
        assert isinstance(meals, dict), "Each hall should map meal names to dish dicts"
        for _meal, dishes in meals.items():
            assert isinstance(dishes, dict), "Each meal should map dish names to metadata"
            for dish_name, details in dishes.items():
                assert isinstance(dish_name, str) and dish_name.strip()
                assert isinstance(details, dict), f"Dish {dish_name!r} should have metadata dict"
                total_dishes += 1

    assert total_dishes > 0, "Expected at least one dish across halls/meals for today"
