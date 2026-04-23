"""Supabase: confirms today's menu rows exist in `daily_menu` (post-ingest pipeline)."""

from __future__ import annotations

import os
from datetime import date

import pytest

pytest.importorskip("psycopg2")

pytestmark = pytest.mark.integration


def test_daily_menu_has_rows_for_today():
    """
    Requires DATABASE_URL (same as GitHub Actions ingest step).
    Uses a read-only COUNT; does not mutate data.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set (add it for CI or local Supabase checks)")

    today = date.today().isoformat()

    import psycopg2

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM daily_menu WHERE date_served = %s::date",
                (today,),
            )
            row = cur.fetchone()
            assert row is not None
            count = int(row[0])
    finally:
        conn.close()

    assert count > 0, f"Expected rows in daily_menu for date_served={today}, got {count}"
