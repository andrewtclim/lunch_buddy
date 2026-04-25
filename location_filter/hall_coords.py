"""
Static lookup table: dining hall name (as it appears in the database)
# -> (latitude, longitude) in decimal degrees.
#
# These coordinates never change, so a plain dict is faster and simpler
# than a database lookup on every request.
#
# NOTE: coordinates verified via Google Maps right-click (Apr 2026).
# If a hall moves or a new one opens, update this file and bump the
# version comment below.
#
Version: 2026-04-23 (verified coordinates -- 8 halls)
"""

# ------------------------------------------------------------------
# Primary name mapping
# Keys match the `dining_hall` column in the `daily_menu` table exactly.
# ------------------------------------------------------------------
HALL_COORDS: dict[str, tuple[float, float]] = {

    # South / central dorm cluster
    "Arrillaga Family Dining Commons": (37.42560002091392,  -122.164133525439),
    "Wilbur Dining":                   (37.42416195987942,  -122.1631657087292),
    "Stern Dining":                    (37.42507655383086,  -122.16564625712856),

    # Central / Row-adjacent dorms
    "Gerhard Casper Dining":           (37.426035684919206, -122.16180681109593),
    "Ricker Dining":                   (37.425928723643814, -122.18064640315967),

    # North dorm cluster
    "Branner Dining":                  (37.42641642487774,  -122.16249198040775),

    # West / southwest
    "Florence Moore Dining":           (37.42266474148614,  -122.17167671109841),
    "Lakeside Dining":                 (37.42484510865807,  -122.17632896396873),
}

# ------------------------------------------------------------------
# Aliases
# The backfill_menu table uses a shortened name for Arrillaga.
# We map it to the same coordinates so it is never silently dropped.
# Long-term fix: normalize the name during ingest (noted in log).
# ------------------------------------------------------------------
HALL_COORDS_ALIASES: dict[str, str] = {
    # alias                    -> canonical name in HALL_COORDS
    "Arrillaga": "Arrillaga Family Dining Commons",
}


def get_coords(hall_name: str) -> tuple[float, float] | None:
    """
    Return (lat, lon) for a dining hall name.

    Checks the primary dict first, then falls back to aliases.
    Returns None if the name is not recognised -- callers should
    handle this gracefully rather than crashing.

    Example:
        >>> get_coords("Arrillaga")
        (37.42560002091392, -122.164133525439)
        >>> get_coords("Unknown Hall")
        None
    """
    # 1. Try exact match in the primary dict
    if hall_name in HALL_COORDS:
        return HALL_COORDS[hall_name]

    # 2. Try alias -- resolve to canonical name, then look up coordinates
    canonical = HALL_COORDS_ALIASES.get(hall_name)
    if canonical:
        return HALL_COORDS[canonical]

    # 3. Unknown name -- return None so the caller can decide what to do
    return None
