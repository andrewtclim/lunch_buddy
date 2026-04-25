"""
Proximity utilities for location-aware dining hall filtering.

Two public functions:
    haversine_distance(lat1, lon1, lat2, lon2) -> meters
    filter_nearby_halls(user_lat, user_lon, radius_m)  -> list[str]
"""

import math
from location_filter.hall_coords import HALL_COORDS, HALL_COORDS_ALIASES


# ------------------------------------------------------------------
# Haversine formula
# ------------------------------------------------------------------

# Earth's mean radius in meters -- used by the haversine formula below.
_EARTH_RADIUS_M = 6_371_000


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    Compute the straight-line surface distance (in meters) between two
    points on Earth using the Haversine formula.

    Haversine is accurate to within ~0.3% for distances under 1000 km,
    which is more than sufficient for a campus-scale radius filter.

    Args:
        lat1, lon1: coordinates of point A (decimal degrees)
        lat2, lon2: coordinates of point B (decimal degrees)

    Returns:
        Distance in meters (float).

    Example:
        >>> haversine_distance(37.4237, -122.1697, 37.4271, -122.1648)
        ~560.0  # Arrillaga -> Florence Moore, roughly
    """
    # Convert all four values from degrees to radians -- math.* trig
    # functions expect radians, not degrees.
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    # Delta (difference) between the two latitudes and longitudes in radians
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    # Core haversine calculation:
    # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r)
         * math.sin(d_lon / 2) ** 2)

    # c = 2 * arctan2(√a, √(1−a))  -- the angular distance in radians
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Multiply by Earth's radius to get the distance in meters
    return _EARTH_RADIUS_M * c


# ------------------------------------------------------------------
# Hall filter
# ------------------------------------------------------------------

def filter_nearby_halls(user_lat: float, user_lon: float,
                        radius_m: float = 800.0) -> list[str]:
    """
    Return all hall names (as stored in the database) that are within
    `radius_m` meters of the user's location.

    Checks both the primary HALL_COORDS dict and all aliases so that
    every name variant in the database is covered.

    Args:
        user_lat:  user's latitude in decimal degrees
        user_lon:  user's longitude in decimal degrees
        radius_m:  walking radius in meters (default 800 -- ~10 min walk)

    Returns:
        List of hall name strings within radius. May be empty if the
        user is far from all halls. The caller (recommend.py) handles
        the empty-list fallback.

    Example:
        >>> filter_nearby_halls(37.4248, -122.1655, radius_m=500)
        ['Wilbur Dining', 'Stern Dining', 'Gerhard Casper Dining']
    """
    nearby = []

    # Check every canonical hall in the primary dict
    for hall_name, (lat, lon) in HALL_COORDS.items():
        dist = haversine_distance(user_lat, user_lon, lat, lon)
        if dist <= radius_m:
            nearby.append(hall_name)           # add the canonical name

    # Also check aliases -- add the alias string itself (not the canonical
    # name) because the database may store the alias variant
    for alias, canonical in HALL_COORDS_ALIASES.items():
        lat, lon = HALL_COORDS[canonical]      # resolve alias -> coordinates
        dist = haversine_distance(user_lat, user_lon, lat, lon)
        if dist <= radius_m and alias not in nearby:
            nearby.append(alias)               # add alias so SQL WHERE matches

    return nearby
