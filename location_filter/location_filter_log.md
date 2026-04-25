# Location Filter — Design Log

**Branch:** `feat/location-filter`
**Contributors:** Andrew Lim
**Started:** 2026-04-15

---

## Why This Exists

The original recommendation pipeline pulls from all 8 dining halls equally,
regardless of where the user is standing. A student at Stern has no reason
to get a recommendation for Lakeside if it's a 15-minute walk away.

This module adds a proximity layer on top of the existing `recommend()` pipeline:
filter the dish pool down to halls within a user-defined walking radius *before*
embeddings are searched and Gemini is called. If no halls are nearby (or the
user opts out), it falls back to the full pool gracefully.

---

## Module Structure

```
location_filter/
    hall_coords.py   -- static lat/lon for each known dining hall
    proximity.py     -- haversine distance + nearby-hall filter
    recommend.py     -- wrapper: location filter -> gemini_flash_rag recommend()
    demo.py          -- interactive demo with --lat/--lon CLI flags + live prompt
```

---

## How It Fits Into the Project

```
User request (lat, lon, radius_m)
        |
        v
proximity.filter_nearby_halls()     <-- new layer (this module)
        |
        v
recommend() in gemini_flash_rag     <-- existing pipeline, unchanged
        |
        v
Top-3 recommendations from halls within walking distance
```

The existing `gemini_flash_rag/recommend.py` is not modified. This module
wraps it and passes a `halls` filter list, which gets injected into the
Supabase cosine search query as a `WHERE dining_hall = ANY(...)` clause.

---

## Design Decisions

### Radius default: 800m
800 meters is roughly a 10-minute walk at a comfortable pace. Configurable
per call via `radius_m` parameter.

### Graceful fallback
If `restrict_to_nearby=True` but no halls fall within the radius, the filter
is silently dropped and the full pool is used. This prevents the system from
returning nothing for off-campus users or edge cases.

### Static coordinates
Dining hall locations do not change. Hardcoded lat/lon in `hall_coords.py`
is simpler and faster than a database lookup on every request. If a new hall
opens, one line gets added to the dict.

### Name aliasing
The `backfill_menu` table uses `"Arrillaga"` while `daily_menu` uses
`"Arrillaga Family Dining Commons"` for the same physical hall. Both aliases
map to the same coordinates so neither is silently dropped.

---

## Known Data Inconsistency

`"Arrillaga"` (backfill_menu) and `"Arrillaga Family Dining Commons"`
(daily_menu) are the same hall. Both are mapped in `hall_coords.py`.
Long-term fix: normalize the name during ingest.

---

## Future Work — On-Campus Cafes and Restaurants

The Stanford eateries map (https://lbre-sites-prod.stanford.edu/stanford_eateries/index.html)
lists many more food options beyond the 8 residential dining halls:
Starbucks, Kikka Sushi, CoHo (Coffee House), Decadence, Coupa Cafe, and others.

These are not scraped yet because they each have independent menu sources
and do not rotate daily the way dining halls do. Proposed approach:

### Step 1 — Get coordinates for all campus eateries
The eateries map is powered by an Esri ArcGIS feature service. The underlying
REST API returns all location names and lat/lon coordinates as JSON without
needing a browser. Pattern to query:
```
GET https://[esri-server]/arcgis/rest/services/.../FeatureServer/0/query
    ?where=1=1&outFields=*&f=json
```
Network tab in browser dev tools will reveal the exact endpoint. This gives
us a clean list of every campus food location with coordinates in one call.

### Step 2 — Source menus per venue
Each venue needs its own strategy:

| Venue | Menu Source | Frequency |
|---|---|---|
| CoHo (Coffee House) | coho.stanford.edu — has a menu page | Weekly |
| Starbucks | Standard Starbucks menu — hardcode once | Static |
| Kikka Sushi | Fixed set menu — hardcode or scrape CoHo site | Static |
| Decadence | Check venue page for menu | Static or weekly |
| Coupa Cafe | Stanford-specific menu — check site | Static |

### Step 3 — New Supabase table: `cafe_menu`
Same schema as `daily_menu` but with a `is_static BOOLEAN` column.
Static venues are embedded once and refreshed monthly (or on change).
Dynamic venues (like CoHo) are scraped on their own schedule.

### Step 4 — Union at query time
At recommend time, query `daily_menu UNION cafe_menu` before the cosine
search. Location filter applies to both tables equally.

### Step 5 — Register a new MLflow model version
Once cafe menus are live, bump to `gemini_flash_rag_v3` reflecting the
expanded retrieval pool.

---

## Experiment Log

| Date | Change | Result |
|---|---|---|
| 2026-04-15 | Module scaffolded, hall_coords.py written | 8 halls mapped |
