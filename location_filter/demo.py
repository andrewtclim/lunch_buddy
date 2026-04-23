"""
Location-aware Lunch Buddy demo.

Wraps the gemini_flash_rag pipeline with proximity filtering so
recommendations only surface dishes from halls within walking distance.

Usage:
    # today's menu, prompt for location at startup
    python demo.py

    # pass location directly (no interactive prompt)
    python demo.py --lat 37.4248 --lon -122.1655

    # custom radius (default is 800m ~ 10 min walk)
    python demo.py --lat 37.4248 --lon -122.1655 --radius 500

    # skip location filter entirely (same as original pipeline)
    python demo.py --no-location

    # 10-day learning loop with location
    python demo.py --table backfill_menu --lat 37.4248 --lon -122.1655

    # include mood + location
    python demo.py --mood "something light" --lat 37.4248 --lon -122.1655

    # load/save a stored Supabase profile
    python demo.py --user_id <supabase-uuid> --lat 37.4248 --lon -122.1655
"""

import sys
import argparse
import json
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- same pattern as recommend.py
# ---------------------------------------------------------------------------

# Add the project root (lunch_buddy/) to sys.path so that both
# models.gemini_flash_rag and location_filter can be imported as packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# location-aware pipeline -- our new wrapper
from location_filter.recommend import recommend_with_location   # noqa: E402

# helpers from the original pipeline -- reused unchanged
from models.gemini_flash_rag.recommend import (                 # noqa: E402
    get_embedding,
    get_available_dates,
    fetch_dish_embedding,
    update_preference_vector,
    summarize_preferences,
    cosine_similarity,
)

# user profile persistence -- load/save preference vectors from Supabase
from models.gemini_flash_rag.user_prefs import (                # noqa: E402
    load_user_pref,
    save_user_pref,
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Lunch Buddy -- location-aware demo")

parser.add_argument(
    "--table",
    default="daily_menu",
    choices=["daily_menu", "backfill_menu"],
    help="Supabase table to pull dishes from (default: daily_menu)",
)
parser.add_argument(
    "--mood",
    default=None,
    help="Optional daily mood string, e.g. 'something light' (daily_menu only)",
)
parser.add_argument(
    "--user_id",
    default=None,
    help="Supabase user UUID -- loads stored profile on start, saves after each pick",
)

# --- location flags ---
parser.add_argument(
    "--lat",
    type=float,
    default=None,
    help="Your latitude in decimal degrees (e.g. 37.4248)",
)
parser.add_argument(
    "--lon",
    type=float,
    default=None,
    help="Your longitude in decimal degrees (e.g. -122.1655)",
)
parser.add_argument(
    "--radius",
    type=float,
    default=800.0,
    help="Walking radius in meters (default: 800 -- approx 10 min walk)",
)
parser.add_argument(
    "--no-location",
    action="store_true",           # flag only, no value -- sets args.no_location = True
    help="Skip location filter entirely and use the full dish pool",
)

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def prompt_for_location() -> tuple[float | None, float | None]:
    """
    Ask the user for their coordinates interactively.

    Called once at session start when --lat/--lon were not passed as CLI flags.
    The user can press Enter to skip and use the full hall pool instead.

    Returns:
        (lat, lon) as floats, or (None, None) if the user skips.
    """
    print("Enter your location to get recommendations from nearby halls.")
    print("(Press Enter to skip and search all halls.)\n")

    raw = input("Your coordinates (lat,lon)  e.g. 37.4248,-122.1655:\n> ").strip()

    if not raw:
        # user pressed Enter -- no location filter this session
        return None, None

    try:
        # split on comma and parse both parts as floats
        parts = raw.split(",")
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except (ValueError, IndexError):
        # bad format -- warn and skip rather than crashing
        print("  Could not parse coordinates -- skipping location filter.\n")
        return None, None


def fmt_location(lat: float | None, lon: float | None,
                 radius_m: float, active: bool) -> str:
    """
    One-line summary of the current location state, printed each round.

    Examples:
        Location: (37.4248, -122.1655)  radius 800m  [ON]
        Location: not set  [OFF]
        Location: (37.4248, -122.1655)  radius 800m  [OFF -- use 'loc on' to enable]
    """
    if not active:
        if lat is not None:
            return f"Location: ({lat:.4f}, {lon:.4f})  radius {radius_m:.0f}m  [OFF -- type 'loc on' to enable]"
        return "Location: not set  [OFF]"
    return f"Location: ({lat:.4f}, {lon:.4f})  radius {radius_m:.0f}m  [ON]"


def handle_loc_command(raw: str, lat: float | None, lon: float | None,
                       radius_m: float, active: bool
                       ) -> tuple[float | None, float | None, float, bool]:
    """
    Parse a 'loc' command typed mid-session and update location state.

    Supported commands:
        loc                -- print current location state
        loc off            -- disable location filter for remaining rounds
        loc on             -- re-enable location filter (if coords are set)
        loc 37.42,-122.16  -- set new coordinates (keeps current radius)
        loc 37.42,-122.16 500  -- set new coordinates and new radius

    Returns updated (lat, lon, radius_m, active).
    """
    # split into tokens after stripping the leading 'loc' word
    tokens = raw.strip().split()   # e.g. ['loc', '37.42,-122.16', '500']

    if len(tokens) == 1:
        # just 'loc' -- print current state and return unchanged
        print(f"  {fmt_location(lat, lon, radius_m, active)}")
        return lat, lon, radius_m, active

    sub = tokens[1].lower()

    if sub == "off":
        print("  Location filter disabled.")
        return lat, lon, radius_m, False

    if sub == "on":
        if lat is None:
            print("  No coordinates set. Use: loc <lat>,<lon>")
        else:
            print(f"  Location filter enabled: ({lat:.4f}, {lon:.4f}) radius {radius_m:.0f}m")
            active = True
        return lat, lon, radius_m, active

    # otherwise interpret tokens[1] as new coordinates
    try:
        parts = tokens[1].split(",")
        new_lat = float(parts[0].strip())
        new_lon = float(parts[1].strip())
    except (ValueError, IndexError):
        print("  Usage: loc <lat>,<lon>  or  loc <lat>,<lon> <radius_m>")
        return lat, lon, radius_m, active

    # optional third token is a new radius
    new_radius = radius_m
    if len(tokens) >= 3:
        try:
            new_radius = float(tokens[2])
        except ValueError:
            print(f"  Bad radius '{tokens[2]}' -- keeping {radius_m:.0f}m")

    print(f"  Location updated: ({new_lat:.4f}, {new_lon:.4f}) radius {new_radius:.0f}m")
    return new_lat, new_lon, new_radius, True   # updating coords implicitly turns filter on


# ---------------------------------------------------------------------------
# Shared onboarding
# ---------------------------------------------------------------------------

def onboard() -> tuple[str, list[str], object]:
    """
    Collect (or load) the user's preference profile.

    If --user_id was passed and a stored profile exists in Supabase,
    load it directly. Otherwise prompt for signup text and allergens.

    Returns:
        (signup_text, allergens, stored_vec)
        stored_vec is None if no profile was loaded -- caller will embed
        signup_text to create a fresh vector.
    """
    if args.user_id:
        profile = load_user_pref(args.user_id)
        if profile:
            print(f"Loaded profile for user '{args.user_id}'.")
            print(f"  Summary:   {profile['preference_summary']}")
            print(f"  Allergens: {profile['allergens'] or 'none'}\n")
            return (profile["preference_summary"],
                    profile["allergens"],
                    profile["preference_vector"])
        else:
            print(f"No profile found for '{args.user_id}' -- starting fresh.\n")

    # no stored profile -- ask the user directly
    signup_text   = input("Describe your food preferences:\n> ").strip()
    allergens_raw = input("\nAllergens to avoid? (comma-separated, or Enter to skip):\n> ").strip()
    allergens     = [a.strip() for a in allergens_raw.split(",") if a.strip()]
    return signup_text, allergens, None


# ---------------------------------------------------------------------------
# One-shot mode (daily_menu)
# ---------------------------------------------------------------------------

def run_oneshot(signup_text: str, allergens: list[str], stored_vec,
                lat: float | None, lon: float | None,
                radius_m: float, restrict: bool) -> None:
    """Run one recommendation round against today's live dining menu."""

    date_str = str(date.today())   # always today for daily_menu

    print(f"\nEmbedding your preferences...", flush=True)
    # use stored vector if loaded from user_pref; otherwise embed signup text
    pref_vec = stored_vec if stored_vec is not None else get_embedding(signup_text)

    print(f"Fetching recommendations for {date_str}...", flush=True)
    print(fmt_location(lat, lon, radius_m, restrict), flush=True)
    print()

    recs, alts, _, halls_used = recommend_with_location(
        pref_vec=pref_vec,
        preference_summary=signup_text,
        user_allergens=allergens,
        date_str=date_str,
        daily_mood=args.mood,
        user_lat=lat,
        user_lon=lon,
        radius_m=radius_m,
        restrict_to_nearby=restrict,
        table="daily_menu",
    )

    # tell the user which halls were actually searched
    if halls_used == ["all halls"]:
        print("Searched: all halls\n", flush=True)
    else:
        print(f"Searched {len(halls_used)} nearby hall(s): {', '.join(halls_used)}\n", flush=True)

    if not recs:
        print("No recommendations returned -- the nearby halls may have no menu today.", flush=True)
        return

    # combine recs + alts into one numbered list
    all_options = recs + alts

    print("Today's recommendations:\n", flush=True)
    for i, rec in enumerate(recs):
        print(f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}", flush=True)
        print(f"     {rec['reason']}\n", flush=True)

    if alts:
        print("Or some other options you might like:\n", flush=True)
        for i, alt in enumerate(alts):
            print(f"  {len(recs)+i+1}. {alt['dish_name']} at {alt['dining_hall']}", flush=True)
            print(f"     {alt['reason']}\n", flush=True)

    # user picks one
    while True:
        pick = input(f"Which do you pick? (1-{len(all_options)}):\n> ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(all_options):
            break
        print(f"  Please enter a number between 1 and {len(all_options)}.")

    chosen   = all_options[int(pick) - 1]
    dish_vec = fetch_dish_embedding(chosen["dish_name"], date_str, table="daily_menu")

    if dish_vec is None:
        print(f"Warning: embedding not found for '{chosen['dish_name']}'.", flush=True)
        return

    updated_vec = update_preference_vector(pref_vec, dish_vec)

    before = cosine_similarity(pref_vec, dish_vec)
    after  = cosine_similarity(updated_vec, dish_vec)
    print(f"\nYou picked: {chosen['dish_name']}", flush=True)
    print(f"Vector similarity to chosen dish -- before: {before:.4f}  after: {after:.4f}", flush=True)

    if args.user_id:
        save_user_pref(args.user_id, updated_vec, signup_text, allergens)
        print(f"Profile saved to user_pref for '{args.user_id}'.", flush=True)

    print("\nPipeline complete.", flush=True)


# ---------------------------------------------------------------------------
# Multi-day learning loop (backfill_menu)
# ---------------------------------------------------------------------------

def run_multiday(signup_text: str, allergens: list[str], stored_vec,
                 lat: float | None, lon: float | None,
                 radius_m: float, restrict: bool) -> None:
    """
    Run a 10-day EMA learning loop over the backfill_menu table.

    Location state (lat, lon, radius, restrict) can be updated mid-loop
    by typing 'loc <command>' at the mood prompt.
    """

    print(f"\nEmbedding your preferences...", flush=True)
    initial_pref_vec   = stored_vec if stored_vec is not None else get_embedding(signup_text)
    pref_vec           = initial_pref_vec.copy()   # evolves each day
    preference_summary = signup_text

    dates = get_available_dates(table="backfill_menu")
    print(f"Dates available: {dates[0]} -> {dates[-1]} ({len(dates)} days)\n", flush=True)
    print('Type "loc" at any prompt to check or update your location.\n', flush=True)

    chosen_dishes = []

    for day_num, date_str in enumerate(dates):
        print(f"--- Day {day_num+1} ({date_str}) ---", flush=True)
        print(f'Preference summary: "{preference_summary}"\n', flush=True)
        print(fmt_location(lat, lon, radius_m, restrict), flush=True)

        # ask for today's mood -- also accept 'loc' commands here
        mood_raw = input("\nWhat are you in the mood for today? (Enter to skip / 'loc' to update location):\n> ").strip()

        # handle loc commands entered at the mood prompt
        while mood_raw.lower().startswith("loc"):
            lat, lon, radius_m, restrict = handle_loc_command(
                mood_raw, lat, lon, radius_m, restrict)
            mood_raw = input("> ").strip()

        daily_mood = mood_raw if mood_raw else None

        print(f"\nFetching recommendations...", flush=True)

        recs, alts, _, halls_used = recommend_with_location(
            pref_vec=pref_vec,
            preference_summary=preference_summary,
            user_allergens=allergens,
            date_str=date_str,
            daily_mood=daily_mood,
            user_lat=lat,
            user_lon=lon,
            radius_m=radius_m,
            restrict_to_nearby=restrict,
            table="backfill_menu",
        )

        if halls_used == ["all halls"]:
            print("Searched: all halls\n", flush=True)
        else:
            print(f"Searched {len(halls_used)} nearby hall(s): {', '.join(halls_used)}\n", flush=True)

        if not recs:
            print("  No recommendations for today -- skipping.\n", flush=True)
            continue

        all_options = recs + alts

        print("Today's recommendations:\n", flush=True)
        for i, rec in enumerate(recs):
            print(f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}", flush=True)
            print(f"     {rec['reason']}\n", flush=True)

        if alts:
            print("Or some other options you might like:\n", flush=True)
            for i, alt in enumerate(alts):
                print(f"  {len(recs)+i+1}. {alt['dish_name']} at {alt['dining_hall']}", flush=True)
                print(f"     {alt['reason']}\n", flush=True)

        while True:
            pick = input(f"Which do you pick? (1-{len(all_options)}):\n> ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(all_options):
                break
            print(f"  Please enter a number between 1 and {len(all_options)}.")

        chosen_name = all_options[int(pick) - 1]["dish_name"]
        chosen_dishes.append(chosen_name)
        print(f"\nYou picked: {chosen_name}\n", flush=True)

        # EMA update
        dish_vec = fetch_dish_embedding(chosen_name, date_str, table="backfill_menu")
        if dish_vec is not None:
            pref_vec = update_preference_vector(pref_vec, dish_vec)
        else:
            print(f"  Warning: embedding not found for '{chosen_name}', vector unchanged.", flush=True)

        # keep the natural language summary in sync with the evolving vector
        print("Updating preference summary...", flush=True)
        preference_summary = summarize_preferences(signup_text, chosen_dishes)
        print(f'New summary: "{preference_summary}"\n', flush=True)

        if args.user_id:
            save_user_pref(args.user_id, pref_vec, preference_summary, allergens)
            print(f"  Profile saved for '{args.user_id}'.\n", flush=True)

    # --- final convergence reveal ---
    print("=== All days complete! ===\n", flush=True)
    true_text = input("Reveal your true food preferences (be specific):\n> ").strip()

    print("\nEmbedding true preferences...", flush=True)
    true_vec      = get_embedding(true_text)
    initial_score = cosine_similarity(initial_pref_vec, true_vec)
    final_score   = cosine_similarity(pref_vec, true_vec)
    delta         = final_score - initial_score

    print(f"\n--- Results ---", flush=True)
    print(f"Initial similarity to true preferences: {initial_score:.4f}", flush=True)
    print(f"Final similarity to true preferences:   {final_score:.4f}", flush=True)
    print(f"Improvement after {len(dates)} days:              {delta:+.4f}", flush=True)

    # save full log locally for review
    log = {
        "signup_text":       signup_text,
        "true_preferences":  true_text,
        "allergens":         allergens,
        "chosen_dishes":     chosen_dishes,
        "initial_score":     initial_score,
        "final_score":       final_score,
        "score_improvement": delta,
    }
    log_path = Path(__file__).parent / "demo_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nFull log saved to {log_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n=== Lunch Buddy  |  table: {args.table} ===\n", flush=True)

    # collect preference profile (or load from Supabase)
    signup_text, allergens, stored_vec = onboard()

    # determine location state for this session
    if args.no_location:
        # user explicitly passed --no-location -- skip filter for entire session
        user_lat, user_lon, restrict = None, None, False
        print("\nLocation filter: OFF (--no-location flag set)\n", flush=True)
    elif args.lat is not None and args.lon is not None:
        # coordinates passed as CLI flags -- use them directly
        user_lat, user_lon, restrict = args.lat, args.lon, True
        print(f"\nLocation: ({user_lat:.4f}, {user_lon:.4f})  radius {args.radius:.0f}m\n", flush=True)
    else:
        # no CLI flags -- prompt once interactively
        print()
        user_lat, user_lon = prompt_for_location()
        restrict = user_lat is not None   # only restrict if the user gave coordinates

    radius_m = args.radius

    if args.table == "daily_menu":
        run_oneshot(signup_text, allergens, stored_vec,
                    user_lat, user_lon, radius_m, restrict)
    else:
        run_multiday(signup_text, allergens, stored_vec,
                     user_lat, user_lon, radius_m, restrict)
