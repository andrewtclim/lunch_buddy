# models/gemini_flash_rag/demo.py
# End-to-end demo of the gemini_flash_rag pipeline.
#
# Two modes depending on --table:
#
#   daily_menu (default) -- one-shot demo using today's real Stanford menu.
#     Embeds preferences, retrieves dishes, gets Gemini recommendations,
#     user picks one, vector updates. Proves the full pipeline works live.
#
#   backfill_menu -- full 10-day learning loop using historical data.
#     Vector and preference summary both update after each pick.
#     At the end, user reveals true preferences and we score convergence.
#     This is the correct mode for testing EMA learning behavior.
#
# Usage:
#   python demo.py                                    # today's menu, one round
#   python demo.py --table backfill_menu              # 10-day learning loop
#   python demo.py --mood "something light today"     # test mood blending (daily only)

import argparse
import json
from datetime import date
from pathlib import Path
from recommend import (
    get_embedding,
    get_available_dates,
    blend_mood,
    recommend,
    fetch_dish_embedding,
    update_preference_vector,
    summarize_preferences,
    cosine_similarity,
)
from user_prefs import load_user_pref, save_user_pref

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Lunch Buddy recommendation demo")
parser.add_argument(
    "--table",
    default="daily_menu",
    choices=["daily_menu", "backfill_menu"],
    help="Which Supabase table to pull dishes from",
)
parser.add_argument(
    "--mood",
    default=None,
    help="Optional daily mood string (daily_menu mode only)",
)
parser.add_argument(
    "--user_id",
    default=None,
    help="Optional Supabase user UUID -- loads stored profile on start, saves after each pick",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Shared onboarding
# ---------------------------------------------------------------------------

def onboard() -> tuple[str, list[str], object]:
    # if --user_id was given, try to load their stored profile from user_pref
    if args.user_id:
        profile = load_user_pref(args.user_id)
        if profile:
            print(f"Loaded profile for user '{args.user_id}'.", flush=True)
            print(f"  Summary:  {profile['preference_summary']}", flush=True)
            print(f"  Allergens: {profile['allergens'] or 'none'}\n", flush=True)
            # return the stored summary as signup_text so the summarizer has context
            return profile["preference_summary"], profile["allergens"], profile["preference_vector"]
        else:
            print(f"No profile found for '{args.user_id}' -- starting fresh.\n", flush=True)

    # no stored profile -- collect from user
    signup_text   = input("Describe your food preferences:\n> ").strip()
    allergens_raw = input("\nAllergens to avoid? (comma-separated, or Enter to skip):\n> ").strip()
    allergens     = [a.strip() for a in allergens_raw.split(",") if a.strip()]
    return signup_text, allergens, None   # None = no stored vector, embed from scratch


# ---------------------------------------------------------------------------
# One-shot mode (daily_menu)
# ---------------------------------------------------------------------------

def run_oneshot(signup_text: str, allergens: list[str], stored_vec) -> None:
    date_str = str(date.today())   # always today for daily_menu

    print(f"\nEmbedding your preferences...", flush=True)
    # use stored vector if loaded from user_pref, otherwise embed the signup text
    pref_vec = stored_vec if stored_vec is not None else get_embedding(signup_text)

    # show mood blend effect if --mood was passed
    if args.mood:
        mood_vec = get_embedding(args.mood)
        blended  = blend_mood(pref_vec, mood_vec)
        sim      = cosine_similarity(pref_vec, blended)
        print(f"Mood blended (cosine similarity to base profile: {sim:.4f})", flush=True)

    print(f"\nFetching recommendations for {date_str}...\n", flush=True)

    recs, alts, _ = recommend(
        pref_vec=pref_vec,
        preference_summary=signup_text,
        user_allergens=allergens,
        date_str=date_str,
        daily_mood=args.mood,
        table="daily_menu",
    )

    if not recs:
        print("No recommendations returned -- check that today's menu has been scraped.", flush=True)
        return

    # combine recs + alts into one numbered list
    all_options = recs + alts

    # display top 3
    print("Today's recommendations:\n", flush=True)
    for i, rec in enumerate(recs):
        print(f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}", flush=True)
        print(f"     {rec['reason']}\n", flush=True)

    # display diverse alternatives (numbered continuing from top 3)
    if alts:
        print("Or some other options you might like:\n", flush=True)
        for i, alt in enumerate(alts):
            print(f"  {len(recs) + i + 1}. {alt['dish_name']} at {alt['dining_hall']}", flush=True)
            print(f"     {alt['reason']}\n", flush=True)

    # user picks from any of the options shown
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

    # show that the vector actually moved toward the chosen dish
    before = cosine_similarity(pref_vec, dish_vec)
    after  = cosine_similarity(updated_vec, dish_vec)
    print(f"\nYou picked: {chosen['dish_name']}", flush=True)
    print(f"Vector similarity to chosen dish -- before: {before:.4f}  after: {after:.4f}", flush=True)

    # save updated vector back to user_pref if a user_id was provided
    if args.user_id:
        save_user_pref(args.user_id, updated_vec, signup_text, allergens)
        print(f"Profile saved to user_pref for '{args.user_id}'.", flush=True)

    print("\nPipeline complete -- all steps verified.", flush=True)


# ---------------------------------------------------------------------------
# Multi-day learning loop (backfill_menu)
# ---------------------------------------------------------------------------

def run_multiday(signup_text: str, allergens: list[str], stored_vec) -> None:
    print(f"\nEmbedding your preferences...", flush=True)
    initial_pref_vec   = stored_vec if stored_vec is not None else get_embedding(signup_text)
    pref_vec           = initial_pref_vec.copy()   # this one evolves each day
    preference_summary = signup_text                   # Gemini uses this as context

    # fetch all available dates from backfill_menu (2026-03-29 to 2026-04-07)
    dates = get_available_dates(table="backfill_menu")
    print(f"Dates available: {dates[0]} -> {dates[-1]} ({len(dates)} days)\n", flush=True)

    chosen_dishes = []   # running list of picks -- fed to the summarizer each day

    # --- day loop ---
    for day_num, date_str in enumerate(dates):
        print(f"--- Day {day_num + 1} ({date_str}) ---", flush=True)
        print(f"Preference summary: \"{preference_summary}\"\n", flush=True)

        # ask for today's mood -- optional, Enter to skip
        mood_raw   = input("What are you in the mood for today? (or Enter to skip):\n> ").strip()
        daily_mood = mood_raw if mood_raw else None

        recs, alts, _ = recommend(
            pref_vec=pref_vec,
            preference_summary=preference_summary,
            user_allergens=allergens,
            date_str=date_str,
            daily_mood=daily_mood,     # blended into query vector if provided
            table="backfill_menu",
        )

        if not recs:
            print("  No recommendations for today, skipping.\n", flush=True)
            continue

        # combine recs + alts into one numbered list for display and picking
        all_options = recs + alts

        # display top 3
        print("Today's recommendations:\n", flush=True)
        for i, rec in enumerate(recs):
            print(f"  {i+1}. {rec['dish_name']} at {rec['dining_hall']}", flush=True)
            print(f"     {rec['reason']}\n", flush=True)

        # display diverse alternatives (numbered continuing from top 3)
        if alts:
            print("Or some other options you might like:\n", flush=True)
            for i, alt in enumerate(alts):
                print(f"  {len(recs) + i + 1}. {alt['dish_name']} at {alt['dining_hall']}", flush=True)
                print(f"     {alt['reason']}\n", flush=True)

        # user picks from any of the options shown
        while True:
            pick = input(f"Which do you pick? (1-{len(all_options)}):\n> ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(all_options):
                break
            print(f"  Please enter a number between 1 and {len(all_options)}.")

        chosen_name = all_options[int(pick) - 1]["dish_name"]
        chosen_dishes.append(chosen_name)
        print(f"\nYou picked: {chosen_name}\n", flush=True)

        # update preference vector toward the chosen dish
        dish_vec = fetch_dish_embedding(chosen_name, date_str, table="backfill_menu")
        if dish_vec is not None:
            pref_vec = update_preference_vector(pref_vec, dish_vec)
        else:
            print(f"  Warning: embedding not found for '{chosen_name}', vector unchanged.", flush=True)

        # rewrite the preference summary so Gemini's prompt stays in sync
        print("Updating preference summary...", flush=True)
        preference_summary = summarize_preferences(signup_text, chosen_dishes)
        print(f"New summary: \"{preference_summary}\"\n", flush=True)

        # save updated vector after every pick if a user_id was provided
        if args.user_id:
            save_user_pref(args.user_id, pref_vec, preference_summary, allergens)
            print(f"  Profile saved to user_pref for '{args.user_id}'.\n", flush=True)

    # --- final reveal ---
    print("=== All days complete! ===\n", flush=True)
    true_text = input("Reveal your true food preferences (be specific):\n> ").strip()

    print("\nEmbedding true preferences...", flush=True)
    true_vec      = get_embedding(true_text)
    initial_score = cosine_similarity(initial_pref_vec, true_vec)
    final_score   = cosine_similarity(pref_vec, true_vec)
    delta         = final_score - initial_score

    print(f"\n--- Results ---", flush=True)
    print(f"Initial vector similarity to true preferences: {initial_score:.4f}", flush=True)
    print(f"Final vector similarity to true preferences:   {final_score:.4f}", flush=True)
    print(f"Improvement after {len(dates)} days of learning:          {delta:+.4f}", flush=True)

    # save full log locally
    log = {
        "signup_text":        signup_text,
        "true_preferences":   true_text,
        "allergens":          allergens,
        "chosen_dishes":      chosen_dishes,
        "initial_score":      initial_score,
        "final_score":        final_score,
        "score_improvement":  delta,
    }
    log_path = Path(__file__).parent / "demo_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nFull log saved to {log_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n=== Lunch Buddy Demo  |  table: {args.table} ===\n", flush=True)
    signup_text, allergens, stored_vec = onboard()

    if args.table == "daily_menu":
        run_oneshot(signup_text, allergens, stored_vec)
    else:
        run_multiday(signup_text, allergens, stored_vec)
