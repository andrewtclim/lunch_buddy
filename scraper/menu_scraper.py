import requests
from bs4 import BeautifulSoup
from datetime import date
import json
import os

BASE_URL = "https://rdeapps.stanford.edu/dininghallmenu/"


def get_page_state():
    """
    GET the page once to capture:
      - Hidden ASP.NET tokens required on every POST
      - Valid dropdown options for dining halls, meal periods, and today's date
    Returns: (session, hidden_tokens, dining_halls, meal_periods, today_value)
    """
    session = requests.Session()                            # persist cookies across requests (ASP.NET needs this)
    # Mimic a real browser — the server silently returns a blank page for non-browser user-agents
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36"
    })
    response = session.get(BASE_URL)                        # initial GET — loads page with dropdowns pre-populated
    # crash early if the site returns 4xx/5xx
    response.raise_for_status()

    # parse raw HTML into a queryable tree
    soup = BeautifulSoup(response.text, "html.parser")

    # ASP.NET embeds these hidden fields on every page; the server rejects POSTs that don't echo them back
    hidden = {
        "__VIEWSTATE":          soup.find("input", {"id": "__VIEWSTATE"})["value"],
        "__VIEWSTATEGENERATOR": soup.find("input", {"id": "__VIEWSTATEGENERATOR"})["value"],
        "__EVENTVALIDATION":    soup.find("input", {"id": "__EVENTVALIDATION"})["value"],
    }

    # Dining hall dropdown → {display name: option value}, e.g. {"Arrillaga Family Dining Commons": "01"}
    loc_select = soup.find("select", {"id": "MainContent_lstLocations"})
    dining_halls = {
        opt.text.strip(): opt["value"]
        for opt in loc_select.find_all("option")
        # skip blank placeholder <option> with no value
        if opt.get("value")
    }

    # Meal period dropdown → {display name: option value}, e.g. {"Lunch": "L"}
    meal_select = soup.find("select", {"id": "MainContent_lstMealType"})
    meal_periods = {
        opt.text.strip(): opt["value"]
        for opt in meal_select.find_all("option")
        if opt.get("value")
    }

    # Find today's date value in the dropdown — format is "3/29/2026" (no zero-padding)
    today     = date.today()
    today_str = f"{today.month}/{today.day}/{today.year}"
    day_select   = soup.find("select", {"id": "MainContent_lstDay"})
    today_option = day_select.find("option", string=lambda t: t and t.strip() == today_str)
    # If today isn't in the list, fall back to the first option that has a non-empty value
    today_value  = today_option["value"] if today_option else next(
        opt["value"] for opt in day_select.find_all("option") if opt.get("value")
    )

    return session, hidden, dining_halls, meal_periods, today_value


def fetch_menu(session, hidden, location_value, meal_value, day_value):
    """
    POST the form with a specific dining hall + meal period + date selection.
    ASP.NET returns a full new HTML page with the menu rendered in it.
    Returns: BeautifulSoup object of the response page.
    """
    payload = {
        # echo back the three required ASP.NET tokens
        **hidden,
        # ASP.NET forms POST using the `name` attribute (ctl00$MainContent$...), not the `id`
        "ctl00$MainContent$lstLocations":  location_value,  # selected dining hall
        "ctl00$MainContent$lstMealType":   meal_value,      # selected meal period
        "ctl00$MainContent$lstDay":        day_value,       # selected date
        # Simulate clicking the "Refresh" submit button — a real control the server trusts.
        # Using __doPostBack with a custom event target fails __EVENTVALIDATION; the button does not.
        "__EVENTTARGET":                   "",              # empty = submitted via button, not __doPostBack
        "__EVENTARGUMENT":                 "",
        "ctl00$MainContent$btnRefresh":    "Refresh",      # the button's name + value, as a browser would send
    }

    # POST to same URL — session keeps cookies alive
    response = session.post(BASE_URL, data=payload)
    response.raise_for_status()

    # return parsed HTML ready for menu extraction
    return BeautifulSoup(response.text, "html.parser")


def parse_menu(soup):
    """
    Extract dish names from a menu response page.
    The site renders all dishes as a flat <ul> of <li class="clsMenuItem"> elements —
    there are no section/station groupings in the HTML, so all dishes go under "Items".
    Returns: {"Items": [dish_name, ...]} or {} if the meal period has no dishes.
    """
    dishes = []

    for item in soup.find_all("li", class_="clsMenuItem"):  # each <li> is one dish
        # Dish name lives in <h3 class="clsLabel_Name">
        name_tag = item.find("h3", class_="clsLabel_Name")
        if not name_tag:
            continue
        dish_name = name_tag.get_text(strip=True)
        if dish_name:
            dishes.append(dish_name)

    return {"Items": dishes} if dishes else {}              # empty dict signals "no menu for this meal"


def scrape_all_menus():
    """
    Orchestrates the full scrape: iterates every dining hall × meal period
    combination for today's date and assembles the complete menu structure.
    Returns: nested dict ready to be serialised to JSON.
             {date_str: {hall_name: {meal_name: {section: [dishes]}}}}
    """
    session, hidden, dining_halls, meal_periods, today_value = get_page_state()

    # e.g. "2026-03-29" — used as the top-level key
    today_str = date.today().isoformat()
    # root of the final JSON
    output = {today_str: {}}

    for hall_name, hall_value in dining_halls.items():
        print(f"  Scraping: {hall_name}")
        # each hall gets its own sub-dict
        output[today_str][hall_name] = {}

        for meal_name, meal_value in meal_periods.items():
            soup = fetch_menu(                              # POST for this specific hall + meal combination
                session, hidden,
                location_value=hall_value,
                meal_value=meal_value,
                day_value=today_value,
            )

            # extract {section: [dishes]} from the response
            sections = parse_menu(soup)

            if sections:                                    # only include meal periods that returned data
                output[today_str][hall_name][meal_name] = sections

        if not output[today_str][hall_name]:
            # Hall has no meals today (closed or no data) — remove it to keep output clean
            del output[today_str][hall_name]

    return output


def save_menu(data, output_dir="."):
    """
    Serialise the scraped menu dict to a JSON file named menu_YYYY-MM-DD.json.
    output_dir defaults to the current directory but can be overridden (e.g. for Docker volumes).
    Returns the path of the file written.
    """
    today_str = date.today().strftime("%Y-%m-%d")          # e.g. "2026-03-29"
    filename = f"menu_{today_str}.json"
    # join dir + filename safely on any OS
    filepath = os.path.join(output_dir, filename)

    # create output dir if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        # indent=2 for readable output; ensure_ascii=False preserves special chars in dish names
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved: {filepath}")
    return filepath


if __name__ == "__main__":
    # Entry point — run directly or via cron: `python scraper/menu_scraper.py`
    # Output always lands in scraper/data/ regardless of working directory

    print(f"Scraping Stanford dining menus for {date.today().isoformat()} ...")

    # fetch and parse all halls + meal periods
    menu_data = scrape_all_menus()

    # Save relative to this script's location so output is always scraper/data/,
    # regardless of which directory the script is invoked from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_menu(menu_data, output_dir=os.path.join(script_dir, "data"))

    print("Done.")
