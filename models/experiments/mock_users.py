# models/experiments/mock_users.py
# Generates 15 synthetic users for recommendation evaluation.
# Each user has a vague signup_text (what the model sees) and a hidden_profile
# (ground truth used only for scoring — never passed to the model).

import json
from pathlib import Path

MOCK_USERS = [
    {
        "user_id": "mock_user_01",
        "signup_text": "I like Asian food and prefer vegetarian options.",
        "hidden_profile": "loves mapo tofu, spicy noodles, kimchi, bold umami flavors",
        "allergens": ["shellfish"]
    },
    {
        "user_id": "mock_user_02",
        "signup_text": "I enjoy hearty meals and American comfort food.",
        "hidden_profile": "loves burgers, BBQ ribs, mac and cheese, fried chicken",
        "allergens": []
    },
    {
        "user_id": "mock_user_03",
        "signup_text": "I eat vegan and care about whole foods.",
        "hidden_profile": "loves grain bowls, roasted vegetables, lentil soup, avocado",
        "allergens": ["dairy", "eggs"]
    },
    {
        "user_id": "mock_user_04",
        "signup_text": "I prefer light meals, mostly Mediterranean.",
        "hidden_profile": "loves hummus, falafel, grilled fish, tabbouleh, olive oil",
        "allergens": ["peanuts"]
    },
    {
        "user_id": "mock_user_05",
        "signup_text": "I like spicy food a lot.",
        "hidden_profile": "loves thai curry, spicy ramen, hot wings, chili dishes",
        "allergens": []
    },
    {
        "user_id": "mock_user_06",
        "signup_text": "I follow a halal diet and enjoy Middle Eastern cuisine.",
        "hidden_profile": "loves shawarma, kabobs, rice pilaf, lentil dishes",
        "allergens": []
    },
    {
        "user_id": "mock_user_07",
        "signup_text": "I try to eat low carb and high protein.",
        "hidden_profile": "loves grilled chicken, steak, eggs, salmon, salads with protein",
        "allergens": ["gluten"]
    },
    {
        "user_id": "mock_user_08",
        "signup_text": "I love seafood and coastal food.",
        "hidden_profile": "loves salmon, shrimp tacos, clam chowder, grilled fish",
        "allergens": []
    },
    {
        "user_id": "mock_user_09",
        "signup_text": "I enjoy Japanese and Korean food.",
        "hidden_profile": "loves sushi, ramen, bibimbap, teriyaki, miso soup",
        "allergens": ["shellfish"]
    },
    {
        "user_id": "mock_user_10",
        "signup_text": "I like Mexican and Latin food.",
        "hidden_profile": "loves tacos, enchiladas, tamales, rice and beans, guacamole",
        "allergens": []
    },
    {
        "user_id": "mock_user_11",
        "signup_text": "I eat gluten free and prefer simple meals.",
        "hidden_profile": "loves rice bowls, grilled proteins, roasted veggies, soups",
        "allergens": ["gluten", "wheat"]
    },
    {
        "user_id": "mock_user_12",
        "signup_text": "I am lactose intolerant and like fresh food.",
        "hidden_profile": "loves salads, grilled chicken, fruit, light soups, sushi",
        "allergens": ["dairy"]
    },
    {
        "user_id": "mock_user_13",
        "signup_text": "I like trying new foods from different cultures.",
        "hidden_profile": "loves Ethiopian injera, Indian curry, Moroccan tagine, Vietnamese pho",
        "allergens": []
    },
    {
        "user_id": "mock_user_14",
        "signup_text": "I prefer vegetarian food, mostly Indian cuisine.",
        "hidden_profile": "loves dal, paneer dishes, vegetable biryani, samosas, naan",
        "allergens": ["eggs"]
    },
    {
        "user_id": "mock_user_15",
        "signup_text": "I like pasta and Italian food.",
        "hidden_profile": "loves spaghetti bolognese, risotto, pizza, bruschetta, tiramisu",
        "allergens": ["shellfish", "peanuts"]
    },
]


def save_mock_users(output_path: str = "models/experiments/mock_users.json"):
    # write the list to a JSON file so other scripts can load it without importing this module
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(MOCK_USERS, f, indent=2)
    print(f"Saved {len(MOCK_USERS)} mock users to {path}")


if __name__ == "__main__":
    save_mock_users()
