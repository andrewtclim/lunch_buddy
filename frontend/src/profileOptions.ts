/** Allergen and diet chips for the profile page — extend these arrays as new options ship. */
export const PROFILE_ALLERGEN_OPTIONS = [
  "Egg",
  "Milk",
  "Soy",
  "Wheat",
  "Sesame",
  "Fish",
  "Shellfish",
  "Coconut",
] as const;

/** Reserved for future allergens; render in the same chip grid when non-empty. */
export const PROFILE_ALLERGEN_OPTIONS_EXTRA: readonly string[] = [];

export const PROFILE_DIET_OPTIONS = ["Vegetarian", "Vegan", "Halal"] as const;

/** Reserved for future dietary labels. */
export const PROFILE_DIET_OPTIONS_EXTRA: readonly string[] = [];
