import type { ProfileState } from "./profileTypes";

const STORAGE_PREFIX = "lunchBuddy.profile.";

function key(userId: string) {
  return `${STORAGE_PREFIX}${userId}`;
}

export function loadProfile(userId: string): ProfileState {
  try {
    const raw = localStorage.getItem(key(userId));
    if (!raw) {
      return { displayName: "", allergens: [], diets: [] };
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") {
      return { displayName: "", allergens: [], diets: [] };
    }
    const o = parsed as Record<string, unknown>;
    const displayName = typeof o.displayName === "string" ? o.displayName : "";
    const allergens = Array.isArray(o.allergens)
      ? o.allergens.filter((x): x is string => typeof x === "string")
      : [];
    const diets = Array.isArray(o.diets) ? o.diets.filter((x): x is string => typeof x === "string") : [];
    return { displayName, allergens, diets };
  } catch {
    return { displayName: "", allergens: [], diets: [] };
  }
}

export function saveProfile(userId: string, next: ProfileState) {
  localStorage.setItem(
    key(userId),
    JSON.stringify({
      displayName: next.displayName.trim(),
      allergens: next.allergens,
      diets: next.diets,
    }),
  );
}
