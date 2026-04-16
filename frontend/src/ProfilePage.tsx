import { useEffect, useState } from "react";
import {
  PROFILE_ALLERGEN_OPTIONS,
  PROFILE_ALLERGEN_OPTIONS_EXTRA,
  PROFILE_DIET_OPTIONS,
  PROFILE_DIET_OPTIONS_EXTRA,
} from "./profileOptions";
import { loadProfile, saveProfile } from "./profileStorage";
import type { ProfileState } from "./profileTypes";

function toggleInList(list: string[], value: string): string[] {
  if (list.includes(value)) return list.filter((x) => x !== value);
  return [...list, value];
}

type ProfilePageProps = {
  userId: string;
  email: string | undefined;
  onBack: () => void;
  onOpenTasteTuning: () => void;
  onProfileSaved?: (state: ProfileState) => void;
};

export default function ProfilePage({
  userId,
  email,
  onBack,
  onOpenTasteTuning,
  onProfileSaved,
}: ProfilePageProps) {
  const [displayName, setDisplayName] = useState("");
  const [allergens, setAllergens] = useState<string[]>([]);
  const [diets, setDiets] = useState<string[]>([]);
  const [savedHint, setSavedHint] = useState(false);

  useEffect(() => {
    const p = loadProfile(userId);
    setDisplayName(p.displayName);
    setAllergens(p.allergens);
    setDiets(p.diets);
  }, [userId]);

  function persist(next: ProfileState, showNameSavedHint: boolean) {
    saveProfile(userId, next);
    onProfileSaved?.(next);
    if (showNameSavedHint) {
      setSavedHint(true);
      window.setTimeout(() => setSavedHint(false), 2000);
    }
  }

  function handleSaveDisplayName(e: React.FormEvent) {
    e.preventDefault();
    persist({ displayName, allergens, diets }, true);
  }

  function handleAllergenToggle(value: string) {
    const nextAllergens = toggleInList(allergens, value);
    setAllergens(nextAllergens);
    persist({ displayName, allergens: nextAllergens, diets }, false);
  }

  function handleDietToggle(value: string) {
    const nextDiets = toggleInList(diets, value);
    setDiets(nextDiets);
    persist({ displayName, allergens, diets: nextDiets }, false);
  }

  const allergenChoices = [...PROFILE_ALLERGEN_OPTIONS, ...PROFILE_ALLERGEN_OPTIONS_EXTRA];
  const dietChoices = [...PROFILE_DIET_OPTIONS, ...PROFILE_DIET_OPTIONS_EXTRA];

  return (
    <div className="app">
      <header className="header">
        <button type="button" className="link-back" onClick={onBack}>
          ← Back to lunch
        </button>
        <h1>Your profile</h1>
        <p className="tagline">
          {email ? <span className="muted">{email}</span> : null}
        </p>
      </header>

      <form className="card profile-section" onSubmit={handleSaveDisplayName}>
        <h2 className="section-title">Display name</h2>
        <p className="section-help">Shown in the app; synced to your account when backend storage is wired up.</p>
        <label className="field">
          <span className="label">Name</span>
          <input
            type="text"
            autoComplete="nickname"
            placeholder="How should we call you?"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
        </label>
        <button type="submit" className="primary">
          Save display name
        </button>
        {savedHint && (
          <p className="saved-hint" role="status">
            Saved locally on this device.
          </p>
        )}
      </form>

      <section className="card profile-section">
        <h2 className="section-title">Allergens</h2>
        <p className="section-help">We’ll use these to filter recommendations. More options may be added later.</p>
        <div className="chips chips-extended">
          {allergenChoices.map((opt) => (
            <label key={opt} className="chip">
              <input
                type="checkbox"
                checked={allergens.includes(opt)}
                onChange={() => handleAllergenToggle(opt)}
              />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="card profile-section">
        <h2 className="section-title">Dietary restrictions</h2>
        <p className="section-help">Vegetarian, vegan, halal — additional diets can be added over time.</p>
        <div className="chips chips-extended">
          {dietChoices.map((opt) => (
            <label key={opt} className="chip">
              <input
                type="checkbox"
                checked={diets.includes(opt)}
                onChange={() => handleDietToggle(opt)}
              />
              <span>{opt}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="card profile-section placeholder-section">
        <h2 className="section-title">Taste profile</h2>
        <p className="section-help">
          Soon you’ll be able to run a short exercise to initialize or tune your preference vector from a few choices.
        </p>
        <button type="button" className="link-standalone" onClick={onOpenTasteTuning}>
          Tune your taste profile →
        </button>
      </section>
    </div>
  );
}
