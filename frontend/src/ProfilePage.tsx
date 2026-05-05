import { useEffect, useState } from "react";
import { PROFILE_ALLERGEN_OPTIONS, PROFILE_DIET_OPTIONS } from "./profileOptions";
import { updateAllergens } from "./api";

type ProfilePageProps = {
  userId: string;
  email: string | undefined;
  accessToken?: string | null;
  onBack: () => void;
};

export default function ProfilePage({
  userId,
  email,
  accessToken,
  onBack,
}: ProfilePageProps) {
  const [displayName, setDisplayName] = useState("");
  const [allergens, setAllergens] = useState<string[]>([]);
  const [diets, setDiets] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [savedHint, setSavedHint] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(`lunchBuddy.profile.${userId}`);
      if (raw) {
        const parsed = JSON.parse(raw);
        setDisplayName(parsed.displayName || "");
        setAllergens(parsed.allergens || []);
        setDiets(parsed.diets || []);
      }
    } catch { /* ignore */ }
  }, [userId]);

  function persistLocal(nextAllergens: string[], nextDiets: string[], nextName: string) {
    localStorage.setItem(
      `lunchBuddy.profile.${userId}`,
      JSON.stringify({ displayName: nextName, allergens: nextAllergens, diets: nextDiets }),
    );
  }

  function toggleAllergen(value: string) {
    const next = allergens.includes(value)
      ? allergens.filter((x) => x !== value)
      : [...allergens, value];
    setAllergens(next);
    persistLocal(next, diets, displayName);
  }

  async function saveAllergens() {
    setSaving(true);
    try {
      await updateAllergens(allergens, accessToken);
      setSavedHint(true);
      setTimeout(() => setSavedHint(false), 2000);
    } catch { /* silent fail */ }
    finally { setSaving(false); }
  }

  function toggleDiet(value: string) {
    const next = diets.includes(value)
      ? diets.filter((x) => x !== value)
      : [...diets, value];
    setDiets(next);
    persistLocal(allergens, next, displayName);
    // TODO: sync dietary restrictions to backend once pipeline supports them
  }

  function saveDisplayName(e: React.FormEvent) {
    e.preventDefault();
    persistLocal(allergens, diets, displayName.trim());
    setSavedHint(true);
    setTimeout(() => setSavedHint(false), 2000);
    // TODO: sync display name to profiles table in Supabase
  }

  return (
    <div className="auth-page">
      <div className="auth-container profile-container">
        <h1 className="auth-title">Your Profile</h1>
        <p className="auth-subtitle">{email}</p>

        <form className="profile-section-styled" onSubmit={saveDisplayName}>
          <span className="auth-label">Display name</span>
          <input
            className="profile-input"
            type="text"
            placeholder="What should we call you?"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
          <button type="submit" className="profile-save-btn">Save</button>
        </form>

        <div className="profile-section-styled">
          <span className="auth-label">Allergens</span>
          <p className="profile-help">We filter these out of your recommendations</p>
          <div className="onboard-chips">
            {PROFILE_ALLERGEN_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`onboard-chip ${allergens.includes(opt) ? "active" : ""}`}
                onClick={() => toggleAllergen(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="profile-save-btn"
            onClick={saveAllergens}
            disabled={saving}
          >
            {saving ? "Saving..." : "Save allergens"}
          </button>
        </div>

        <div className="profile-section-styled">
          <span className="auth-label">Dietary restrictions</span>
          <div className="onboard-chips">
            {PROFILE_DIET_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`onboard-chip ${diets.includes(opt) ? "active" : ""}`}
                onClick={() => toggleDiet(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {savedHint && <p className="profile-saved">Saved</p>}

        <div className="profile-actions">
          <button type="button" className="secondary" onClick={onBack}>
            Back to lunch
          </button>
        </div>
      </div>
    </div>
  );
}
