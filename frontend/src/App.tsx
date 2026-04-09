import { useMemo, useState } from "react";
import { predict } from "./api";
import { ALLERGEN_OPTIONS, RESTRICTION_OPTIONS } from "./options";
import "./App.css";

function toggleInSet(set: Set<string>, value: string): Set<string> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export default function App() {
  const [userId, setUserId] = useState("");
  const [allergens, setAllergens] = useState<Set<string>>(() => new Set());
  const [restrictions, setRestrictions] = useState<Set<string>>(() => new Set());
  const [preferencesText, setPreferencesText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ suggestions: string[]; rationale?: string | null } | null>(
    null,
  );

  const constraints = useMemo(
    () => [...allergens, ...restrictions],
    [allergens, restrictions],
  );

  async function onRecommend(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    const preferences = preferencesText.trim() ? [preferencesText.trim()] : [];
    const body = {
      user_id: userId.trim() || null,
      preferences,
      constraints,
    };
    try {
      const data = await predict(body);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Lunch Buddy</h1>
        <p className="tagline">Set allergens, restrictions, and preferences — then get a recommendation.</p>
      </header>

      <form className="card" onSubmit={onRecommend}>
        <label className="field">
          <span className="label">User ID</span>
          <input
            type="text"
            name="user_id"
            autoComplete="username"
            placeholder="e.g. demo-user-1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
        </label>

        <fieldset className="fieldset">
          <legend className="legend">Allergens</legend>
          <div className="chips">
            {ALLERGEN_OPTIONS.map((opt) => (
              <label key={opt} className="chip">
                <input
                  type="checkbox"
                  checked={allergens.has(opt)}
                  onChange={() => setAllergens((s) => toggleInSet(s, opt))}
                />
                <span>{opt}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="fieldset">
          <legend className="legend">Restrictions</legend>
          <div className="chips">
            {RESTRICTION_OPTIONS.map((opt) => (
              <label key={opt} className="chip">
                <input
                  type="checkbox"
                  checked={restrictions.has(opt)}
                  onChange={() => setRestrictions((s) => toggleInSet(s, opt))}
                />
                <span>{opt}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <label className="field">
          <span className="label">Preferences</span>
          <textarea
            name="preferences"
            rows={3}
            placeholder="Anything else we should know (e.g. loves spicy food, quick lunch)…"
            value={preferencesText}
            onChange={(e) => setPreferencesText(e.target.value)}
          />
        </label>

        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Working…" : "Recommend"}
        </button>
      </form>

      {error && (
        <div className="card message error" role="alert">
          <strong>Error</strong>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="card message success">
          <strong>Suggestions</strong>
          <ul className="suggestions">
            {result.suggestions.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
          {result.rationale != null && result.rationale !== "" && (
            <p className="rationale">
              <strong>Why:</strong> {result.rationale}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
