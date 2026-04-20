import { useEffect, useMemo, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { predict } from "./api";
import AuthPage from "./AuthPage";
import ProfilePage from "./ProfilePage";
import TasteTuningPlaceholder from "./TasteTuningPlaceholder";
import { loadProfile } from "./profileStorage";
import type { ProfileState } from "./profileTypes";
import { supabase } from "./supabaseClient";
import "./App.css";

type AppView = "home" | "profile" | "tuning";

const emptyProfile: ProfileState = { displayName: "", allergens: [], diets: [] };

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [view, setView] = useState<AppView>("home");
  const [profile, setProfile] = useState<ProfileState>(emptyProfile);
  const [preferencesText, setPreferencesText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ suggestions: string[]; rationale?: string | null } | null>(
    null,
  );

  const userId = session?.user?.id ?? null;

  const constraints = useMemo(
    () => [...profile.allergens, ...profile.diets],
    [profile.allergens, profile.diets],
  );

  useEffect(() => {
    let mounted = true;

    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session ?? null);
      setAuthReady(true);
    });

    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setAuthReady(true);
    });

    return () => {
      mounted = false;
      data.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!userId) {
      setProfile(emptyProfile);
      return;
    }
    setProfile(loadProfile(userId));
  }, [userId]);

  async function onRecommend(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    const preferences = preferencesText.trim() ? [preferencesText.trim()] : [];
    const body = {
      user_id: userId,
      preferences,
      constraints,
    };
    try {
      const data = await predict(body, session?.access_token);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function onSignOut() {
    await supabase.auth.signOut();
    setView("home");
  }

  function handleProfileSaved(next: ProfileState) {
    setProfile(next);
  }

  if (!authReady) {
    return (
      <div className="app">
        <div className="card message">
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return <AuthPage onAuthSuccess={() => undefined} />;
  }

  if (!userId) {
    return (
      <div className="app">
        <div className="card message">
          <p>Missing user session.</p>
        </div>
      </div>
    );
  }

  if (view === "profile") {
    return (
      <ProfilePage
        userId={userId}
        email={session.user.email}
        onBack={() => setView("home")}
        onOpenTasteTuning={() => setView("tuning")}
        onProfileSaved={handleProfileSaved}
      />
    );
  }

  if (view === "tuning") {
    return <TasteTuningPlaceholder onBack={() => setView("profile")} />;
  }

  const displayLabel =
    profile.displayName.trim() ||
    session.user.email?.split("@")[0] ||
    "there";

  return (
    <div className="app">
      <header className="header">
        <h1>Lunch Buddy</h1>
        <p className="tagline">Hi, {displayLabel} — what sounds good today?</p>
        <div className="auth-row">
          <button type="button" className="link-inline" onClick={() => setView("profile")}>
            Profile
          </button>
          <span className="auth-user">{session.user.email}</span>
          <button type="button" className="secondary" onClick={onSignOut}>
            Sign Out
          </button>
        </div>
      </header>

      <form className="card" onSubmit={onRecommend}>
        <h2 className="home-prompt">What do you want for lunch?</h2>
        <label className="field">
          <span className="label">Your mood or cravings</span>
          <textarea
            name="preferences"
            rows={4}
            placeholder="e.g. something light and warm, near Stern Hall…"
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
