import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { predict } from "./api";
import type { PredictResponseBody } from "./api";
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
  const [moodText, setMoodText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponseBody | null>(null);

  const userId = session?.user?.id ?? null;

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
    const body = {
      mood: moodText.trim() || undefined,
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

  function onPick(dishName: string, diningHall: string) {
    console.log("[pick]", { dishName, diningHall, userId });
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
          <span className="label">Your mood or cravings (optional)</span>
          <textarea
            name="mood"
            rows={3}
            placeholder="e.g. something spicy, light and warm, craving sushi..."
            value={moodText}
            onChange={(e) => setMoodText(e.target.value)}
          />
        </label>

        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Finding dishes..." : "Get Recommendations"}
        </button>
      </form>

      {error && (
        <div className="card message error" role="alert">
          <strong>Error</strong>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <>
          <p className="preference-hint">
            Your taste profile: {result.preference_summary}
          </p>
          <div className="dish-cards">
            {[...result.recommendations, ...result.alternatives].map((dish) => (
              <div className="card dish-card" key={`${dish.dish_name}-${dish.dining_hall}`}>
                <h3 className="dish-name">{dish.dish_name}</h3>
                <p className="dish-hall">{dish.dining_hall}</p>
                <p className="dish-reason">{dish.reason}</p>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onPick(dish.dish_name, dish.dining_hall)}
                >
                  Pick this
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
