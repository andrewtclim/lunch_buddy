import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { predict, pickDish, onboard, checkProfile } from "./api";
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
  const [pickedDish, setPickedDish] = useState<string | null>(null);
  const [pickMsg, setPickMsg] = useState<string | null>(null);
  const [needsOnboarding, setNeedsOnboarding] = useState<boolean | null>(null);
  const [onboardBlurb, setOnboardBlurb] = useState("");
  const [onboardLoading, setOnboardLoading] = useState(false);
  const [onboardError, setOnboardError] = useState<string | null>(null);

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

  // check if the logged-in user has a profile yet
  useEffect(() => {
    const token = session?.access_token;
    if (!token) {
      setNeedsOnboarding(null);
      return;
    }
    checkProfile(token).then((has) => setNeedsOnboarding(!has));
  }, [session?.access_token]);

  async function onOnboard(e: React.FormEvent) {
    e.preventDefault();
    if (!onboardBlurb.trim()) return;
    setOnboardLoading(true);
    setOnboardError(null);
    try {
      await onboard({ blurb: onboardBlurb.trim() }, session?.access_token);
      setNeedsOnboarding(false);
    } catch (err) {
      setOnboardError(err instanceof Error ? err.message : "Onboarding failed");
    } finally {
      setOnboardLoading(false);
    }
  }

  async function onRecommend(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setPickedDish(null);
    setPickMsg(null);
    setLoading(true);
    const body = {
      mood: moodText.trim() || undefined,
    };
    try {
      const data = await predict(body, session?.access_token);
      setResult(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      if (msg.includes("taste profile")) {
        setNeedsOnboarding(true);
        return;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  async function onPick(dishName: string, diningHall: string) {
    setPickedDish(dishName);
    setPickMsg(null);
    try {
      const data = await pickDish(
        { dish_name: dishName, dining_hall: diningHall },
        session?.access_token,
      );
      setPickMsg(data.message);
    } catch (err) {
      setPickMsg(err instanceof Error ? err.message : "Pick failed");
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

  if (needsOnboarding === null) {
    return (
      <div className="app">
        <div className="card message">
          <p>Loading your profile...</p>
        </div>
      </div>
    );
  }

  if (needsOnboarding) {
    return (
      <div className="app">
        <header className="header">
          <h1>Lunch Buddy</h1>
          <p className="tagline">Welcome! Tell us what you like to eat.</p>
        </header>

        <form className="card" onSubmit={onOnboard}>
          <h2 className="home-prompt">Set up your taste profile</h2>
          <label className="field">
            <span className="label">
              Describe your food preferences in a sentence or two
            </span>
            <textarea
              name="blurb"
              rows={4}
              placeholder="e.g. I love spicy noodles, grilled chicken, and anything with garlic. Not a fan of seafood."
              value={onboardBlurb}
              onChange={(e) => setOnboardBlurb(e.target.value)}
            />
          </label>

          <button
            type="submit"
            className="primary"
            disabled={onboardLoading || !onboardBlurb.trim()}
          >
            {onboardLoading ? "Setting up..." : "Save & Continue"}
          </button>
        </form>

        {onboardError && (
          <div className="card message error" role="alert">
            <strong>Error</strong>
            <p>{onboardError}</p>
          </div>
        )}
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

      {!pickMsg && <form className="card" onSubmit={onRecommend}>
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
      </form>}

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

          {pickMsg ? (
            // after picking: show only the chosen dish, centered
            <div className="pick-confirmation">
              {[...result.recommendations, ...result.alternatives]
                .filter((dish) => dish.dish_name === pickedDish)
                .map((dish) => (
                  <div className="card dish-card picked" key={`${dish.dish_name}-${dish.dining_hall}`}>
                    <h3 className="dish-name">{dish.dish_name}</h3>
                    <p className="dish-hall">{dish.dining_hall}</p>
                    <p className="dish-reason">{dish.reason}</p>
                    <p className="pick-msg">{pickMsg}</p>
                  </div>
                ))}
              <button
                type="button"
                className="secondary"
                onClick={() => { setPickedDish(null); setPickMsg(null); }}
              >
                Back to results
              </button>
            </div>
          ) : (
            // before picking: show all dish cards
            <div className="dish-cards">
              {[...result.recommendations, ...result.alternatives].map((dish) => (
                <div className="card dish-card" key={`${dish.dish_name}-${dish.dining_hall}`}>
                  <h3 className="dish-name">{dish.dish_name}</h3>
                  <p className="dish-hall">{dish.dining_hall}</p>
                  <p className="dish-reason">{dish.reason}</p>
                  <button
                    type="button"
                    className="secondary"
                    disabled={pickedDish !== null}
                    onClick={() => onPick(dish.dish_name, dish.dining_hall)}
                  >
                    {pickedDish === dish.dish_name ? "Picking..." : "Pick this"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
