import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { predict, pickDish, onboard, checkProfile } from "./api";
import type { PredictRequestBody, PredictResponseBody } from "./api";
import AuthPage from "./AuthPage";
import ProfilePage from "./ProfilePage";
import TasteTuningPlaceholder from "./TasteTuningPlaceholder";
import { loadProfile } from "./profileStorage";
import type { ProfileState } from "./profileTypes";
import { PROFILE_ALLERGEN_OPTIONS } from "./profileOptions";
import DishCard from "./DishCard";
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
  const [userLocation, setUserLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [locationEnabled, setLocationEnabled] = useState(true);
  const [needsOnboarding, setNeedsOnboarding] = useState<boolean | null>(null);
  const [onboardBlurb, setOnboardBlurb] = useState("");
  const [onboardAllergens, setOnboardAllergens] = useState<string[]>([]);
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

  // auto-detect location on mount; dev override via VITE_DEV_LOCATION env var
  // usage: VITE_DEV_LOCATION=37.4248,-122.1655 npm run dev
  useEffect(() => {
    const devLoc = import.meta.env.VITE_DEV_LOCATION as string | undefined;
    if (devLoc) {
      const [lat, lon] = devLoc.split(",").map(Number);
      if (!isNaN(lat) && !isNaN(lon)) {
        setUserLocation({ lat, lon });
        return;
      }
    }
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setUserLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude });
      },
      () => {},
      { enableHighAccuracy: false, timeout: 5000 },
    );
  }, []);

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
      await onboard({ blurb: onboardBlurb.trim(), allergens: onboardAllergens }, session?.access_token);
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
    const body: PredictRequestBody = {
      mood: moodText.trim() || undefined,
    };
    if (locationEnabled && userLocation) {
      body.latitude = userLocation.lat;
      body.longitude = userLocation.lon;
    }
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
      setPickMsg(`Enjoy your ${dishName}!`);
    }
  }

  async function onSignOut() {
    await supabase.auth.signOut();
    setView("home");
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
    function toggleAllergen(value: string) {
      setOnboardAllergens((prev) =>
        prev.includes(value) ? prev.filter((x) => x !== value) : [...prev, value]
      );
    }

    return (
      <div className="auth-page">
        <div className="auth-container onboard-container">
          <h1 className="auth-title">Lunch Buddy</h1>
          <p className="auth-subtitle">Tell us what you like to eat</p>

          <form className="auth-form" onSubmit={onOnboard}>
            <label className="auth-field">
              <span className="auth-label">Describe your food preferences</span>
              <textarea
                className="onboard-textarea"
                name="blurb"
                rows={3}
                placeholder="e.g. I love spicy noodles, grilled chicken, and anything with garlic."
                value={onboardBlurb}
                onChange={(e) => setOnboardBlurb(e.target.value)}
              />
            </label>

            <div className="onboard-section">
              <span className="auth-label">Any allergens?</span>
              <div className="onboard-chips">
                {PROFILE_ALLERGEN_OPTIONS.map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    className={`onboard-chip ${onboardAllergens.includes(opt) ? "active" : ""}`}
                    onClick={() => toggleAllergen(opt)}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="submit"
              className="auth-submit"
              disabled={onboardLoading || !onboardBlurb.trim()}
            >
              {onboardLoading ? "Setting up..." : "Save & Continue"}
            </button>
          </form>

          {onboardError && (
            <div className="auth-message" role="alert">
              <p>{onboardError}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (view === "profile") {
    return (
      <ProfilePage
        userId={userId}
        email={session.user.email}
        accessToken={session.access_token}
        onBack={() => setView("home")}
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
        <p className="tagline">{displayLabel}'s table</p>
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

      {!result && !pickMsg && (
        <form onSubmit={onRecommend}>
          <div className="plate-wrapper">
            {/* Fork */}
            <svg className="plate-fork" viewBox="0 0 32 180" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Tines - slim pills with rounded ends */}
              <rect x="9" y="6" width="3" height="44" rx="1.5" fill="#D0CCC5"/>
              <rect x="10" y="6" width="1.5" height="44" rx="0.75" fill="#B8B4AD"/>
              <rect x="13.5" y="6" width="3" height="44" rx="1.5" fill="#D0CCC5"/>
              <rect x="14.5" y="6" width="1.5" height="44" rx="0.75" fill="#B8B4AD"/>
              <rect x="18" y="6" width="3" height="44" rx="1.5" fill="#D0CCC5"/>
              <rect x="19" y="6" width="1.5" height="44" rx="0.75" fill="#B8B4AD"/>
              <rect x="22.5" y="6" width="3" height="44" rx="1.5" fill="#D0CCC5"/>
              <rect x="23.5" y="6" width="1.5" height="44" rx="0.75" fill="#B8B4AD"/>
              {/* Neck - narrow connector */}
              <path d="M9 50 C9 60 13 66 16 66 C19 66 23 60 25.5 50" fill="#D0CCC5"/>
              <path d="M16 50 C17 50 23 60 25.5 50" fill="#B8B4AD"/>
              {/* Neck stem */}
              <rect x="14" y="66" width="4" height="10" rx="2" fill="#D0CCC5"/>
              <rect x="16" y="66" width="2" height="10" rx="1" fill="#B8B4AD"/>
              {/* Handle - wider */}
              <rect x="11" y="76" width="10" height="96" rx="5" fill="#D0CCC5"/>
              <rect x="16" y="76" width="5" height="96" rx="2.5" fill="#B8B4AD"/>
            </svg>

            <div className="plate">
              <div className="plate-inner">
                <h2>What's on your mind?</h2>
                <textarea
                  name="mood"
                  rows={3}
                  placeholder="spicy, light, craving sushi..."
                  value={moodText}
                  onChange={(e) => setMoodText(e.target.value)}
                />
                <div className="plate-controls">
                  <label className="location-toggle">
                    <input
                      type="checkbox"
                      checked={locationEnabled}
                      onChange={(e) => setLocationEnabled(e.target.checked)}
                    />
                    <span>Nearby halls only</span>
                  </label>
                  <button type="submit" className="primary" disabled={loading}>
                    {loading ? "Finding..." : "Get Recs"}
                  </button>
                </div>
              </div>
            </div>

            {/* Knife */}
            <svg className="plate-knife" viewBox="0 0 32 180" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Blade - straight spine, wider and longer with gentle curve to tip */}
              <path d="M12 4 L12 82 C12 82 13 84 16 84 C19 84 23 81 23 76 C23 66 22 46 21 28 C20 16 17 7 12 4 Z" fill="#D0CCC5"/>
              <path d="M16 84 C19 84 23 81 23 76 C23 66 22 46 21 28 C20 16 17 7 12 4 L12 10 C15 16 18 28 19.5 46 C20.5 64 20 78 16 84 Z" fill="#B8B4AD"/>
              {/* Spine line */}
              <line x1="12" y1="4" x2="12" y2="84" stroke="#A8A49D" strokeWidth="0.5"/>
              {/* Bolster */}
              <rect x="12" y="84" width="8" height="6" rx="2" fill="#D0CCC5"/>
              <rect x="16" y="84" width="4" height="6" rx="2" fill="#B8B4AD"/>
              {/* Handle - wider */}
              <rect x="11" y="90" width="10" height="82" rx="5" fill="#D0CCC5"/>
              <rect x="16" y="90" width="5" height="82" rx="2.5" fill="#B8B4AD"/>
            </svg>
          </div>
        </form>
      )}

      {error && (
        <div className="card message error" role="alert">
          <strong>Error</strong>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <>
          {!pickMsg && (
            <p className="preference-hint">
              Here are some options nearby you might enjoy
            </p>
          )}

          {pickMsg ? (
            <div className="pick-confirmation">
              {[...result.recommendations, ...result.alternatives]
                .filter((dish) => dish.dish_name === pickedDish)
                .map((dish) => (
                  <div className="dish-card" key={`${dish.dish_name}-${dish.dining_hall}`}>
                    <h3 className="dish-card__name">{dish.dish_name}</h3>
                    <p className="dish-card__hall">{dish.dining_hall}</p>
                    <p className="dish-card__reason">{dish.reason}</p>
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
              <button
                type="button"
                className="secondary"
                onClick={() => { setResult(null); setPickedDish(null); setPickMsg(null); }}
              >
                New search
              </button>
            </div>
          ) : (
            <>
              <div className="dish-cards">
                {[...result.recommendations, ...result.alternatives].map((dish) => (
                  <DishCard
                    key={`${dish.dish_name}-${dish.dining_hall}`}
                    dish_name={dish.dish_name}
                    dining_hall={dish.dining_hall}
                    reason={dish.reason}
                    distance_m={dish.distance_m}
                    picked={pickedDish === dish.dish_name}
                    disabled={pickedDish !== null}
                    onPick={() => onPick(dish.dish_name, dish.dining_hall)}
                  />
                ))}
              </div>
              <button
                type="button"
                className="secondary back-to-plate"
                onClick={() => { setResult(null); setPickedDish(null); setPickMsg(null); }}
              >
                Change my mood
              </button>
            </>
          )}
        </>
      )}
    </div>
  );
}
