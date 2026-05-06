import { useState } from "react";
import { supabase } from "./supabaseClient";

type AuthPageProps = {
  onAuthSuccess: () => void;
};

export default function AuthPage({ onAuthSuccess }: AuthPageProps) {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setMessage(null);
    setLoading(true);

    try {
      if (mode === "signin") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        onAuthSuccess();
      } else {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setMessage("Account created. If email confirmation is enabled, check your inbox.");
        onAuthSuccess();
      }
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h1 className="auth-title">Lunch Buddy</h1>
        <p className="auth-subtitle">The friend that always knows where to go</p>

        <form className="auth-form" onSubmit={onSubmit}>

          <label className="auth-field">
            <span className="auth-label">Email</span>
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>

          <label className="auth-field">
            <span className="auth-label">Password</span>
            <input
              type="password"
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              minLength={6}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? "Working..." : mode === "signin" ? "Sign In" : "Create Account"}
          </button>

          <button
            type="button"
            className="auth-toggle"
            onClick={() => setMode((m) => (m === "signin" ? "signup" : "signin"))}
            disabled={loading}
          >
            {mode === "signin" ? "Need an account? Create one" : "Already have an account? Sign in"}
          </button>
        </form>

        {message && (
          <div className="auth-message" role="status">
            <p>{message}</p>
          </div>
        )}
      </div>
    </div>
  );
}
