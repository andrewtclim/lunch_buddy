const base = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type DishCard = {
  dish_name: string;
  dining_hall: string;
  reason: string;
};

export type PredictRequestBody = {
  mood?: string;
  date?: string;
};

export type PredictResponseBody = {
  recommendations: DishCard[];
  alternatives: DishCard[];
  preference_summary: string;
};

// --- /pick: tell the backend which dish the user chose ---

export type PickRequestBody = {
  dish_name: string;
  dining_hall: string;
};

export type PickResponseBody = {
  status: string;
  message: string;
};

export async function pickDish(
  body: PickRequestBody,
  accessToken?: string | null,
): Promise<PickResponseBody> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }

  const res = await fetch(`${base}/pick`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`Invalid JSON from server (${res.status})`);
  }
  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : text || res.statusText;
    throw new Error(detail);
  }
  return data as PickResponseBody;
}

// --- /me/profile: lightweight profile check ---

export async function checkProfile(
  accessToken: string,
): Promise<boolean> {
  const res = await fetch(`${base}/me/profile`, {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) return false;  // network/auth error -- assume no profile
  const data = await res.json();
  return data.has_profile === true;
}

// --- /onboard: first-time taste profile setup ---

export type OnboardRequestBody = {
  blurb: string;  // free-text signup description
};

export type OnboardResponseBody = {
  status: string;
  preference_summary: string;
};

export async function onboard(
  body: OnboardRequestBody,
  accessToken?: string | null,
): Promise<OnboardResponseBody> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }

  const res = await fetch(`${base}/onboard`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`Invalid JSON from server (${res.status})`);
  }
  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : text || res.statusText;
    throw new Error(detail);
  }
  return data as OnboardResponseBody;
}

// --- /predict: get recommendations ---

export async function predict(
  body: PredictRequestBody,
  accessToken?: string | null,
): Promise<PredictResponseBody> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }

  const res = await fetch(`${base}/predict`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`Invalid JSON from server (${res.status})`);
  }
  if (!res.ok) {
    const detail =
      typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : text || res.statusText;
    throw new Error(detail);
  }
  return data as PredictResponseBody;
}
