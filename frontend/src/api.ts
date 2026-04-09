const base = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export type PredictRequestBody = {
  user_id: string | null;
  preferences: string[];
  constraints: string[];
};

export type PredictResponseBody = {
  suggestions: string[];
  rationale?: string | null;
};

export async function predict(body: PredictRequestBody): Promise<PredictResponseBody> {
  const res = await fetch(`${base}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
