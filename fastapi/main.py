# main.py — skeleton; adapt schemas and inference to your LLM wrapper

import os
import sys
import time
from datetime import date
from urllib.error import URLError
from urllib.request import urlopen
from contextlib import asynccontextmanager
from typing import Any, Optional

# make models/gemini_flash_rag importable from fastapi/
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "models", "gemini_flash_rag"),
)
from recommend import (
    recommend,
    fetch_dish_embedding,
    update_preference_vector,
    summarize_preferences,
)  # noqa: E402
from user_prefs import load_user_pref, save_user_pref  # noqa: E402

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Optional: only import mlflow when you wire the real registry ---
try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None  # type: ignore

MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/LunchBuddyModel/Production")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)  # e.g. https://your-mlflow-server
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")

if SUPABASE_URL:
    JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    JWT_ISSUER = f"{SUPABASE_URL}/auth/v1"
else:
    JWKS_URL = None
    JWT_ISSUER = None

# Global loaded model (set in lifespan)
_model: Any = None
_load_error: Optional[str] = None
_jwks_cache: dict[str, Any] = {"keys": None, "expires_at": 0}
_jwks_ttl_seconds = 3600
_bearer = HTTPBearer(auto_error=False)
_jwt_algorithms = ["ES256", "RS256"]


class AuthenticatedUser(BaseModel):
    user_id: str
    email: Optional[str] = None


def load_model_from_registry() -> Any:
    if MLFLOW_TRACKING_URI and mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow is None:
        raise RuntimeError("mlflow not installed")
    return mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _load_error
    _load_error = None
    try:
        if os.getenv("USE_STUB_MODEL", "").lower() in ("1", "true", "yes"):
            _model = None  # health can still report "degraded" or you use a tiny stub
            _load_error = "stub mode: no MLflow model"
        else:
            _model = load_model_from_registry()
    except Exception as e:
        _model = None
        _load_error = str(e)
    yield
    _model = None


app = FastAPI(title="Lunch Buddy API", lifespan=lifespan)

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic: change fields to match what your model expects/returns ---


class PredictRequest(BaseModel):
    mood: Optional[str] = None  # free-text mood for today, e.g. "something spicy"
    date: Optional[str] = None  # "YYYY-MM-DD"; defaults to today server-side if omitted


class DishCard(BaseModel):
    dish_name: str  # e.g. "Gochujang Spiced Chicken"
    dining_hall: str  # e.g. "Arrillaga"
    reason: str  # one-sentence explanation from Gemini


class PredictResponse(BaseModel):
    recommendations: list[DishCard]  # top 3 matches
    alternatives: list[DishCard]  # 2 diverse alternatives
    preference_summary: str  # user's current taste profile sentence


def _fetch_jwks() -> dict[str, Any]:
    now = int(time.time())
    cached_keys = _jwks_cache.get("keys")
    expires_at = int(_jwks_cache.get("expires_at", 0))
    if cached_keys and now < expires_at:
        return {"keys": cached_keys}

    if not JWKS_URL:
        raise HTTPException(
            status_code=500, detail="SUPABASE_URL is not configured on the API"
        )

    try:
        with urlopen(JWKS_URL, timeout=5) as resp:
            payload = resp.read().decode("utf-8")
    except URLError as exc:
        raise HTTPException(
            status_code=503, detail="Unable to fetch Supabase JWKS"
        ) from exc

    import json

    jwks = json.loads(payload)
    keys = jwks.get("keys")
    if not isinstance(keys, list) or not keys:
        raise HTTPException(
            status_code=503, detail="Supabase JWKS response was missing keys"
        )

    _jwks_cache["keys"] = keys
    _jwks_cache["expires_at"] = now + _jwks_ttl_seconds
    return {"keys": keys}


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = credentials.credentials
    jwks = _fetch_jwks()

    try:
        claims = jwt.decode(
            token,
            jwks,
            algorithms=_jwt_algorithms,
            audience=SUPABASE_JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"verify_aud": bool(SUPABASE_JWT_AUDIENCE)},
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=401, detail="Token missing user id")

    email = claims.get("email")
    return AuthenticatedUser(
        user_id=user_id, email=email if isinstance(email, str) else None
    )


@app.get("/")
def root():
    return {"message": "Welcome to Lunch Buddy API"}


@app.get("/health")
def health():
    # Rubric: "confirms the model is loaded successfully"
    if _model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "detail": _load_error or "model not loaded",
        }
    return {"status": "healthy", "model_loaded": True, "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> PredictResponse:
    # 1. load user profile from Supabase
    profile = load_user_pref(current_user.user_id)
    if profile is None:
        raise HTTPException(
            status_code=422,
            detail="No preference profile found. Please complete your taste profile to get recommendations.",
        )

    # 2. default date to today if client didn't send one
    date_str = body.date or date.today().isoformat()

    # 3. call the real recommendation pipeline
    recs, alts, _query_vec = recommend(
        pref_vec=profile["preference_vector"],
        preference_summary=profile["preference_summary"],
        user_allergens=profile["allergens"],
        date_str=date_str,
        daily_mood=body.mood,
        table="daily_menu",
        original_profile_vec=profile["original_profile_vector"],
        recent_choices_vecs=profile["recent_choices_vecs"],
    )

    # 4. return structured response -- no EMA update here (deferred to /pick)
    return PredictResponse(
        recommendations=[DishCard(**d) for d in recs],
        alternatives=[DishCard(**d) for d in alts],
        preference_summary=profile["preference_summary"],
    )


# ---------------------------------------------------------------------------
# POST /pick -- EMA update after user picks a dish
# ---------------------------------------------------------------------------


class PickRequest(BaseModel):
    dish_name: str  # from the DishCard the user tapped
    dining_hall: str  # included for logging / future use
    date: Optional[str] = None  # defaults to today if omitted


class PickResponse(BaseModel):
    status: str  # "ok" or "error"
    message: str  # human-readable confirmation


@app.post("/pick", response_model=PickResponse)
def pick(
    body: PickRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> PickResponse:
    # resolve the date -- default to today if the frontend didn't send one
    date_str = body.date or date.today().isoformat()

    # load the user's current preference profile
    profile = load_user_pref(current_user.user_id)
    if profile is None:
        raise HTTPException(
            status_code=422,
            detail="No preference profile found -- complete your taste profile first",
        )

    # look up the picked dish's embedding vector
    dish_vec = fetch_dish_embedding(body.dish_name, date_str)
    if dish_vec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dish '{body.dish_name}' not found in menu for {date_str}",
        )

    # EMA update: nudge the preference vector toward the picked dish
    new_vec = update_preference_vector(profile["preference_vector"], dish_vec)

    # prepend new pick to recent_choices and trim to 5 (most-recent first)
    recent = [dish_vec.tolist()] + (profile.get("recent_choices_raw") or [])
    recent = recent[:5]

    # persist updated vector and recent choices immediately (summary unchanged for now)
    save_user_pref(
        current_user.user_id,
        new_vec,
        profile["preference_summary"],
        profile["allergens"],
        recent_choices=recent,
    )

    # regenerate summary in a background thread so the user isn't blocked
    # delay 5s to avoid Gemini 429 rate limit from the /predict call
    import threading

    def _update_summary():
        time.sleep(5)
        new_summary = summarize_preferences(
            profile["preference_summary"],
            [body.dish_name],
        )
        save_user_pref(
            current_user.user_id,
            new_vec,
            new_summary,
            profile["allergens"],
        )

    threading.Thread(target=_update_summary, daemon=True).start()

    return PickResponse(
        status="ok",
        message=f"Enjoy your {body.dish_name}!",
    )
