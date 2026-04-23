# main.py — skeleton; adapt schemas and inference to your LLM wrapper

import os
import time
from urllib.error import URLError
from urllib.request import urlopen
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Optional: only import mlflow when you wire the real registry ---
try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None  # type: ignore

MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/LunchBuddyModel/Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # e.g. https://your-mlflow-server
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
    user_id: Optional[str] = None
    preferences: list[str] = Field(..., description="Free text or structured prefs")
    constraints: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    suggestions: list[str]
    rationale: Optional[str] = None


def _fetch_jwks() -> dict[str, Any]:
    now = int(time.time())
    cached_keys = _jwks_cache.get("keys")
    expires_at = int(_jwks_cache.get("expires_at", 0))
    if cached_keys and now < expires_at:
        return {"keys": cached_keys}

    if not JWKS_URL:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured on the API")

    try:
        with urlopen(JWKS_URL, timeout=5) as resp:
            payload = resp.read().decode("utf-8")
    except URLError as exc:
        raise HTTPException(status_code=503, detail="Unable to fetch Supabase JWKS") from exc

    import json

    jwks = json.loads(payload)
    keys = jwks.get("keys")
    if not isinstance(keys, list) or not keys:
        raise HTTPException(status_code=503, detail="Supabase JWKS response was missing keys")

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
    return AuthenticatedUser(user_id=user_id, email=email if isinstance(email, str) else None)


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
    return {"status": "ok", "model_loaded": True, "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model not loaded; check /health and MLflow config",
        )
    # Build a dict or DataFrame in the shape your pyfunc model was trained with
    payload = {
        "preferences": body.preferences,
        "constraints": body.constraints,
        # Never trust client-supplied user_id for auth; use token subject.
        "user_id": current_user.user_id,
    }
    raw = _model.predict(payload)  # or pd.DataFrame([payload]) for tabular pyfunc
    # Normalize MLflow output into PredictResponse (depends on your model)
    if isinstance(raw, list) and raw:
        first = raw[0]
    else:
        first = raw
    # Example mapping — replace with your real structure
    if isinstance(first, dict):
        return PredictResponse(
            suggestions=first.get("suggestions", []),
            rationale=first.get("rationale"),
        )
    return PredictResponse(suggestions=[str(first)])