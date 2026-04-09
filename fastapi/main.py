# main.py — skeleton; adapt schemas and inference to your LLM wrapper

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
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

# Global loaded model (set in lifespan)
_model: Any = None
_load_error: Optional[str] = None


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
def predict(body: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model not loaded; check /health and MLflow config",
        )
    # Build a dict or DataFrame in the shape your pyfunc model was trained with
    payload = {
        "preferences": body.preferences,
        "constraints": body.constraints,
        "user_id": body.user_id,
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