"""FastAPI serving endpoint for DNS threat classification.

Accepts raw domain strings and returns predictions (benign / DGA /
exfiltration) with confidence scores.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.training.models import LABEL_MAP
from src.training.train import build_features

logger = logging.getLogger(__name__)


# ── Request / Response schemas ──────────────────────────────────────


class PredictRequest(BaseModel):
    """Request body for /predict."""

    domains: list[str] = Field(..., min_length=1, description="DNS domain strings to classify")


class PredictionResult(BaseModel):
    """Single prediction result."""

    domain: str
    label: int
    label_name: str
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    """Response body for /predict."""

    predictions: list[PredictionResult]
    inference_time_ms: float
    model_type: str


# ── App state ───────────────────────────────────────────────────────


_state: dict = {
    "model": None,
    "tokenizer": None,
    "config": {},
    "start_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and vectorizer at startup."""
    _state["start_time"] = time.time()

    model_path = Path(_state["config"].get("model_path", "models/registry/model.pkl"))
    vec_path = Path(_state["config"].get("vectorizer_path", "models/registry/vectorizer.pkl"))

    if model_path.exists() and vec_path.exists():
        _state["model"] = joblib.load(model_path)
        _state["tokenizer"] = joblib.load(vec_path)
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("Model files not found — serving in degraded mode")

    yield
    _state.clear()


app = FastAPI(
    title="SeqThreat — DNS Threat Classifier",
    description="Classify DNS queries as benign, DGA, or exfiltration",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Service health check."""
    uptime = time.time() - _state["start_time"] if _state["start_time"] else 0
    model_loaded = _state["model"] is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": type(_state["model"]).__name__ if model_loaded else None,
        "uptime_sec": round(uptime, 1),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Classify DNS domains."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = _state["model"]
    tokenizer = _state["tokenizer"]

    t0 = time.time()

    # Build feature matrix
    X = build_features(request.domains, tokenizer, _state.get("config", {}), fit=False)

    # Predict
    labels = model.predict(X)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

    elapsed_ms = (time.time() - t0) * 1000

    results = []
    for i, domain in enumerate(request.domains):
        prob_dict = None
        if probabilities is not None:
            prob_dict = {
                LABEL_MAP.get(j, str(j)): round(float(p), 4)
                for j, p in enumerate(probabilities[i])
            }
        results.append(
            PredictionResult(
                domain=domain,
                label=int(labels[i]),
                label_name=LABEL_MAP.get(int(labels[i]), "unknown"),
                probabilities=prob_dict,
            )
        )

    return PredictResponse(
        predictions=results,
        inference_time_ms=round(elapsed_ms, 2),
        model_type=type(model).__name__,
    )


@app.get("/model/info")
def model_info():
    """Return model metadata."""
    if _state["model"] is None:
        return {"loaded": False}

    model = _state["model"]
    tokenizer = _state["tokenizer"]

    info = {
        "loaded": True,
        "model_type": type(model).__name__,
        "n_features": tokenizer.n_features if tokenizer else None,
        "ngram_range": tokenizer.ngram_range if tokenizer else None,
        "classes": [LABEL_MAP.get(i, str(i)) for i in range(3)],
    }

    if hasattr(model, "get_params"):
        info["params"] = {
            k: v for k, v in model.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }

    return info
