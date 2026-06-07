"""
app/main.py
-----------
FastAPI REST API for the Energy Demand Forecasting project.

Endpoints:
    GET  /            — health check
    GET  /models      — list available models
    POST /predict     — generate forecast
    POST /retrain     — retrain a model on new data (async)
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="Energy Demand Forecasting API",
    description=(
        "Hourly electricity demand forecasting for Germany "
        "using SARIMA, Prophet, XGBoost, LSTM, and Ensemble models."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Model registry ───────────────────────────────────────────────────────────
MODELS: dict = {}

def load_models():
    """Load all fitted models from disk at startup."""
    for name in ["sarima", "prophet", "xgboost", "ensemble"]:
        path = MODEL_DIR / f"{name}.joblib"
        if path.exists():
            MODELS[name] = joblib.load(path)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found on disk: {name} (run notebooks first)")


@app.on_event("startup")
def startup_event():
    load_models()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model: Literal["sarima", "prophet", "xgboost", "lstm", "ensemble"] = Field(
        default="ensemble",
        description="Which forecasting model to use.",
    )
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of hours to forecast (1–168).",
    )
    start_datetime: Optional[str] = Field(
        default=None,
        description="ISO 8601 start datetime for the forecast window (default: now).",
    )

    @validator("start_datetime", pre=True, always=True)
    def parse_datetime(cls, v):
        if v is None:
            return datetime.utcnow().isoformat()
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("start_datetime must be ISO 8601 format")
        return v


class PredictResponse(BaseModel):
    model:         str
    horizon_hours: int
    start_datetime: str
    timestamps:    list[str]
    forecast_mw:   list[float]
    unit:          str = "MW"


class ModelInfo(BaseModel):
    name:      str
    available: bool
    type:      str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Energy Demand Forecasting API is running.",
        "docs": "/docs",
    }


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
def list_models():
    model_types = {
        "sarima":   "Statistical (SARIMA)",
        "prophet":  "Statistical (Prophet)",
        "xgboost":  "Machine Learning (XGBoost)",
        "lstm":     "Deep Learning (LSTM)",
        "ensemble": "Ensemble (Prophet + XGBoost)",
    }
    return [
        ModelInfo(name=name, available=name in MODELS, type=mtype)
        for name, mtype in model_types.items()
    ]


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
def predict(req: PredictRequest):
    if req.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' is not loaded. "
                   f"Available: {list(MODELS.keys())}",
        )

    model = MODELS[req.model]

    try:
        raw_preds = model.predict(horizon=req.horizon_hours)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Clip negative forecasts
    raw_preds = np.clip(raw_preds, 0, None)

    start = datetime.fromisoformat(req.start_datetime)
    timestamps = [
        (start + timedelta(hours=i)).isoformat()
        for i in range(req.horizon_hours)
    ]

    return PredictResponse(
        model=req.model,
        horizon_hours=req.horizon_hours,
        start_datetime=req.start_datetime,
        timestamps=timestamps,
        forecast_mw=[round(float(v), 1) for v in raw_preds],
    )


@app.post("/retrain", tags=["Models"])
async def retrain(
    model_name: Literal["xgboost", "prophet", "ensemble"],
    background_tasks: BackgroundTasks,
):
    """Trigger a background retrain (placeholder for real pipeline)."""
    def _retrain_job(name: str):
        logger.info(f"Background retrain started for {name}")
        # In production: pull fresh data, retrain, save, reload MODELS
        logger.info(f"Background retrain completed for {name}")

    background_tasks.add_task(_retrain_job, model_name)
    return {"status": "queued", "model": model_name}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
