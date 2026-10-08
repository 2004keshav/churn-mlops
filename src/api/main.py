import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import time
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Literal

import config
from src.api.schema import CustomerRecord, PredictionResponse
from src.api import inference
from src.data.clean import standardize_senior_citizen
from src.monitoring import db, drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        inference.get_model("champion")
        inference.get_model("challenger")
        logger.info("Both models loaded successfully at startup.")
    except Exception as e:
        logger.error(f"FATAL: model loading failed at startup: {e}")
        raise
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

REFERENCE_STATS_PATH = str(config.MODEL_DIR / "reference_stats.json")


@app.get("/health")
def health():
    return {"status": "ok"}


def _normalize_for_logging(raw_dict: dict) -> dict:
    """Convert SeniorCitizen 0/1 -> 'No'/'Yes' for consistency with
    reference_stats.json's categorical distributions (which were built
    from the already-standardized churn_processed.csv)."""
    df = pd.DataFrame([raw_dict])
    df = standardize_senior_citizen(df)
    return df.iloc[0].to_dict()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerRecord,
    model: Literal["champion", "challenger"] = Query("champion"),
):
    start_time = time.perf_counter()
    try:
        raw_dict = customer.model_dump()
        result = inference.predict(raw_dict, alias=model)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log to Postgres. Never let logging failure break the response —
        # db.insert_prediction_log() is designed to catch its own errors
        # and return None rather than raise.
        try:
            normalized_input = _normalize_for_logging(raw_dict)
            db.insert_prediction_log(
                raw_input=normalized_input,
                prediction=result["prediction"],
                churn_probability=result["churn_probability"],
                model_alias=result["model_alias"],
                model_version=str(result["model_version"]),
                run_id=result["run_id"],
                latency_ms=latency_ms,
            )
        except Exception as log_err:
            logger.error(f"Prediction logging failed (non-fatal): {log_err}")

        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error. Please contact support if this persists.",
        )


@app.get("/monitoring/stats")
def monitoring_stats():
    """Quick sanity check: how many predictions have been logged so far."""
    total = db.count_logs()
    return {"total_logged_predictions": total}


@app.get("/monitoring/drift")
def monitoring_drift(n: int = Query(500, ge=30, le=5000)):
    """
    Compare the most recent N logged predictions against the training
    reference distribution using KS test (numeric) + PSI (all features).
    Returns 'insufficient_data' status if fewer than 30 logs exist.
    """
    try:
        live_df = db.fetch_recent_logs(n=n)
        report = drift.run_drift_report(live_df, REFERENCE_STATS_PATH)
        return report
    except Exception as e:
        logger.error(f"Drift check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Drift analysis failed due to an internal error.",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )
