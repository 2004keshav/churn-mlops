import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Literal

from src.api.schema import CustomerRecord, PredictionResponse
from src.api import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.on_event("startup")
def load_models_on_startup():
    # Fail-fast: if models can't load, the container should not start
    # serving. Render will show a failed deploy rather than a silently
    # broken /health endpoint.
    try:
        inference.get_model("champion")
        inference.get_model("challenger")
        logger.info("Both models loaded successfully at startup.")
    except Exception as e:
        logger.error(f"FATAL: model loading failed at startup: {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerRecord,
    model: Literal["champion", "challenger"] = Query("champion"),
):
    try:
        result = inference.predict(customer.model_dump(), alias=model)
        return result
    except Exception as e:
        # Log full detail server-side only; never leak internals to the client
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error. Please contact support if this persists.",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Catches anything not already handled above (e.g. errors outside /predict)
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )
