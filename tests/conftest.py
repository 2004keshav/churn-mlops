"""
Shared pytest fixtures for Phase 8 unit tests.
No external systems (Postgres, MLflow) are touched by any fixture here.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Sample valid CustomerRecord payload (matches src/api/schema.py exactly)
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_customer_payload() -> dict:
    """A valid raw customer record, all 19 fields, matching CustomerRecord schema."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


# ---------------------------------------------------------------------------
# Fake inference.predict() return value (shape matches src/api/inference.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_prediction_result() -> dict:
    return {
        "prediction": 1,
        "churn_probability": 0.8047,
        "model_alias": "champion",
        "model_version": 3,
        "run_id": "6c90c1eb1e2b48f7b62f8db8ca0966b8",
    }


# ---------------------------------------------------------------------------
# Fake MLflow native model object with a working predict_proba()
# ---------------------------------------------------------------------------
class FakeModel:
    """Mimics a loaded sklearn/xgboost model's predict_proba interface."""
    def __init__(self, proba_churn: float = 0.8047):
        self.proba_churn = proba_churn

    def predict_proba(self, X):
        n = len(X)
        return np.array([[1 - self.proba_churn, self.proba_churn]] * n)


@pytest.fixture
def fake_model():
    return FakeModel()


# ---------------------------------------------------------------------------
# Minimal reference_stats.json-shaped dict for drift.py tests
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_reference_stats() -> dict:
    rng = np.random.default_rng(42)
    tenure_raw = rng.normal(loc=32, scale=24, size=500).clip(0, 72).tolist()
    bin_edges = np.linspace(0, 72, 11).tolist()
    bin_counts, _ = np.histogram(tenure_raw, bins=bin_edges)

    return {
        "numeric": {
            "tenure": {
                "mean": float(np.mean(tenure_raw)),
                "std": float(np.std(tenure_raw)),
                "min": float(np.min(tenure_raw)),
                "max": float(np.max(tenure_raw)),
                "raw_values": tenure_raw,
                "bin_edges": bin_edges,
                "bin_counts": bin_counts.tolist(),
            },
            "MonthlyCharges": {
                "mean": 64.76, "std": 30.09, "min": 18.25, "max": 118.75,
                "raw_values": rng.normal(64.76, 30.09, 500).clip(18.25, 118.75).tolist(),
                "bin_edges": np.linspace(18.25, 118.75, 11).tolist(),
                "bin_counts": [50] * 10,
            },
            "TotalCharges": {
                "mean": 2283.3, "std": 2266.77, "min": 0.0, "max": 8684.8,
                "raw_values": rng.normal(2283.3, 2266.77, 500).clip(0, 8684.8).tolist(),
                "bin_edges": np.linspace(0, 8684.8, 11).tolist(),
                "bin_counts": [50] * 10,
            },
        },
        "categorical": {
            "gender": {"Female": 0.5, "Male": 0.5},
            "SeniorCitizen": {"No": 0.84, "Yes": 0.16},
            "Partner": {"Yes": 0.48, "No": 0.52},
            "Dependents": {"Yes": 0.30, "No": 0.70},
            "PhoneService": {"Yes": 0.90, "No": 0.10},
            "MultipleLines": {"No": 0.48, "Yes": 0.42, "No phone service": 0.10},
            "InternetService": {"Fiber optic": 0.44, "DSL": 0.34, "No": 0.22},
            "OnlineSecurity": {"No": 0.50, "Yes": 0.29, "No internet service": 0.21},
            "OnlineBackup": {"No": 0.44, "Yes": 0.35, "No internet service": 0.21},
            "DeviceProtection": {"No": 0.44, "Yes": 0.35, "No internet service": 0.21},
            "TechSupport": {"No": 0.49, "Yes": 0.29, "No internet service": 0.21},
            "StreamingTV": {"No": 0.40, "Yes": 0.38, "No internet service": 0.21},
            "StreamingMovies": {"No": 0.40, "Yes": 0.39, "No internet service": 0.21},
            "Contract": {"Month-to-month": 0.55, "Two year": 0.24, "One year": 0.21},
            "PaperlessBilling": {"Yes": 0.59, "No": 0.41},
            "PaymentMethod": {
                "Electronic check": 0.34, "Mailed check": 0.23,
                "Bank transfer (automatic)": 0.22, "Credit card (automatic)": 0.22,
            },
        },
    }


# ---------------------------------------------------------------------------
# FastAPI TestClient — plain, NOT a context manager (confirmed in Step 2:
# this skips the startup event, so no real MLflow load is attempted).
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)
