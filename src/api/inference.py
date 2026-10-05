"""
Raw customer dict -> churn prediction, reusing Phase 1 cleaning + Phase 2
encoder/scaler. Loads native flavor model directly for predict_proba
(pyfunc .predict() returns hard labels only, per MLflow 3.x behavior).
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import mlflow
import config
from src.data.clean import standardize_senior_citizen, fix_total_charges
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

_encoder = joblib.load(config.MODEL_DIR / "onehot_encoder.joblib")
_scaler = joblib.load(config.MODEL_DIR / "standard_scaler.joblib")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
_model_cache = {}

def _load_native_model(alias: str, model_name: str = "churn_predictor"):
    """Load native flavor (not pyfunc) to access predict_proba.
    Tries sklearn flavor first, falls back to xgboost flavor."""
    uri = f"models:/{model_name}@{alias}"
    try:
        return mlflow.sklearn.load_model(uri), "sklearn"
    except Exception:
        return mlflow.xgboost.load_model(uri), "xgboost"

def _get_version_and_run(alias: str, model_name: str = "churn_predictor"):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    return mv.version, mv.run_id

def get_model(alias: str = "champion"):
    if alias not in _model_cache:
        model, flavor = _load_native_model(alias)
        version, run_id = _get_version_and_run(alias)
        _model_cache[alias] = {
            "model": model, "flavor": flavor,
            "version": version, "run_id": run_id,
        }
    return _model_cache[alias]

def preprocess(raw_record: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_record])
    df = standardize_senior_citizen(df)
    df = fix_total_charges(df)

    cat_cols = _encoder.feature_names_in_.tolist()
    encoded = _encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=_encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )

    scaled = _scaler.transform(df[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled, columns=NUMERIC_COLS, index=df.index)

    final_df = pd.concat(
        [scaled_df, encoded_df.drop(columns=NUMERIC_COLS, errors="ignore")],
        axis=1,
    )
    return final_df

def predict(raw_record: dict, alias: str = "champion") -> dict:
    entry = get_model(alias)
    model = entry["model"]
    X = preprocess(raw_record)
    proba = float(model.predict_proba(X)[0][1])  # P(churn=1)
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "churn_probability": proba,
        "model_alias": alias,
        "model_version": entry["version"],
        "run_id": entry["run_id"],
    }
