"""
Numeric scaling module.
Fits a StandardScaler on TRAIN data only, then transforms both train and test
to prevent data leakage. Scaler is saved for reuse during inference.
"""

import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fit_scaler(X_train: pd.DataFrame, numeric_cols: list) -> StandardScaler:
    """Fits a StandardScaler on the training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])
    logger.info(f"Scaler fitted on {len(numeric_cols)} numeric columns")
    return scaler


def transform_numeric(
    df: pd.DataFrame, scaler: StandardScaler, numeric_cols: list
) -> pd.DataFrame:
    """
    Applies a FITTED scaler to a dataframe (train or test).
    Returns a copy of df with numeric_cols replaced by scaled values.
    """
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    logger.info(f"Scaled columns: {numeric_cols}")
    return df


def save_scaler(scaler: StandardScaler, path: str):
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)
