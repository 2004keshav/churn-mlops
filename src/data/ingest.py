import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_raw_data(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d columns", df.shape[0], df.shape[1])

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns: {missing_cols}")

    if df.empty:
        raise ValueError("Raw data file is empty.")

    return df
