"""
Feature engineering orchestrator.
Runs: load processed data -> split -> encode categoricals -> scale numerics
-> save final train/test sets + encoder/scaler artifacts.

Entry point: python -m src.features.build_features
"""

import logging
import pandas as pd
import os

from config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TARGET_COL,
    ID_COL,
    RANDOM_STATE,
)
from src.features.split_data import split_data
from src.features.encode import (
    get_categorical_columns,
    fit_encoder,
    transform_categorical,
    save_encoder,
)
from src.features.scale import fit_scaler, transform_numeric, save_scaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_features():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Load cleaned data from Phase 1
    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Step 2: Split (before any fitting, to prevent leakage)
    logger.info("Splitting into train/test...")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=TARGET_COL, id_col=ID_COL, random_state=RANDOM_STATE
    )

    # Step 3: Encode categoricals (fit on train only)
    logger.info("Encoding categorical features...")
    categorical_cols = get_categorical_columns(X_train)
    encoder = fit_encoder(X_train, categorical_cols)
    X_train_encoded = transform_categorical(X_train, encoder, categorical_cols)
    X_test_encoded = transform_categorical(X_test, encoder, categorical_cols)
    save_encoder(encoder, f"{MODEL_DIR}/onehot_encoder.joblib")

    # Step 4: Scale numerics (fit on train only)
    logger.info("Scaling numeric features...")
    scaler = fit_scaler(X_train_encoded, NUMERIC_COLS)
    X_train_final = transform_numeric(X_train_encoded, scaler, NUMERIC_COLS)
    X_test_final = transform_numeric(X_test_encoded, scaler, NUMERIC_COLS)
    save_scaler(scaler, f"{MODEL_DIR}/standard_scaler.joblib")

    # Step 5: Save final feature sets to disk (for Phase 3 to consume directly)
    X_train_final.to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    X_test_final.to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)

    logger.info(f"Final X_train shape: {X_train_final.shape}")
    logger.info(f"Final X_test shape: {X_test_final.shape}")
    logger.info("Feature engineering complete. Files saved to data/processed/")

    return X_train_final, X_test_final, y_train, y_test


if __name__ == "__main__":
    build_features()
