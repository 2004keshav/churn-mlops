"""
Train/Test split module.
Splits cleaned data into train and test sets BEFORE any encoding/scaling,
to prevent data leakage. Uses stratified split to preserve churn ratio.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits dataframe into train/test sets.

    - Drops id_col (no predictive value).
    - Stratifies on target_col to preserve class ratio in both splits.

    Returns:
        X_train, X_test, y_train, y_test (all pandas objects)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    df = df.drop(columns=[id_col], errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # preserves churn ratio in both sets
    )

    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.4f} | Test churn rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test
