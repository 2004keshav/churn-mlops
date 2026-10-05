"""
Categorical encoding module.
Fits a OneHotEncoder on TRAIN data only, then transforms both train and test
to prevent data leakage. Encoder is saved for reuse during inference.
"""

import logging
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Returns list of object/category dtype columns."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def fit_encoder(X_train: pd.DataFrame, categorical_cols: list) -> OneHotEncoder:
    """
    Fits a OneHotEncoder on the training data only.
    - drop='if_binary': avoids redundant column for 2-category features.
    - handle_unknown='ignore': prevents crash if test/inference data has
      an unseen category (encodes it as all zeros instead of erroring).
    """
    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False,
    )
    encoder.fit(X_train[categorical_cols])
    logger.info(f"Encoder fitted on {len(categorical_cols)} categorical columns")
    return encoder


def transform_categorical(
    df: pd.DataFrame, encoder: OneHotEncoder, categorical_cols: list
) -> pd.DataFrame:
    """
    Applies a FITTED encoder to a dataframe (train or test).
    Returns a dataframe with categorical columns replaced by encoded columns,
    numeric columns left untouched.
    """
    encoded_array = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    numeric_df = df.drop(columns=categorical_cols)
    result = pd.concat([numeric_df, encoded_df], axis=1)

    logger.info(f"Transformed shape: {result.shape}")
    return result


def save_encoder(encoder: OneHotEncoder, path: str):
    joblib.dump(encoder, path)
    logger.info(f"Encoder saved to {path}")


def load_encoder(path: str) -> OneHotEncoder:
    return joblib.load(path)
