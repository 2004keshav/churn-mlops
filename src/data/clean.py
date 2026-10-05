import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        logger.info("Filling %d missing TotalCharges values with 0", n_missing)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    return df


def standardize_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Dropped %d exact duplicate rows", removed)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_exact_duplicates(df)
    df = fix_total_charges(df)
    df = standardize_senior_citizen(df)
    df = encode_target(df)
    return df
