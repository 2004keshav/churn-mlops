"""
Sanity-check script for Phase 1 (Data Cleaning) and Phase 2 (Feature Engineering).
Verifies outputs exist, have correct shape/dtype, and contain no unexpected issues.
Run: python -m scripts.verify_pipeline
"""

import os
import sys
import pandas as pd
import joblib

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TARGET_COL,
)

checks_passed = 0
checks_failed = 0


def check(description: str, condition: bool, details: str = ""):
    global checks_passed, checks_failed
    status = "PASS" if condition else "FAIL"
    symbol = "✅" if condition else "❌"
    print(f"{symbol} [{status}] {description}" + (f" — {details}" if details and not condition else ""))
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1


def verify_phase1():
    print("\n" + "=" * 60)
    print("PHASE 1 — Data Ingestion + Cleaning")
    print("=" * 60)

    check("Raw data file exists", os.path.exists(RAW_DATA_PATH))
    check("Processed data file exists", os.path.exists(PROCESSED_DATA_PATH))

    if not os.path.exists(PROCESSED_DATA_PATH):
        print("Skipping remaining Phase 1 checks — processed file missing.")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)

    check("Processed data has 7043 rows", df.shape[0] == 7043, f"got {df.shape[0]}")
    check("Processed data has 21 columns", df.shape[1] == 21, f"got {df.shape[1]}")
    check("TotalCharges is numeric (float64)", df["TotalCharges"].dtype == "float64",
          f"got {df['TotalCharges'].dtype}")
    check("Churn is encoded as int (0/1)", df[TARGET_COL].dtype in ["int64", "int32"],
          f"got {df[TARGET_COL].dtype}")
    check("Churn values are only 0 and 1", set(df[TARGET_COL].unique()) == {0, 1},
          f"got {set(df[TARGET_COL].unique())}")
    check("No missing values in TotalCharges", df["TotalCharges"].isna().sum() == 0,
          f"{df['TotalCharges'].isna().sum()} missing")
    check("No duplicate customerIDs", df["customerID"].duplicated().sum() == 0,
          f"{df['customerID'].duplicated().sum()} duplicates")

    churn_rate = df[TARGET_COL].mean()
    check("Churn rate is reasonable (20-35%)", 0.20 <= churn_rate <= 0.35,
          f"got {churn_rate:.4f}")


def verify_phase2():
    print("\n" + "=" * 60)
    print("PHASE 2 — Feature Engineering")
    print("=" * 60)

    files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    paths = {f: f"{PROCESSED_DATA_DIR}/{f}" for f in files}

    for f, p in paths.items():
        check(f"{f} exists", os.path.exists(p))

    if not all(os.path.exists(p) for p in paths.values()):
        print("Skipping remaining Phase 2 checks — one or more files missing.")
        return

    X_train = pd.read_csv(paths["X_train.csv"])
    X_test = pd.read_csv(paths["X_test.csv"])
    y_train = pd.read_csv(paths["y_train.csv"])
    y_test = pd.read_csv(paths["y_test.csv"])

    check("X_train and X_test have same columns",
          list(X_train.columns) == list(X_test.columns))
    check("X_train row count matches y_train", len(X_train) == len(y_train),
          f"{len(X_train)} vs {len(y_train)}")
    check("X_test row count matches y_test", len(X_test) == len(y_test),
          f"{len(X_test)} vs {len(y_test)}")
    check("No NaN values in X_train", X_train.isna().sum().sum() == 0,
          f"{X_train.isna().sum().sum()} NaNs")
    check("No NaN values in X_test", X_test.isna().sum().sum() == 0,
          f"{X_test.isna().sum().sum()} NaNs")
    check("customerID not in X_train columns", "customerID" not in X_train.columns)
    check("Churn not in X_train columns", TARGET_COL not in X_train.columns)
    check("All X_train columns are numeric",
          all(X_train.dtypes.apply(lambda d: d.kind in "if")))

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    train_means = X_train[numeric_cols].mean()
    check("Scaled numeric columns have mean ~0 in train",
          all(abs(train_means) < 0.01), f"means: {train_means.to_dict()}")

    encoder_path = f"{MODEL_DIR}/onehot_encoder.joblib"
    scaler_path = f"{MODEL_DIR}/standard_scaler.joblib"
    check("Encoder file exists", os.path.exists(encoder_path))
    check("Scaler file exists", os.path.exists(scaler_path))

    if os.path.exists(encoder_path):
        try:
            joblib.load(encoder_path)
            check("Encoder file loads without error", True)
        except Exception as e:
            check("Encoder file loads without error", False, str(e))

    if os.path.exists(scaler_path):
        try:
            joblib.load(scaler_path)
            check("Scaler file loads without error", True)
        except Exception as e:
            check("Scaler file loads without error", False, str(e))


def main():
    verify_phase1()
    verify_phase2()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {checks_passed} passed, {checks_failed} failed")
    print("=" * 60)

    if checks_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
