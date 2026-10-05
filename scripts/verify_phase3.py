"""
Verification script for Phase 3 — Model Training + MLflow Tracking.
Run standalone: python scripts/verify_phase3.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR

checks_passed = 0
checks_failed = 0
failures = []


def check(description, condition):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
        print(f"  PASS: {description}")
    else:
        checks_failed += 1
        failures.append(description)
        print(f"  FAIL: {description}")


# ------------------------------------------------------------
print("\n[1] File existence checks")
train_py = ROOT_DIR / "src" / "models" / "train.py"
requirements_txt = ROOT_DIR / "requirements.txt"
mlflow_db = ROOT_DIR / "mlflow.db"
mlruns_dir = ROOT_DIR / "mlruns"

check("src/models/train.py exists", train_py.exists())
check("requirements.txt exists", requirements_txt.exists())
check("mlflow.db exists", mlflow_db.exists())
check("mlruns/ directory exists", mlruns_dir.exists())

# ------------------------------------------------------------
print("\n[2] train.py code structure checks")
train_code = train_py.read_text() if train_py.exists() else ""

required_functions = [
    "def train_logistic_regression",
    "def train_random_forest",
    "def train_xgboost",
    "def evaluate_model",
    "def train_and_log_model",
    "def main",
]
for fn in required_functions:
    check(f"train.py contains '{fn}'", fn in train_code)

check("CLI entry point guard present", 'if __name__ == "__main__"' in train_code)
check("class_weight='balanced' used (imbalance handling)", "class_weight" in train_code)
check("scale_pos_weight used for XGBoost (imbalance handling)", "scale_pos_weight" in train_code)
# FIXED: file now correctly ends with the CLI guard calling main(), not the training function
check(
    "File not truncated (ends with CLI entry point)",
    train_code.strip().endswith('if __name__ == "__main__":\n    main()')
)

# ------------------------------------------------------------
print("\n[3] train.py syntax validity")
try:
    compile(train_code, str(train_py), "exec")
    check("train.py has valid Python syntax", True)
except SyntaxError as e:
    check(f"train.py has valid Python syntax (error: {e})", False)

# ------------------------------------------------------------
print("\n[4] MLflow run data checks")
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
experiment = mlflow.get_experiment_by_name("churn_prediction_baseline")

check("Experiment 'churn_prediction_baseline' exists", experiment is not None)

if experiment is not None:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    check("At least 1 run logged", len(runs) >= 1)

    model_types_found = set(r.data.params.get("model_type") for r in runs)
    expected_models = {"LogisticRegression", "RandomForest", "XGBoost"}
    check(
        f"All 3 model types present in runs (found: {model_types_found})",
        expected_models.issubset(model_types_found)
    )

    required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    if runs:
        sample_run = runs[0]
        for metric in required_metrics:
            check(f"Sample run has metric '{metric}' logged", metric in sample_run.data.metrics)

        for metric in required_metrics:
            val = sample_run.data.metrics.get(metric)
            if val is not None:
                check(f"Metric '{metric}' is in valid range [0,1] (value={val:.4f})", 0.0 <= val <= 1.0)

        # FIXED: MLflow 3.x logs models as "Logged Model" entities, not plain run artifacts.
        # Check via run.outputs.model_outputs (the correct 3.x API), with filesystem fallback.
        model_artifact_ok = False
        model_detail = ""
        try:
            model_outputs = getattr(sample_run.outputs, "model_outputs", None)
            if model_outputs:
                model_id = model_outputs[0].model_id
                logged_model = client.get_logged_model(model_id)
                model_artifact_ok = True
                model_detail = f"model_id={model_id}"
        except Exception as e:
            model_detail = f"API fallback triggered ({e})"

        if not model_artifact_ok:
            models_dir = ROOT_DIR / "mlruns" / "models"
            mlmodel_files = list(models_dir.glob("m-*/artifacts/MLmodel")) if models_dir.exists() else []
            model_artifact_ok = len(mlmodel_files) >= 3
            model_detail = f"{len(mlmodel_files)} MLmodel files found on disk (filesystem check)"

        check(f"Model artifacts logged as MLflow 3.x Logged Models ({model_detail})", model_artifact_ok)

    logreg_runs = [r for r in runs if r.data.params.get("model_type") == "LogisticRegression"]
    c_values_logged = set(r.data.params.get("C") for r in logreg_runs if r.data.params.get("C") is not None)
    check(
        f"Multiple distinct C values logged for LogisticRegression (found: {c_values_logged})",
        len(c_values_logged) >= 2
    )

    all_metrics_present = all(
        all(m in r.data.metrics for m in required_metrics) for r in runs
    )
    check("All logged runs have complete metric sets (no partial/failed runs)", all_metrics_present)

# ------------------------------------------------------------
print("\n[5] requirements.txt content checks")
req_content = requirements_txt.read_text() if requirements_txt.exists() else ""
required_packages = ["mlflow", "xgboost", "pandas", "numpy", "scikit-learn", "joblib"]
for pkg in required_packages:
    check(f"requirements.txt contains '{pkg}'", pkg in req_content)

# ------------------------------------------------------------
print("\n[6] Data leakage sanity check (Phase 2 carryover, re-verified)")
import pandas as pd
X_train_check = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
X_test_check = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
check("X_train and X_test have same number of columns", X_train_check.shape[1] == X_test_check.shape[1])
check("X_train and X_test have different row counts", X_train_check.shape[0] != X_test_check.shape[0])

# ------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PHASE 3 VERIFICATION SUMMARY: {checks_passed} passed, {checks_failed} failed")
print("=" * 60)

if checks_failed > 0:
    print("\nFailed checks:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nAll Phase 3 checks passed. Phase 3 is verified complete.")
    sys.exit(0)
