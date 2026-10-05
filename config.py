from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "churn_processed.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

MODEL_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"

TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

# MLflow tracking (SQLite backend — local file, no hosted server needed)
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = str(MLRUNS_DIR)
