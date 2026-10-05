"""
Model training functions for churn prediction.
Each train_* function returns a FITTED model (pure, no MLflow logging here).
evaluate_model() computes standard classification metrics.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import RANDOM_STATE


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    """Train Logistic Regression with class_weight='balanced' to handle imbalance."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train Random Forest with class_weight='balanced' to handle imbalance."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost. Uses scale_pos_weight (XGBoost's equivalent of class_weight)
    to handle imbalance: ratio of negative to positive class count.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Compute standard classification metrics for imbalanced churn problem.
    Returns a dict ready for MLflow logging.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of churn=1

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


# ============================================================
# MLflow logging wrapper
# ============================================================

MODEL_BUILDERS = {
    "LogisticRegression": train_logistic_regression,
    "RandomForest": train_random_forest,
    "XGBoost": train_xgboost,
}


def _log_sklearn_model(model, name="model"):
    """Use cloudpickle (not skops) to avoid untrusted-type checks on our own trusted models."""
    mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")


def _log_xgboost_model(model, name="model"):
    mlflow.xgboost.log_model(model, name=name)


MODEL_LOG_FUNCTIONS = {
    "LogisticRegression": _log_sklearn_model,
    "RandomForest": _log_sklearn_model,
    "XGBoost": _log_xgboost_model,
}


def train_and_log_model(model_name, X_train, y_train, X_test, y_test, params=None, run_name=None):
    """
    Trains a model (by name), evaluates it, and logs everything to MLflow:
    - params used
    - evaluation metrics
    - the fitted model itself as an artifact

    Parameters
    ----------
    model_name : str
        One of "LogisticRegression", "RandomForest", "XGBoost"
    params : dict, optional
        Hyperparameters to pass to the training function (e.g. {"C": 0.5})
    run_name : str, optional
        Display name for this MLflow run. Defaults to model_name.

    Returns
    -------
    model : fitted model object
    metrics : dict of evaluation metrics
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(MODEL_BUILDERS.keys())}")

    params = params or {}
    run_name = run_name or model_name
    build_fn = MODEL_BUILDERS[model_name]
    log_fn = MODEL_LOG_FUNCTIONS[model_name]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        model = build_fn(X_train, y_train, **params)
        metrics = evaluate_model(model, X_test, y_test)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        log_fn(model, name="model")

        run_id = mlflow.active_run().info.run_id
        print(f"[{model_name}] run_id={run_id} | roc_auc={metrics['roc_auc']:.4f} | recall={metrics['recall']:.4f}")

    return model, metrics


# ============================================================
# CLI entry point: python -m src.models.train
# ============================================================

def main():
    """
    Runs training for all 3 models with default hyperparameters,
    logging each to MLflow. Entry point for `python -m src.models.train`.
    """
    import pandas as pd
    from config import PROCESSED_DATA_DIR, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction_baseline")

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    print("Loaded data:", X_train.shape, X_test.shape)
    print("Training all models...\n")

    for model_name in MODEL_BUILDERS.keys():
        train_and_log_model(model_name, X_train, y_train, X_test, y_test)

    print("\nAll models trained and logged to MLflow.")


if __name__ == "__main__":
    main()


# ============================================================
# CLI entry point: python -m src.models.train
# ============================================================

def main():
    """
    Runs training for all 3 models with default hyperparameters,
    logging each to MLflow. Entry point for `python -m src.models.train`.
    """
    import pandas as pd
    from config import PROCESSED_DATA_DIR, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction_baseline")

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    print("Loaded data:", X_train.shape, X_test.shape)
    print("Training all models...\n")

    for model_name in MODEL_BUILDERS.keys():
        train_and_log_model(model_name, X_train, y_train, X_test, y_test)

    print("\nAll models trained and logged to MLflow.")


if __name__ == "__main__":
    main()
