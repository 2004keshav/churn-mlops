"""
Model Registry module — register models, manage aliases, load by alias.
Reused in Phase 5 for serving.
"""
import mlflow
from mlflow.tracking import MlflowClient
import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

MODEL_NAME = "churn_predictor"


def get_client():
    return MlflowClient()


def register_model_version(run_id, model_name=MODEL_NAME):
    """Register a model version from a run_id."""
    version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )
    return version


def set_alias(alias, version, model_name=MODEL_NAME):
    """Point an alias at a specific version. Overwrites if alias already exists."""
    client = get_client()
    client.set_registered_model_alias(model_name, alias, version)
    print(f"Alias '{alias}' -> version {version}")


def load_model_by_alias(alias, model_name=MODEL_NAME):
    """Load the model currently pointed to by an alias. Used in Phase 5 serving."""
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def list_versions(model_name=MODEL_NAME):
    """List all versions of a registered model with run_id, status, and which aliases point to them."""
    client = get_client()
    rm = client.get_registered_model(model_name)
    alias_map = rm.aliases  # {alias_name: version}

    # Invert: version -> list of aliases pointing to it
    version_to_aliases = {}
    for alias, ver in alias_map.items():
        version_to_aliases.setdefault(str(ver), []).append(alias)

    versions = client.search_model_versions(f"name='{model_name}'")
    result = []
    for v in versions:
        result.append({
            "version": v.version,
            "run_id": v.run_id,
            "status": v.status,
            "aliases": version_to_aliases.get(str(v.version), [])
        })
    return result


if __name__ == "__main__":
    print(f"--- Versions of '{MODEL_NAME}' ---")
    for v in list_versions():
        print(v)
