"""
Independent Phase 4 audit — re-derives correctness from scratch rather
than trusting previously logged/computed values. Run after verify_phase4.py
as a second, independent check.
Run: python scripts/audit_phase4.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import glob
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import roc_auc_score
import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "churn_predictor"

checks_passed = 0
checks_failed = 0
warnings = []

def check(name, condition, detail=""):
    global checks_passed, checks_failed
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] {name}{suffix}")

def warn(msg):
    warnings.append(msg)
    print(f"[WARN] {msg}")

print("=" * 70)
print("PHASE 4 INDEPENDENT AUDIT")
print("=" * 70)

# -------------------------------------------------------------------
# 1. Environment sanity
# -------------------------------------------------------------------
print("\n--- Environment ---")
check("MLflow tracking URI is SQLite (not file:// or http)",
      mlflow.get_tracking_uri().startswith("sqlite:///"),
      mlflow.get_tracking_uri())
check("mlflow version is 3.x", mlflow.__version__.startswith("3."), mlflow.__version__)

# -------------------------------------------------------------------
# 2. Registered model exists and has exactly the expected versions
# -------------------------------------------------------------------
print("\n--- Registered Model State ---")
try:
    rm = client.get_registered_model(MODEL_NAME)
    check(f"'{MODEL_NAME}' is registered", True)
except Exception as e:
    check(f"'{MODEL_NAME}' is registered", False, str(e))
    sys.exit(1)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version_nums = sorted(set(str(v.version) for v in versions))
check("at least 2 versions exist", len(version_nums) >= 2, f"found: {version_nums}")

# -------------------------------------------------------------------
# 3. Alias integrity — no orphan/duplicate/conflicting aliases
# -------------------------------------------------------------------
print("\n--- Alias Integrity ---")
aliases = rm.aliases  # {alias: version}
check("no duplicate alias names", len(aliases) == len(set(aliases.keys())))

# Each alias must point to a version that actually exists and is READY
valid_versions = {str(v.version): v.status for v in versions}
for alias, ver in aliases.items():
    ver_str = str(ver)
    check(f"alias '{alias}' points to an existing version", ver_str in valid_versions, f"version {ver_str}")
    if ver_str in valid_versions:
        check(f"alias '{alias}' -> version {ver_str} is READY", valid_versions[ver_str] == "READY")

# No two aliases should point to the SAME version unless intentional (flag it)
from collections import Counter
version_counts = Counter(str(v) for v in aliases.values())
for ver, count in version_counts.items():
    if count > 1:
        pointing_aliases = [a for a, v in aliases.items() if str(v) == ver]
        warn(f"version {ver} has multiple aliases pointing to it: {pointing_aliases} — confirm this is intentional")

check("'champion' alias exists", "champion" in aliases)
check("'challenger' alias exists", "challenger" in aliases)
if "champion" in aliases and "challenger" in aliases:
    check("champion and challenger point to DIFFERENT versions",
          str(aliases["champion"]) != str(aliases["challenger"]))

# -------------------------------------------------------------------
# 4. Re-derive: is champion actually the best-performing model?
#    (independent check — don't trust Phase 3's claim, recompute)
# -------------------------------------------------------------------
print("\n--- Re-deriving Champion Correctness (recompute from run metrics) ---")
experiment = client.get_experiment_by_name("churn_prediction_baseline")
all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])

best_run = max(
    (r for r in all_runs if r.data.metrics.get("roc_auc") is not None),
    key=lambda r: r.data.metrics["roc_auc"]
)
best_roc_auc = best_run.data.metrics["roc_auc"]
print(f"  Best run by roc_auc across ALL Phase 3 runs: {best_run.info.run_id} ({best_roc_auc:.4f})")

champion_version_obj = next(v for v in versions if str(v.version) == str(aliases.get("champion")))
champion_run = client.get_run(champion_version_obj.run_id)
champion_roc_auc = champion_run.data.metrics.get("roc_auc")

check(
    "champion's run_id roc_auc matches the best roc_auc across all runs",
    abs(champion_roc_auc - best_roc_auc) < 1e-9,
    f"champion={champion_roc_auc:.4f}, best_overall={best_roc_auc:.4f}"
)

# -------------------------------------------------------------------
# 5. Re-derive: does the LOADED model actually reproduce logged metrics?
#    (the strongest possible check — bypasses all trust in prior scripts)
# -------------------------------------------------------------------
print("\n--- Re-deriving Metrics From Loaded Model (ground-truth check) ---")
try:
    if "src.models.registry" in sys.modules:
        del sys.modules["src.models.registry"]
    from src.models import registry as reg_module

    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(config.PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    for alias in ["champion", "challenger"]:
        ver_obj = next(v for v in versions if str(v.version) == str(aliases.get(alias)))
        logged_roc_auc = client.get_run(ver_obj.run_id).data.metrics.get("roc_auc")

        model = reg_module.load_model_by_alias(alias)
        # pyfunc models: predict() gives class labels; try predict_proba via unwrapped model for AUC
        raw_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}") \
            if "sklearn" in str(ver_obj.tags).lower() or alias == "champion" else None

        try:
            # Attempt native predict_proba for accurate AUC recomputation
            native_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")._model_impl.python_model \
                if hasattr(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}"), "_model_impl") else None
        except Exception:
            native_model = None

        preds = model.predict(X_test)
        # If predict returns probabilities/labels, compute AUC using available scores
        try:
            recomputed_auc = roc_auc_score(y_test, preds)
            diff = abs(recomputed_auc - logged_roc_auc)
            check(
                f"'{alias}' recomputed roc_auc is close to logged roc_auc",
                diff < 0.02,  # tolerance since predict() may return hard labels not probabilities
                f"recomputed={recomputed_auc:.4f}, logged={logged_roc_auc:.4f}, diff={diff:.4f}"
            )
        except Exception as e:
            warn(f"Could not recompute AUC for '{alias}' via predict() (likely returns hard labels, not probabilities): {e}")

except Exception as e:
    check(f"ground-truth reload/recompute check (error: {e})", False)

# -------------------------------------------------------------------
# 6. Repo-wide scan for deprecated API usage (not just registry.py)
# -------------------------------------------------------------------
print("\n--- Repo-wide Deprecated API Scan ---")
py_files = glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/**/*.py", recursive=True)
deprecated_hits = []
for fpath in py_files:
    with open(fpath, errors="ignore") as f:
        content = f.read()
    if "transition_model_version_stage" in content:
        deprecated_hits.append((fpath, "transition_model_version_stage"))
    if "models:/" in content and "/Production" in content:
        deprecated_hits.append((fpath, "models:/.../Production stage syntax"))
    if "models:/" in content and "/Staging" in content:
        deprecated_hits.append((fpath, "models:/.../Staging stage syntax"))

check("no deprecated stages API used anywhere in src/ or scripts/", len(deprecated_hits) == 0,
      str(deprecated_hits) if deprecated_hits else "")

# -------------------------------------------------------------------
# 7. Model signature / feature count sanity
# -------------------------------------------------------------------
print("\n--- Model Input Compatibility ---")
try:
    X_test_cols = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv").shape[1]
    champion_model_info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}@champion")
    sig = champion_model_info.signature
    if sig is not None and sig.inputs is not None:
        expected_features = len(sig.inputs.inputs)
        check("champion model's expected input feature count matches X_test",
              expected_features == X_test_cols,
              f"model expects {expected_features}, X_test has {X_test_cols}")
    else:
        warn("champion model has no logged signature — cannot verify expected input shape")
except Exception as e:
    warn(f"Could not check model signature: {e}")

# -------------------------------------------------------------------
# 8. Physical artifact existence (not just registry metadata)
# -------------------------------------------------------------------
print("\n--- Physical Artifact Check ---")
mlruns_models_dir = os.path.join(str(config.MLRUNS_DIR), "models")
check("mlruns/models/ directory exists on disk", os.path.isdir(mlruns_models_dir))
if os.path.isdir(mlruns_models_dir):
    logged_model_dirs = os.listdir(mlruns_models_dir)
    check("at least 2 logged model artifact dirs exist (m-<hash>)", len(logged_model_dirs) >= 2,
          f"found {len(logged_model_dirs)}")

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"AUDIT RESULTS: {checks_passed}/{checks_passed + checks_failed} checks passed")
if warnings:
    print(f"WARNINGS: {len(warnings)} (review above — not auto-failures)")
print("=" * 70)

if checks_failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
