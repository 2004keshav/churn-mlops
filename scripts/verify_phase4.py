"""
Phase 4 verification script — Model Registry checks.
Run: python scripts/verify_phase4.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import mlflow
from mlflow.tracking import MlflowClient
import config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "churn_predictor"
EXPECTED_CHAMPION_RUN_ID = "6c90c1eb1e2b48f7b62f8db8ca0966b8"
EXPECTED_CHALLENGER_RUN_ID = "2e5b7bacfcb74e39aaa3cbde1b6fc1bf"

checks_passed = 0
checks_failed = 0

def check(name, condition):
    global checks_passed, checks_failed
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1
    print(f"[{status}] {name}")

print("=" * 70)
print("PHASE 4 VERIFICATION")
print("=" * 70)

# 1. File existence
print("\n--- File Existence ---")
check("registry.py exists", os.path.exists("src/models/registry.py"))

# 2. Code structure
print("\n--- Code Structure ---")
with open("src/models/registry.py") as f:
    registry_code = f.read()
check("register_model_version function defined", "def register_model_version" in registry_code)
check("set_alias function defined", "def set_alias" in registry_code)
check("load_model_by_alias function defined", "def load_model_by_alias" in registry_code)
check("list_versions function defined", "def list_versions" in registry_code)
check("uses models:/ alias syntax (not stages)", "models:/" in registry_code and "@" in registry_code)
check("does NOT use deprecated stages API", "transition_model_version_stage" not in registry_code)
check("does NOT use deprecated Production stage syntax", '"Production"' not in registry_code and "'Production'" not in registry_code)
check("uses pyfunc (flavor-agnostic) loading", "mlflow.pyfunc.load_model" in registry_code)

# 3. Syntax validity
print("\n--- Syntax Validity ---")
try:
    compile(registry_code, "registry.py", "exec")
    check("registry.py compiles without syntax errors", True)
except SyntaxError:
    check("registry.py compiles without syntax errors", False)

# 4. Registered model exists
print("\n--- Registry State ---")
try:
    rm = client.get_registered_model(MODEL_NAME)
    check(f"registered model '{MODEL_NAME}' exists", True)
except Exception:
    check(f"registered model '{MODEL_NAME}' exists", False)
    rm = None

# 5. Aliases correctness
if rm is not None:
    aliases = rm.aliases
    check("'champion' alias exists", "champion" in aliases)
    check("'challenger' alias exists", "challenger" in aliases)
    check("champion points to version 1", str(aliases.get("champion")) == "1")
    check("challenger points to version 2", str(aliases.get("challenger")) == "2")

# 6. Version -> run_id integrity (traceability chain)
print("\n--- Traceability: run_id -> version -> alias ---")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
version_map = {str(v.version): v.run_id for v in versions}
status_map = {str(v.version): v.status for v in versions}

check("version 1 exists and is READY", status_map.get("1") == "READY")
check("version 2 exists and is READY", status_map.get("2") == "READY")
check(
    "version 1 run_id matches expected champion run_id",
    version_map.get("1") == EXPECTED_CHAMPION_RUN_ID
)
check(
    "version 2 run_id matches expected challenger run_id",
    version_map.get("2") == EXPECTED_CHALLENGER_RUN_ID
)

# 7. Functional check: load_model_by_alias actually loads a working model
print("\n--- Functional: load_model_by_alias ---")
try:
    if "src.models.registry" in sys.modules:
        del sys.modules["src.models.registry"]
    from src.models import registry as reg_module

    champion_model = reg_module.load_model_by_alias("champion")
    check("load_model_by_alias('champion') returns a model", champion_model is not None)
    check("champion model has predict method", hasattr(champion_model, "predict"))

    challenger_model = reg_module.load_model_by_alias("challenger")
    check("load_model_by_alias('challenger') returns a model", challenger_model is not None)
    check("challenger model has predict method", hasattr(challenger_model, "predict"))

    # Functional smoke test: actually run inference on real test data
    import pandas as pd
    X_test = pd.read_csv(config.PROCESSED_DATA_DIR / "X_test.csv")
    champ_preds = champion_model.predict(X_test.head(5))
    chall_preds = challenger_model.predict(X_test.head(5))
    check("champion model produces predictions on real data", len(champ_preds) == 5)
    check("challenger model produces predictions on real data", len(chall_preds) == 5)

except Exception as e:
    check(f"load_model_by_alias functional test (error: {e})", False)

# Summary
print("\n" + "=" * 70)
print(f"RESULTS: {checks_passed}/{checks_passed + checks_failed} checks passed")
print("=" * 70)

if checks_failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
