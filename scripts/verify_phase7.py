"""
Phase 7 verification: prediction logging (Postgres) + drift detection.
Runs live HTTP checks against the deployed Render service.
"""
import requests
import time

BASE_URL = "https://churn-mlops-hm0l.onrender.com"

passed = 0
failed = 0
failures = []

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        failures.append(name)
        print(f"  ❌ {name}")

VALID_RECORD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}

print("=" * 60)
print("PHASE 7 VERIFICATION — Prediction Logging + Drift Detection")
print("=" * 60)

print("\n[1] Health check")
r = requests.get(f"{BASE_URL}/health", timeout=15)
check("GET /health returns 200", r.status_code == 200)
check("GET /health body is {'status': 'ok'}", r.json() == {"status": "ok"})

print("\n[2] /monitoring/stats basic shape")
r = requests.get(f"{BASE_URL}/monitoring/stats", timeout=15)
check("GET /monitoring/stats returns 200", r.status_code == 200)
stats_before = r.json()
check("'total_logged_predictions' key present", "total_logged_predictions" in stats_before)
check("total_logged_predictions is a non-negative int",
      isinstance(stats_before.get("total_logged_predictions"), int)
      and stats_before["total_logged_predictions"] >= 0)
count_before = stats_before["total_logged_predictions"]

print("\n[3] /predict correctness (known baseline)")
r = requests.post(f"{BASE_URL}/predict?model=champion", json=VALID_RECORD, timeout=15)
check("POST /predict (champion) returns 200", r.status_code == 200)
pred = r.json()
check("churn_probability matches known baseline (~0.8047)",
      abs(pred.get("churn_probability", 0) - 0.8047) < 0.01)
check("model_alias == 'champion'", pred.get("model_alias") == "champion")
check("response has all required PredictionResponse fields",
      all(k in pred for k in ["prediction", "churn_probability", "model_alias", "model_version", "run_id"]))

r2 = requests.post(f"{BASE_URL}/predict?model=challenger", json=VALID_RECORD, timeout=15)
check("POST /predict (challenger) returns 200", r2.status_code == 200)
check("challenger model_alias == 'challenger'", r2.json().get("model_alias") == "challenger")

print("\n[4] Logging side-effect verification")
time.sleep(1)
r = requests.get(f"{BASE_URL}/monitoring/stats", timeout=15)
count_after_2_calls = r.json()["total_logged_predictions"]
check(f"count increased by exactly 2 after 2 /predict calls (before={count_before}, after={count_after_2_calls})",
      count_after_2_calls == count_before + 2)

print("\n[5] Invalid requests are rejected and NOT logged")
bad_record = dict(VALID_RECORD)
bad_record["Contract"] = "Not a real contract type"
r = requests.post(f"{BASE_URL}/predict?model=champion", json=bad_record, timeout=15)
check("invalid enum value returns 422", r.status_code == 422)

missing_record = dict(VALID_RECORD)
del missing_record["tenure"]
r = requests.post(f"{BASE_URL}/predict?model=champion", json=missing_record, timeout=15)
check("missing required field returns 422", r.status_code == 422)

time.sleep(1)
r = requests.get(f"{BASE_URL}/monitoring/stats", timeout=15)
count_after_invalid = r.json()["total_logged_predictions"]
check("count UNCHANGED after invalid requests (no false logging)",
      count_after_invalid == count_after_2_calls)

# --- 6. /monitoring/drift: n below the API's own minimum (ge=30) is rejected
# at the query-param validation layer with 422 — this is correct, intentional
# behavior baked into main.py's Query(..., ge=30), NOT a path that reaches
# run_drift_report()'s internal "insufficient_data" logic. That internal path
# is only reachable when n>=30 is requested but fewer than 30 rows actually
# EXIST in the table (a state we can't force via query params alone once
# we're past 30 logged rows, so it's covered by drift.py's own unit tests
# from Colab instead, not re-tested here).
print("\n[6] /monitoring/drift — n below API minimum (ge=30) rejected at query-param level")
r = requests.get(f"{BASE_URL}/monitoring/drift?n=5", timeout=15)
check("GET /monitoring/drift?n=5 returns 422 (FastAPI Query ge=30 constraint)", r.status_code == 422)

print("\n[7] /monitoring/drift — full report path")
r = requests.get(f"{BASE_URL}/monitoring/drift", timeout=20)
check("GET /monitoring/drift (default n) returns 200", r.status_code == 200)
report = r.json()
check("status == 'ok' (enough samples logged)", report.get("status") == "ok")
check("'overall_drift_detected' is a bool", isinstance(report.get("overall_drift_detected"), bool))
check("'features' is a non-empty list", isinstance(report.get("features"), list) and len(report["features"]) > 0)

feature_names = {f["feature"] for f in report.get("features", [])}
expected_numeric = {"tenure", "MonthlyCharges", "TotalCharges"}
expected_categorical = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
}
check("all 3 numeric features present in report", expected_numeric.issubset(feature_names))
check("all 16 categorical features present in report", expected_categorical.issubset(feature_names))

for f in report.get("features", []):
    if f["type"] == "numeric":
        check(f"  {f['feature']}: drift_detected requires BOTH ks_pvalue<0.05 AND psi>0.2 (AND-logic)",
              f["drift_detected"] == (f["ks_pvalue"] < 0.05 and f["psi"] > 0.2))
    else:
        check(f"  {f['feature']}: categorical drift_detected matches psi>0.2 threshold",
              f["drift_detected"] == (f["psi"] > 0.2))

print("\n[8] /monitoring/drift query param bounds")
r = requests.get(f"{BASE_URL}/monitoring/drift?n=10000", timeout=15)
check("n above max (5000) rejected with 422", r.status_code == 422)
r = requests.get(f"{BASE_URL}/monitoring/drift?n=30", timeout=15)
check("n at exact minimum (30) returns 200", r.status_code == 200)

print("\n[9] No internal detail leakage")
r = requests.post(f"{BASE_URL}/predict?model=nonexistent", json=VALID_RECORD, timeout=15)
check("invalid model query param rejected (422)", r.status_code == 422)
body_text = r.text.lower()
check("no file paths leaked in error response",
      "/app/" not in body_text and "/content/" not in body_text and "traceback" not in body_text)

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed (out of {passed + failed})")
if failures:
    print("\nFailed checks:")
    for f in failures:
        print(f"  - {f}")
print("=" * 60)
