"""
Phase 6 verification: Docker containerization, portable paths, hardened
error handling, and live Render deployment. Tests against the LIVE
Render URL - this IS the Docker verification, since local docker
build/run was unavailable in this environment (Colab-only, no Docker
daemon access).
"""
import requests
import time

BASE_URL = "https://churn-mlops-hm0l.onrender.com"

CHAMPION_RUN_ID = "6c90c1eb1e2b48f7b62f8db8ca0966b8"
CHALLENGER_RUN_ID = "2e5b7bacfcb74e39aaa3cbde1b6fc1bf"

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

results = []

def check(description, condition):
    status = "PASS" if condition else "FAIL"
    results.append((description, status))
    print(f"[{status}] {description}")

def safe_post(payload, model="champion", timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            return requests.post(f"{BASE_URL}/predict", json=payload,
                                  params={"model": model}, timeout=timeout)
        except Exception as e:
            if attempt == retries:
                print(f"  (safe_post failed after {retries+1} attempts: {type(e).__name__}: {e})")
                return None
            time.sleep(2)

def safe_get(path, timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            return requests.get(f"{BASE_URL}{path}", timeout=timeout)
        except Exception as e:
            if attempt == retries:
                print(f"  (safe_get failed after {retries+1} attempts: {type(e).__name__}: {e})")
                return None
            time.sleep(2)


print("=" * 60)
print("PHASE 6 VERIFICATION - Live Render Deployment")
print(f"Target: {BASE_URL}")
print("=" * 60)

# --- 1. Health check ---
r = safe_get("/health")
check("GET /health returns 200", r is not None and r.status_code == 200)
check("GET /health body is {'status': 'ok'}", r is not None and r.ok and r.json() == {"status": "ok"})

# --- 2. Champion prediction correctness ---
r = safe_post(SAMPLE_CUSTOMER, model="champion")
champion_ok = r is not None and r.status_code == 200
check("POST /predict?model=champion returns 200", champion_ok)
if champion_ok:
    body = r.json()
    check("Champion response has all expected keys",
          set(body.keys()) == {"prediction", "churn_probability", "model_alias", "model_version", "run_id"})
    check("Champion run_id matches known value", body.get("run_id") == CHAMPION_RUN_ID)
    check("Champion model_alias == 'champion'", body.get("model_alias") == "champion")
    check("Champion churn_probability in valid range [0,1]",
          isinstance(body.get("churn_probability"), float) and 0.0 <= body["churn_probability"] <= 1.0)
    check("Champion prediction is 0 or 1", body.get("prediction") in (0, 1))
    check("Champion probability matches known verified value (~0.8047)",
          abs(body.get("churn_probability", -1) - 0.8047499944163649) < 1e-6)

# --- 3. Challenger prediction correctness ---
r = safe_post(SAMPLE_CUSTOMER, model="challenger")
challenger_ok = r is not None and r.status_code == 200
check("POST /predict?model=challenger returns 200", challenger_ok)
if challenger_ok:
    body = r.json()
    check("Challenger run_id matches known value", body.get("run_id") == CHALLENGER_RUN_ID)
    check("Challenger model_alias == 'challenger'", body.get("model_alias") == "challenger")
    check("Challenger churn_probability in valid range [0,1]",
          isinstance(body.get("churn_probability"), float) and 0.0 <= body["churn_probability"] <= 1.0)

# --- 4. Invalid model query param ---
r = safe_post(SAMPLE_CUSTOMER, model="nonexistent_model")
check("Invalid model query param returns 422", r is not None and r.status_code == 422)

# --- 5. Missing required field ---
incomplete = dict(SAMPLE_CUSTOMER)
del incomplete["tenure"]
r = requests.post(f"{BASE_URL}/predict", json=incomplete, params={"model": "champion"})
check("Missing required field (tenure) returns 422", r.status_code == 422)

# --- 6. Invalid enum value ---
bad_enum = dict(SAMPLE_CUSTOMER)
bad_enum["Contract"] = "Not-A-Real-Contract"
r = requests.post(f"{BASE_URL}/predict", json=bad_enum, params={"model": "champion"})
check("Invalid enum value (Contract) returns 422", r.status_code == 422)

# --- 7. Wrong type ---
wrong_type = dict(SAMPLE_CUSTOMER)
wrong_type["tenure"] = "not_a_number"
r = requests.post(f"{BASE_URL}/predict", json=wrong_type, params={"model": "champion"})
check("Wrong type (tenure as string) returns 422", r.status_code == 422)

# --- 8. Out-of-range values (Phase 6 new bounds) ---
oob_monthly = dict(SAMPLE_CUSTOMER)
oob_monthly["MonthlyCharges"] = 99999
r = requests.post(f"{BASE_URL}/predict", json=oob_monthly, params={"model": "champion"})
check("Out-of-range MonthlyCharges (>500) returns 422", r.status_code == 422)

oob_total = dict(SAMPLE_CUSTOMER)
oob_total["TotalCharges"] = 999999
r = requests.post(f"{BASE_URL}/predict", json=oob_total, params={"model": "champion"})
check("Out-of-range TotalCharges (>20000) returns 422", r.status_code == 422)

oob_tenure = dict(SAMPLE_CUSTOMER)
oob_tenure["tenure"] = 500
r = requests.post(f"{BASE_URL}/predict", json=oob_tenure, params={"model": "champion"})
check("Out-of-range tenure (>100) returns 422", r.status_code == 422)

negative_charge = dict(SAMPLE_CUSTOMER)
negative_charge["MonthlyCharges"] = -10
r = requests.post(f"{BASE_URL}/predict", json=negative_charge, params={"model": "champion"})
check("Negative MonthlyCharges returns 422", r.status_code == 422)

# --- 9. Error message does not leak internals (hardened error handling) ---
r = requests.post(f"{BASE_URL}/predict", json=bad_enum, params={"model": "champion"})
error_body_text = r.text.lower()
check("422 error body does not contain file paths",
      "/app/" not in error_body_text and "/content/" not in error_body_text and "traceback" not in error_body_text)

# --- 10. Response time sanity (cold start tolerance for free tier) ---
start = time.time()
r = safe_get("/health")
elapsed = time.time() - start
check(f"Health check responds within 60s (actual: {elapsed:.1f}s)", elapsed < 60)

# --- Summary ---
print("=" * 60)
passed = sum(1 for _, s in results if s == "PASS")
total = len(results)
print(f"RESULT: {passed}/{total} checks passed")
print("=" * 60)

if passed != total:
    print("\nFAILED CHECKS:")
    for desc, status in results:
        if status == "FAIL":
            print(f"  - {desc}")
