"""
Phase 5 verification: sends real HTTP requests to a running FastAPI
instance and checks responses. Assumes server is already running on
localhost:8000 (started separately via uvicorn in the host notebook).
"""
import sys
import requests

BASE_URL = "http://localhost:8000"

VALID_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

EXPECTED_CHAMPION_PROBA = 0.8047499944163649
EXPECTED_CHALLENGER_PROBA = 0.8130133152008057
EXPECTED_CHAMPION_RUN_ID = "6c90c1eb1e2b48f7b62f8db8ca0966b8"
EXPECTED_CHALLENGER_RUN_ID = "2e5b7bacfcb74e39aaa3cbde1b6fc1bf"

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"PASS: {name}")
        passed += 1
    else:
        print(f"FAIL: {name}")
        failed += 1

r = requests.get(f"{BASE_URL}/health")
check("health status 200", r.status_code == 200)
check("health body correct", r.json() == {"status": "ok"})

r = requests.post(f"{BASE_URL}/predict", json=VALID_CUSTOMER)
check("predict champion status 200", r.status_code == 200)
body = r.json()
check("champion has prediction field", "prediction" in body)
check("champion has churn_probability field", "churn_probability" in body)
check("champion prediction is 1", body.get("prediction") == 1)
check("champion probability matches expected",
      abs(body.get("churn_probability", 0) - EXPECTED_CHAMPION_PROBA) < 1e-6)
check("champion model_alias correct", body.get("model_alias") == "champion")
check("champion run_id correct", body.get("run_id") == EXPECTED_CHAMPION_RUN_ID)

r = requests.post(f"{BASE_URL}/predict?model=challenger", json=VALID_CUSTOMER)
check("predict challenger status 200", r.status_code == 200)
body = r.json()
check("challenger probability matches expected",
      abs(body.get("churn_probability", 0) - EXPECTED_CHALLENGER_PROBA) < 1e-6)
check("challenger model_alias correct", body.get("model_alias") == "challenger")
check("challenger run_id correct", body.get("run_id") == EXPECTED_CHALLENGER_RUN_ID)

r = requests.post(f"{BASE_URL}/predict?model=invalid_alias", json=VALID_CUSTOMER)
check("invalid model query param rejected (422)", r.status_code == 422)

bad_customer = dict(VALID_CUSTOMER)
bad_customer["gender"] = "Other"
r = requests.post(f"{BASE_URL}/predict", json=bad_customer)
check("invalid gender rejected (422)", r.status_code == 422)

incomplete_customer = dict(VALID_CUSTOMER)
del incomplete_customer["tenure"]
r = requests.post(f"{BASE_URL}/predict", json=incomplete_customer)
check("missing field rejected (422)", r.status_code == 422)

bad_type_customer = dict(VALID_CUSTOMER)
bad_type_customer["MonthlyCharges"] = "not_a_number"
r = requests.post(f"{BASE_URL}/predict", json=bad_type_customer)
check("invalid type rejected (422)", r.status_code == 422)

negative_tenure_customer = dict(VALID_CUSTOMER)
negative_tenure_customer["tenure"] = -5
r = requests.post(f"{BASE_URL}/predict", json=negative_tenure_customer)
check("negative tenure rejected (422)", r.status_code == 422)

print(f"\n{'='*40}")
print(f"TOTAL: {passed} passed, {failed} failed")
print(f"{'='*40}")

if failed > 0:
    sys.exit(1)
