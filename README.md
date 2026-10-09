# Churn MLOps

End-to-end production-style ML system for customer churn prediction, built on the IBM Telco Customer Churn dataset.

## Live Service

- API: https://churn-mlops-hm0l.onrender.com
- Health check: `GET /health`
- Predict: `POST /predict?model=champion|challenger`
- Monitoring: `GET /monitoring/stats`, `GET /monitoring/drift?n=500`

## Stack

- **Model training & tracking:** scikit-learn, XGBoost, MLflow (local SQLite backend + Model Registry with aliases)
- **Serving:** FastAPI, served via Docker on Render
- **Monitoring:** Postgres-backed prediction logging, KS-test + PSI drift detection
- **Testing:** pytest, 111 unit tests covering data cleaning, inference, API, and monitoring logic
- **CI/CD:** GitHub Actions runs the test suite on every push to main; on success, triggers a gated deploy via a Render Deploy Hook (Auto-Deploy is disabled)

## Project Structure

```
src/
  data/         # cleaning (fix_total_charges, standardize_senior_citizen)
  api/          # FastAPI app, schema, inference logic
  monitoring/   # prediction logging (Postgres) + drift detection
  models/       # MLflow registry helpers
tests/unit/     # pytest suite (111 tests)
scripts/        # live-deployment verification scripts
.github/workflows/ci.yml   # test-gated CI/CD pipeline
```

## Models

- **Champion:** Logistic Regression (ROC-AUC 0.8415)
- **Challenger:** XGBoost (ROC-AUC 0.8332)

Both served via MLflow Model Registry aliases, loaded via their native flavor (not pyfunc) to support predict_proba.
