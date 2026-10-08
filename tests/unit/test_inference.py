"""
Unit tests for src/api/inference.py.
preprocess() uses the REAL encoder/scaler (joblib artifacts load fine in
this environment, confirmed via diagnostic) — no mocking needed there.
predict() is tested with get_model() MOCKED (no real MLflow model load).
"""
import numpy as np
import pandas as pd
import pytest

from src.api import inference


EXPECTED_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'SeniorCitizen_Yes', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
]


class TestPreprocess:
    def test_output_shape(self, valid_customer_payload):
        X = inference.preprocess(valid_customer_payload)
        assert X.shape == (1, 40)

    def test_output_columns_match_training_schema(self, valid_customer_payload):
        X = inference.preprocess(valid_customer_payload)
        assert X.columns.tolist() == EXPECTED_COLUMNS

    def test_output_all_float_dtype(self, valid_customer_payload):
        X = inference.preprocess(valid_customer_payload)
        assert all(dt == np.float64 for dt in X.dtypes)

    def test_senior_citizen_zero_handled(self, valid_customer_payload):
        """SeniorCitizen=0 (raw int from API) must be standardized to 'No'
        internally before encoding, not crash or silently produce NaN."""
        payload = dict(valid_customer_payload, SeniorCitizen=0)
        X = inference.preprocess(payload)
        assert X["SeniorCitizen_Yes"].iloc[0] == 0.0
        assert not X.isna().any().any()

    def test_senior_citizen_one_handled(self, valid_customer_payload):
        payload = dict(valid_customer_payload, SeniorCitizen=1)
        X = inference.preprocess(payload)
        assert X["SeniorCitizen_Yes"].iloc[0] == 1.0

    def test_binary_fields_produce_single_column(self, valid_customer_payload):
        """drop='if_binary' means Yes/No fields produce exactly ONE column,
        not two — this was a Phase 2 design decision, regression-guard it."""
        X = inference.preprocess(valid_customer_payload)
        assert "PhoneService_No" not in X.columns
        assert "PhoneService_Yes" in X.columns

    def test_multicategory_fields_produce_all_columns(self, valid_customer_payload):
        """Non-binary fields (3+ categories) must produce one column PER
        category, unlike binary fields."""
        X = inference.preprocess(valid_customer_payload)
        contract_cols = [c for c in X.columns if c.startswith("Contract_")]
        assert len(contract_cols) == 3

    def test_does_not_mutate_input_dict(self, valid_customer_payload):
        original = dict(valid_customer_payload)
        inference.preprocess(valid_customer_payload)
        assert valid_customer_payload == original


class TestPredict:
    def test_predict_returns_correct_response_shape(
        self, mocker, valid_customer_payload, fake_model
    ):
        mocker.patch(
            "src.api.inference.get_model",
            return_value={
                "model": fake_model,
                "flavor": "sklearn",
                "version": 3,
                "run_id": "6c90c1eb1e2b48f7b62f8db8ca0966b8",
            },
        )
        result = inference.predict(valid_customer_payload, alias="champion")

        assert set(result.keys()) == {
            "prediction", "churn_probability", "model_alias",
            "model_version", "run_id",
        }
        assert isinstance(result["prediction"], int)
        assert isinstance(result["churn_probability"], float)
        assert result["model_alias"] == "champion"
        assert result["model_version"] == 3
        assert result["run_id"] == "6c90c1eb1e2b48f7b62f8db8ca0966b8"

    def test_predict_high_proba_gives_prediction_one(
        self, mocker, valid_customer_payload
    ):
        high_proba_model = inference_FakeModel(proba_churn=0.8047)
        mocker.patch(
            "src.api.inference.get_model",
            return_value={
                "model": high_proba_model, "flavor": "sklearn",
                "version": 3, "run_id": "test-run-id",
            },
        )
        result = inference.predict(valid_customer_payload, alias="champion")
        assert result["prediction"] == 1
        assert result["churn_probability"] == pytest.approx(0.8047)

    def test_predict_low_proba_gives_prediction_zero(
        self, mocker, valid_customer_payload
    ):
        low_proba_model = inference_FakeModel(proba_churn=0.15)
        mocker.patch(
            "src.api.inference.get_model",
            return_value={
                "model": low_proba_model, "flavor": "sklearn",
                "version": 4, "run_id": "test-run-id-2",
            },
        )
        result = inference.predict(valid_customer_payload, alias="challenger")
        assert result["prediction"] == 0
        assert result["churn_probability"] == pytest.approx(0.15)

    def test_predict_boundary_proba_exactly_half(
        self, mocker, valid_customer_payload
    ):
        """proba == 0.5 exactly: code uses >= 0.5, so prediction should be 1."""
        boundary_model = inference_FakeModel(proba_churn=0.5)
        mocker.patch(
            "src.api.inference.get_model",
            return_value={
                "model": boundary_model, "flavor": "sklearn",
                "version": 3, "run_id": "test-run-id-3",
            },
        )
        result = inference.predict(valid_customer_payload, alias="champion")
        assert result["prediction"] == 1

    def test_predict_passes_correct_alias_to_get_model(
        self, mocker, valid_customer_payload, fake_model
    ):
        mock_get_model = mocker.patch(
            "src.api.inference.get_model",
            return_value={
                "model": fake_model, "flavor": "sklearn",
                "version": 4, "run_id": "test-run-id",
            },
        )
        inference.predict(valid_customer_payload, alias="challenger")
        mock_get_model.assert_called_once_with("challenger")


# Local copy of FakeModel (independent of conftest's fixture instance) so
# tests above can construct it with custom proba_churn values directly.
class inference_FakeModel:
    def __init__(self, proba_churn: float = 0.8047):
        self.proba_churn = proba_churn

    def predict_proba(self, X):
        n = len(X)
        return np.array([[1 - self.proba_churn, self.proba_churn]] * n)


class TestGetModelCachingAndFallback:
    def test_get_model_caches_after_first_call(self, mocker):
        """get_model() must only call _load_native_model and
        _get_version_and_run ONCE per alias — subsequent calls return the
        cached entry, not reload from MLflow every time."""
        inference._model_cache.clear()

        mock_load = mocker.patch(
            "src.api.inference._load_native_model",
            return_value=(mocker.MagicMock(), "sklearn"),
        )
        mock_version = mocker.patch(
            "src.api.inference._get_version_and_run",
            return_value=(3, "run-id-abc"),
        )

        result1 = inference.get_model("champion")
        result2 = inference.get_model("champion")

        assert result1 is result2  # same cached dict object
        mock_load.assert_called_once_with("champion")
        mock_version.assert_called_once_with("champion")

        inference._model_cache.clear()

    def test_load_native_model_falls_back_to_xgboost_when_sklearn_fails(self, mocker):
        """_load_native_model tries sklearn flavor first; if that raises,
        it must fall back to xgboost flavor rather than propagating the
        sklearn error. Patch target is the library's own dotted path
        ('mlflow.sklearn.load_model'), not prefixed through
        src.api.inference — confirmed via diagnostic that the prefixed
        path does not intercept the call."""
        mocker.patch(
            "mlflow.sklearn.load_model",
            side_effect=Exception("not an sklearn flavor model"),
        )
        fake_xgb_model = mocker.MagicMock()
        mock_xgb_load = mocker.patch(
            "mlflow.xgboost.load_model",
            return_value=fake_xgb_model,
        )

        model, flavor = inference._load_native_model("challenger")

        assert flavor == "xgboost"
        assert model is fake_xgb_model
        mock_xgb_load.assert_called_once_with("models:/churn_predictor@challenger")

    def test_load_native_model_uses_sklearn_when_it_succeeds(self, mocker):
        """When sklearn flavor load succeeds, xgboost fallback must NOT
        be attempted."""
        fake_sklearn_model = mocker.MagicMock()
        mocker.patch(
            "mlflow.sklearn.load_model",
            return_value=fake_sklearn_model,
        )
        mock_xgb_load = mocker.patch("mlflow.xgboost.load_model")

        model, flavor = inference._load_native_model("champion")

        assert flavor == "sklearn"
        assert model is fake_sklearn_model
        mock_xgb_load.assert_not_called()
