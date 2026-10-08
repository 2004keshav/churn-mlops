"""
Unit tests for /predict and /health in src/api/main.py.
Uses FastAPI's TestClient WITHOUT the `with` context manager (confirmed in
diagnostics: this skips the startup event, so no real MLflow model load is
attempted). inference.predict and db.insert_prediction_log are mocked at
their point of use inside src.api.main.
"""
import pytest


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPredictSuccess:
    def test_predict_with_valid_payload_champion(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id-123")

        resp = client.post("/predict", json=valid_customer_payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == fake_prediction_result["prediction"]
        assert body["churn_probability"] == fake_prediction_result["churn_probability"]
        assert body["model_alias"] == "champion"
        assert body["model_version"] == fake_prediction_result["model_version"]
        assert body["run_id"] == fake_prediction_result["run_id"]
        mock_log.assert_called_once()

    def test_predict_with_challenger_model_param(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        mock_predict = mocker.patch(
            "src.api.main.inference.predict",
            return_value=dict(fake_prediction_result, model_alias="challenger"),
        )
        mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id")

        resp = client.post("/predict?model=challenger", json=valid_customer_payload)

        assert resp.status_code == 200
        assert resp.json()["model_alias"] == "challenger"
        mock_predict.assert_called_once()
        call_kwargs = mock_predict.call_args
        assert call_kwargs.kwargs.get("alias") == "challenger" or "challenger" in call_kwargs.args

    def test_predict_defaults_to_champion_when_model_param_omitted(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        mock_predict = mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id")

        resp = client.post("/predict", json=valid_customer_payload)
        assert resp.status_code == 200
        call_kwargs = mock_predict.call_args
        assert call_kwargs.kwargs.get("alias") == "champion" or "champion" in call_kwargs.args

    def test_predict_rejects_invalid_model_param(
        self, client, mocker, valid_customer_payload
    ):
        """Only 'champion'/'challenger' are valid per Literal type."""
        resp = client.post("/predict?model=nonexistent", json=valid_customer_payload)
        assert resp.status_code == 422


class TestPredictLoggingSideEffect:
    def test_logging_called_with_normalized_senior_citizen_zero(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        """SeniorCitizen=0 (raw int) must be normalized to 'No' (string)
        before being passed to db.insert_prediction_log, per Phase 7 design
        (matches reference_stats.json's string-based categorical format)."""
        mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id")

        payload = dict(valid_customer_payload, SeniorCitizen=0)
        resp = client.post("/predict", json=payload)

        assert resp.status_code == 200
        _, kwargs = mock_log.call_args
        logged_raw_input = kwargs["raw_input"]
        assert logged_raw_input["SeniorCitizen"] == "No"

    def test_logging_called_with_normalized_senior_citizen_one(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id")

        payload = dict(valid_customer_payload, SeniorCitizen=1)
        resp = client.post("/predict", json=payload)

        assert resp.status_code == 200
        _, kwargs = mock_log.call_args
        logged_raw_input = kwargs["raw_input"]
        assert logged_raw_input["SeniorCitizen"] == "Yes"

    def test_predict_response_not_affected_when_logging_fails(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        """THE key design guarantee from Phase 7: a logging failure must
        NEVER break the /predict response — main.py wraps the logging call
        in its own try/except."""
        mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mocker.patch(
            "src.api.main.db.insert_prediction_log",
            side_effect=Exception("Postgres is down"),
        )

        resp = client.post("/predict", json=valid_customer_payload)

        assert resp.status_code == 200
        assert resp.json()["prediction"] == fake_prediction_result["prediction"]

    def test_logging_includes_latency_ms(
        self, client, mocker, valid_customer_payload, fake_prediction_result
    ):
        mocker.patch(
            "src.api.main.inference.predict",
            return_value=fake_prediction_result,
        )
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log", return_value="req-id")

        client.post("/predict", json=valid_customer_payload)

        _, kwargs = mock_log.call_args
        assert "latency_ms" in kwargs
        assert isinstance(kwargs["latency_ms"], float)
        assert kwargs["latency_ms"] >= 0


class TestPredictValidationErrors:
    def test_missing_required_field_rejected_and_not_logged(
        self, client, mocker, valid_customer_payload
    ):
        mock_predict = mocker.patch("src.api.main.inference.predict")
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log")

        payload = dict(valid_customer_payload)
        del payload["gender"]
        resp = client.post("/predict", json=payload)

        assert resp.status_code == 422
        mock_predict.assert_not_called()
        mock_log.assert_not_called()

    def test_invalid_enum_value_rejected_and_not_logged(
        self, client, mocker, valid_customer_payload
    ):
        mock_predict = mocker.patch("src.api.main.inference.predict")
        mock_log = mocker.patch("src.api.main.db.insert_prediction_log")

        payload = dict(valid_customer_payload, Contract="Three year")
        resp = client.post("/predict", json=payload)

        assert resp.status_code == 422
        mock_predict.assert_not_called()
        mock_log.assert_not_called()

    def test_out_of_bounds_tenure_rejected(
        self, client, mocker, valid_customer_payload
    ):
        mocker.patch("src.api.main.inference.predict")
        payload = dict(valid_customer_payload, tenure=999)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_prediction_failure_returns_generic_500_no_internal_details(
        self, client, mocker, valid_customer_payload
    ):
        """Hardening from Phase 6: no leaked paths/tracebacks in error
        responses."""
        mocker.patch(
            "src.api.main.inference.predict",
            side_effect=Exception("Some internal MLflow path error: /app/mlruns/..."),
        )
        resp = client.post("/predict", json=valid_customer_payload)

        assert resp.status_code == 500
        body = resp.json()
        assert "mlruns" not in body.get("detail", "")
        assert "/app/" not in body.get("detail", "")
