"""
Unit tests for /monitoring/stats and /monitoring/drift in src/api/main.py.
db.count_logs, db.fetch_recent_logs, and drift.run_drift_report are all
mocked — no real Postgres or reference_stats.json file access.
"""
import pandas as pd
import pytest


class TestMonitoringStats:
    def test_returns_total_logged_predictions(self, client, mocker):
        mocker.patch("src.api.main.db.count_logs", return_value=76)
        resp = client.get("/monitoring/stats")
        assert resp.status_code == 200
        assert resp.json() == {"total_logged_predictions": 76}

    def test_returns_zero_when_no_logs(self, client, mocker):
        mocker.patch("src.api.main.db.count_logs", return_value=0)
        resp = client.get("/monitoring/stats")
        assert resp.status_code == 200
        assert resp.json() == {"total_logged_predictions": 0}


class TestMonitoringDriftQueryParamValidation:
    def test_n_below_30_rejected_by_fastapi_itself(self, client, mocker):
        """n < 30 is rejected by FastAPI's Query(ge=30) BEFORE reaching
        drift.py's internal insufficient_data branch — this test confirms
        the 422 comes from param validation, not from calling drift logic."""
        mock_fetch = mocker.patch("src.api.main.db.fetch_recent_logs")
        mock_report = mocker.patch("src.api.main.drift.run_drift_report")

        resp = client.get("/monitoring/drift?n=29")

        assert resp.status_code == 422
        mock_fetch.assert_not_called()
        mock_report.assert_not_called()

    def test_n_above_5000_rejected_by_fastapi(self, client, mocker):
        mock_fetch = mocker.patch("src.api.main.db.fetch_recent_logs")
        resp = client.get("/monitoring/drift?n=5001")
        assert resp.status_code == 422
        mock_fetch.assert_not_called()

    def test_n_exactly_30_accepted(self, client, mocker):
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame())
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            return_value={"status": "insufficient_data", "n_samples": 0,
                          "message": "...", "min_required": 30},
        )
        resp = client.get("/monitoring/drift?n=30")
        assert resp.status_code == 200

    def test_n_exactly_5000_accepted(self, client, mocker):
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame())
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            return_value={"status": "insufficient_data", "n_samples": 0,
                          "message": "...", "min_required": 30},
        )
        resp = client.get("/monitoring/drift?n=5000")
        assert resp.status_code == 200

    def test_default_n_used_when_omitted(self, client, mocker):
        """Default n=500 per main.py's Query(500, ge=30, le=5000)."""
        mock_fetch = mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame())
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            return_value={"status": "insufficient_data", "n_samples": 0,
                          "message": "...", "min_required": 30},
        )
        client.get("/monitoring/drift")
        mock_fetch.assert_called_once_with(n=500)


class TestMonitoringDriftReportStructure:
    def test_insufficient_data_status_passed_through(self, client, mocker):
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame())
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            return_value={
                "status": "insufficient_data",
                "message": "Need at least 30 logged predictions...",
                "n_samples": 5,
                "min_required": 30,
            },
        )
        resp = client.get("/monitoring/drift?n=30")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "insufficient_data"
        assert body["n_samples"] == 5
        assert body["min_required"] == 30

    def test_full_drift_report_structure_passed_through(self, client, mocker):
        fake_report = {
            "status": "ok",
            "n_samples": 500,
            "overall_drift_detected": False,
            "features": [
                {"feature": "tenure", "type": "numeric", "ks_statistic": 0.02,
                 "ks_pvalue": 0.8, "psi": 0.05, "drift_detected": False,
                 "live_mean": 32.0, "reference_mean": 32.4},
                {"feature": "gender", "type": "categorical", "psi": 0.01,
                 "drift_detected": False,
                 "live_distribution": {"Female": 0.5, "Male": 0.5},
                 "reference_distribution": {"Female": 0.5, "Male": 0.5}},
            ],
        }
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame({"a": [1]}))
        mocker.patch("src.api.main.drift.run_drift_report", return_value=fake_report)

        resp = client.get("/monitoring/drift?n=500")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["overall_drift_detected"] is False
        assert len(body["features"]) == 2

    def test_overall_drift_detected_true_when_any_feature_drifts(self, client, mocker):
        fake_report = {
            "status": "ok",
            "n_samples": 500,
            "overall_drift_detected": True,
            "features": [
                {"feature": "tenure", "type": "numeric", "ks_statistic": 0.5,
                 "ks_pvalue": 0.001, "psi": 0.5, "drift_detected": True,
                 "live_mean": 10.0, "reference_mean": 32.4},
            ],
        }
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame({"a": [1]}))
        mocker.patch("src.api.main.drift.run_drift_report", return_value=fake_report)

        resp = client.get("/monitoring/drift?n=500")
        assert resp.json()["overall_drift_detected"] is True


class TestMonitoringDriftErrorHandling:
    def test_drift_check_failure_returns_generic_500(self, client, mocker):
        """Hardening: a failure inside drift computation must not leak
        internal details either."""
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame({"a": [1]}))
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            side_effect=Exception("reference_stats.json not found at /app/models/..."),
        )
        resp = client.get("/monitoring/drift?n=500")
        assert resp.status_code == 500
        body = resp.json()
        assert "/app/" not in body.get("detail", "")
        assert "reference_stats.json" not in body.get("detail", "")

    def test_fetch_logs_failure_does_not_crash_endpoint(self, client, mocker):
        """db.fetch_recent_logs itself never raises (returns empty DataFrame
        on failure, per Phase 7 design) — but confirm the endpoint still
        works correctly when it receives that empty DataFrame."""
        mocker.patch("src.api.main.db.fetch_recent_logs", return_value=pd.DataFrame())
        mocker.patch(
            "src.api.main.drift.run_drift_report",
            return_value={"status": "insufficient_data", "n_samples": 0,
                          "message": "...", "min_required": 30},
        )
        resp = client.get("/monitoring/drift?n=30")
        assert resp.status_code == 200
        assert resp.json()["status"] == "insufficient_data"
