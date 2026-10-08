"""
Unit tests for src/monitoring/db.py.
Mocks psycopg2.connect (the external system boundary) — no real Postgres
connection is made. Verifies the Phase 7 design guarantee: logging
functions must never raise, even when the DB is unreachable.
"""
import os
import pandas as pd
import pytest

from src.monitoring import db


@pytest.fixture
def mock_db_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@fake-host/fake-db")


class TestGetConnection:
    def test_raises_if_database_url_not_set(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            db.get_connection()

    def test_calls_psycopg2_connect_with_url(self, mocker, mock_db_env):
        mock_connect = mocker.patch("src.monitoring.db.psycopg2.connect")
        db.get_connection()
        mock_connect.assert_called_once_with("postgresql://fake:fake@fake-host/fake-db")


class TestInsertPredictionLog:
    def test_successful_insert_returns_request_id(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        result = db.insert_prediction_log(
            raw_input={"gender": "Female"},
            prediction=1,
            churn_probability=0.8,
            model_alias="champion",
            model_version="3",
            run_id="abc123",
            latency_ms=12.5,
        )

        assert result is not None
        assert isinstance(result, str)  # UUID string
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_never_raises_when_connect_fails(self, mocker, mock_db_env):
        """THE key design guarantee from Phase 7: a DB outage must never
        propagate as an exception out of insert_prediction_log."""
        mocker.patch(
            "src.monitoring.db.psycopg2.connect",
            side_effect=Exception("connection refused"),
        )
        result = db.insert_prediction_log(
            raw_input={"gender": "Female"},
            prediction=1,
            churn_probability=0.8,
            model_alias="champion",
            model_version="3",
            run_id="abc123",
        )
        assert result is None  # fails gracefully, no exception raised

    def test_never_raises_when_execute_fails(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute.side_effect = Exception("insert failed")
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        result = db.insert_prediction_log(
            raw_input={"gender": "Female"},
            prediction=1,
            churn_probability=0.8,
            model_alias="champion",
            model_version="3",
            run_id="abc123",
        )
        assert result is None

    def test_never_raises_when_database_url_missing(self, monkeypatch):
        """get_connection() raises RuntimeError when DATABASE_URL is unset —
        insert_prediction_log must catch this too, not just connect() failures."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = db.insert_prediction_log(
            raw_input={"gender": "Female"},
            prediction=1,
            churn_probability=0.8,
            model_alias="champion",
            model_version="3",
            run_id="abc123",
        )
        assert result is None


class TestFetchRecentLogs:
    def test_successful_fetch_expands_raw_input_jsonb(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "request_id": "req-1",
                "timestamp": "2026-10-08T00:00:00Z",
                "raw_input": {"gender": "Female", "tenure": 5},
                "prediction": 1,
                "churn_probability": 0.8,
                "model_alias": "champion",
                "model_version": "3",
                "run_id": "abc123",
                "latency_ms": 10.0,
            }
        ]
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        result = db.fetch_recent_logs(n=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "raw_input" not in result.columns  # expanded, not kept as-is
        assert "gender" in result.columns
        assert result["gender"].iloc[0] == "Female"
        assert "tenure" in result.columns

    def test_empty_result_returns_empty_dataframe(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        result = db.fetch_recent_logs(n=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_empty_dataframe_not_exception_on_failure(self, mocker, mock_db_env):
        """THE key design guarantee: a DB outage during /monitoring/drift
        must not crash the endpoint — must degrade to empty DataFrame."""
        mocker.patch(
            "src.monitoring.db.psycopg2.connect",
            side_effect=Exception("connection refused"),
        )
        result = db.fetch_recent_logs(n=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_respects_limit_parameter(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        db.fetch_recent_logs(n=250)
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == (250,)


class TestCountLogs:
    def test_successful_count(self, mocker, mock_db_env):
        mock_conn = mocker.MagicMock()
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value = mock_cursor
        mocker.patch("src.monitoring.db.psycopg2.connect", return_value=mock_conn)

        result = db.count_logs()
        assert result == 42

    def test_returns_zero_not_exception_on_failure(self, mocker, mock_db_env):
        mocker.patch(
            "src.monitoring.db.psycopg2.connect",
            side_effect=Exception("connection refused"),
        )
        result = db.count_logs()
        assert result == 0
