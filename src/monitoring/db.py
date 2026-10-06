"""
Postgres logging layer for prediction monitoring.
Connection string comes from DATABASE_URL env var (set in Render dashboard for prod,
or set manually in Colab for local testing).
"""
import os
import json
import uuid
import logging
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import pandas as pd

logger = logging.getLogger(__name__)


def get_connection():
    """Create a new Postgres connection using DATABASE_URL env var."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def insert_prediction_log(
    raw_input: dict,
    prediction: int,
    churn_probability: float,
    model_alias: str,
    model_version: str,
    run_id: str,
    latency_ms: float = None,
) -> str:
    """
    Insert one prediction record into the predictions table.
    Returns the request_id (str) on success, or None on failure.
    Never raises — logging failure must not break the /predict response.
    """
    request_id = str(uuid.uuid4())
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions
                (request_id, timestamp, raw_input, prediction, churn_probability,
                 model_alias, model_version, run_id, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request_id,
                datetime.now(timezone.utc),
                json.dumps(raw_input),
                prediction,
                churn_probability,
                model_alias,
                model_version,
                run_id,
                latency_ms,
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return request_id
    except Exception as e:
        logger.error(f"Failed to log prediction to Postgres: {e}")
        return None


def fetch_recent_logs(n: int = 500) -> pd.DataFrame:
    """
    Fetch the N most recent prediction logs, newest first.
    Returns a DataFrame with raw_input expanded into separate columns.
    Returns an empty DataFrame (not an exception) if fetch fails.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """
            SELECT request_id, timestamp, raw_input, prediction, churn_probability,
                   model_alias, model_version, run_id, latency_ms
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (n,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Expand raw_input JSONB into separate columns
        raw_input_df = pd.json_normalize(df["raw_input"])
        df = pd.concat([df.drop(columns=["raw_input"]), raw_input_df], axis=1)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch logs from Postgres: {e}")
        return pd.DataFrame()


def count_logs() -> int:
    """Return total number of logged predictions. Returns 0 on failure."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to count logs: {e}")
        return 0
