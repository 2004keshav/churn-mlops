FROM python:3.12-slim

WORKDIR /app

# System deps for building some Python packages (xgboost, scikit-learn wheels are usually prebuilt, but keep minimal build tools just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY src/ ./src/
COPY models/ ./models/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
