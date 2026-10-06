"""
Drift detection: compares live logged predictions against the training
reference distribution using KS test (numeric) and PSI (numeric + categorical).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MIN_SAMPLES_FOR_DRIFT = 30

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_reference_stats(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _psi_from_counts(expected_pct: np.ndarray, actual_pct: np.ndarray, eps: float = 1e-4) -> float:
    """Core PSI formula: sum((actual% - expected%) * ln(actual% / expected%)) per bin."""
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compute_numeric_drift(col: str, live_values: np.ndarray, ref_stats: dict) -> dict:
    ref = ref_stats["numeric"][col]

    # KS test: use the REAL training values directly (not a binned reconstruction —
    # binning with only 10 bins was shown to produce false-positive drift signals)
    ref_sample = np.array(ref["raw_values"])
    ks_stat, ks_pvalue = ks_2samp(ref_sample, live_values)

    # PSI: binning is appropriate here (PSI is defined as a binned comparison)
    bin_edges = np.array(ref["bin_edges"])
    ref_counts = np.array(ref["bin_counts"], dtype=float)
    ref_pct = ref_counts / ref_counts.sum()
    live_counts, _ = np.histogram(live_values, bins=bin_edges)
    live_pct = live_counts / live_counts.sum() if live_counts.sum() > 0 else live_counts.astype(float)
    psi = _psi_from_counts(ref_pct, live_pct)

    # Require BOTH KS and PSI to agree before flagging drift. Verified empirically
    # (5-seed test on known-no-drift samples) that PSI alone is noisy at small
    # sample sizes (n~100) and produces false positives; KS alone can also be
    # sensitive to small shifts that aren't practically meaningful. Requiring
    # agreement between two independent tests reduces false alarms.
    return {
        "feature": col,
        "type": "numeric",
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "psi": psi,
        "drift_detected": bool(ks_pvalue < 0.05 and psi > 0.2),
        "live_mean": float(np.mean(live_values)),
        "reference_mean": ref["mean"],
    }


def compute_categorical_drift(col: str, live_series: pd.Series, ref_stats: dict) -> dict:
    ref_dist = ref_stats["categorical"][col]
    categories = list(ref_dist.keys())

    ref_pct = np.array([ref_dist[c] for c in categories])
    live_counts = live_series.value_counts(normalize=True)
    live_pct = np.array([live_counts.get(c, 0.0) for c in categories])

    psi = _psi_from_counts(ref_pct, live_pct)

    # Categorical has only PSI (no KS equivalent for categories), so we keep
    # the standard PSI threshold alone here.
    return {
        "feature": col,
        "type": "categorical",
        "psi": psi,
        "drift_detected": bool(psi > 0.2),
        "live_distribution": {c: float(live_counts.get(c, 0.0)) for c in categories},
        "reference_distribution": ref_dist,
    }


def run_drift_report(live_df: pd.DataFrame, reference_stats_path: str) -> dict:
    """
    Main entry point. Returns a dict report, or an 'insufficient_data' status
    if live_df has fewer than MIN_SAMPLES_FOR_DRIFT rows.
    """
    n_samples = len(live_df)
    if n_samples < MIN_SAMPLES_FOR_DRIFT:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_SAMPLES_FOR_DRIFT} logged predictions for a "
                       f"reliable drift check, found {n_samples}.",
            "n_samples": n_samples,
            "min_required": MIN_SAMPLES_FOR_DRIFT,
        }

    ref_stats = load_reference_stats(reference_stats_path)

    results = []
    for col in NUMERIC_COLS:
        live_values = pd.to_numeric(live_df[col], errors="coerce").dropna().values
        if len(live_values) > 0:
            results.append(compute_numeric_drift(col, live_values, ref_stats))

    for col in CATEGORICAL_COLS:
        if col in live_df.columns:
            results.append(compute_categorical_drift(col, live_df[col], ref_stats))

    any_drift = any(r["drift_detected"] for r in results)

    return {
        "status": "ok",
        "n_samples": n_samples,
        "overall_drift_detected": any_drift,
        "features": results,
    }
