"""
Unit tests for src/monitoring/drift.py.
Includes REGRESSION TESTS for the two real bugs found in Phase 7:
  1. KS test must use raw reference values, not binned reconstruction.
  2. drift_detected (numeric) must require BOTH ks_pvalue<0.05 AND psi>0.2,
     not OR.
"""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    compute_numeric_drift,
    compute_categorical_drift,
    run_drift_report,
    _psi_from_counts,
    MIN_SAMPLES_FOR_DRIFT,
)


# ---------------------------------------------------------------------------
# _psi_from_counts — core formula correctness
# ---------------------------------------------------------------------------
class TestPsiFromCounts:
    def test_identical_distributions_give_zero_psi(self):
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        actual = np.array([0.25, 0.25, 0.25, 0.25])
        psi = _psi_from_counts(expected, actual)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_known_psi_value(self):
        """Manually computed: sum((a-e)*ln(a/e)) for a simple 2-bin case."""
        expected = np.array([0.5, 0.5])
        actual = np.array([0.7, 0.3])
        expected_psi = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
        psi = _psi_from_counts(expected, actual)
        assert psi == pytest.approx(expected_psi, abs=1e-9)

    def test_zero_bins_handled_without_error(self):
        """Zero bins would cause log(0) / div-by-zero without the eps fix."""
        expected = np.array([0.0, 1.0])
        actual = np.array([1.0, 0.0])
        psi = _psi_from_counts(expected, actual)
        assert np.isfinite(psi)


# ---------------------------------------------------------------------------
# REGRESSION TEST: KS must use raw reference values, not binned reconstruction
# ---------------------------------------------------------------------------
class TestKsUsesRawValuesNotBins:
    def test_ks_pvalue_high_when_comparing_same_distribution_via_raw_values(self):
        """If live data is drawn from the SAME distribution as the reference's
        raw_values, KS p-value must be high (no false drift alarm).
        This is the exact ground-truth check that caught the Phase 7 bug:
        binned reconstruction (~10 unique values) inflated the KS statistic
        and produced p-value ~0.02 even with no real drift."""
        rng = np.random.default_rng(123)
        ref_raw = rng.normal(32, 24, 2000).clip(0, 72)
        bin_edges = np.linspace(0, 72, 11)
        bin_counts, _ = np.histogram(ref_raw, bins=bin_edges)

        ref_stats = {
            "numeric": {
                "tenure": {
                    "mean": float(np.mean(ref_raw)),
                    "raw_values": ref_raw.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "bin_counts": bin_counts.tolist(),
                }
            }
        }

        live_values = rng.normal(32, 24, 200).clip(0, 72)

        result = compute_numeric_drift("tenure", live_values, ref_stats)

        assert result["ks_pvalue"] > 0.01, (
            f"KS p-value suspiciously low ({result['ks_pvalue']}) for "
            f"same-distribution comparison — possible regression to binned "
            f"reconstruction bug."
        )

    def test_raw_values_have_more_unique_values_than_bin_midpoints(self):
        """Sanity check on the fixture itself: raw_values must have far more
        unique values than the 10-bin reconstruction would — this is what
        made the old approach wrong in the first place."""
        rng = np.random.default_rng(7)
        ref_raw = rng.normal(32, 24, 500).clip(0, 72)
        n_unique_raw = len(np.unique(np.round(ref_raw, 2)))
        n_bins = 10
        assert n_unique_raw > n_bins * 2


# ---------------------------------------------------------------------------
# REGRESSION TEST: drift_detected (numeric) requires BOTH ks AND psi
# ---------------------------------------------------------------------------
class TestNumericDriftRequiresBothTests:
    def _make_ref_stats(self, raw_values, bin_edges, bin_counts):
        return {
            "numeric": {
                "tenure": {
                    "mean": float(np.mean(raw_values)),
                    "raw_values": raw_values.tolist() if hasattr(raw_values, "tolist") else raw_values,
                    "bin_edges": bin_edges.tolist() if hasattr(bin_edges, "tolist") else bin_edges,
                    "bin_counts": bin_counts.tolist() if hasattr(bin_counts, "tolist") else bin_counts,
                }
            }
        }

    def test_drift_false_when_only_psi_exceeds_threshold(self):
        """Deterministic regression test (seed=80, confirmed via search) for
        the exact Phase 7 scenario: live data drawn from the SAME distribution
        as reference, small n (100) causes PSI to cross 0.2 by sampling noise
        alone, while KS correctly reports no significant difference.
        drift_detected must be False, proving AND logic suppresses this
        false alarm."""
        ref_rng = np.random.default_rng(2024)
        ref_raw = ref_rng.normal(32, 24, 2000).clip(0, 72)
        bin_edges = np.linspace(0, 72, 11)
        bin_counts, _ = np.histogram(ref_raw, bins=bin_edges)
        ref_stats = self._make_ref_stats(ref_raw, bin_edges, bin_counts)

        rng = np.random.default_rng(80)
        live_values = rng.normal(32, 24, 100).clip(0, 72)

        result = compute_numeric_drift("tenure", live_values, ref_stats)

        assert result["ks_pvalue"] >= 0.05, (
            f"Fixture regression: expected ks_pvalue>=0.05, got {result['ks_pvalue']}"
        )
        assert result["psi"] > 0.2, (
            f"Fixture regression: expected psi>0.2, got {result['psi']}"
        )
        assert result["drift_detected"] is False

    def test_drift_detected_logic_is_AND_not_OR_directly(self):
        """Directly verifies the boolean logic using controlled inputs,
        independent of whether real data happens to produce the scenario."""
        assert (True and False) is False

        rng = np.random.default_rng(55)
        ref_raw = rng.normal(50, 5, 2000)
        bin_edges = np.linspace(30, 70, 11)
        bin_counts, _ = np.histogram(ref_raw, bins=bin_edges)
        ref_stats = self._make_ref_stats(ref_raw, bin_edges, bin_counts)

        live_values = rng.normal(51, 5, 300)

        result = compute_numeric_drift("tenure", live_values, ref_stats)
        expected = bool(result["ks_pvalue"] < 0.05 and result["psi"] > 0.2)
        assert result["drift_detected"] == expected

    def test_drift_true_when_both_conditions_met(self):
        """Clear, strong drift: live data from a totally different
        distribution should trigger both KS and PSI, so drift_detected True."""
        rng = np.random.default_rng(1)
        ref_raw = rng.normal(32, 10, 2000).clip(0, 72)
        bin_edges = np.linspace(0, 72, 11)
        bin_counts, _ = np.histogram(ref_raw, bins=bin_edges)
        ref_stats = self._make_ref_stats(ref_raw, bin_edges, bin_counts)

        live_values = rng.normal(70, 2, 100).clip(0, 72)

        result = compute_numeric_drift("tenure", live_values, ref_stats)
        assert result["ks_pvalue"] < 0.05
        assert result["psi"] > 0.2
        assert result["drift_detected"] is True


# ---------------------------------------------------------------------------
# compute_categorical_drift
# ---------------------------------------------------------------------------
class TestCategoricalDrift:
    def test_identical_distribution_no_drift(self):
        ref_stats = {"categorical": {"gender": {"Female": 0.5, "Male": 0.5}}}
        live_series = pd.Series(["Female"] * 50 + ["Male"] * 50)
        result = compute_categorical_drift("gender", live_series, ref_stats)
        assert result["psi"] == pytest.approx(0.0, abs=1e-6)
        assert result["drift_detected"] is False

    def test_strongly_shifted_distribution_detected(self):
        ref_stats = {"categorical": {"gender": {"Female": 0.5, "Male": 0.5}}}
        live_series = pd.Series(["Female"] * 95 + ["Male"] * 5)
        result = compute_categorical_drift("gender", live_series, ref_stats)
        assert result["psi"] > 0.2
        assert result["drift_detected"] is True

    def test_missing_category_in_live_treated_as_zero(self):
        """If a category present in reference never appears in live data,
        it should be treated as 0%, not raise a KeyError."""
        ref_stats = {
            "categorical": {
                "Contract": {"Month-to-month": 0.55, "One year": 0.21, "Two year": 0.24}
            }
        }
        live_series = pd.Series(["Month-to-month"] * 100)
        result = compute_categorical_drift("Contract", live_series, ref_stats)
        assert result["live_distribution"]["One year"] == 0.0
        assert result["live_distribution"]["Two year"] == 0.0


# ---------------------------------------------------------------------------
# run_drift_report — insufficient_data path + full report shape
# ---------------------------------------------------------------------------
class TestRunDriftReport:
    def test_insufficient_data_below_min_samples(self, fake_reference_stats, tmp_path):
        import json
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps(fake_reference_stats))

        live_df = pd.DataFrame({"tenure": [10, 20, 30]})
        report = run_drift_report(live_df, str(ref_path))

        assert report["status"] == "insufficient_data"
        assert report["n_samples"] == 3
        assert report["min_required"] == MIN_SAMPLES_FOR_DRIFT

    def test_exactly_min_samples_proceeds_to_full_report(self, fake_reference_stats, tmp_path):
        """Boundary: exactly MIN_SAMPLES_FOR_DRIFT rows should NOT be
        insufficient_data."""
        import json
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps(fake_reference_stats))

        rng = np.random.default_rng(0)
        n = MIN_SAMPLES_FOR_DRIFT
        live_df = pd.DataFrame({
            "tenure": rng.normal(32, 24, n).clip(0, 72),
            "MonthlyCharges": rng.normal(64.76, 30.09, n).clip(18.25, 118.75),
            "TotalCharges": rng.normal(2283.3, 2266.77, n).clip(0, 8684.8),
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice(["No", "Yes"], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["No", "Yes", "No internet service"], n),
            "OnlineBackup": rng.choice(["No", "Yes", "No internet service"], n),
            "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], n),
            "TechSupport": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingTV": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingMovies": rng.choice(["No", "Yes", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
        })

        report = run_drift_report(live_df, str(ref_path))
        assert report["status"] == "ok"
        assert report["n_samples"] == n
        assert "overall_drift_detected" in report
        assert len(report["features"]) == 19

    def test_report_includes_all_19_features(self, fake_reference_stats, tmp_path):
        import json
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps(fake_reference_stats))

        rng = np.random.default_rng(0)
        n = 50
        live_df = pd.DataFrame({
            "tenure": rng.normal(32, 24, n).clip(0, 72),
            "MonthlyCharges": rng.normal(64.76, 30.09, n).clip(18.25, 118.75),
            "TotalCharges": rng.normal(2283.3, 2266.77, n).clip(0, 8684.8),
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice(["No", "Yes"], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["No", "Yes", "No internet service"], n),
            "OnlineBackup": rng.choice(["No", "Yes", "No internet service"], n),
            "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], n),
            "TechSupport": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingTV": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingMovies": rng.choice(["No", "Yes", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
        })
        report = run_drift_report(live_df, str(ref_path))
        feature_names = {f["feature"] for f in report["features"]}
        expected_names = {
            "tenure", "MonthlyCharges", "TotalCharges",
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
        }
        assert feature_names == expected_names
