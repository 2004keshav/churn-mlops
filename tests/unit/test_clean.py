"""
Unit tests for src/data/clean.py.
No external systems involved — pure pandas transformations.
"""
import pandas as pd
import pytest

from src.data.clean import (
    fix_total_charges,
    standardize_senior_citizen,
    drop_exact_duplicates,
    encode_target,
    clean_data,
)


# ---------------------------------------------------------------------------
# fix_total_charges
# ---------------------------------------------------------------------------
class TestFixTotalCharges:
    def test_blank_string_filled_with_zero(self):
        """The real Phase 1 bug: TotalCharges stored as ' ' (single space)
        for 11 rows in the original dataset, not NaN or empty string."""
        df = pd.DataFrame({"TotalCharges": [" ", "29.85", "1889.5"]})
        result = fix_total_charges(df)
        assert result["TotalCharges"].iloc[0] == 0.0
        assert result["TotalCharges"].dtype == float

    def test_numeric_strings_converted_correctly(self):
        df = pd.DataFrame({"TotalCharges": ["29.85", "1889.5", "108.15"]})
        result = fix_total_charges(df)
        assert result["TotalCharges"].tolist() == [29.85, 1889.5, 108.15]

    def test_already_numeric_passthrough(self):
        df = pd.DataFrame({"TotalCharges": [29.85, 1889.5]})
        result = fix_total_charges(df)
        assert result["TotalCharges"].tolist() == [29.85, 1889.5]

    def test_does_not_mutate_input(self):
        """fix_total_charges uses df.copy() — original df must be untouched."""
        df = pd.DataFrame({"TotalCharges": [" "]})
        fix_total_charges(df)
        assert df["TotalCharges"].iloc[0] == " "

    def test_multiple_blanks(self):
        """Mirrors the real dataset: 11 blank rows among many valid ones."""
        df = pd.DataFrame({"TotalCharges": [" "] * 11 + ["100.0"] * 5})
        result = fix_total_charges(df)
        assert (result["TotalCharges"].iloc[:11] == 0.0).all()
        assert result["TotalCharges"].isna().sum() == 0


# ---------------------------------------------------------------------------
# standardize_senior_citizen
# ---------------------------------------------------------------------------
class TestStandardizeSeniorCitizen:
    def test_zero_maps_to_no(self):
        df = pd.DataFrame({"SeniorCitizen": [0]})
        result = standardize_senior_citizen(df)
        assert result["SeniorCitizen"].iloc[0] == "No"

    def test_one_maps_to_yes(self):
        df = pd.DataFrame({"SeniorCitizen": [1]})
        result = standardize_senior_citizen(df)
        assert result["SeniorCitizen"].iloc[0] == "Yes"

    def test_mixed_values(self):
        df = pd.DataFrame({"SeniorCitizen": [0, 1, 0, 1]})
        result = standardize_senior_citizen(df)
        assert result["SeniorCitizen"].tolist() == ["No", "Yes", "No", "Yes"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"SeniorCitizen": [0]})
        standardize_senior_citizen(df)
        assert df["SeniorCitizen"].iloc[0] == 0


# ---------------------------------------------------------------------------
# drop_exact_duplicates
# ---------------------------------------------------------------------------
class TestDropExactDuplicates:
    def test_removes_exact_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = drop_exact_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = drop_exact_duplicates(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------
class TestEncodeTarget:
    def test_yes_no_mapped_to_int(self):
        df = pd.DataFrame({"Churn": ["Yes", "No", "Yes"]})
        result = encode_target(df)
        assert result["Churn"].tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------
# clean_data (full pipeline)
# ---------------------------------------------------------------------------
class TestCleanData:
    def test_full_pipeline_on_messy_sample(self):
        """Mirrors real dataset messiness: blank TotalCharges, numeric
        SeniorCitizen, string Churn, one duplicate row."""
        df = pd.DataFrame({
            "TotalCharges": [" ", "100.5", "100.5"],
            "SeniorCitizen": [0, 1, 1],
            "Churn": ["No", "Yes", "Yes"],
        })
        result = clean_data(df)
        # duplicate row (index 1 and 2 identical) should be dropped
        assert len(result) == 2
        assert result["TotalCharges"].isna().sum() == 0
        assert set(result["SeniorCitizen"].unique()) <= {"No", "Yes"}
        assert set(result["Churn"].unique()) <= {0, 1}
