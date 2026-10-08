"""
Unit tests for src/api/schema.py — CustomerRecord Pydantic validation.
"""
import pytest
from pydantic import ValidationError

from src.api.schema import CustomerRecord


def valid_payload(**overrides):
    base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    base.update(overrides)
    return base


class TestValidPayloads:
    def test_minimal_valid_payload(self):
        record = CustomerRecord(**valid_payload())
        assert record.gender == "Female"
        assert record.tenure == 1

    def test_senior_citizen_one_accepted(self):
        record = CustomerRecord(**valid_payload(SeniorCitizen=1))
        assert record.SeniorCitizen == 1

    @pytest.mark.parametrize("method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    def test_all_payment_methods_accepted(self, method):
        record = CustomerRecord(**valid_payload(PaymentMethod=method))
        assert record.PaymentMethod == method

    @pytest.mark.parametrize("contract", ["Month-to-month", "One year", "Two year"])
    def test_all_contracts_accepted(self, contract):
        record = CustomerRecord(**valid_payload(Contract=contract))
        assert record.Contract == contract


class TestInvalidCategoricalValues:
    def test_invalid_gender_rejected(self):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(gender="Other"))

    def test_invalid_senior_citizen_value_rejected(self):
        """SeniorCitizen must be literal 0 or 1 — not 2, not 'Yes'."""
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(SeniorCitizen=2))

    def test_senior_citizen_string_yes_rejected(self):
        """Schema expects int 0/1, not pre-normalized 'Yes'/'No' strings —
        normalization to Yes/No happens downstream in clean.py, not here."""
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(SeniorCitizen="Yes"))

    def test_invalid_internet_service_rejected(self):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(InternetService="Satellite"))

    def test_invalid_payment_method_rejected(self):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(PaymentMethod="Cash"))

    def test_invalid_contract_rejected(self):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(Contract="Three year"))


class TestNumericBounds:
    @pytest.mark.parametrize("tenure", [0, 100])
    def test_tenure_boundary_values_accepted(self, tenure):
        record = CustomerRecord(**valid_payload(tenure=tenure))
        assert record.tenure == tenure

    @pytest.mark.parametrize("tenure", [-1, 101])
    def test_tenure_outside_bounds_rejected(self, tenure):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(tenure=tenure))

    @pytest.mark.parametrize("charge", [0, 500])
    def test_monthly_charges_boundary_values_accepted(self, charge):
        record = CustomerRecord(**valid_payload(MonthlyCharges=charge))
        assert record.MonthlyCharges == charge

    @pytest.mark.parametrize("charge", [-0.01, 500.01])
    def test_monthly_charges_outside_bounds_rejected(self, charge):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(MonthlyCharges=charge))

    @pytest.mark.parametrize("charge", [0, 20000])
    def test_total_charges_boundary_values_accepted(self, charge):
        record = CustomerRecord(**valid_payload(TotalCharges=charge))
        assert record.TotalCharges == charge

    @pytest.mark.parametrize("charge", [-0.01, 20000.01])
    def test_total_charges_outside_bounds_rejected(self, charge):
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(TotalCharges=charge))


class TestMissingRequiredFields:
    def test_missing_gender_rejected(self):
        payload = valid_payload()
        del payload["gender"]
        with pytest.raises(ValidationError):
            CustomerRecord(**payload)

    def test_missing_tenure_rejected(self):
        payload = valid_payload()
        del payload["tenure"]
        with pytest.raises(ValidationError):
            CustomerRecord(**payload)

    def test_missing_total_charges_rejected(self):
        """Note: schema requires TotalCharges as float — the real dataset's
        blank-string TotalCharges issue is handled upstream in clean.py,
        not accepted here as a missing/blank field."""
        payload = valid_payload()
        del payload["TotalCharges"]
        with pytest.raises(ValidationError):
            CustomerRecord(**payload)


class TestTotalChargesBlankStringRejectedAtSchemaLevel:
    def test_blank_string_total_charges_rejected(self):
        """The schema expects a float, so the raw ' ' blank-string value
        from the original dataset would be REJECTED here if sent directly
        to the API — this is why main.py/inference.py's cleaning must run
        BEFORE schema validation in the real pipeline, not after. The API
        contract requires callers to send numeric TotalCharges."""
        with pytest.raises(ValidationError):
            CustomerRecord(**valid_payload(TotalCharges=" "))
