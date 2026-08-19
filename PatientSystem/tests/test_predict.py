"""Unit tests for src/predict.py.

The range math (_encode_row, predict_los, predict_cost) is tested against a
fake model + real fitted encoders, monkeypatched in place of the actual
trained artefacts on disk. This keeps the tests deterministic and independent
of whatever the currently-trained models happen to predict, and means they
don't require running train_models.py first.
"""
import os
import sys

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.predict as predict
import config


class _FakeModel:
    """Stands in for a fitted RandomForestRegressor: always predicts a fixed value."""

    def __init__(self, value: float):
        self._value = value

    def predict(self, X):
        return np.array([self._value] * len(X))


def _fitted_encoders():
    severity_enc = LabelEncoder().fit(config.SEVERITY_OPTIONS)
    admission_enc = LabelEncoder().fit(config.ADMISSION_TYPE_OPTIONS)
    department_enc = LabelEncoder().fit(config.DEPARTMENT_OPTIONS)
    return {
        "severity": severity_enc,
        "admission_type": admission_enc,
        "department_name": department_enc,
    }


# ── _encode_row ───────────────────────────────────────────────────────────────

def test_encode_row_column_order_and_names():
    encoders = _fitted_encoders()
    row = predict._encode_row(
        age=40, severity="Mild", admission_type="Elective",
        has_chronic=True, department=config.DEPARTMENT_OPTIONS[0],
        encoders=encoders,
    )
    assert list(row.columns) == [
        "age", "has_chronic_condition", "severity",
        "admission_type", "department_name",
    ]
    assert row["age"].iloc[0] == 40
    assert row["has_chronic_condition"].iloc[0] == 1  # bool -> int


def test_encode_row_has_chronic_false_encodes_as_zero():
    encoders = _fitted_encoders()
    row = predict._encode_row(
        age=25, severity="Mild", admission_type="Elective",
        has_chronic=False, department=config.DEPARTMENT_OPTIONS[0],
        encoders=encoders,
    )
    assert row["has_chronic_condition"].iloc[0] == 0


def test_encode_row_unknown_category_falls_back_instead_of_raising():
    encoders = _fitted_encoders()
    # A department that was never fitted must not raise — it should fall
    # back to the encoder's first known class rather than crash the app.
    row = predict._encode_row(
        age=30, severity="Mild", admission_type="Elective",
        department="Some Department That Does Not Exist",
        has_chronic=False, encoders=encoders,
    )
    expected_fallback = encoders["department_name"].transform(
        [encoders["department_name"].classes_[0]]
    )[0]
    assert row["department_name"].iloc[0] == expected_fallback


# ── predict_los / predict_cost range math ──────────────────────────────────────

def test_predict_los_returns_mid_plus_minus_mae(monkeypatch):
    encoders = _fitted_encoders()
    monkeypatch.setattr(
        predict, "_los_artefacts",
        lambda: (_FakeModel(10.0), encoders, 2.0),
    )
    low, high = predict.predict_los(
        age=40, severity="Mild", admission_type="Elective",
        has_chronic=False, department=config.DEPARTMENT_OPTIONS[0],
    )
    assert (low, high) == (8.0, 12.0)


def test_predict_los_low_end_floors_at_half_a_day(monkeypatch):
    encoders = _fitted_encoders()
    # mid - mae = 3 - 10 = -7, which must be floored to 0.5, not go negative
    monkeypatch.setattr(
        predict, "_los_artefacts",
        lambda: (_FakeModel(3.0), encoders, 10.0),
    )
    low, high = predict.predict_los(
        age=40, severity="Mild", admission_type="Elective",
        has_chronic=False, department=config.DEPARTMENT_OPTIONS[0],
    )
    assert low == 0.5
    assert high == 13.0


def test_predict_cost_returns_mid_plus_minus_mae(monkeypatch):
    encoders = _fitted_encoders()
    monkeypatch.setattr(
        predict, "_cost_artefacts",
        lambda: (_FakeModel(50000.0), encoders, 15000.0),
    )
    low, high = predict.predict_cost(
        age=40, severity="Severe", admission_type="Emergency",
        has_chronic=True, department=config.DEPARTMENT_OPTIONS[0],
    )
    assert (low, high) == (35000.0, 65000.0)


def test_predict_cost_low_end_floors_at_zero(monkeypatch):
    encoders = _fitted_encoders()
    # mid - mae = 5000 - 20000 = -15000, must be floored to 0.0, never negative
    monkeypatch.setattr(
        predict, "_cost_artefacts",
        lambda: (_FakeModel(5000.0), encoders, 20000.0),
    )
    low, high = predict.predict_cost(
        age=40, severity="Mild", admission_type="Elective",
        has_chronic=False, department=config.DEPARTMENT_OPTIONS[0],
    )
    assert low == 0.0
    assert high == 25000.0


def test_predict_los_and_cost_are_independent_of_argument_order_stability():
    # Same inputs, called twice, must be perfectly reproducible (no hidden
    # randomness / mutable-default-argument bugs in the encoding path).
    encoders = _fitted_encoders()
    row_a = predict._encode_row(30, "Moderate", "Referral", True,
                                 config.DEPARTMENT_OPTIONS[1], encoders)
    row_b = predict._encode_row(30, "Moderate", "Referral", True,
                                 config.DEPARTMENT_OPTIONS[1], encoders)
    assert row_a.equals(row_b)


# ── models_ready ────────────────────────────────────────────────────────────

def test_models_ready_false_when_model_files_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOS_MODEL_PATH", str(tmp_path / "missing_los.joblib"))
    monkeypatch.setattr(config, "COST_MODEL_PATH", str(tmp_path / "missing_cost.joblib"))
    assert predict.models_ready() is False


def test_models_ready_true_when_both_model_files_exist(monkeypatch, tmp_path):
    los_path = tmp_path / "los.joblib"
    cost_path = tmp_path / "cost.joblib"
    los_path.write_bytes(b"placeholder")
    cost_path.write_bytes(b"placeholder")
    monkeypatch.setattr(config, "LOS_MODEL_PATH", str(los_path))
    monkeypatch.setattr(config, "COST_MODEL_PATH", str(cost_path))
    assert predict.models_ready() is True
