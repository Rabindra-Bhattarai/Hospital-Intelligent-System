"""
Inference layer — loads saved models once (Streamlit cache), returns RANGES.
predict_los()  → (low_days, high_days)
predict_cost() → (low_npr,  high_npr)
"""
import os, sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Lazy-load helpers ───────────────────────────────────────────────────────

def _models_exist() -> bool:
    return (
        os.path.exists(config.LOS_MODEL_PATH)
        and os.path.exists(config.COST_MODEL_PATH)
    )


def _load(path):
    return joblib.load(path)


try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _los_artefacts():
        return (_load(config.LOS_MODEL_PATH),
                _load(config.LOS_ENCODER_PATH),
                _load(config.LOS_MAE_PATH))

    @st.cache_resource(show_spinner=False)
    def _cost_artefacts():
        return (_load(config.COST_MODEL_PATH),
                _load(config.COST_ENCODER_PATH),
                _load(config.COST_MAE_PATH))

except Exception:
    def _los_artefacts():
        return (_load(config.LOS_MODEL_PATH),
                _load(config.LOS_ENCODER_PATH),
                _load(config.LOS_MAE_PATH))

    def _cost_artefacts():
        return (_load(config.COST_MODEL_PATH),
                _load(config.COST_ENCODER_PATH),
                _load(config.COST_MAE_PATH))


# ── Feature encoding (mirrors train_models logic) ──────────────────────────

def _encode_row(age: int, severity: str, admission_type: str,
                has_chronic: bool, department: str, encoders: dict):
    """Return a single-row DataFrame with encoded features (preserves column names)."""
    import pandas as pd
    row = {
        "age":                   age,
        "has_chronic_condition": int(has_chronic),
        "severity":              severity,
        "admission_type":        admission_type,
        "department_name":       department,
    }
    categorical = ["severity", "admission_type", "department_name"]
    for col in categorical:
        le    = encoders[col]
        known = set(le.classes_)
        val   = row[col] if row[col] in known else le.classes_[0]
        row[col] = int(le.transform([val])[0])

    feature_order = ["age", "has_chronic_condition",
                     "severity", "admission_type", "department_name"]
    return pd.DataFrame([[row[f] for f in feature_order]], columns=feature_order)


# ── Public API ──────────────────────────────────────────────────────────────

def predict_los(age: int, severity: str, admission_type: str,
                has_chronic: bool, department: str) -> tuple[float, float]:
    """Return (low_days, high_days) as prediction ± MAE."""
    model, encoders, mae = _los_artefacts()
    X   = _encode_row(age, severity, admission_type, has_chronic, department, encoders)
    mid = float(model.predict(X)[0])
    return max(0.5, mid - mae), mid + mae


def predict_cost(age: int, severity: str, admission_type: str,
                 has_chronic: bool, department: str) -> tuple[float, float]:
    """Return (low_npr, high_npr) as prediction ± MAE."""
    model, encoders, mae = _cost_artefacts()
    X   = _encode_row(age, severity, admission_type, has_chronic, department, encoders)
    mid = float(model.predict(X)[0])
    return max(0.0, mid - mae), mid + mae


def models_ready() -> bool:
    """True if model files exist on disk."""
    return _models_exist()


def model_metrics() -> dict:
    """Live accuracy figures for the trained RF models, read from the
    joblib artefacts saved by train_models.py. R² is unavailable for
    models trained before the r2 artefact existed."""
    metrics = {}
    if os.path.exists(config.LOS_MODEL_PATH):
        model, _, mae = _los_artefacts()
        metrics["los"] = {
            "mae": float(mae),
            "r2":  float(_load(config.LOS_R2_PATH)) if os.path.exists(config.LOS_R2_PATH) else None,
            "accuracy_pct": float(_load(config.LOS_ACC_PATH)) if os.path.exists(config.LOS_ACC_PATH) else None,
            "n_estimators": model.n_estimators,
            "max_depth":    model.max_depth,
            "trained_at":   os.path.getmtime(config.LOS_MODEL_PATH),
        }
    if os.path.exists(config.COST_MODEL_PATH):
        model, _, mae = _cost_artefacts()
        metrics["cost"] = {
            "mae": float(mae),
            "r2":  float(_load(config.COST_R2_PATH)) if os.path.exists(config.COST_R2_PATH) else None,
            "accuracy_pct": float(_load(config.COST_ACC_PATH)) if os.path.exists(config.COST_ACC_PATH) else None,
            "n_estimators": model.n_estimators,
            "max_depth":    model.max_depth,
            "trained_at":   os.path.getmtime(config.COST_MODEL_PATH),
        }
    return metrics
