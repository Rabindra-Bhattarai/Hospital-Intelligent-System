"""
Train Random Forest models for LOS and Cost prediction.
Saves model + encoder + MAE to models/ via joblib.

Run:  python src/train_models.py
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Feature list ────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["severity", "admission_type", "department_name"]
NUMERIC_FEATURES     = ["age", "has_chronic_condition"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS              = {"los": "length_of_stay_days", "cost": "total_bill_npr"}


def _prepare_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Encode categoricals and return (X, encoders).
    If fit=False, use provided encoders (for inference).
    """
    X = df[ALL_FEATURES].copy()

    # Boolean → int
    X["has_chronic_condition"] = X["has_chronic_condition"].astype(int)

    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_FEATURES:
            le = encoders[col]
            known = set(le.classes_)
            X[col] = X[col].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            X[col] = le.transform(X[col])

    return X, encoders


def train_and_save(verbose: bool = True) -> dict:
    """Train both models, print metrics, save artefacts. Returns metrics dict."""
    # Merge admissions + patients
    print("[train] Loading data …")
    adm = pd.read_csv(config.ADMISSIONS_CSV)
    pat = pd.read_csv(config.PATIENTS_CSV)
    df  = adm.merge(pat, on="patient_id", how="left")

    # Drop rows with missing target or features
    df = df.dropna(subset=ALL_FEATURES + list(TARGETS.values()))

    if verbose:
        print(f"[train] Merged dataset: {len(df):,} rows, {len(ALL_FEATURES)} features")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    results = {}

    for key, target_col in TARGETS.items():
        print(f"\n[train] -- Training {key.upper()} model (target: {target_col}) --")
        X, encoders = _prepare_features(df, fit=True)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)

        if verbose:
            print(f"  MAE  : {mae:.4f}")
            print(f"  R2   : {r2:.4f}")

        # Save artefacts
        if key == "los":
            joblib.dump(rf,       config.LOS_MODEL_PATH)
            joblib.dump(encoders, config.LOS_ENCODER_PATH)
            joblib.dump(mae,      config.LOS_MAE_PATH)
        else:
            joblib.dump(rf,       config.COST_MODEL_PATH)
            joblib.dump(encoders, config.COST_ENCODER_PATH)
            joblib.dump(mae,      config.COST_MAE_PATH)

        results[key] = {"mae": mae, "r2": r2}

    print("\n[train] All models saved to", config.MODELS_DIR)
    return results


if __name__ == "__main__":
    metrics = train_and_save(verbose=True)
    print("\n[train] Summary:")
    for k, v in metrics.items():
        print(f"  {k.upper():5s}  MAE={v['mae']:.3f}  R2={v['r2']:.3f}")
