"""
Cached CSV loading. All reads go through here so Streamlit caches them once.
Returns cleaned DataFrames; never mutates the originals.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Raw loaders (no caching — called by cached wrappers below) ─────────────

def _load_admissions_raw() -> pd.DataFrame:
    df = pd.read_csv(config.ADMISSIONS_CSV)
    df["admission_date"]  = pd.to_datetime(df["admission_date"],  errors="coerce")
    df["discharge_date"]  = pd.to_datetime(df["discharge_date"],  errors="coerce")
    df["admission_month"] = df["admission_date"].dt.month
    df["admission_year"]  = df["admission_date"].dt.year
    return df


def _load_occupancy_raw() -> pd.DataFrame:
    df = pd.read_csv(config.OCCUPANCY_CSV)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    return df


def _load_occupancy_with_hospital_raw() -> pd.DataFrame:
    occ  = _load_occupancy_raw()
    try:
        dept = pd.read_csv(config.DEPARTMENTS_CSV)[["department_id", "hospital_id"]]
        occ  = occ.merge(dept, on="department_id", how="left")
    except Exception:
        occ["hospital_id"] = None
    return occ


def _load_flow_raw() -> pd.DataFrame:
    df = pd.read_csv(config.PATIENT_FLOW_CSV)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    return df


def _hospital_stats_raw() -> pd.DataFrame:
    """Pre-compute one row per hospital: avg_wait_triage, avg_wait_bed,
    avg_cost, recovery_rate, doctor_count, avg_doctor_exp."""
    adm   = _load_admissions_raw()
    flow  = _load_flow_raw()
    dept  = pd.read_csv(config.DEPARTMENTS_CSV)[["department_id", "hospital_id"]]
    hosp  = pd.read_csv(config.HOSPITALS_CSV)

    flow_adm = flow.merge(
        adm[["admission_id", "hospital_id", "department_id"]], on="admission_id", how="left"
    )
    triage = (flow_adm[flow_adm["event_type"] == "Triage"]
              .groupby("hospital_id")["wait_minutes"].mean().rename("avg_wait_triage"))
    bed_assign = (flow_adm[flow_adm["event_type"] == "Bed Assigned"]
                  .groupby("hospital_id")["wait_minutes"].mean().rename("avg_wait_bed"))

    adm_stats = adm.groupby("hospital_id").agg(
        avg_cost=("total_bill_npr", "mean"),
        recovery_rate=("discharge_outcome", lambda x: (x == "Recovered").mean()),
    )

    try:
        staff = pd.read_csv(config.STAFF_CSV)
        _doc_roles = {"Resident Doctor", "Consultant", "Medical Officer"}
        docs  = (staff[staff["role"].isin(_doc_roles)]
                 .groupby("hospital_id")
                 .agg(doctor_count=("staff_id", "count"),
                      avg_exp=("years_experience", "mean")))
    except Exception:
        docs = pd.DataFrame(
            columns=["hospital_id", "doctor_count", "avg_exp"]
        ).set_index("hospital_id")

    stats = (hosp.set_index("hospital_id")
             .join(triage).join(bed_assign).join(adm_stats).join(docs))
    return stats.reset_index()


# ── Department helpers (no Streamlit dependency) ───────────────────────────

def departments_by_hospital() -> dict:
    try:
        dept = pd.read_csv(config.DEPARTMENTS_CSV)
        return dept.groupby("hospital_id")["department_name"].apply(list).to_dict()
    except Exception:
        return {}


def hospitals_with_department(dept_name: str) -> list:
    try:
        dept = pd.read_csv(config.DEPARTMENTS_CSV)
        return dept[dept["department_name"] == dept_name]["hospital_id"].tolist()
    except Exception:
        return []


def merged_admissions_patients() -> pd.DataFrame:
    adm = load_admissions()
    pat = load_patients()
    return adm.merge(pat, on="patient_id", how="left")


def kpi_summary(adm: pd.DataFrame, occ: pd.DataFrame) -> dict:
    return {
        "total_admissions": len(adm),
        "avg_occupancy":    occ["occupancy_rate"].mean(),
        "avg_los":          adm["length_of_stay_days"].mean(),
        "mortality_rate":   (adm["discharge_outcome"] == "Died").mean(),
    }


# ── Thesis-wide model metrics (read-only; feeds the Tableau side) ──────────

def _metric_series(filename: str):
    path = os.path.join(config.ML_OUTPUTS_DIR, filename)
    return pd.read_csv(path, index_col="metric")["value"]


def _thesis_model_metrics_raw() -> dict:
    """Read-only summary of all 5 models trained for the Tableau thesis
    analysis (Raw_data_outputs/ml_models/) — separate from the 2 models
    (LOS, Cost) this app actually serves to patients. Regression MAE is
    normalized against the real mean so every model reports one comparable
    accuracy percentage."""
    out = {}

    try:
        los = _metric_series("los_model_metrics.csv")
        los_mean = pd.read_csv(
            os.path.join(config.CLEANING_DIR, "admissions_master.csv")
        )["length_of_stay_days"].mean()
        out["los"] = {
            "label": "Length of Stay",
            "accuracy_pct": max(0.0, 100 * (1 - los["MAE_days"] / los_mean)),
            "detail": f"MAE {los['MAE_days']:.1f}d · R² {los['R2_Score']:.2f}",
        }
    except Exception:
        pass

    try:
        occ = _metric_series("occupancy_model_metrics.csv")
        occ_mean = pd.read_csv(
            os.path.join(config.CLEANING_DIR, "bed_occupancy_master.csv")
        )["occupancy_rate"].mean()
        out["occupancy"] = {
            "label": "Bed Occupancy",
            "accuracy_pct": max(0.0, 100 * (1 - occ["MAE"] / occ_mean)),
            "detail": f"MAE {occ['MAE']:.1%} · R² {occ['R2_Score']:.2f}",
        }
    except Exception:
        pass

    try:
        disc = _metric_series("discharge_overall_metrics.csv")
        out["discharge"] = {
            "label": "Discharge Outcome",
            "accuracy_pct": disc["Overall_Accuracy"] * 100,
            "detail": "5-class outcome · skews to the majority class",
        }
    except Exception:
        pass

    try:
        ot = _metric_series("overtime_model_metrics.csv")
        out["overtime"] = {
            "label": "Staff Overtime",
            "accuracy_pct": ot["Accuracy"] * 100,
            "detail": f"Precision {ot['Precision']:.2f} · Recall {ot['Recall']:.2f}",
        }
    except Exception:
        pass

    try:
        rd = _metric_series("readmission_model_metrics.csv")
        out["readmission"] = {
            "label": "Readmission",
            "accuracy_pct": rd["Accuracy"] * 100,
            "detail": f"Precision {rd['Precision']:.2f} · near-random, imbalanced classes",
        }
    except Exception:
        pass

    return out


# ── Cached public API (single try/except wraps all cache decorators) ────────

try:
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def load_admissions() -> pd.DataFrame:
        return _load_admissions_raw()

    @st.cache_data(show_spinner=False)
    def load_patients() -> pd.DataFrame:
        return pd.read_csv(config.PATIENTS_CSV)

    @st.cache_data(show_spinner=False)
    def load_occupancy() -> pd.DataFrame:
        return _load_occupancy_raw()

    @st.cache_data(show_spinner=False)
    def load_occupancy_with_hospital() -> pd.DataFrame:
        return _load_occupancy_with_hospital_raw()

    @st.cache_data(show_spinner=False)
    def load_flow_events() -> pd.DataFrame:
        return _load_flow_raw()

    @st.cache_data(show_spinner=False)
    def hospital_stats() -> pd.DataFrame:
        return _hospital_stats_raw()

    @st.cache_data(show_spinner=False)
    def thesis_model_metrics() -> dict:
        return _thesis_model_metrics_raw()

except Exception:
    def load_admissions() -> pd.DataFrame:
        return _load_admissions_raw()

    def load_patients() -> pd.DataFrame:
        return pd.read_csv(config.PATIENTS_CSV)

    def load_occupancy() -> pd.DataFrame:
        return _load_occupancy_raw()

    def load_occupancy_with_hospital() -> pd.DataFrame:
        return _load_occupancy_with_hospital_raw()

    def load_flow_events() -> pd.DataFrame:
        return _load_flow_raw()

    def hospital_stats() -> pd.DataFrame:
        return _hospital_stats_raw()

    def thesis_model_metrics() -> dict:
        return _thesis_model_metrics_raw()
