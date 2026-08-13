"""Central configuration: paths, thresholds, default credentials.

Secrets are read from the environment, never hardcoded. For local dev,
copy .env.example to .env and fill in real values — it's loaded below
and is gitignored, so it never gets committed.
"""
import os

# ── Base paths ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
AVATARS_DIR = os.path.join(BASE_DIR, "static", "avatars")


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (no external dependency). Existing env vars win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

# ── MongoDB ─────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB  = os.environ.get("MONGO_DB", "hospital_intelligence")

# ── Google SSO (Google Cloud Console → APIs & Services → Credentials) ───────
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
AUTO_DIR   = os.path.join(BASE_DIR, "automation")
LOG_PATH   = os.path.join(AUTO_DIR, "refresh.log")

# ── Raw / upstream data (used by refresh pipeline) ─────────────────────────
RAW_DATA_DIR    = os.path.join(BASE_DIR, "..", "Raw_Data")
CLEANING_DIR    = os.path.join(BASE_DIR, "..", "Raw_data_outputs", "cleaning")
KPI_DIR         = os.path.join(BASE_DIR, "..", "Raw_data_outputs", "kpi")
ML_OUTPUTS_DIR  = os.path.join(BASE_DIR, "..", "Raw_data_outputs", "ml_models")

# ── Data files ──────────────────────────────────────────────────────────────
ADMISSIONS_CSV    = os.path.join(DATA_DIR, "admissions.csv")
OCCUPANCY_CSV     = os.path.join(DATA_DIR, "bed_occupancy_daily.csv")
PATIENTS_CSV      = os.path.join(DATA_DIR, "patients.csv")

# ── Model artefacts ─────────────────────────────────────────────────────────
LOS_MODEL_PATH      = os.path.join(MODELS_DIR, "los_model.joblib")
LOS_ENCODER_PATH    = os.path.join(MODELS_DIR, "los_encoder.joblib")
LOS_MAE_PATH        = os.path.join(MODELS_DIR, "los_mae.joblib")

COST_MODEL_PATH     = os.path.join(MODELS_DIR, "cost_model.joblib")
COST_ENCODER_PATH   = os.path.join(MODELS_DIR, "cost_encoder.joblib")
COST_MAE_PATH       = os.path.join(MODELS_DIR, "cost_mae.joblib")

# ── WHO / operational thresholds ───────────────────────────────────────────
WHO_OCCUPANCY_THRESHOLD   = 0.85   # surge above this
WARN_OCCUPANCY_THRESHOLD  = 0.75   # warning band
OVERTIME_THRESHOLD        = 0.20   # >20 % of shifts are overtime → breach
WHO_LOS_DAYS              = 7      # WHO benchmark LOS

# ── Default admin seed (only used the very first time the DB is empty) ──────
DEFAULT_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "bkt_his_admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
DEFAULT_ADMIN_ROLE     = "admin"

# ── Categorical feature choices (used by patient planner UI) ───────────────
SEVERITY_OPTIONS        = ["Mild", "Moderate", "Severe", "Critical"]
ADMISSION_TYPE_OPTIONS  = ["Walk-in", "Elective", "Emergency", "Referral"]
DEPARTMENT_OPTIONS = [
    "Burn Unit", "Cardiology", "Dermatology", "Dialysis Unit", "ENT",
    "Emergency", "Gastroenterology", "General Medicine", "General Surgery",
    "HDU", "ICU", "Maternity & Obstetrics", "Neonatal ICU", "Nephrology",
    "Neurology", "Oncology", "Ophthalmology", "Orthopedics", "Pediatrics",
    "Psychiatry", "Pulmonology", "Urology",
]

# ── Hospital registry — loaded from hospitals.csv ───────────────────────────
HOSPITALS_CSV        = os.path.join(DATA_DIR, "hospitals.csv")
DEPARTMENTS_CSV      = os.path.join(DATA_DIR, "departments.csv")
PATIENT_FLOW_CSV     = os.path.join(DATA_DIR, "patient_flow_events.csv")
STAFF_CSV            = os.path.join(DATA_DIR, "staff.csv")
WARDS_BEDS_CSV       = os.path.join(DATA_DIR, "wards_beds.csv")

def _load_hospital_registry():
    import csv
    names, locations, beds, levels, types_ = {}, {}, {}, {}, {}
    try:
        with open(HOSPITALS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hid = row["hospital_id"]
                names[hid]     = row["hospital_name"]
                locations[hid] = row["district"]
                beds[hid]      = int(row["total_beds"])
                levels[hid]    = row["level"]
                types_[hid]    = row["hospital_type"]
    except Exception:
        pass
    return names, locations, beds, levels, types_

HOSPITAL_NAMES, HOSPITAL_LOCATIONS, HOSPITAL_BEDS, HOSPITAL_LEVELS, HOSPITAL_TYPES = (
    _load_hospital_registry()
)

# ── Estimate disclaimer ─────────────────────────────────────────────────────
ESTIMATE_DISCLAIMER = (
    "These figures are a rough guide to help you plan — not a diagnosis, "
    "a guarantee, or a final bill. Your actual stay and cost may differ. "
    "Always talk to your doctor for exact advice."
)
