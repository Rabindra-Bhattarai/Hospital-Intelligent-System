"""
Automation pipeline: raw → clean → KPI → retrain → predict.
Runs a hash check on raw CSVs; if anything changed it re-runs the full chain.

Run:  python automation/refresh_pipeline.py
"""
import os, sys, hashlib, json, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

LOG_PATH   = config.LOG_PATH
HASH_PATH  = os.path.join(config.AUTO_DIR, "raw_hashes.json")

RAW_FILES = [
    os.path.join(config.RAW_DATA_DIR, "admissions.csv"),
    os.path.join(config.RAW_DATA_DIR, "bed_occupancy_daily.csv"),
    os.path.join(config.RAW_DATA_DIR, "patients.csv"),
]


# ── Utilities ───────────────────────────────────────────────────────────────

def _log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs(config.AUTO_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_saved_hashes() -> dict:
    if not os.path.exists(HASH_PATH):
        return {}
    with open(HASH_PATH, "r") as f:
        return json.load(f)


def _save_hashes(hashes: dict):
    os.makedirs(config.AUTO_DIR, exist_ok=True)
    with open(HASH_PATH, "w") as f:
        json.dump(hashes, f, indent=2)


def _raw_files_changed() -> bool:
    saved   = _load_saved_hashes()
    current = {p: _file_hash(p) for p in RAW_FILES}
    return any(current.get(p) != saved.get(p) for p in RAW_FILES), current


# ── Pipeline steps ──────────────────────────────────────────────────────────

def step_copy_data():
    """Copy latest raw CSVs into PatientSystem/data/."""
    _log("STEP 1 — copying raw CSVs into PatientSystem/data/ …")
    for src in RAW_FILES:
        if os.path.exists(src):
            dst = os.path.join(config.DATA_DIR, os.path.basename(src))
            shutil.copy2(src, dst)
            _log(f"  copied {os.path.basename(src)}")
        else:
            _log(f"  WARNING: {src} not found — skipping")


def step_retrain():
    """Retrain ML models."""
    _log("STEP 2 — retraining ML models …")
    from src.train_models import train_and_save
    metrics = train_and_save(verbose=False)
    for k, v in metrics.items():
        _log(f"  {k.upper()} → MAE={v['mae']:.3f}  R²={v['r2']:.3f}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    _log("=" * 60)
    _log("Refresh pipeline started")

    changed, current_hashes = _raw_files_changed()

    if not changed and os.path.exists(config.LOS_MODEL_PATH):
        _log("No raw-file changes detected and models exist — nothing to do.")
        _log("=" * 60)
        return

    reason = "raw files changed" if changed else "models missing"
    _log(f"Running full pipeline ({reason}) …")

    step_copy_data()
    step_retrain()

    _save_hashes(current_hashes)
    _log("Hashes updated.")
    _log("Pipeline completed successfully.")
    _log("=" * 60)


if __name__ == "__main__":
    run()
