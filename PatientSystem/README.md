# Hospital Intelligence System

Companion software for the Bagmati Region Hospital Operations Tableau thesis.  
Synthetic dataset — 5 hospitals, 2021–2024.

---

## Quick start

```bash
pip install -r requirements.txt
cp ../.env.example ../.env    # fill in Mongo URI, Google OAuth creds, admin username/password
python src/train_models.py    # train RF models (first-time setup, ~30 s)
streamlit run app.py          # launch the app
```

Auth data lives in MongoDB (see `MONGO_URI`/`MONGO_DB` in `.env`), not a local file.
On first run, if no admin account exists yet, one is seeded from `ADMIN_USERNAME`/
`ADMIN_PASSWORD` in `.env` — leave `ADMIN_PASSWORD` blank and no admin is created.

> Admins cannot be created via the UI — only seeded via `.env`, or inserted directly
> into the `users` collection.

---

## Folder structure

```
PatientSystem/
├── app.py                    # Streamlit entry point
├── config.py                 # paths, thresholds, env-driven config
├── requirements.txt
├── data/                     # admissions.csv  bed_occupancy_daily.csv  patients.csv
├── models/                   # trained .joblib artefacts
├── src/
│   ├── auth.py               # MongoDB auth, salted SHA-256 password hashing
│   ├── train_models.py       # RF training for LOS + Cost
│   ├── predict.py            # inference → ranges
│   ├── decision_engine.py    # threshold rules → alerts
│   └── data_loader.py        # cached CSV loading
└── automation/
    └── refresh_pipeline.py   # raw→retrain pipeline with hash check

.env                          # local secrets (gitignored) — see .env.example
```

---

## Running the automation pipeline

```bash
python automation/refresh_pipeline.py
```

Re-copies raw CSVs from `../Raw_Data/`, retrains models, and logs a timestamped
line to `automation/refresh.log`.  Uses SHA-256 hash checks to skip reruns when
nothing has changed.

---

## Ethical design

This system was built with the following ethical principles:

1. **No individual mortality/readmission predictions to patients.**  
   The readmission model has precision ≈ 0.08 (near random). Showing it to
   patients would cause unwarranted anxiety without clinical value.

2. **Patient predictions are always RANGES, not point values.**  
   Every estimate is shown as `point ± MAE` with a permanent disclaimer:
   *"Not a diagnosis, prediction of your outcome, or final bill. Always consult a doctor."*

3. **No patient health data is stored.**  
   The only persisted account data is `username`, `password_hash`, and `role`.
   No names, diagnoses, age, or personal information are saved.

4. **Passwords are always hashed.**  
   SHA-256 with a per-user random salt. Plaintext passwords are never stored or logged.

5. **Admin and patient modes are fully decoupled.**  
   Neither mode can escalate to the other's permissions.  
   Patients cannot self-register as admin.

6. **Data is synthetic.**  
   All hospital records are generated synthetic data for academic research purposes.
   No real patient data is used.

---

## Model accuracy

| Model             | MAE           | R²    | Used for                |
|-------------------|---------------|-------|-------------------------|
| LOS (RF)          | ~6.9 days     | ~0.23 | Patient stay planner     |
| Cost (RF)         | ~NPR 9,200    | ~0.55 | Patient cost planner     |
| Bed Occupancy (RF)| ~7%           | ~0.55 | Admin surge forecast     |
| Readmission       | weak (P≈0.08) | —     | Admin analytics only     |

Actual live MAE values are displayed in the Admin → Model Performance tab after training.

---

## Technology stack

- **Frontend:** Streamlit (single-tier)
- **ML:** scikit-learn RandomForestRegressor + joblib
- **Auth:** MongoDB + salted SHA-256 hashing (stdlib `hashlib`)
- **Visualisation:** Plotly Express / Graph Objects
- **Data:** pandas, numpy
