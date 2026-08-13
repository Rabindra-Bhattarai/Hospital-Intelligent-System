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

## Running with Docker

From the repo root, with `.env` filled in (see above):

```bash
docker compose up --build
```

Starts the app (`localhost:8501`) and a MongoDB container together — no local
Mongo install needed. `docker-compose.yml` overrides `MONGO_URI` to point at
the `mongo` service; every other variable comes from `.env`.

To build/run just the app image against an external Mongo:

```bash
cd PatientSystem
docker build -t his-app .
docker run -p 8501:8501 --env-file ../.env -e MONGO_URI=mongodb://host.docker.internal:27017/ his-app
```

Known limitation: patient avatars are written to the container's local
filesystem (`static/avatars/`), so they don't persist across redeploys or
survive a multi-replica setup. Fine for a single-instance demo; would need
object storage (S3-compatible, etc.) for real production use.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR touching `PatientSystem/`:
compiles all source, spins up a real MongoDB service container, and runs the
`AppTest`-based smoke tests in `PatientSystem/tests/` (landing page, login,
and — with credentials seeded — the full admin dashboard, asserting each
renders with no exception). A separate job does a Docker build-only check so
a broken `Dockerfile` fails CI too.

`.github/workflows/docker-publish.yml` builds and pushes the image to
`ghcr.io/<owner>/<repo>:latest` on every push to `main`. No secrets to
configure — it authenticates with the automatically-provided
`GITHUB_TOKEN`. Pull it with:

```bash
docker pull ghcr.io/rabindra-bhattarai/thesis:latest
```

(The package may need to be set to public in the repo's Packages settings
the first time, or you'll need to `docker login ghcr.io` first.)

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
