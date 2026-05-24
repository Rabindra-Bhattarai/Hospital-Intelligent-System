# Design and Development of a Data-Driven Hospital Dashboard for Optimizing Bed Occupancy, Patient Flow and Staff Allocation using Local Hospital Data

## Synthetic Relational Dataset, Documentation and Metadata Package

---

## 1. Overview

This package provides a **synthetic, relational, multi-table dataset of approximately 500,000 records** designed to support a thesis on building a **data-driven hospital operations dashboard** that optimizes the three areas named in the title: **bed occupancy, patient flow, and staff allocation**.

The data models a small **local hospital network** (5 hospitals) in the Bagmati region of Nepal over **four years (January 2021 to December 2024)**, structured the way a hospital information system would store it: separate, ID-linked tables for hospitals, departments, beds, staff, patients, admissions, and the daily operational signals a dashboard would visualize.

> **IMPORTANT — DATA NATURE**
> This dataset is **SYNTHETIC / SIMULATED**. It is **not** real patient, staff, or hospital records. Real medical records are highly sensitive and legally protected; using synthetic data is the correct and ethical choice for thesis work. All values are generated programmatically with engineered, realistic statistical correlations. Do **not** use it clinically or present it as real hospital statistics. Any resemblance to real persons or institutions is coincidental.

---

## 2. Files in This Package

### Relational tables (these are the dataset)

| File | Rows | Role |
|---|---:|---|
| `hospitals.csv` | 5 | Hospital network reference |
| `departments.csv` | 110 | Departments per hospital (22 types) |
| `wards_beds.csv` | 660 | Every physical bed, ward, bed type |
| `staff.csv` | 1,200 | Doctors, nurses, technicians, support |
| `patients.csv` | 80,000 | Patient master with demographics |
| `admissions.csv` | 120,000 | Core admissions: LOS, severity, outcome, bill |
| `bed_occupancy_daily.csv` | 150,000 | Daily per-department occupancy snapshot |
| `patient_flow_events.csv` | 95,000 | Patient movement / flow events |
| `staff_shifts.csv` | 53,000 | Daily staff roster and workload |
| **Total** | **~500,000** | |

### Convenience and documentation

| File | Description |
|---|---|
| `admissions_flat.csv` | Pre-merged wide table (admission + patient + department + hospital), 120,000 x 31 |
| `admissions_flat.csv.gz`, `patient_flow_events.csv.gz` | Compressed versions of large files |
| `README.md` | This document |
| `data_dictionary.md` | Every column in every table defined |
| `ER_relationships.md` | Entity relationships and join keys |
| `metadata.json` | Machine-readable schema, keys, checksums, provenance |
| `generate_dataset.py` | Reproducible seeded generation script |

Load a compressed file directly:

```python
import pandas as pd
flat = pd.read_csv("admissions_flat.csv.gz", compression="gzip")
```

---

## 3. How the Dataset Maps to the Thesis Goals

### 3.1 Bed Occupancy Optimization

`bed_occupancy_daily.csv` is the heart of the dashboard. Each row is a department-day with `total_beds`, `occupied_beds`, `available_beds`, `occupancy_rate`, plus same-day `admissions_count` and `discharges_count`. Engineered so that critical-care departments (ICU, HDU, NICU, Burn) run hotter (around 86% average occupancy) than general inpatient wards (around 71%), with seasonal peaks in monsoon and winter. `wards_beds.csv` gives the physical bed inventory by type for capacity analysis.

### 3.2 Patient Flow Optimization

`admissions.csv` provides `length_of_stay_days`, `severity`, `admission_type`, and `discharge_outcome`. Length of stay rises strongly with severity (about 5 days for Mild up to about 21 for Critical) and with age. `patient_flow_events.csv` tracks movement (ER Arrival, Triage, Bed Assigned, Ward Transfer, Discharge, Referral) with `wait_minutes` that increase with severity, so the dashboard can surface bottlenecks and ER wait problems.

### 3.3 Staff Allocation Optimization

`staff.csv` and `staff_shifts.csv` give roster and per-shift workload. `patients_handled` per shift and the `is_overtime` flag rise during monsoon and winter, creating a visible staffing-strain signal you can analyze against occupancy to recommend better allocation. This directly supports the allocation-optimization objective.

---

## 4. Suggested Dashboard KPIs

- **Occupancy:** occupancy rate by department/ward/day, available beds trend, ICU vs general utilization, seasonal peak detection
- **Flow:** average length of stay by department and severity, ER wait time, admissions vs discharges balance, readmission rate, discharge-outcome mix
- **Staff:** patients handled per shift, overtime rate, staff-to-occupancy ratio, monsoon/winter strain

---

## 5. Suggested Machine Learning / Analytics

| Task | How to use this data |
|---|---|
| Time-series forecasting | Forecast daily occupancy or admissions from `bed_occupancy_daily` (ARIMA, Prophet, LSTM) |
| Regression | Predict `length_of_stay_days` or `total_bill_npr` from admission + patient features |
| Classification | Predict `discharge_outcome`, `readmission_flag`, or `severity` |
| Optimization | Match forecasted patient load to staff roster to minimize overtime / understaffing |
| Clustering | Segment departments by workload or patients by risk profile |

Recommended: a temporal train/test split (train on 2021 to 2023, test on 2024) so forecasting is evaluated honestly.

---

## 6. Engineered Realism (so the dashboard shows real signal)

- Critical departments run at higher occupancy than general wards.
- Length of stay rises with severity and patient age.
- Mortality (Expired outcome) rises with severity (about 0.6% Mild to about 24% Critical).
- Monsoon (Jun to Sep) and winter (Dec to Jan) increase admissions and per-shift staff load.
- ER and triage wait times rise with severity.
- Bill amount correlates with severity, length of stay, and critical care.
- Realistic missing values in optional fields (insurance, blood group).

---

## 7. Reproducibility

`generate_dataset.py` uses a fixed seed (`20240519`) and regenerates the dataset identically. Per-file MD5 checksums are recorded in `metadata.json` under `file_integrity`.

---

## 8. Ethical and Academic Integrity Statement

This dataset is fully synthetic. No real patient, staff, or hospital records were accessed or used. This is essential: real medical data is legally protected and ethically sensitive, and a synthetic dataset lets you build and demonstrate the full dashboard and modeling pipeline without any privacy risk. In your thesis, state clearly that the data is synthetic and generated for system development and evaluation. Methods, dashboard design, and model results are valid academic contributions; the numbers are not real hospital statistics.

---

## 9. License

Released under **Creative Commons Attribution 4.0 (CC BY 4.0)** for academic use with attribution.

---

*Version 1.0.0. See `metadata.json` for the full machine-readable schema and `data_dictionary.md` for column-level documentation.*
