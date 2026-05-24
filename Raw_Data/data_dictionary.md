# Data Dictionary

Dataset: **Hospital Operations Dashboard (Synthetic)** | ~500,000 rows across 9 linked tables.

> Synthetic data for thesis/dashboard development. Not real patient records.

## hospitals.csv
Rows: **5** | Columns: **6** | Primary key: `hospital_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `hospital_id` | str | 0.0 | Unique hospital ID. PK of hospitals. |
| `hospital_name` | str | 0.0 | Hospital name. |
| `hospital_type` | str | 0.0 | Public / Private / Teaching. |
| `district` | str | 0.0 | District location. |
| `total_beds` | int64 | 0.0 | Total bed capacity. |
| `level` | str | 0.0 | Care level (Secondary/Tertiary). |

## departments.csv
Rows: **110** | Columns: **4** | Primary key: `department_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `department_name` | str | 0.0 | Department name. |
| `department_type` | str | 0.0 | Critical/Inpatient/Outpatient/Diagnostic. |
| `hospital_id` | str | 0.0 | Unique hospital ID. PK of hospitals. |

## wards_beds.csv
Rows: **660** | Columns: **6** | Primary key: `bed_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `bed_id` | str | 0.0 | Unique bed ID. PK of wards_beds. |
| `ward_name` | str | 0.0 | Ward the bed belongs to. |
| `bed_type` | str | 0.0 | ICU/HDU/NICU/General/Private/Maternity/Emergency/Dialysis. |
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `hospital_id` | str | 0.0 | Unique hospital ID. PK of hospitals. |
| `is_functional` | bool | 0.0 | Whether the bed is currently usable. |

## staff.csv
Rows: **1,200** | Columns: **9** | Primary key: `staff_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `staff_id` | str | 0.0 | Unique staff ID. PK of staff. |
| `role` | str | 0.0 | Job role. |
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `hospital_id` | str | 0.0 | Unique hospital ID. PK of hospitals. |
| `specialization` | str | 0.0 | Clinical specialization (doctors only). |
| `shift_type` | str | 0.0 | Usual shift pattern. |
| `employment_type` | str | 0.0 | Permanent/Contract/Visiting/Intern. |
| `years_experience` | float64 | 0.0 | Years of experience. |
| `on_roster` | bool | 0.0 | Currently active on roster. |

## patients.csv
Rows: **80,000** | Columns: **9** | Primary key: `patient_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `patient_id` | str | 0.0 | Unique patient ID. PK of patients. |
| `age` | int64 | 0.0 | Patient age in years. |
| `age_group` | str | 0.0 | Banded age group. |
| `gender` | str | 0.0 | Gender. |
| `district` | str | 0.0 | District location. |
| `address_type` | str | 0.0 | Urban/Semi-Urban/Rural. |
| `blood_group` | str | 3.0 | Blood group (nullable). |
| `insurance_type` | str | 47.1 | Insurance coverage (nullable). |
| `has_chronic_condition` | bool | 0.0 | Has a chronic condition flag. |

## admissions.csv
Rows: **120,000** | Columns: **17** | Primary key: `admission_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `admission_id` | str | 0.0 | Unique admission ID. PK of admissions. |
| `patient_id` | str | 0.0 | Unique patient ID. PK of patients. |
| `hospital_id` | str | 0.0 | Unique hospital ID. PK of hospitals. |
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `department_name` | str | 0.0 | Department name. |
| `bed_id` | str | 0.0 | Unique bed ID. PK of wards_beds. |
| `admission_date` | str | 0.0 | Date of admission. |
| `admission_hour` | int64 | 0.0 | Hour of admission (0-23). |
| `discharge_date` | str | 0.0 | Date of discharge (>= admission_date). |
| `length_of_stay_days` | int64 | 0.0 | Length of stay in days. KEY flow metric. |
| `admission_type` | str | 0.0 | Emergency/Elective/Referral/Walk-in. |
| `diagnosis_category` | str | 0.0 | Grouped diagnosis category. |
| `severity` | str | 0.0 | Mild/Moderate/Severe/Critical. |
| `attending_staff_id` | str | 0.0 | Attending doctor (FK to staff). |
| `discharge_outcome` | str | 0.0 | Recovered/Referred/LAMA/Transferred/Expired. |
| `readmission_flag` | bool | 0.0 | Readmitted within follow-up window. |
| `total_bill_npr` | int64 | 0.0 | Total bill in NPR. |

## bed_occupancy_daily.csv
Rows: **150,000** | Columns: **11** | Primary key: `record_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `record_id` | str | 0.0 | Unique daily occupancy record. PK of bed_occupancy_daily. |
| `date` | str | 0.0 | Calendar date of the snapshot. |
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `total_beds` | int64 | 0.0 | Total bed capacity. |
| `occupied_beds` | int64 | 0.0 | Beds occupied that day. |
| `available_beds` | int64 | 0.0 | Free beds that day. |
| `occupancy_rate` | float64 | 0.0 | occupied/total (0-1). KEY occupancy metric. |
| `admissions_count` | int64 | 0.0 | Admissions that day in the department. |
| `discharges_count` | int64 | 0.0 | Discharges that day. |
| `is_weekend` | bool | 0.0 | Weekend flag. |
| `season` | str | 0.0 | Monsoon/Winter/Spring/Autumn. |

## patient_flow_events.csv
Rows: **95,000** | Columns: **8** | Primary key: `event_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `event_id` | str | 0.0 | Unique flow event ID. PK of patient_flow_events. |
| `admission_id` | str | 0.0 | Unique admission ID. PK of admissions. |
| `patient_id` | str | 0.0 | Unique patient ID. PK of patients. |
| `event_type` | str | 0.0 | ER Arrival/Triage/Bed Assigned/Ward Transfer/Discharge/Referral. |
| `event_timestamp` | str | 0.0 | Timestamp of the event. |
| `from_department` | str | 0.0 | Origin department. |
| `to_department` | str | 0.0 | Destination department (for transfers/referrals). |
| `wait_minutes` | int64 | 0.0 | Wait time for the event in minutes. |

## staff_shifts.csv
Rows: **53,000** | Columns: **7** | Primary key: `shift_id`

| Column | Type | Null % | Description |
|---|---|---:|---|
| `shift_id` | str | 0.0 | Unique shift ID. PK of staff_shifts. |
| `staff_id` | str | 0.0 | Unique staff ID. PK of staff. |
| `department_id` | str | 0.0 | Unique department ID. PK of departments. |
| `shift_date` | str | 0.0 | Date of the shift. |
| `shift_type` | str | 0.0 | Usual shift pattern. |
| `patients_handled` | int64 | 0.0 | Patients handled in the shift. KEY staffing metric. |
| `is_overtime` | bool | 0.0 | Shift exceeded normal load. |

## admissions_flat.csv (merged convenience file)

Admission-grain wide table: 120,000 rows x 31 cols. Combines admissions + patient + department + hospital columns for quick modeling. See individual tables for the normalized form.

## Notes

- All primary keys are unique; all listed foreign keys validated.
- `discharge_date` is always on or after `admission_date`.
- `occupancy_rate` = occupied_beds / total_beds.
- Engineered correlations: occupancy by dept type, LOS by severity, mortality by severity, seasonal staffing strain. See README section 6.
- Missing values are intentional in optional fields for a realistic cleaning exercise.