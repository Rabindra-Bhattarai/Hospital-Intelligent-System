# Entity Relationships

## Tables and Keys

| Table | Primary Key | Grain |
|---|---|---|
| hospitals | hospital_id | one row per hospital |
| departments | department_id | one row per department per hospital |
| wards_beds | bed_id | one row per physical bed |
| staff | staff_id | one row per staff member |
| patients | patient_id | one row per patient |
| admissions | admission_id | one row per admission |
| bed_occupancy_daily | record_id | one row per (department, day) |
| patient_flow_events | event_id | one row per flow event |
| staff_shifts | shift_id | one row per (staff, shift) |

## Foreign Keys

- `departments.hospital_id -> hospitals.hospital_id`
- `wards_beds.department_id -> departments.department_id`
- `staff.department_id -> departments.department_id`
- `admissions.patient_id -> patients.patient_id`
- `admissions.department_id -> departments.department_id`
- `admissions.bed_id -> wards_beds.bed_id`
- `admissions.attending_staff_id -> staff.staff_id`
- `bed_occupancy_daily.department_id -> departments.department_id`
- `patient_flow_events.admission_id -> admissions.admission_id`
- `patient_flow_events.patient_id -> patients.patient_id`
- `staff_shifts.staff_id -> staff.staff_id`
- `staff_shifts.department_id -> departments.department_id`

## Diagram

```
hospitals --< departments --< wards_beds
                  |   \
                  |    +--< staff --< staff_shifts
                  |
patients --< admissions >-- departments
                 |   \__ bed (wards_beds), attending (staff)
                 +--< patient_flow_events
departments --< bed_occupancy_daily
```

## Typical Joins

```python
import pandas as pd
adm=pd.read_csv('admissions.csv'); pat=pd.read_csv('patients.csv')
dep=pd.read_csv('departments.csv')
df=adm.merge(pat,on='patient_id').merge(dep,on='department_id')
occ=pd.read_csv('bed_occupancy_daily.csv')  # dashboard time series
```
