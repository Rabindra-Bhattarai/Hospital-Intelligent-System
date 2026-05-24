import pandas as pd
import numpy as np
import os


# SCRIPT 01: DATA CLEANING
# Hospital Thesis - Rabindra Bhattarai


Data_Path = "Raw_Data/"
print("Loading all data files...")

admissions          = pd.read_csv(Data_Path + "admissions.csv")
bed_occupancy_daily = pd.read_csv(Data_Path + "bed_occupancy_daily.csv")
departments         = pd.read_csv(Data_Path + "departments.csv")
hospitals           = pd.read_csv(Data_Path + "hospitals.csv")
patient_flow_events = pd.read_csv(Data_Path + "patient_flow_events.csv")
patients            = pd.read_csv(Data_Path + "patients.csv")
staff_shifts        = pd.read_csv(Data_Path + "staff_shifts.csv")
staff               = pd.read_csv(Data_Path + "staff.csv")
wards_beds          = pd.read_csv(Data_Path + "wards_beds.csv")

print("All files loaded successfully!")
print(f"  hospitals:          {len(hospitals)} rows")
print(f"  departments:        {len(departments)} rows")
print(f"  wards_beds:         {len(wards_beds)} rows")
print(f"  staff:              {len(staff)} rows")
print(f"  patients:           {len(patients)} rows")
print(f"  admissions:         {len(admissions)} rows")
print(f"  bed_occupancy:      {len(bed_occupancy_daily)} rows")
print(f"  patient_flow:       {len(patient_flow_events)} rows")
print(f"  staff_shifts:       {len(staff_shifts)} rows")



# STEP 2: CHECK MISSING VALUES

print("\nChecking for missing values...")

for name, df in {
    "hospitals": hospitals,
    "departments": departments,
    "wards_beds": wards_beds,
    "staff": staff,
    "patients": patients,
    "admissions": admissions,
    "bed_occupancy_daily": bed_occupancy_daily,
    "patient_flow_events": patient_flow_events,
    "staff_shifts": staff_shifts
}.items():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n{name}:")
        print(missing)
    else:
        print(f"  {name}: no missing values")



# STEP 3: FIX MISSING VALUES

print("\nFixing missing values...")

patients['blood_group']    = patients['blood_group'].fillna('Unknown')
patients['insurance_type'] = patients['insurance_type'].fillna('No Insurance')

print(f"  blood_group missing:    {patients['blood_group'].isnull().sum()}")
print(f"  insurance_type missing: {patients['insurance_type'].isnull().sum()}")
print("Missing values fixed!")



# STEP 4: FIX DATE FORMATS

print("\nFixing date formats...")

admissions['admission_date']           = pd.to_datetime(admissions['admission_date'], dayfirst=True)
admissions['discharge_date']           = pd.to_datetime(admissions['discharge_date'], dayfirst=True)
bed_occupancy_daily['date']            = pd.to_datetime(bed_occupancy_daily['date'])
patient_flow_events['event_timestamp'] = pd.to_datetime(patient_flow_events['event_timestamp'])
staff_shifts['shift_date']             = pd.to_datetime(staff_shifts['shift_date'])

print(f"  admissions date sample:    {admissions['admission_date'].iloc[0]}")
print(f"  bed_occupancy date sample: {bed_occupancy_daily['date'].iloc[0]}")
print(f"  staff_shifts date sample:  {staff_shifts['shift_date'].iloc[0]}")
print("Date formats fixed!")



# STEP 5: MERGE TABLES

print("\nMerging tables...")


# ADMISSIONS MASTER
# Before merging we rename district in patients to patient_district
# and district in hospitals to hospital_district
# so they dont clash after merging
patients_clean = patients.rename(columns={'district': 'patient_district'})

hospitals_clean = hospitals.rename(columns={'district': 'hospital_district'})

# Also drop department_name from departments before merging
# because admissions already has department_name
# keeping both would create department_name_x and department_name_y
departments_clean = departments[['department_id', 'department_type']].copy()

admissions_master = admissions.merge(
    patients_clean,
    on='patient_id',
    how='left'
).merge(
    departments_clean,
    on='department_id',
    how='left'
).merge(
    hospitals_clean[['hospital_id', 'hospital_name',
                     'hospital_type', 'hospital_district', 'level']],
    on='hospital_id',
    how='left'
)

print(f"  Admissions master:    {len(admissions_master)} rows, {len(admissions_master.columns)} columns")
print(f"  Columns: {list(admissions_master.columns)}")


# BED OCCUPANCY MASTER
bed_occupancy_master = bed_occupancy_daily.merge(
    departments[['department_id', 'department_name', 'department_type', 'hospital_id']],
    on='department_id',
    how='left'
).merge(
    hospitals_clean[['hospital_id', 'hospital_name',
                     'hospital_type', 'hospital_district']],
    on='hospital_id',
    how='left'
)

print(f"  Bed occupancy master: {len(bed_occupancy_master)} rows, {len(bed_occupancy_master.columns)} columns")
print(f"  Columns: {list(bed_occupancy_master.columns)}")


# STAFF SHIFTS MASTER
# Remove department_id from staff to avoid conflict
# because staff_shifts already has department_id
staff_clean = staff[['staff_id', 'role', 'hospital_id',
                      'employment_type', 'years_experience']].copy()

staff_shifts_master = staff_shifts.merge(
    staff_clean,
    on='staff_id',
    how='left'
).merge(
    departments[['department_id', 'department_name', 'department_type']],
    on='department_id',
    how='left'
).merge(
    hospitals_clean[['hospital_id', 'hospital_name',
                     'hospital_type', 'hospital_district']],
    on='hospital_id',
    how='left'
)

print(f"  Staff shifts master:  {len(staff_shifts_master)} rows, {len(staff_shifts_master.columns)} columns")
print(f"  Columns: {list(staff_shifts_master.columns)}")


# FLOW MASTER
flow_master = patient_flow_events.merge(
    admissions[['admission_id', 'severity', 'admission_type',
                'department_name', 'hospital_id']],
    on='admission_id',
    how='left'
).merge(
    patients_clean[['patient_id', 'age', 'age_group',
                    'gender', 'patient_district', 'address_type']],
    on='patient_id',
    how='left'
).merge(
    hospitals_clean[['hospital_id', 'hospital_name', 'hospital_district']],
    on='hospital_id',
    how='left'
)

print(f"  Flow master:          {len(flow_master)} rows, {len(flow_master.columns)} columns")
print(f"  Columns: {list(flow_master.columns)}")

print("All tables merged successfully!")



# STEP 6: FINAL CHECK - NO DUPLICATE COLUMNS

print("\nFinal check for duplicate columns...")

for name, df in {
    "admissions_master": admissions_master,
    "bed_occupancy_master": bed_occupancy_master,
    "staff_shifts_master": staff_shifts_master,
    "flow_master": flow_master
}.items():
    cols = df.columns.tolist()
    duplicates = [col for col in cols if col.endswith('_x') or col.endswith('_y')]
    if duplicates:
        print(f"  {name}: DUPLICATE COLUMNS FOUND: {duplicates}")
    else:
        print(f"  {name}: no duplicate columns")



# STEP 7: SAVE CLEANED FILES

print("\nSaving cleaned files...")

os.makedirs("Raw_data_outputs/cleaning", exist_ok=True)

admissions_master.to_csv("Raw_data_outputs/cleaning/admissions_master.csv",       index=False)
bed_occupancy_master.to_csv("Raw_data_outputs/cleaning/bed_occupancy_master.csv", index=False)
staff_shifts_master.to_csv("Raw_data_outputs/cleaning/staff_shifts_master.csv",   index=False)
flow_master.to_csv("Raw_data_outputs/cleaning/flow_master.csv",                   index=False)

print("  admissions_master.csv    saved")
print("  bed_occupancy_master.csv saved")
print("  staff_shifts_master.csv  saved")
print("  flow_master.csv          saved")

print("\nScript 01 Complete!")
print("All files clean, no duplicate columns.")
