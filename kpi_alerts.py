import pandas as pd
import numpy as np
import os


# SCRIPT 02: KPI CALCULATION AND ALERTS
# Hospital Thesis - Rabindra Bhattarai

print("Loading cleaned master files...")

admissions_master    = pd.read_csv("Raw_data_outputs/cleaning/admissions_master.csv")
bed_occupancy_master = pd.read_csv("Raw_data_outputs/cleaning/bed_occupancy_master.csv")
staff_shifts_master  = pd.read_csv("Raw_data_outputs/cleaning/staff_shifts_master.csv")
flow_master          = pd.read_csv("Raw_data_outputs/cleaning/flow_master.csv")

admissions_master['admission_date']  = pd.to_datetime(admissions_master['admission_date'])
admissions_master['discharge_date']  = pd.to_datetime(admissions_master['discharge_date'])
bed_occupancy_master['date']         = pd.to_datetime(bed_occupancy_master['date'])
staff_shifts_master['shift_date']    = pd.to_datetime(staff_shifts_master['shift_date'])
flow_master['event_timestamp']       = pd.to_datetime(flow_master['event_timestamp'])

print("All master files loaded!")
print(f"  admissions_master:    {len(admissions_master)} rows")
print(f"  bed_occupancy_master: {len(bed_occupancy_master)} rows")
print(f"  staff_shifts_master:  {len(staff_shifts_master)} rows")
print(f"  flow_master:          {len(flow_master)} rows")



# STEP 1: DEFINE BENCHMARKS

# Published international and national standards
# WHO  = World Health Organization
# NHS  = National Health Service UK
# MoHP = Ministry of Health and Population Nepal

print("\nSetting benchmarks...")

WHO_OCCUPANCY  = 0.85
NHS_OCCUPANCY  = 0.82
MOPH_OCCUPANCY = 0.75

WHO_LOS  = 7.0
NHS_LOS  = 5.8
MOPH_LOS = 4.8

WHO_WAIT  = 240
NHS_WAIT  = 240
MOPH_WAIT = 60

WHO_OVERTIME  = 0.20
NHS_OVERTIME  = 0.15
MOPH_OVERTIME = 0.25

WHO_READMISSION  = 0.10
NHS_READMISSION  = 0.05
MOPH_READMISSION = 0.08

print("Benchmarks set!")
print(f"  Occupancy:   WHO={WHO_OCCUPANCY*100}%  NHS={NHS_OCCUPANCY*100}%  MoHP={MOPH_OCCUPANCY*100}%")
print(f"  LOS:         WHO={WHO_LOS}days  NHS={NHS_LOS}days  MoHP={MOPH_LOS}days")
print(f"  Wait time:   WHO={WHO_WAIT}min  NHS={NHS_WAIT}min  MoHP={MOPH_WAIT}min")
print(f"  Overtime:    WHO={WHO_OVERTIME*100}%  NHS={NHS_OVERTIME*100}%  MoHP={MOPH_OVERTIME*100}%")
print(f"  Readmission: WHO={WHO_READMISSION*100}%  NHS={NHS_READMISSION*100}%  MoHP={MOPH_READMISSION*100}%")



# STEP 2: BED OCCUPANCY KPIs

print("\nCalculating Bed Occupancy KPIs...")

occupancy_by_dept = bed_occupancy_master.groupby(
    ['department_name', 'department_type', 'hospital_name']
).agg(
    avg_occupancy      = ('occupancy_rate', 'mean'),
    max_occupancy      = ('occupancy_rate', 'max'),
    avg_available_beds = ('available_beds', 'mean'),
    total_admissions   = ('admissions_count', 'sum'),
    total_discharges   = ('discharges_count', 'sum')
).reset_index()

occupancy_by_dept['avg_occupancy'] = occupancy_by_dept['avg_occupancy'].round(4)
occupancy_by_dept['max_occupancy'] = occupancy_by_dept['max_occupancy'].round(4)

occupancy_by_season = bed_occupancy_master.groupby('season').agg(
    avg_occupancy  = ('occupancy_rate', 'mean'),
    avg_admissions = ('admissions_count', 'mean')
).reset_index()
occupancy_by_season['avg_occupancy'] = occupancy_by_season['avg_occupancy'].round(4)

occupancy_by_dept['vs_WHO']  = (occupancy_by_dept['avg_occupancy'] - WHO_OCCUPANCY).round(4)
occupancy_by_dept['vs_NHS']  = (occupancy_by_dept['avg_occupancy'] - NHS_OCCUPANCY).round(4)
occupancy_by_dept['vs_MoHP'] = (occupancy_by_dept['avg_occupancy'] - MOPH_OCCUPANCY).round(4)

def occupancy_alert(rate):
    if rate >= WHO_OCCUPANCY:
        return 'RED'
    elif rate >= MOPH_OCCUPANCY:
        return 'YELLOW'
    else:
        return 'GREEN'

occupancy_by_dept['alert_status'] = occupancy_by_dept['avg_occupancy'].apply(occupancy_alert)

print("Bed Occupancy KPIs done!")
print(f"  RED alerts:    {(occupancy_by_dept['alert_status']=='RED').sum()}")
print(f"  YELLOW alerts: {(occupancy_by_dept['alert_status']=='YELLOW').sum()}")
print(f"  GREEN alerts:  {(occupancy_by_dept['alert_status']=='GREEN').sum()}")



# STEP 3: PATIENT FLOW KPIs

print("\nCalculating Patient Flow KPIs...")

los_by_severity = admissions_master.groupby('severity').agg(
    avg_los = ('length_of_stay_days', 'mean'),
    max_los = ('length_of_stay_days', 'max'),
    count   = ('admission_id', 'count')
).reset_index()
los_by_severity['avg_los'] = los_by_severity['avg_los'].round(2)

los_by_dept = admissions_master.groupby(
    ['department_name', 'hospital_name']
).agg(
    avg_los = ('length_of_stay_days', 'mean'),
    count   = ('admission_id', 'count')
).reset_index()
los_by_dept['avg_los'] = los_by_dept['avg_los'].round(2)

los_by_dept['vs_WHO']  = (los_by_dept['avg_los'] - WHO_LOS).round(2)
los_by_dept['vs_NHS']  = (los_by_dept['avg_los'] - NHS_LOS).round(2)
los_by_dept['vs_MoHP'] = (los_by_dept['avg_los'] - MOPH_LOS).round(2)

def los_alert(days):
    if days >= WHO_LOS * 2:
        return 'RED'
    elif days >= WHO_LOS:
        return 'YELLOW'
    else:
        return 'GREEN'

los_by_dept['alert_status'] = los_by_dept['avg_los'].apply(los_alert)

outcomes = admissions_master['discharge_outcome'].value_counts().reset_index()
outcomes.columns = ['outcome', 'count']
outcomes['percentage'] = (outcomes['count'] / len(admissions_master) * 100).round(2)

overall_readmission_rate = admissions_master['readmission_flag'].mean()

readmission_by_severity = admissions_master.groupby('severity').agg(
    readmission_rate = ('readmission_flag', 'mean'),
    count            = ('admission_id', 'count')
).reset_index()
readmission_by_severity['readmission_rate'] = readmission_by_severity['readmission_rate'].round(4)

def readmission_alert(rate):
    if rate >= WHO_READMISSION:
        return 'RED'
    elif rate >= NHS_READMISSION:
        return 'YELLOW'
    else:
        return 'GREEN'

readmission_by_severity['alert_status'] = readmission_by_severity['readmission_rate'].apply(readmission_alert)

er_wait = flow_master[flow_master['event_type'] == 'ER Arrival'].groupby(
    ['hospital_name', 'severity']
).agg(
    avg_wait_minutes = ('wait_minutes', 'mean'),
    max_wait_minutes = ('wait_minutes', 'max')
).reset_index()
er_wait['avg_wait_minutes'] = er_wait['avg_wait_minutes'].round(2)

def wait_alert(minutes):
    if minutes >= WHO_WAIT:
        return 'RED'
    elif minutes >= MOPH_WAIT:
        return 'YELLOW'
    else:
        return 'GREEN'

er_wait['alert_status'] = er_wait['avg_wait_minutes'].apply(wait_alert)

print("Patient Flow KPIs done!")
print(f"  Overall readmission rate: {round(overall_readmission_rate*100,2)}%")
print(f"  vs WHO  ({WHO_READMISSION*100}%):  {'ABOVE' if overall_readmission_rate > WHO_READMISSION else 'BELOW'}")
print(f"  vs NHS  ({NHS_READMISSION*100}%):  {'ABOVE' if overall_readmission_rate > NHS_READMISSION else 'BELOW'}")
print(f"  vs MoHP ({MOPH_READMISSION*100}%): {'ABOVE' if overall_readmission_rate > MOPH_READMISSION else 'BELOW'}")



# STEP 4: STAFF ALLOCATION KPIs

print("\nCalculating Staff Allocation KPIs...")

overtime_by_dept = staff_shifts_master.groupby(
    ['department_name', 'hospital_name']
).agg(
    overtime_rate        = ('is_overtime', 'mean'),
    avg_patients_handled = ('patients_handled', 'mean'),
    total_shifts         = ('shift_id', 'count')
).reset_index()
overtime_by_dept['overtime_rate']        = overtime_by_dept['overtime_rate'].round(4)
overtime_by_dept['avg_patients_handled'] = overtime_by_dept['avg_patients_handled'].round(2)

overtime_by_dept['vs_WHO']  = (overtime_by_dept['overtime_rate'] - WHO_OVERTIME).round(4)
overtime_by_dept['vs_NHS']  = (overtime_by_dept['overtime_rate'] - NHS_OVERTIME).round(4)
overtime_by_dept['vs_MoHP'] = (overtime_by_dept['overtime_rate'] - MOPH_OVERTIME).round(4)

def overtime_alert(rate):
    if rate >= WHO_OVERTIME:
        return 'RED'
    elif rate >= MOPH_OVERTIME:
        return 'YELLOW'
    else:
        return 'GREEN'

overtime_by_dept['alert_status'] = overtime_by_dept['overtime_rate'].apply(overtime_alert)

overtime_by_shift = staff_shifts_master.groupby('shift_type').agg(
    overtime_rate        = ('is_overtime', 'mean'),
    avg_patients_handled = ('patients_handled', 'mean')
).reset_index()
overtime_by_shift['overtime_rate'] = overtime_by_shift['overtime_rate'].round(4)

staff_role_dist = staff_shifts_master.groupby('role').agg(
    total_shifts         = ('shift_id', 'count'),
    overtime_rate        = ('is_overtime', 'mean'),
    avg_patients_handled = ('patients_handled', 'mean')
).reset_index()
staff_role_dist['overtime_rate'] = staff_role_dist['overtime_rate'].round(4)

print("Staff Allocation KPIs done!")
print(f"  RED alerts:    {(overtime_by_dept['alert_status']=='RED').sum()}")
print(f"  YELLOW alerts: {(overtime_by_dept['alert_status']=='YELLOW').sum()}")
print(f"  GREEN alerts:  {(overtime_by_dept['alert_status']=='GREEN').sum()}")



# STEP 5: OVERALL HOSPITAL SUMMARY

print("\nCalculating Hospital Summary...")

hospital_summary = admissions_master.groupby(
    ['hospital_name', 'hospital_type', 'hospital_district', 'level']
).agg(
    total_admissions = ('admission_id', 'count'),
    avg_los          = ('length_of_stay_days', 'mean'),
    readmission_rate = ('readmission_flag', 'mean'),
    mortality_rate   = ('discharge_outcome', lambda x: (x == 'Expired').mean()),
    avg_bill_npr     = ('total_bill_npr', 'mean')
).reset_index()

hospital_summary['avg_los']          = hospital_summary['avg_los'].round(2)
hospital_summary['readmission_rate'] = hospital_summary['readmission_rate'].round(4)
hospital_summary['mortality_rate']   = hospital_summary['mortality_rate'].round(4)
hospital_summary['avg_bill_npr']     = hospital_summary['avg_bill_npr'].round(0)
hospital_summary['readmission_alert'] = hospital_summary['readmission_rate'].apply(readmission_alert)

print("Hospital Summary done!")
print(hospital_summary[['hospital_name', 'total_admissions',
                         'avg_los', 'readmission_rate',
                         'mortality_rate', 'readmission_alert']].to_string())



# STEP 6: SAVE ALL KPI FILES

print("\nSaving KPI files...")

os.makedirs("Raw_data_outputs/kpi", exist_ok=True)

occupancy_by_dept.to_csv("Raw_data_outputs/kpi/kpi_occupancy_by_dept.csv",         index=False)
occupancy_by_season.to_csv("Raw_data_outputs/kpi/kpi_occupancy_by_season.csv",     index=False)
los_by_severity.to_csv("Raw_data_outputs/kpi/kpi_los_by_severity.csv",             index=False)
los_by_dept.to_csv("Raw_data_outputs/kpi/kpi_los_by_dept.csv",                     index=False)
outcomes.to_csv("Raw_data_outputs/kpi/kpi_discharge_outcomes.csv",                 index=False)
readmission_by_severity.to_csv("Raw_data_outputs/kpi/kpi_readmission.csv",         index=False)
er_wait.to_csv("Raw_data_outputs/kpi/kpi_er_wait_times.csv",                       index=False)
overtime_by_dept.to_csv("Raw_data_outputs/kpi/kpi_overtime_by_dept.csv",           index=False)
overtime_by_shift.to_csv("Raw_data_outputs/kpi/kpi_overtime_by_shift.csv",         index=False)
staff_role_dist.to_csv("Raw_data_outputs/kpi/kpi_staff_role_distribution.csv",     index=False)
hospital_summary.to_csv("Raw_data_outputs/kpi/kpi_hospital_summary.csv",           index=False)

print("  kpi_occupancy_by_dept.csv       saved")
print("  kpi_occupancy_by_season.csv     saved")
print("  kpi_los_by_severity.csv         saved")
print("  kpi_los_by_dept.csv             saved")
print("  kpi_discharge_outcomes.csv      saved")
print("  kpi_readmission.csv             saved")
print("  kpi_er_wait_times.csv           saved")
print("  kpi_overtime_by_dept.csv        saved")
print("  kpi_overtime_by_shift.csv       saved")
print("  kpi_staff_role_distribution.csv saved")
print("  kpi_hospital_summary.csv        saved")

print("\nScript 02 Complete!")
print("All KPIs saved to Raw_data_outputs/kpi/ folder")
