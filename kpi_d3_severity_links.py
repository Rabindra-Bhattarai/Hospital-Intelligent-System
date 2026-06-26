import pandas as pd
import os

# ============================================
# SCRIPT: LOS + DISCHARGE OUTCOMES (D3 INTERACTIVITY FIX)
# Hospital Thesis - Rabindra Bhattarai
# ============================================
# Purpose: build two aggregated KPI files so that
# Dashboard 3 (LOS & Discharge) sheets can cross-filter:
#   - kpi_los_by_dept_severity.csv: department_name + severity
#   - kpi_discharge_outcomes_by_severity.csv: severity + outcome
# severity is the shared field linking all 3 D3 sheets.

print("Loading admissions_master.csv...")
df = pd.read_csv("Raw_data_outputs/cleaning/admissions_master.csv")
print(f"Loaded: {len(df)} rows")


# STEP 1: LOS BY DEPARTMENT + SEVERITY

print("\nAggregating LOS by department_name + severity...")

los_by_dept_severity = df.groupby(['department_name', 'severity']).agg(
    avg_los = ('length_of_stay_days', 'mean'),
    count   = ('admission_id', 'count')
).reset_index()
los_by_dept_severity['avg_los'] = los_by_dept_severity['avg_los'].round(2)

print(f"  Result: {len(los_by_dept_severity)} rows")


# STEP 2: DISCHARGE OUTCOMES BY SEVERITY

print("\nAggregating discharge outcomes by severity...")

outcomes_by_severity = df.groupby(['severity', 'discharge_outcome']).agg(
    count = ('admission_id', 'count')
).reset_index()

totals = outcomes_by_severity.groupby('severity')['count'].transform('sum')
outcomes_by_severity['percentage'] = (outcomes_by_severity['count'] / totals * 100).round(2)

print(f"  Result: {len(outcomes_by_severity)} rows")


# STEP 3: SAVE

print("\nSaving files...")

os.makedirs("Raw_data_outputs/kpi", exist_ok=True)

los_by_dept_severity.to_csv(
    "Raw_data_outputs/kpi/kpi_los_by_dept_severity.csv", index=False)

outcomes_by_severity.to_csv(
    "Raw_data_outputs/kpi/kpi_discharge_outcomes_by_severity.csv", index=False)

print("  kpi_los_by_dept_severity.csv          saved")
print("  kpi_discharge_outcomes_by_severity.csv saved")

print("\nDone! Connect both files in Tableau as new data sources for D3.")