import pandas as pd
import os

# ============================================
# SCRIPT: OCCUPANCY BY DEPARTMENT + SEASON (2024)
# Hospital Thesis - Rabindra Bhattarai
# ============================================
# Purpose: build a department+season occupancy
# summary from the 2024 occupancy forecast test
# results (occupancy_predictions.csv), so the
# Tableau dashboard (D2) can cross-filter between
# the department-level occupancy sheet and a
# seasonal occupancy view using a shared field
# (department_name).
#
# This sheet shows ACTUAL vs PREDICTED occupancy
# for 2024 (the model's test year), by department
# and season - distinct from kpi_occupancy_by_season.csv
# which covers 2021-2024 with no department breakdown.

print("Loading occupancy predictions (2024 test results)...")

occupancy_predictions = pd.read_csv("Raw_data_outputs/ml_models/occupancy_predictions.csv")

print(f"Loaded: {len(occupancy_predictions)} rows")
print(f"Columns: {list(occupancy_predictions.columns)}")


# STEP 1: AGGREGATE BY DEPARTMENT + SEASON

print("\nAggregating by department_name, department_type, season...")

occupancy_by_dept_season = occupancy_predictions.groupby(
    ['department_name', 'department_type', 'season']
).agg(
    avg_occupancy           = ('actual_occupancy', 'mean'),
    avg_predicted_occupancy = ('predicted_occupancy', 'mean'),
    record_count            = ('actual_occupancy', 'count')
).reset_index()

occupancy_by_dept_season['avg_occupancy'] = occupancy_by_dept_season['avg_occupancy'].round(4)
occupancy_by_dept_season['avg_predicted_occupancy'] = occupancy_by_dept_season['avg_predicted_occupancy'].round(4)

print(f"  Result: {len(occupancy_by_dept_season)} rows "
      f"({occupancy_by_dept_season['department_name'].nunique()} departments "
      f"x {occupancy_by_dept_season['season'].nunique()} seasons)")


# STEP 2: SAVE

print("\nSaving kpi_occupancy_by_dept_season.csv...")

os.makedirs("Raw_data_outputs/kpi", exist_ok=True)

occupancy_by_dept_season.to_csv(
    "Raw_data_outputs/kpi/kpi_occupancy_by_dept_season.csv", index=False)

print("  kpi_occupancy_by_dept_season.csv saved")
print("\nDone! Connect this file in Tableau as a new data source for D2.")