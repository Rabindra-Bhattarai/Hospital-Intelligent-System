import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             confusion_matrix)

# ============================================
# SCRIPT 07: STAFF OVERTIME PREDICTION MODEL
# Hospital Thesis - Rabindra Bhattarai
# ============================================
# Predicts whether a staff shift will go into
# overtime BEFORE the shift starts.
# Uses only information available before shift begins.
# patients_handled is EXCLUDED because it is
# recorded after the shift ends (data leakage).

print("Loading staff shifts data...")

staff_shifts_master = pd.read_csv("Raw_data_outputs/cleaning/staff_shifts_master.csv")
staff_shifts_master['shift_date'] = pd.to_datetime(staff_shifts_master['shift_date'])

print(f"Loaded: {len(staff_shifts_master)} rows")
print(f"Overtime rate: {staff_shifts_master['is_overtime'].mean()*100:.2f}%")
print(f"Overtime shifts:     {staff_shifts_master['is_overtime'].sum()}")
print(f"Not overtime shifts: {(staff_shifts_master['is_overtime']==False).sum()}")


# ============================================
# STEP 1: PREPARE FEATURES
# ============================================
print("\nPreparing features...")

# Shift type
shift_map = {'Morning': 1, 'Evening': 2, 'Night': 3}
staff_shifts_master['shift_type_num'] = staff_shifts_master['shift_type'].map(shift_map)

# Department type
dept_type_map = {'Outpatient': 1, 'Diagnostic': 2, 'Inpatient': 3, 'Critical': 4}
staff_shifts_master['dept_type_num'] = staff_shifts_master['department_type'].map(dept_type_map)

# Hospital type
hospital_map = {'Public': 1, 'Private': 2, 'Teaching': 3}
staff_shifts_master['hospital_type_num'] = staff_shifts_master['hospital_type'].map(hospital_map)

# Staff role - all 9 roles included
role_map = {
    'Staff Nurse':          1,
    'Senior Nurse':         2,
    'Medical Officer':      3,
    'Resident Doctor':      4,
    'Consultant':           5,
    'Pharmacist':           6,
    'Lab Technician':       7,
    'Radiology Technician': 8,
    'Support Staff':        9
}
staff_shifts_master['role_num'] = staff_shifts_master['role'].map(role_map)

# Employment type
employment_map = {
    'Permanent': 1,
    'Contract':  2,
    'Visiting':  3,
    'Intern':    4
}
staff_shifts_master['employment_num'] = staff_shifts_master['employment_type'].map(employment_map)

# Date features
staff_shifts_master['shift_year']        = staff_shifts_master['shift_date'].dt.year
staff_shifts_master['shift_month']       = staff_shifts_master['shift_date'].dt.month
staff_shifts_master['shift_day_of_week'] = staff_shifts_master['shift_date'].dt.dayofweek
staff_shifts_master['shift_quarter']     = staff_shifts_master['shift_date'].dt.quarter

# Season from month
def get_season(month):
    if month in [6, 7, 8, 9]:
        return 2  # Monsoon
    elif month in [12, 1]:
        return 4  # Winter
    elif month in [2, 3, 4, 5]:
        return 1  # Spring
    else:
        return 3  # Autumn

staff_shifts_master['season_num'] = staff_shifts_master['shift_month'].apply(get_season)

print("Features prepared!")


# ============================================
# STEP 2: CHECK FOR MISSING VALUES
# ============================================
print("\nChecking for missing values in features...")

check_cols = ['shift_type_num', 'dept_type_num',
              'hospital_type_num', 'role_num', 'employment_num']

for col in check_cols:
    nulls = staff_shifts_master[col].isnull().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} missing → filling with mode")
        mode_val = staff_shifts_master[col].mode()[0]
        staff_shifts_master[col] = staff_shifts_master[col].fillna(mode_val)
    else:
        print(f"  {col}: no missing values")


# ============================================
# STEP 3: FEATURES AND TARGET
# ============================================
print("\nDefining features and target...")

# Only features known BEFORE shift starts
# patients_handled EXCLUDED - data leakage
features = [
    'shift_type_num',
    'dept_type_num',
    'hospital_type_num',
    'role_num',
    'employment_num',
    'years_experience',
    'shift_year',
    'shift_month',
    'shift_day_of_week',
    'shift_quarter',
    'season_num'
]

target = 'is_overtime'

X = staff_shifts_master[features]
y = staff_shifts_master[target].astype(int)

print(f"  Features: {len(features)}")
print(f"  Target: {target}")
print(f"  X shape: {X.shape}")
print(f"  Note: patients_handled excluded to prevent data leakage")


# ============================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================
print("\nSplitting into train and test...")

train_data = staff_shifts_master[staff_shifts_master['shift_year'] <= 2023]
test_data  = staff_shifts_master[staff_shifts_master['shift_year'] == 2024]

X_train = train_data[features]
y_train = train_data[target].astype(int)

X_test  = test_data[features]
y_test  = test_data[target].astype(int)

print(f"  Training: {len(X_train)} rows (2021-2023)")
print(f"  Testing:  {len(X_test)} rows (2024)")
print(f"  Overtime in training: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Overtime in testing:  {y_test.sum()} ({y_test.mean()*100:.1f}%)")


# ============================================
# STEP 5: TRAIN MODEL
# ============================================
print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1
)

model.fit(X_train, y_train)
print("Model trained successfully!")


# ============================================
# STEP 6: EVALUATE
# ============================================
print("\nEvaluating model on 2024 test data...")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n  Accuracy:  {round(accuracy, 4)}")
print(f"  Precision: {round(precision, 4)}")
print(f"  Recall:    {round(recall, 4)}")
print(f"  F1 Score:  {round(f1, 4)}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negatives  (correctly NOT overtime): {cm[0][0]}")
print(f"  False Positives (wrongly flagged):         {cm[0][1]}")
print(f"  False Negatives (missed overtime):         {cm[1][0]}")
print(f"  True Positives  (correctly caught):        {cm[1][1]}")

print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Overtime', 'Overtime']))


# ============================================
# STEP 7: FEATURE IMPORTANCE
# ============================================
print("\nFeature Importance (what predicts overtime?)...")

importance_df = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df['importance'] = importance_df['importance'].round(4)
print(importance_df.to_string())


# ============================================
# STEP 8: OVERTIME RISK ANALYSIS
# ============================================
print("\nOvertime risk analysis...")

test_results = test_data[['shift_id', 'staff_id',
                           'department_name', 'hospital_name',
                           'department_type', 'role',
                           'shift_type', 'shift_date',
                           'years_experience']].copy()

test_results['actual_overtime']      = y_test.values
test_results['predicted_overtime']   = y_pred
test_results['overtime_probability'] = y_pred_prob.round(4)

def overtime_risk(prob):
    if prob >= 0.70:
        return 'HIGH RISK'
    elif prob >= 0.40:
        return 'MEDIUM RISK'
    else:
        return 'LOW RISK'

test_results['overtime_risk'] = test_results['overtime_probability'].apply(overtime_risk)

print("\nOvertime Risk Distribution:")
print(test_results['overtime_risk'].value_counts())

print("\nActual overtime rate by risk category:")
for risk in ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK']:
    subset = test_results[test_results['overtime_risk'] == risk]
    if len(subset) > 0:
        rate = subset['actual_overtime'].mean() * 100
        print(f"  {risk}: {rate:.1f}% actual overtime ({len(subset)} shifts)")

print("\nOvertime by department type:")
print(round(test_results.groupby('department_type')['actual_overtime'].mean()*100, 2))

print("\nOvertime by shift type:")
print(round(test_results.groupby('shift_type')['actual_overtime'].mean()*100, 2))

print("\nOvertime by role:")
print(round(test_results.groupby('role')['actual_overtime'].mean()*100, 2))


# ============================================
# STEP 9: SAVE RESULTS
# ============================================
print("\nSaving results...")

os.makedirs("Raw_data_outputs/ml_models", exist_ok=True)

test_results.to_csv(
    "Raw_data_outputs/ml_models/overtime_predictions.csv", index=False)

importance_df.to_csv(
    "Raw_data_outputs/ml_models/overtime_feature_importance.csv", index=False)

metrics_df = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
    'value':  [round(accuracy,4), round(precision,4),
               round(recall,4),   round(f1,4)]
})
metrics_df.to_csv(
    "Raw_data_outputs/ml_models/overtime_model_metrics.csv", index=False)

print("  overtime_predictions.csv         saved")
print("  overtime_feature_importance.csv  saved")
print("  overtime_model_metrics.csv       saved")

print("\nScript 07 Complete!")
print("\n=== ALL 5 ML MODELS COMPLETE ===")
print("  Model 1: Occupancy Forecasting       (Script 03)")
print("  Model 2: LOS Prediction              (Script 04)")
print("  Model 3: Readmission Classification  (Script 05)")
print("  Model 4: Discharge Outcome           (Script 06)")
print("  Model 5: Staff Overtime Prediction   (Script 07)")
print("\nNext step: Tableau Dashboard")