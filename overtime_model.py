import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score,classification_report,confusion_matrix)


# SCRIPT 07: STAFF OVERTIME PREDICTION MODEL
# Hospital Thesis - Rabindra Bhattarai

# This model predicts whether a staff shift
# will go into overtime before it happens.
# This directly supports staff allocation
# optimization — one of our 3 thesis objectives.

print("Loading staff shifts data...")

staff_shifts_master = pd.read_csv("Raw_data_outputs/cleaning/staff_shifts_master.csv")
staff_shifts_master['shift_date'] = pd.to_datetime(staff_shifts_master['shift_date'])

print(f"Loaded: {len(staff_shifts_master)} rows")
print(f"\nOvertime distribution:")
print(staff_shifts_master['is_overtime'].value_counts())
print(f"\nOvertime rate: {staff_shifts_master['is_overtime'].mean()*100:.2f}%")
print(f"\nColumns: {list(staff_shifts_master.columns)}")



# STEP 1: PREPARE FEATURES

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

# Staff role
role_map = {
    'Staff Nurse': 1,
    'Medical Officer': 2,
    'Consultant': 3,
    'Technician': 4,
    'Support Staff': 5
}
staff_shifts_master['role_num'] = staff_shifts_master['role'].map(role_map)

# Employment type
employment_map = {
    'Permanent': 1,
    'Contract': 2,
    'Visiting': 3,
    'Intern': 4
}
staff_shifts_master['employment_num'] = staff_shifts_master['employment_type'].map(employment_map)

# Extract date features
staff_shifts_master['shift_year']       = staff_shifts_master['shift_date'].dt.year
staff_shifts_master['shift_month']      = staff_shifts_master['shift_date'].dt.month
staff_shifts_master['shift_day_of_week']= staff_shifts_master['shift_date'].dt.dayofweek
staff_shifts_master['shift_quarter']    = staff_shifts_master['shift_date'].dt.quarter

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

# Years experience (already numeric)
# patients_handled (already numeric)

print("Features prepared!")
print(f"  Unique roles: {staff_shifts_master['role'].unique().tolist()}")
print(f"  Unique shift types: {staff_shifts_master['shift_type'].unique().tolist()}")
print(f"  Unique employment types: {staff_shifts_master['employment_type'].unique().tolist()}")



# STEP 2: CHECK FOR NULLS IN FEATURES

print("\nChecking for missing values in features...")

check_cols = ['shift_type_num', 'dept_type_num', 'hospital_type_num',
              'role_num', 'employment_num']

for col in check_cols:
    nulls = staff_shifts_master[col].isnull().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} missing → filling with 0")
        staff_shifts_master[col] = staff_shifts_master[col].fillna(0)
    else:
        print(f"  {col}: no missing values")



# STEP 3: FEATURES AND TARGET

print("\nDefining features and target...")

features = [
    'shift_type_num',
    'dept_type_num',
    'hospital_type_num',
    'role_num',
    'employment_num',
    'years_experience',
    'patients_handled',
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



# STEP 4: TRAIN/TEST SPLIT

print("\nSplitting into train and test...")

train_data = staff_shifts_master[staff_shifts_master['shift_year'] <= 2023]
test_data  = staff_shifts_master[staff_shifts_master['shift_year'] == 2024]

X_train = train_data[features]
y_train = train_data[target].astype(int)

X_test  = test_data[features]
y_test  = test_data[target].astype(int)

print(f"  Training: {len(X_train)} rows (2021-2023)")
print(f"  Testing:  {len(X_test)} rows (2024)")
print(f"  Overtime in training: {y_train.sum()} shifts ({y_train.mean()*100:.1f}%)")
print(f"  Overtime in testing:  {y_test.sum()} shifts ({y_test.mean()*100:.1f}%)")



# STEP 5: TRAIN MODEL

print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1
)

model.fit(X_train, y_train)
print("Model trained successfully!")



# STEP 6: EVALUATE

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
print(f"  True Negatives  (correctly predicted NOT overtime): {cm[0][0]}")
print(f"  False Positives (wrongly flagged as overtime):      {cm[0][1]}")
print(f"  False Negatives (missed actual overtime):           {cm[1][0]}")
print(f"  True Positives  (correctly caught overtime):        {cm[1][1]}")

print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Overtime', 'Overtime']))



# STEP 7: FEATURE IMPORTANCE

print("\nFeature Importance (what drives overtime?)...")

importance_df = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df['importance'] = importance_df['importance'].round(4)
print(importance_df.to_string())



# STEP 8: OVERTIME RISK BY DEPARTMENT

print("\nOvertime risk analysis by department type...")

test_results = test_data[['shift_id', 'staff_id', 'department_name','hospital_name', 'department_type','role', 'shift_type', 'shift_date','patients_handled']].copy()

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
print(test_results.groupby('department_type')['actual_overtime'].mean().round(4)*100)

print("\nOvertime by shift type:")
print(test_results.groupby('shift_type')['actual_overtime'].mean().round(4)*100)



# STEP 9: SAVE RESULTS

print("\nSaving results...")

os.makedirs("Raw_data_outputs/ml_models", exist_ok=True)

test_results.to_csv(
    "Raw_data_outputs/ml_models/overtime_predictions.csv", index=False)

importance_df.to_csv(
    "Raw_data_outputs/ml_models/overtime_feature_importance.csv", index=False)

metrics_df = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
    'value':  [round(accuracy,4), round(precision,4),
               round(recall,4), round(f1,4)]
})
metrics_df.to_csv(
    "Raw_data_outputs/ml_models/overtime_model_metrics.csv", index=False)

print("\nScript 07 Complete!")
print("All 5 ML models done!")


