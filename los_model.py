# SCRIPT 04: LENGTH OF STAY PREDICTION MODEL
# Hospital Thesis - Rabindra Bhattarai

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



print("Loading admissions data...")

admissions_master = pd.read_csv("Raw_data_outputs/cleaning/admissions_master.csv")
admissions_master['admission_date'] = pd.to_datetime(admissions_master['admission_date'])
admissions_master['discharge_date'] = pd.to_datetime(admissions_master['discharge_date'])

print(f"Loaded: {len(admissions_master)} rows")
print(f"Columns: {list(admissions_master.columns)}")
print(f"Avg LOS: {admissions_master['length_of_stay_days'].mean():.2f} days")


# STEP 1: PREPARE FEATURES

# We convert all text columns to numbers
# because ML models only understand numbers

print("\nPreparing features...")

# Severity: how sick is the patient
severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}
admissions_master['severity_num'] = admissions_master['severity'].map(severity_map)

# Admission type
admission_map = {'Walk-in': 1, 'Elective': 2, 'Referral': 3, 'Emergency': 4}
admissions_master['admission_type_num'] = admissions_master['admission_type'].map(admission_map)

# Department type
dept_type_map = {'Outpatient': 1, 'Diagnostic': 2, 'Inpatient': 3, 'Critical': 4}
admissions_master['dept_type_num'] = admissions_master['department_type'].map(dept_type_map)

# Hospital type
hospital_map = {'Public': 1, 'Private': 2, 'Teaching': 3}
admissions_master['hospital_type_num'] = admissions_master['hospital_type'].map(hospital_map)

# Hospital level
level_map = {'Secondary': 1, 'Tertiary': 2}
admissions_master['level_num'] = admissions_master['level'].map(level_map)

# Address type (urban vs rural)
address_map = {'Rural': 1, 'Semi-Urban': 2, 'Urban': 3}
admissions_master['address_type_num'] = admissions_master['address_type'].map(address_map)

# Gender
gender_map = {'Female': 0, 'Male': 1}
admissions_master['gender_num'] = admissions_master['gender'].map(gender_map)

# Chronic condition (already True/False, convert to 1/0)
admissions_master['chronic_num'] = admissions_master['has_chronic_condition'].astype(int)

# Extract admission month and hour
admissions_master['admission_month'] = admissions_master['admission_date'].dt.month
admissions_master['admission_year']  = admissions_master['admission_date'].dt.year

print("Features prepared!")



# STEP 2: DEFINE FEATURES AND TARGET


print("\nDefining features and target...")

features = [
    'severity_num',
    'admission_type_num',
    'dept_type_num',
    'hospital_type_num',
    'level_num',
    'address_type_num',
    'gender_num',
    'chronic_num',
    'age',
    'admission_hour',
    'admission_month',
    'admission_year'
]

target = 'length_of_stay_days'

X = admissions_master[features]
y = admissions_master[target]

print(f"  Features: {features}")
print(f"  Target: {target}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")



# STEP 3: TRAIN/TEST SPLIT

# Train on 2021-2023, test on 2024
# Same honest temporal split as occupancy model

print("\nSplitting into train and test...")

train_data = admissions_master[admissions_master['admission_year'] <= 2023]
test_data  = admissions_master[admissions_master['admission_year'] == 2024]

X_train = train_data[features]
y_train = train_data[target]

X_test  = test_data[features]
y_test  = test_data[target]

print(f"  Training data: {len(X_train)} rows (2021-2023)")
print(f"  Testing data:  {len(X_test)} rows (2024)")



# STEP 4: TRAIN THE MODEL


print("\nTraining Random Forest model...")
print("This may take 1-2 minutes...")

model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1
)

model.fit(X_train, y_train)
print("Model trained successfully!")



# STEP 5: EVALUATE THE MODEL


print("\nEvaluating model on 2024 test data...")

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  R2 Score: {round(r2, 4)}  (1.0 = perfect)")
print(f"  MAE:      {round(mae, 4)} days (avg error)")
print(f"  RMSE:     {round(rmse, 4)}")
print(f"  In plain terms: predictions are off by {round(mae, 2)} days on average")



# STEP 6: FEATURE IMPORTANCE


print("\nFeature Importance (what drives length of stay?)...")

importance_df = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df['importance'] = importance_df['importance'].round(4)
print(importance_df.to_string())



# STEP 7: SAVE RESULTS
print("\nSaving results...")

os.makedirs("Raw_data_outputs/ml_models", exist_ok=True)

# Save predictions vs actual
test_results = test_data[['admission_id', 'patient_id',
                           'department_name', 'hospital_name',
                           'severity', 'admission_type',
                           'age', 'has_chronic_condition']].copy()

test_results['actual_los']    = y_test.values
test_results['predicted_los'] = y_pred.round(1)
test_results['error_days']    = (y_pred - y_test.values).round(1)

# Add risk category based on predicted LOS
def los_risk(days):
    if days >= 14:
        return 'HIGH'
    elif days >= 7:
        return 'MEDIUM'
    else:
        return 'LOW'

test_results['los_risk_category'] = test_results['predicted_los'].apply(los_risk)

test_results.to_csv("Raw_data_outputs/ml_models/los_predictions.csv", index=False)

# Save feature importance
importance_df.to_csv("Raw_data_outputs/ml_models/los_feature_importance.csv", index=False)

# Save metrics
metrics = pd.DataFrame({
    'metric': ['R2_Score', 'MAE_days', 'RMSE'],
    'value':  [round(r2,4), round(mae,4), round(rmse,4)]
})
metrics.to_csv("Raw_data_outputs/ml_models/los_model_metrics.csv", index=False)

# Show risk category distribution
print("\nLOS Risk Category Distribution in 2024:")
print(test_results['los_risk_category'].value_counts())

print("\n  los_predictions.csv       saved")
print("  los_feature_importance.csv saved")
print("  los_model_metrics.csv      saved")

print("\nScript 04 Complete!")
print("Ready for Script 05: Readmission Classification")