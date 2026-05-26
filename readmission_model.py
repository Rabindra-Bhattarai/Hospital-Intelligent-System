# SCRIPT 05: READMISSION CLASSIFICATION MODEL
# Hospital Thesis - Rabindra Bhattarai

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score,confusion_matrix, classification_report)




print("Loading admissions data...")

admissions_master = pd.read_csv("Raw_data_outputs/cleaning/admissions_master.csv")
admissions_master['admission_date'] = pd.to_datetime(admissions_master['admission_date'])

print(f"Loaded: {len(admissions_master)} rows")
print(f"Readmission rate: {admissions_master['readmission_flag'].mean()*100:.2f}%")
print(f"Readmitted:     {admissions_master['readmission_flag'].sum()} patients")
print(f"Not readmitted: {(admissions_master['readmission_flag']==False).sum()} patients")



# STEP 1: PREPARE FEATURES

print("\nPreparing features...")

# Convert text to numbers
severity_map     = {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4}
admission_map    = {'Walk-in': 1, 'Elective': 2, 'Referral': 3, 'Emergency': 4}
dept_type_map    = {'Outpatient': 1, 'Diagnostic': 2, 'Inpatient': 3, 'Critical': 4}
hospital_map     = {'Public': 1, 'Private': 2, 'Teaching': 3}
level_map        = {'Secondary': 1, 'Tertiary': 2}
address_map      = {'Rural': 1, 'Semi-Urban': 2, 'Urban': 3}
gender_map       = {'Female': 0, 'Male': 1}
outcome_map      = {'Recovered': 1, 'Referred': 2, 'Transferred': 3, 'LAMA': 4, 'Expired': 5}

admissions_master['severity_num']      = admissions_master['severity'].map(severity_map)
admissions_master['admission_type_num']= admissions_master['admission_type'].map(admission_map)
admissions_master['dept_type_num']     = admissions_master['department_type'].map(dept_type_map)
admissions_master['hospital_type_num'] = admissions_master['hospital_type'].map(hospital_map)
admissions_master['level_num']         = admissions_master['level'].map(level_map)
admissions_master['address_type_num']  = admissions_master['address_type'].map(address_map)
admissions_master['gender_num']        = admissions_master['gender'].map(gender_map)
admissions_master['outcome_num']       = admissions_master['discharge_outcome'].map(outcome_map)
admissions_master['chronic_num']       = admissions_master['has_chronic_condition'].astype(int)
admissions_master['admission_month']   = admissions_master['admission_date'].dt.month
admissions_master['admission_year']    = admissions_master['admission_date'].dt.year

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
    'outcome_num',
    'age',
    'length_of_stay_days',
    'admission_hour',
    'admission_month',
    'admission_year',
    'total_bill_npr'
]

target = 'readmission_flag'

X = admissions_master[features]
y = admissions_master[target]

print(f"  Features: {len(features)} features")
print(f"  Target: {target}")
print(f"  X shape: {X.shape}")



# STEP 3: TRAIN/TEST SPLIT

print("\nSplitting into train and test...")

train_data = admissions_master[admissions_master['admission_year'] <= 2023]
test_data  = admissions_master[admissions_master['admission_year'] == 2024]

X_train = train_data[features]
y_train = train_data[target]

X_test  = test_data[features]
y_test  = test_data[target]

print(f"  Training data: {len(X_train)} rows (2021-2023)")
print(f"  Testing data:  {len(X_test)} rows (2024)")
print(f"  Readmissions in test: {y_test.sum()} patients")



# STEP 4: TRAIN THE MODEL

print("\nTraining Random Forest Classifier...")
print("This may take 1-2 minutes...")

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1,
    class_weight = 'balanced'
    # class_weight balanced is important because
    # only 6% of patients are readmitted
    # without this the model would just predict
    # everyone as NOT readmitted and be lazy
)

model.fit(X_train, y_train)
print("Model trained successfully!")



# STEP 5: EVALUATE THE MODEL

# Classification uses different metrics than regression
# Accuracy:  overall how often is model correct
# Precision: when model says readmitted, how often right
# Recall:    of all actual readmissions, how many caught
# F1:        balance between precision and recall

print("\nEvaluating model on 2024 test data...")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"  Accuracy:  {round(accuracy,4)}  (overall correct predictions)")
print(f"  Precision: {round(precision,4)} (when model flags readmission, how often right)")
print(f"  Recall:    {round(recall,4)}  (of all real readmissions, how many caught)")
print(f"  F1 Score:  {round(f1,4)}  (balance of precision and recall)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negatives  (correctly said NOT readmitted): {cm[0][0]}")
print(f"  False Positives (wrongly flagged as readmitted):  {cm[0][1]}")
print(f"  False Negatives (missed actual readmissions):     {cm[1][0]}")
print(f"  True Positives  (correctly caught readmissions):  {cm[1][1]}")



# STEP 6: FEATURE IMPORTANCE

print("\nFeature Importance (what drives readmission?)...")

importance_df = pd.DataFrame({
    'feature':    features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df['importance'] = importance_df['importance'].round(4)
print(importance_df.to_string())



# STEP 7: SAVE RESULTS

print("\nSaving results...")

os.makedirs("Raw_data_outputs/ml_models", exist_ok=True)

# Save predictions with risk probability
test_results = test_data[['admission_id', 'patient_id','department_name', 'hospital_name','severity', 'admission_type','age', 'length_of_stay_days','discharge_outcome']].copy()

test_results['actual_readmission']      = y_test.values
test_results['predicted_readmission']   = y_pred
test_results['readmission_probability'] = y_pred_prob.round(4)

# Add risk label based on probability
def risk_label(prob):
    if prob >= 0.70:
        return 'HIGH RISK'
    elif prob >= 0.40:
        return 'MEDIUM RISK'
    else:
        return 'LOW RISK'

test_results['risk_label'] = test_results['readmission_probability'].apply(risk_label)

test_results.to_csv("Raw_data_outputs/ml_models/readmission_predictions.csv", index=False)

# Save feature importance
importance_df.to_csv("Raw_data_outputs/ml_models/readmission_feature_importance.csv", index=False)

# Save metrics
metrics = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
    'value':  [round(accuracy,4), round(precision,4),round(recall,4), round(f1,4)]
})
metrics.to_csv("Raw_data_outputs/ml_models/readmission_model_metrics.csv", index=False)

# Show risk distribution
print("\nRisk Label Distribution in 2024 predictions:")
print(test_results['risk_label'].value_counts())

print("\n  readmission_predictions.csv       saved")
print("  readmission_feature_importance.csv saved")
print("  readmission_model_metrics.csv      saved")

print("\nScript 05 Complete!")
print("All 3 ML models done!")




