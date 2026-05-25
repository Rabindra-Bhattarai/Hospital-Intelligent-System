import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SCRIPT 03: OCCUPANCY PREDICTION MODEL
# Hospital Thesis - Rabindra Bhattarai


print("Loading bed occupancy data...")

bed_occupancy_master = pd.read_csv("Raw_data_outputs/cleaning/bed_occupancy_master.csv")
bed_occupancy_master['date'] = pd.to_datetime(bed_occupancy_master['date'])

print(f"Loaded: {len(bed_occupancy_master)} rows")
print(f"Date range: {bed_occupancy_master['date'].min()} to {bed_occupancy_master['date'].max()}")
print(f"Columns: {list(bed_occupancy_master.columns)}")



# STEP 1: PREPARE FEATURES FOR ML MODEL
# Machine learning needs numbers, not text.
# We extract useful number features from the date
# and convert text columns to numbers.

print("\nPreparing features...")

# Extract date features
# These help the model learn time patterns
bed_occupancy_master['year']        = bed_occupancy_master['date'].dt.year
bed_occupancy_master['month']       = bed_occupancy_master['date'].dt.month
bed_occupancy_master['day']         = bed_occupancy_master['date'].dt.day
bed_occupancy_master['day_of_week'] = bed_occupancy_master['date'].dt.dayofweek
bed_occupancy_master['quarter']     = bed_occupancy_master['date'].dt.quarter

# Convert season text to numbers
# Model cannot read text, only numbers
season_map = {'Spring': 1, 'Monsoon': 2, 'Autumn': 3, 'Winter': 4}
bed_occupancy_master['season_num'] = bed_occupancy_master['season'].map(season_map)

# Convert department_type text to numbers
dept_type_map = {'Critical': 1, 'Inpatient': 2, 'Outpatient': 3, 'Diagnostic': 4}
bed_occupancy_master['dept_type_num'] = bed_occupancy_master['department_type'].map(dept_type_map)

# Convert is_weekend to number (already True/False, convert to 1/0)
bed_occupancy_master['is_weekend_num'] = bed_occupancy_master['is_weekend'].astype(int)

print("Features prepared!")
print(f"  year, month, day, day_of_week, quarter extracted")
print(f"  season converted to numbers: {season_map}")
print(f"  department_type converted to numbers: {dept_type_map}")



# STEP 2: DEFINE FEATURES AND TARGET
# Features = inputs we give the model (X)
# Target   = what we want to predict (y)

print("\nDefining features and target...")

features = [
    'year',
    'month',
    'day',
    'day_of_week',
    'quarter',
    'season_num',
    'dept_type_num',
    'is_weekend_num',
    'total_beds',
    'admissions_count',
    'discharges_count'
]

target = 'occupancy_rate'

X = bed_occupancy_master[features]
y = bed_occupancy_master[target]

print(f"  Features used: {features}")
print(f"  Target: {target}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")



# STEP 3: TRAIN/TEST SPLIT
# We train on 2021-2023 data
# We test on 2024 data
# This is called temporal split
# It is the honest way to test forecasting

print("\nSplitting data into train and test...")

train_data = bed_occupancy_master[bed_occupancy_master['year'] <= 2023]
test_data  = bed_occupancy_master[bed_occupancy_master['year'] == 2024]

X_train = train_data[features]
y_train = train_data[target]

X_test  = test_data[features]
y_test  = test_data[target]

print(f"  Training data: {len(X_train)} rows (2021-2023)")
print(f"  Testing data:  {len(X_test)} rows (2024)")



# STEP 4: TRAIN THE MODEL
# Random Forest builds many decision trees
# and combines their predictions
# n_estimators = number of trees (100 is standard)
# random_state = fixes randomness so results are reproducible

print("\nTraining Random Forest model...")
print("This may take 1-2 minutes...")

model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1   # use all CPU cores to speed up
)

model.fit(X_train, y_train)

print("Model trained successfully!")


# STEP 5: EVALUATE THE MODEL
# We test the model on 2024 data it has never seen
# R2 Score: how well model fits (1.0 = perfect, 0 = terrible)
# MAE:      average error in occupancy rate
# RMSE:     similar to MAE but punishes big errors more

print("\nEvaluating model on 2024 test data...")

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  R2 Score: {round(r2, 4)}  (1.0 = perfect)")
print(f"  MAE:      {round(mae, 4)}  (avg error in occupancy rate)")
print(f"  RMSE:     {round(rmse, 4)}")
print(f"  In plain terms: model predictions are off by {round(mae*100, 2)}% occupancy on average")



# STEP 6: FEATURE IMPORTANCE

# This tells us which features matter most
# for predicting occupancy
# Very useful finding for thesis

print("\nFeature Importance (what drives occupancy?)...")

importance_df = pd.DataFrame({
    'feature':   features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df['importance'] = importance_df['importance'].round(4)
print(importance_df.to_string())



# STEP 7: SAVE PREDICTIONS AND RESULTS

print("\nSaving predictions...")

os.makedirs("Raw_data_outputs/ml_models", exist_ok=True)

# Save test predictions vs actual
test_results = test_data[['date', 'department_name', 
                           'hospital_name', 'department_type',
                           'season']].copy()
test_results['actual_occupancy']    = y_test.values
test_results['predicted_occupancy'] = y_pred.round(4)
test_results['prediction_error']    = (y_pred - y_test.values).round(4)

test_results.to_csv("Raw_data_outputs/ml_models/occupancy_predictions.csv", index=False)

# Save feature importance
importance_df.to_csv("Raw_data_outputs/ml_models/occupancy_feature_importance.csv", index=False)

# Save model evaluation metrics
metrics = pd.DataFrame({
    'metric': ['R2_Score', 'MAE', 'RMSE'],
    'value':  [round(r2,4), round(mae,4), round(rmse,4)]
})
metrics.to_csv("Raw_data_outputs/ml_models/occupancy_model_metrics.csv", index=False)

print("  occupancy_predictions.csv       saved")
print("  occupancy_feature_importance.csv saved")
print("  occupancy_model_metrics.csv      saved")

print("\nScript 03 Complete!")
