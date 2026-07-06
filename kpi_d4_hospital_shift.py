import pandas as pd
import os

df = pd.read_csv("Raw_data_outputs/cleaning/staff_shifts_master.csv")

result = df.groupby(['hospital_name', 'shift_type']).agg(
    overtime_rate        = ('is_overtime', 'mean'),
    avg_patients_handled = ('patients_handled', 'mean'),
    total_shifts         = ('shift_id', 'count')
).reset_index()

result['overtime_rate']        = result['overtime_rate'].round(4)
result['avg_patients_handled'] = result['avg_patients_handled'].round(2)

os.makedirs("Raw_data_outputs/kpi", exist_ok=True)
result.to_csv("Raw_data_outputs/kpi/kpi_overtime_by_shift_hospital.csv", index=False)
print("Done! kpi_overtime_by_shift_hospital.csv saved.")