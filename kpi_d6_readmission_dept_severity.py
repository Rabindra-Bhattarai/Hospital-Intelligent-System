import pandas as pd, os

# D6: Readmission at department x severity grain (counts, not rate)
# Lets the Readmission sheet cross-filter on BOTH department_name and severity
# Rate is computed in Tableau so it still rolls up to 7.39 / 6.90 / 5.81 / 4.16

df = pd.read_csv("Raw_data_outputs/cleaning/admissions_master.csv")

out = df.groupby(['department_name', 'hospital_name', 'severity']).agg(
    readmitted_count = ('readmission_flag', 'sum'),
    total_count      = ('admission_id', 'count')
).reset_index()

out['readmitted_count'] = out['readmitted_count'].astype(int)

os.makedirs("Raw_data_outputs/kpi", exist_ok=True)
out.to_csv("Raw_data_outputs/kpi/kpi_readmission_by_dept_severity.csv", index=False)

print(f"Saved {len(out)} rows")
print(out.head())