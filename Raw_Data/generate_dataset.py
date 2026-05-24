#!/usr/bin/env python3
"""
Hospital Dashboard - Synthetic Operations Dataset Generator
===========================================================
Relational, multi-table synthetic dataset (~500,000 rows) for a thesis:
"Design and Development of a Data-Driven Hospital Dashboard for Optimizing
 Bed Occupancy, Patient Flow and Staff Allocation using Local Hospital Data."

SYNTHETIC data with engineered, realistic correlations. NOT real patient
records. Reproducible (fixed seed). Time span: 2021-01-01 to 2024-12-31.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 20240519
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
OUT = "/home/claude"

START = datetime(2021, 1, 1)
END   = datetime(2024, 12, 31)
SPAN  = (END - START).days
ALL_DAYS = [START + timedelta(days=d) for d in range(SPAN + 1)]

# row targets (sum ~500,000)
N_PATIENTS   = 80_000
N_ADMISSIONS = 120_000
N_OCC_DAILY  = 150_000
N_FLOW       = 95_000
N_SHIFTS     = 53_000

# ---------------------------------------------------------------------------
# 1. HOSPITALS (small local network)
# ---------------------------------------------------------------------------
HOSP = [
 ("HOSP01","Valley General Hospital","Public","Kathmandu",420,"Tertiary"),
 ("HOSP02","Bagmati Community Hospital","Public","Lalitpur",180,"Secondary"),
 ("HOSP03","Himal Teaching Hospital","Teaching","Kathmandu",520,"Tertiary"),
 ("HOSP04","Bhaktapur District Hospital","Public","Bhaktapur",120,"Secondary"),
 ("HOSP05","Trishuli Referral Hospital","Public","Nuwakot",90,"Secondary"),
]
hospitals = pd.DataFrame(HOSP, columns=[
 "hospital_id","hospital_name","hospital_type","district","total_beds","level"])
hospitals.to_csv(f"{OUT}/hospitals.csv", index=False)
HOSP_IDS = hospitals.hospital_id.to_numpy()
HOSP_W = np.array([0.30,0.16,0.34,0.12,0.08])  # admission share by hospital

# ---------------------------------------------------------------------------
# 2. DEPARTMENTS
# ---------------------------------------------------------------------------
DEPTS = [
 ("Emergency","Critical",0.16),("ICU","Critical",0.05),("HDU","Critical",0.04),
 ("General Medicine","Inpatient",0.15),("General Surgery","Inpatient",0.10),
 ("Orthopedics","Inpatient",0.07),("Maternity & Obstetrics","Inpatient",0.09),
 ("Pediatrics","Inpatient",0.08),("Cardiology","Inpatient",0.05),
 ("Neurology","Inpatient",0.03),("Nephrology","Inpatient",0.025),
 ("Oncology","Inpatient",0.02),("Pulmonology","Inpatient",0.03),
 ("Gastroenterology","Inpatient",0.025),("Urology","Inpatient",0.02),
 ("ENT","Inpatient",0.015),("Ophthalmology","Inpatient",0.012),
 ("Dermatology","Outpatient",0.008),("Psychiatry","Inpatient",0.01),
 ("Burn Unit","Critical",0.008),("Neonatal ICU","Critical",0.015),
 ("Dialysis Unit","Diagnostic",0.012),
]
dep_rows=[]
did=1
for hid in HOSP_IDS:
    for nm,tp,w in DEPTS:
        dep_rows.append((f"DEP{did:04d}",nm,tp,hid,w))
        did+=1
departments = pd.DataFrame(dep_rows, columns=[
 "department_id","department_name","department_type","hospital_id","_w"])
DEP_IDS = departments.department_id.to_numpy()
DEP_NAME = departments.department_name.to_numpy()
DEP_TYPE = departments.department_type.to_numpy()
DEP_HOSP = departments.hospital_id.to_numpy()
dep_w = departments._w.to_numpy().astype(float)
dep_w = dep_w/dep_w.sum()
departments.drop(columns="_w").to_csv(f"{OUT}/departments.csv", index=False)

# ---------------------------------------------------------------------------
# 3. WARDS & BEDS (~650)
# ---------------------------------------------------------------------------
BED_TYPE_BY_DEPT = {
 "ICU":"ICU","HDU":"HDU","Neonatal ICU":"NICU","Burn Unit":"ICU",
 "Emergency":"Emergency","Maternity & Obstetrics":"Maternity",
 "Dialysis Unit":"Dialysis",
}
bed_rows=[]
bid=1
for _,d in departments.iterrows():
    nm=d.department_name
    bt = BED_TYPE_BY_DEPT.get(nm, rng.choice(["General","Private","Semi-Private"],p=[0.6,0.2,0.2]))
    # bed count by dept type
    base = {"Critical":8,"Inpatient":14,"Outpatient":4,"Diagnostic":6}[d.department_type]
    cnt = max(2,int(base + rng.integers(-3,5)))
    for k in range(cnt):
        if bid>660: break
        thisbt = BED_TYPE_BY_DEPT.get(nm,
                  rng.choice(["General","Private","Semi-Private"],p=[0.62,0.18,0.20]))
        ward = f"{nm[:14].strip()} Ward {1+k//8}"
        bed_rows.append((f"BED{bid:05d}",ward,thisbt,d.department_id,d.hospital_id,
                         bool(rng.random()>0.03)))
        bid+=1
wards_beds = pd.DataFrame(bed_rows, columns=[
 "bed_id","ward_name","bed_type","department_id","hospital_id","is_functional"])
wards_beds.to_csv(f"{OUT}/wards_beds.csv", index=False)
BED_IDS = wards_beds.bed_id.to_numpy()
BED_DEP = wards_beds.department_id.to_numpy()

print("hospitals:",hospitals.shape,"| departments:",departments.shape,
      "| wards_beds:",wards_beds.shape)

# ---------------------------------------------------------------------------
# 4. STAFF (~1200)
# ---------------------------------------------------------------------------
ROLE_MIX = [("Consultant",0.10),("Medical Officer",0.16),("Resident Doctor",0.10),
 ("Staff Nurse",0.34),("Senior Nurse",0.08),("Lab Technician",0.06),
 ("Radiology Technician",0.03),("Pharmacist",0.03),("Support Staff",0.10)]
roles=[r for r,_ in ROLE_MIX]; rolep=np.array([w for _,w in ROLE_MIX]); rolep/=rolep.sum()
N_STAFF=1200
staff_dep = rng.choice(len(DEP_IDS), N_STAFF, p=dep_w)
roles_arr = rng.choice(roles, N_STAFF, p=rolep)
SPEC = {"Cardiology":"Cardiologist","Neurology":"Neurologist","ICU":"Intensivist",
 "Pediatrics":"Pediatrician","General Surgery":"Surgeon","Orthopedics":"Orthopedic Surgeon",
 "Maternity & Obstetrics":"Obstetrician","Oncology":"Oncologist","Nephrology":"Nephrologist"}
staff = pd.DataFrame({
 "staff_id":[f"STF{i:05d}" for i in range(1,N_STAFF+1)],
 "role":roles_arr,
 "department_id":DEP_IDS[staff_dep],
 "hospital_id":DEP_HOSP[staff_dep],
 "specialization":[SPEC.get(DEP_NAME[d],"General") if r in
   ("Consultant","Medical Officer","Resident Doctor") else "-"
   for d,r in zip(staff_dep,roles_arr)],
 "shift_type":rng.choice(["Morning","Evening","Night","Rotating"],N_STAFF,p=[0.34,0.22,0.16,0.28]),
 "employment_type":rng.choice(["Permanent","Contract","Visiting","Intern"],N_STAFF,p=[0.55,0.28,0.07,0.10]),
 "years_experience":np.clip(rng.gamma(2.2,4,N_STAFF),0,40).round(1),
 "on_roster":rng.choice([True,False],N_STAFF,p=[0.88,0.12]),
})
staff.to_csv(f"{OUT}/staff.csv", index=False)
STAFF_IDS=staff.staff_id.to_numpy(); STAFF_DEP=staff.department_id.to_numpy()
DOCTOR_MASK = np.isin(staff.role, ["Consultant","Medical Officer","Resident Doctor"])
DOCTOR_IDS = STAFF_IDS[DOCTOR_MASK]; DOCTOR_DEP = STAFF_DEP[DOCTOR_MASK]

# ---------------------------------------------------------------------------
# 5. PATIENTS (80k)
# ---------------------------------------------------------------------------
DISTRICTS=["Kathmandu","Lalitpur","Bhaktapur","Nuwakot","Kavre","Dhading",
 "Makwanpur","Sindhupalchok","Ramechhap","Chitwan","Rasuwa","Sindhuli"]
DIST_W=np.array([0.34,0.16,0.12,0.07,0.07,0.05,0.04,0.04,0.03,0.04,0.02,0.02]); DIST_W/=DIST_W.sum()
age = np.clip(rng.normal(38,21,N_PATIENTS),0,98).astype(int)
def agegrp(a):
    return pd.cut(a,[-1,1,5,12,18,35,50,65,200],
      labels=["Infant","Toddler","Child","Adolescent","Young Adult",
              "Middle Age","Senior","Elderly"]).astype(str)
patients = pd.DataFrame({
 "patient_id":[f"PAT{i:06d}" for i in range(1,N_PATIENTS+1)],
 "age":age,"age_group":agegrp(age),
 "gender":rng.choice(["Male","Female","Other"],N_PATIENTS,p=[0.515,0.48,0.005]),
 "district":rng.choice(DISTRICTS,N_PATIENTS,p=DIST_W),
 "address_type":rng.choice(["Urban","Semi-Urban","Rural"],N_PATIENTS,p=[0.46,0.28,0.26]),
 "blood_group":rng.choice(["O+","A+","B+","AB+","O-","A-","B-","AB-"],
   N_PATIENTS,p=[0.34,0.27,0.22,0.07,0.04,0.03,0.02,0.01]),
 "insurance_type":rng.choice(["None","Government Scheme","Private Insurance","Employer"],
   N_PATIENTS,p=[0.44,0.33,0.13,0.10]),
 "has_chronic_condition":rng.choice([True,False],N_PATIENTS,p=[0.27,0.73]),
})
patients.loc[patients.sample(frac=0.06,random_state=1).index,"insurance_type"]=np.nan
patients.loc[patients.sample(frac=0.03,random_state=2).index,"blood_group"]=np.nan
patients.to_csv(f"{OUT}/patients.csv", index=False)
PAT_IDS=patients.patient_id.to_numpy()
PAT_AGE=patients.age.to_numpy()
PAT_CHRONIC=patients.has_chronic_condition.to_numpy()

print("staff:",staff.shape,"| patients:",patients.shape)

# ---------------------------------------------------------------------------
# 6. ADMISSIONS (120k) - core table w/ engineered LOS, severity, outcome
# ---------------------------------------------------------------------------
NA=N_ADMISSIONS
adm_pat = rng.integers(0,N_PATIENTS,NA)
adm_hosp_i = rng.choice(len(HOSP_IDS),NA,p=HOSP_W)
adm_hosp = HOSP_IDS[adm_hosp_i]
# pick a department within that hospital
dep_by_hosp = {h:np.where(DEP_HOSP==h)[0] for h in HOSP_IDS}
adm_dep_idx = np.empty(NA,dtype=int)
for h in HOSP_IDS:
    m=np.where(adm_hosp==h)[0]
    pool=dep_by_hosp[h]
    pw=dep_w[pool]/dep_w[pool].sum()
    adm_dep_idx[m]=rng.choice(pool,size=len(m),p=pw)
adm_dep = DEP_IDS[adm_dep_idx]
adm_depname = DEP_NAME[adm_dep_idx]

# admission date with seasonal effects
day_w=np.ones(len(ALL_DAYS))
for i,dt in enumerate(ALL_DAYS):
    m=dt.month; w=1.0
    if m in (6,7,8,9): w*=1.30          # monsoon: dengue/waterborne/respiratory
    if m in (12,1): w*=1.18             # winter respiratory
    if m in (10,11): w*=1.12            # festival accidents (Dashain/Tihar)
    if dt.weekday()>=5: w*=0.93         # slightly fewer elective on weekends
    day_w[i]=w
day_w/=day_w.sum()
adm_day=rng.choice(len(ALL_DAYS),NA,p=day_w)
adm_date=np.array(ALL_DAYS,dtype='datetime64[s]')[adm_day]
adm_hour=rng.integers(0,24,NA)

adm_type=rng.choice(["Emergency","Elective","Referral","Walk-in"],NA,p=[0.42,0.24,0.16,0.18])
DIAG=["Respiratory Infection","Gastroenteritis","Cardiovascular","Injury/Trauma",
 "Obstetric","Pediatric Illness","Renal/Urology","Neurological","Infectious Disease",
 "Oncology","Diabetes/Endocrine","Surgical (Elective)","Orthopedic","Other"]
diag=rng.choice(DIAG,NA,p=np.array(
 [0.13,0.11,0.10,0.13,0.09,0.09,0.05,0.05,0.07,0.03,0.04,0.05,0.04,0.02]))
# severity influenced by department type + age
dep_is_crit=np.isin(DEP_TYPE[adm_dep_idx],["Critical"])
sev_score=(rng.normal(1.6,0.7,NA)
           + dep_is_crit*1.4
           + (PAT_AGE[adm_pat]>65)*0.5
           + (adm_type=="Emergency")*0.4
           + PAT_CHRONIC[adm_pat]*0.3)
severity=np.select([sev_score<1.3,sev_score<2.1,sev_score<3.0],
                    ["Mild","Moderate","Severe"],default="Critical")
SEV_N=pd.Series(severity).map({"Mild":1,"Moderate":2,"Severe":3,"Critical":4}).to_numpy()

# length of stay: depends on severity, age, dept
base_los=np.clip(rng.gamma(2.0, 1.6, NA),0.2,None)
los=(base_los*(0.7+0.9*SEV_N)
     *np.where(dep_is_crit,1.6,1.0)
     *(1+(PAT_AGE[adm_pat]>65)*0.25)).round().astype(int)
los=np.clip(los,0,90)
disch_date=adm_date+los.astype('timedelta64[D]')
disch_date=np.minimum(disch_date,np.datetime64(END))

# outcome influenced by severity
out_p=np.zeros((NA,5))
# columns: Recovered, Referred, LAMA, Transferred, Expired
for s in (1,2,3,4):
    m=SEV_N==s
    table={1:[0.93,0.02,0.03,0.015,0.005],
           2:[0.86,0.05,0.04,0.03,0.02],
           3:[0.70,0.10,0.05,0.07,0.08],
           4:[0.52,0.10,0.04,0.10,0.24]}[s]
    out_p[m]=table
u=rng.random(NA)
cum=np.cumsum(out_p,axis=1)
oc=(u[:,None]>cum).sum(1)
OUTC=np.array(["Recovered","Referred","LAMA","Transferred","Expired"])
outcome=OUTC[oc]
readmit=(oc==0)&(rng.random(NA)<0.08)

attending=np.empty(NA,dtype=object)
for i in range(0,NA,40000):
    sl=slice(i,min(i+40000,NA))
    attending[sl]=rng.choice(DOCTOR_IDS,size=sl.stop-sl.start)
bed_assigned=BED_IDS[rng.integers(0,len(BED_IDS),NA)]
bill=( (1500 + SEV_N*4200 + los*2300 + dep_is_crit*9000)
       *(1+rng.normal(0,0.18,NA)) ).round().clip(300).astype(np.int64)

admissions=pd.DataFrame({
 "admission_id":[f"ADM{i:07d}" for i in range(1,NA+1)],
 "patient_id":PAT_IDS[adm_pat],
 "hospital_id":adm_hosp,
 "department_id":adm_dep,
 "department_name":adm_depname,
 "bed_id":bed_assigned,
 "admission_date":pd.to_datetime(adm_date).date,
 "admission_hour":adm_hour,
 "discharge_date":pd.to_datetime(disch_date).date,
 "length_of_stay_days":los,
 "admission_type":adm_type,
 "diagnosis_category":diag,
 "severity":severity,
 "attending_staff_id":attending,
 "discharge_outcome":outcome,
 "readmission_flag":readmit,
 "total_bill_npr":bill,
})
admissions.to_csv(f"{OUT}/admissions.csv", index=False)
print("admissions:",admissions.shape)

# ---------------------------------------------------------------------------
# 7. BED_OCCUPANCY_DAILY (150k) - the dashboard core
#    grain: one row per (department, ward-group, day) sampled
# ---------------------------------------------------------------------------
# bed counts per department
bed_per_dep = wards_beds.groupby("department_id").size().to_dict()
dep_ids_list = list(bed_per_dep.keys())

rows_needed = N_OCC_DAILY
# build (dep, day) combos
combo_dep = rng.choice(len(DEP_IDS), rows_needed)
combo_day = rng.integers(0, len(ALL_DAYS), rows_needed)
occ=[]
dep_type_map=dict(zip(range(len(DEP_IDS)),DEP_TYPE))
for k in range(rows_needed):
    di=combo_dep[k]
    dep_id=DEP_IDS[di]
    total=bed_per_dep.get(dep_id, rng.integers(6,20))
    dt=ALL_DAYS[combo_day[k]]
    m=dt.month
    # base occupancy rate by dept type
    dt_type=DEP_TYPE[di]
    base={"Critical":0.82,"Inpatient":0.68,"Outpatient":0.40,"Diagnostic":0.55}[dt_type]
    seas=1.0
    if m in (6,7,8,9): seas+=0.12
    if m in (12,1): seas+=0.08
    if dt.weekday()>=5 and dt_type=="Inpatient": seas-=0.05
    rate=np.clip(base*seas + rng.normal(0,0.08),0.05,1.0)
    occ_beds=int(round(total*rate))
    occ_beds=min(occ_beds,total)
    adm_c=max(0,int(rng.poisson(total*0.18*seas)))
    dis_c=max(0,int(rng.poisson(total*0.16*seas)))
    occ.append((f"OCC{k+1:07d}",dt.date(),dep_id,total,occ_beds,total-occ_beds,
                round(occ_beds/total,4),adm_c,dis_c,
                dt.weekday()>=5,
                ("Monsoon" if m in (6,7,8,9) else "Winter" if m in (12,1,2)
                 else "Spring" if m in (3,4,5) else "Autumn")))
bed_occupancy_daily=pd.DataFrame(occ,columns=[
 "record_id","date","department_id","total_beds","occupied_beds","available_beds",
 "occupancy_rate","admissions_count","discharges_count","is_weekend","season"])
bed_occupancy_daily.to_csv(f"{OUT}/bed_occupancy_daily.csv", index=False)
print("bed_occupancy_daily:",bed_occupancy_daily.shape)

# ---------------------------------------------------------------------------
# 8. PATIENT_FLOW_EVENTS (95k) - movement through hospital
# ---------------------------------------------------------------------------
NF=N_FLOW
f_adm=rng.integers(0,NA,NF)
EVENTS=["ER Arrival","Triage","Bed Assigned","Ward Transfer","Discharge","Referral"]
ev=rng.choice(EVENTS,NF,p=[0.22,0.20,0.20,0.13,0.18,0.07])
# wait time rises with implied occupancy/severity
sev_of=SEV_N[f_adm]
wait=np.clip(rng.gamma(2.0,18,NF)*(1+0.25*sev_of)
             *np.where(np.isin(ev,["ER Arrival","Triage","Bed Assigned"]),1.4,0.7),
             1,1400).astype(int)
adm_dt=pd.to_datetime(admissions.admission_date.to_numpy()[f_adm])
ev_offset=rng.integers(0,72,NF)  # hours after admission
ev_ts=adm_dt + pd.to_timedelta(ev_offset,unit="h")
from_dep=DEP_NAME[rng.integers(0,len(DEP_IDS),NF)]
to_dep=np.where(np.isin(ev,["Ward Transfer","Referral"]),
                DEP_NAME[rng.integers(0,len(DEP_IDS),NF)],"-")
patient_flow_events=pd.DataFrame({
 "event_id":[f"EVT{i:07d}" for i in range(1,NF+1)],
 "admission_id":admissions.admission_id.to_numpy()[f_adm],
 "patient_id":admissions.patient_id.to_numpy()[f_adm],
 "event_type":ev,
 "event_timestamp":ev_ts,
 "from_department":from_dep,
 "to_department":to_dep,
 "wait_minutes":wait,
})
patient_flow_events.to_csv(f"{OUT}/patient_flow_events.csv", index=False)
patient_flow_events.to_csv(f"{OUT}/patient_flow_events.csv.gz", index=False, compression="gzip")
print("patient_flow_events:",patient_flow_events.shape)

# ---------------------------------------------------------------------------
# 9. STAFF_SHIFTS (53k) - roster w/ staffing-strain signal
# ---------------------------------------------------------------------------
NS=N_SHIFTS
s_staff=rng.integers(0,N_STAFF,NS)
s_day=rng.integers(0,len(ALL_DAYS),NS)
s_date=np.array([ALL_DAYS[d].date() for d in s_day])
s_month=np.array([ALL_DAYS[d].month for d in s_day])
stype=rng.choice(["Morning","Evening","Night"],NS,p=[0.40,0.32,0.28])
# patients handled rises in monsoon/winter (strain) -> serves "staff allocation"
strain=np.ones(NS)
strain[np.isin(s_month,[6,7,8,9])]=1.25
strain[np.isin(s_month,[12,1])]=1.15
base_load=np.where(stype=="Night",6,9)
patients_handled=np.clip((base_load*strain*(0.6+rng.gamma(2.0,0.5,NS)))
                         .round(),0,60).astype(int)
is_ot=patients_handled> (base_load*1.6)
staff_shifts=pd.DataFrame({
 "shift_id":[f"SHF{i:07d}" for i in range(1,NS+1)],
 "staff_id":STAFF_IDS[s_staff],
 "department_id":STAFF_DEP[s_staff],
 "shift_date":s_date,
 "shift_type":stype,
 "patients_handled":patients_handled,
 "is_overtime":is_ot,
})
staff_shifts.to_csv(f"{OUT}/staff_shifts.csv", index=False)
print("staff_shifts:",staff_shifts.shape)

# ---------------------------------------------------------------------------
# 10. MERGED FLAT FILE (admission grain + patient + dept + outcome)
# ---------------------------------------------------------------------------
flat=admissions.merge(patients,on="patient_id",how="left",suffixes=("","_pat"))
flat=flat.merge(departments.drop(columns="_w"),on="department_id",how="left",suffixes=("","_dep"))
flat=flat.merge(hospitals[["hospital_id","hospital_name","hospital_type","level"]],
                on="hospital_id",how="left")
flat.to_csv(f"{OUT}/admissions_flat.csv", index=False)
flat.to_csv(f"{OUT}/admissions_flat.csv.gz", index=False, compression="gzip")
print("admissions_flat:",flat.shape)

TABLES={"hospitals.csv":hospitals,"departments.csv":departments.drop(columns="_w"),
 "wards_beds.csv":wards_beds,"staff.csv":staff,"patients.csv":patients,
 "admissions.csv":admissions,"bed_occupancy_daily.csv":bed_occupancy_daily,
 "patient_flow_events.csv":patient_flow_events,"staff_shifts.csv":staff_shifts,
 "admissions_flat.csv":flat}
tot=sum(len(v) for k,v in TABLES.items() if k!="admissions_flat.csv")
print("TOTAL relational rows (excl flat):", f"{tot:,}")
for k,v in TABLES.items(): print(f"  {k}: {len(v):,} x {v.shape[1]}")
