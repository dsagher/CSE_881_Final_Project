import pandas as pd

# -----------------------------------------------------------
# 1. READ WITH CORRECT FORMATTING
# -----------------------------------------------------------
hospital = pd.read_csv(
    "data/raw/hhs_facility_timeseries.csv",
    quotechar='"',
    dtype=str,                  # read everything as string first
    na_values=["", " "],       # treat empty fields as NaN
    keep_default_na=True
)

state = pd.read_csv(
    "data/raw/hhs_state_timeseries.csv",
    quotechar='"',
    dtype=str,
    na_values=["", " "],
    keep_default_na=True
)

# -----------------------------------------------------------
# 2. PARSE DATETIME COLUMNS
# -----------------------------------------------------------
hospital["collection_week"] = pd.to_datetime(
    hospital["collection_week"], format="%Y-%m-%dT%H:%M:%S.%f"
).dt.date

state["date"] = pd.to_datetime(
    state["date"], format="%Y-%m-%dT%H:%M:%S.%f"
).dt.date

# -----------------------------------------------------------
# 3. SELECT RELEVANT COLUMNS
# -----------------------------------------------------------
hospital_selected = hospital[[
    "state",
    "collection_week",
    "inpatient_beds_7_day_avg",
    "inpatient_beds_used_7_day_avg",
    "inpatient_beds_used_covid_7_day_avg",
    "total_staffed_adult_icu_beds_7_day_avg",
    "icu_beds_used_7_day_avg"
]].copy()

state_selected = state[[
    "state",
    "date",
    "inpatient_beds",
    "inpatient_beds_used",
    "inpatient_beds_used_covid",
    "total_staffed_adult_icu_beds",
    "staffed_adult_icu_bed_occupancy"
]].copy()

# -----------------------------------------------------------
# 4. CAST NUMERIC COLUMNS
# -----------------------------------------------------------
hospital_numeric = [
    "inpatient_beds_7_day_avg",
    "inpatient_beds_used_7_day_avg",
    "inpatient_beds_used_covid_7_day_avg",
    "total_staffed_adult_icu_beds_7_day_avg",
    "icu_beds_used_7_day_avg"
]

state_numeric = [
    "inpatient_beds",
    "inpatient_beds_used",
    "inpatient_beds_used_covid",
    "total_staffed_adult_icu_beds",
    "staffed_adult_icu_bed_occupancy"
]

hospital_selected[hospital_numeric] = hospital_selected[hospital_numeric].apply(
    pd.to_numeric, errors="coerce"   # unparseable strings → NaN
)

state_selected[state_numeric] = state_selected[state_numeric].apply(
    pd.to_numeric, errors="coerce"
)

# -----------------------------------------------------------
# 5. AGGREGATE HOSPITAL TO STATE LEVEL
#    (drop NaN rows before aggregating so they don't skew means)
# -----------------------------------------------------------
hospital_agg = (
    hospital_selected
    .dropna(subset=hospital_numeric, how="all")   # drop rows with ALL nulls
    .groupby(["state", "collection_week"])
    .agg(
        inpatient_beds_avg               = ("inpatient_beds_7_day_avg",               "mean"),
        inpatient_beds_used_avg          = ("inpatient_beds_used_7_day_avg",           "mean"),
        inpatient_beds_used_covid_avg    = ("inpatient_beds_used_covid_7_day_avg",     "mean"),
        total_staffed_adult_icu_beds_avg = ("total_staffed_adult_icu_beds_7_day_avg",  "mean"),
        icu_beds_used_avg                = ("icu_beds_used_7_day_avg",                 "mean"),
        hospital_count                   = ("inpatient_beds_7_day_avg",               "count") # how many hospitals contributed
    )
    .reset_index()
)

# -----------------------------------------------------------
# 6. JOIN ON STATE + DATE
# -----------------------------------------------------------
combined = hospital_agg.merge(
    state_selected,
    left_on  = ["state", "collection_week"],
    right_on = ["state", "date"],
    how      = "inner"
)

# -----------------------------------------------------------
# 7. QUICK SENSE CHECK
# -----------------------------------------------------------
print(combined.shape)
print(combined.dtypes)
print(combined.head())