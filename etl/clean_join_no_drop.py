import pandas as pd
import numpy as np

# Join the dataset without dropping any columns
# It will be needed to assess feature importance in RandomForest

facility_df = pd.read_csv('../data/raw/hhs_facility_timeseries.csv')
state_df = pd.read_csv('../data/raw/hhs_state_timeseries.csv')
facility_df = facility_df.sample(frac=0.05)
state_subset = state_df.copy()
state_subset['date'] = pd.to_datetime(state_subset['date'])

facility_subset = facility_df.copy()
facility_subset['collection_week'] = pd.to_datetime(facility_subset['collection_week'])

# target that I used
facility_subset['available_beds'] = (
    facility_subset['inpatient_beds_7_day_avg'] - 
    facility_subset['inpatient_beds_used_7_day_avg']
)

facility_subset = facility_subset.drop(columns=['inpatient_beds_7_day_avg', 'inpatient_beds_used_7_day_avg'])

state_subset = state_subset.rename(columns={'date': 'state_date'})

merged_df = facility_subset.merge(
    state_subset,
    left_on=['state', 'collection_week'],
    right_on=['state', 'state_date'],
    how='left'
)

merged_df = merged_df.drop(columns=['state_date'])

merged_df.to_csv('../data/clean/preprocessed_data_full.csv', index=False)