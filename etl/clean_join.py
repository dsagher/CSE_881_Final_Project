import pandas as pd
import numpy as np

facility_df = pd.read_csv('../data/raw/hhs_facility_timeseries.csv')
state_df = pd.read_csv('../data/raw/hhs_state_timeseries.csv')


state_features = ['state', 'date', 
                  'inpatient_beds_utilization',      
                  'adult_icu_bed_utilization',         
                  'inpatient_beds_used',                 
                  'percent_of_inpatients_with_covid',     
                  'inpatient_bed_covid_utilization']     

state_subset = state_df[state_features].copy()
state_subset['date'] = pd.to_datetime(state_subset['date'])


facility_features = ['hospital_pk', 'state', 'collection_week',
                     'total_beds_7_day_avg',                                
                     'all_adult_hospital_inpatient_bed_occupied_7_day_avg',  
                     'inpatient_beds_used_7_day_sum',                        
                     'total_icu_beds_7_day_avg',                             
                     'previous_day_total_ed_visits_7_day_sum',               
                     'inpatient_beds_7_day_avg',      
                     'inpatient_beds_used_7_day_avg'] 

facility_subset = facility_df[facility_features].copy()
facility_subset['collection_week'] = pd.to_datetime(facility_subset['collection_week'])

# Create target: available beds
facility_subset['available_beds'] = (facility_subset['inpatient_beds_7_day_avg'] - 
                                      facility_subset['inpatient_beds_used_7_day_avg'])


state_subset = state_subset.rename(columns={'date': 'state_date'})
merged_df = facility_subset.merge(state_subset, 
                                  left_on=['state', 'collection_week'],
                                  right_on=['state', 'state_date'], 
                                  how='left')
merged_df = merged_df.drop(columns=['state_date'])

# keep only what we need
final_columns = ['hospital_pk', 'state', 'collection_week', 'available_beds',
                 'total_icu_beds_7_day_avg',
                 'previous_day_total_ed_visits_7_day_sum',
                 'inpatient_beds_utilization',
                 'adult_icu_bed_utilization',
                 'percent_of_inpatients_with_covid',
                 'inpatient_bed_covid_utilization']

output_df = merged_df[final_columns].copy()
print(output_df.isnull().sum())
output_df.to_csv('../data/clean/clean_cols.csv', index=False)





output_df_clean = output_df.dropna()
print(output_df_clean.select_dtypes(include='number').corr()['previous_day_total_ed_visits_7_day_sum'].sort_values(ascending=False).to_string())
other_cols = [col for col in output_df.columns if col != 'previous_day_total_ed_visits_7_day_sum']

sample_to_impute = output_df.sample(n=5000, random_state=881881, replace=False)
sample_to_impute.to_csv("../imputation/sample_to_impute.csv")
output_df_clean = output_df.dropna(subset=other_cols)
print(output_df_clean.isnull().sum())
sample = output_df_clean.sample(n=5000, random_state=881, replace=False)
print(sample.isnull().sum())
sample.to_csv('../imputation/sample.csv',index=False)