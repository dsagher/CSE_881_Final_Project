import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Analysis of feature importance for Random Forest
# To get preprocessed_data_full.csv, you need to run to
# run 'clean_join_no_drop.py' in "etl" folder

df = pd.read_csv('../data/clean/preprocessed_data_full.csv')
# only take a small sample of 10k
sample = df.sample(n=10000, random_state=42, replace=False)

median_beds = sample['available_beds'].median()
sample['available_beds_class'] = (sample['available_beds'] > median_beds).astype(int)

features = [col for col in sample.columns if col not in 
            ['available_beds', 'available_beds_class', 'hospital_pk', 'state', 'collection_week']]

X = sample[features].select_dtypes(include='number')
y = sample.loc[X.index, 'available_beds_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Accuracy: {rf.score(X_test, y_test):.4f}")

importance_df = pd.DataFrame({'feature':    X.columns,'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
print(importance_df.to_string())