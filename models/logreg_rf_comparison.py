import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

# To obtain "clean_cols.csv" dataset,
# run 'clean_join.py' in the 'etl' folder

df = pd.read_csv('../data/clean/clean_cols.csv')
df = df.dropna()
df = df.sample(n=min(100000, len(df)))
threshold = df['available_beds'].median()
# Features - EXCLUDE columns that make up the target
X = df.drop(columns=[
    'available_beds',        # target
    'hospital_pk',           # ID
    'state',                 # categorical
    'collection_week',       # time
])
y = (df['available_beds'] > threshold).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print(f"Logistic Regression - Accuracy: {accuracy_score(y_test, lr_pred):.4f}, Precision: {precision_score(y_test, lr_pred):.4f}")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest - Accuracy: {accuracy_score(y_test, rf_pred):.4f}, Precision: {precision_score(y_test, rf_pred):.4f}")
print("Train accuracy:", accuracy_score(y_train, rf.predict(X_train))) # check for data leakage