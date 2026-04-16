import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Compares two models before and after hierarchical imputation

def run_models(filepath, label):
    df = pd.read_csv(filepath)
    threshold = df['available_beds'].median()
    X = df.drop(columns=['available_beds', 'hospital_pk', 'state', 'collection_week'])
    y = (df['available_beds'] > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print(f"{label}:")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}, precision: {precision_score(y_test, rf_pred):.4f}")

run_models('data/clean/sample_to_impute.csv', 'Original')
run_models('data/clean/imputed.csv', 'Imputed')