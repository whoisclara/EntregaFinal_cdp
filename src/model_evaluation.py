from sklearn.metrics import classification_report, roc_auc_score

# Cargar modelo
import joblib
model = joblib.load("models/RandomForest.pkl")

# Cargar datos reales desde BigQuery
from google.cloud import bigquery
from dotenv import load_dotenv
import os
load_dotenv()

project_id = os.getenv("PROJECT_ID")
dataset_id = os.getenv("DATASET")
client = bigquery.Client(project=project_id)
df_test = client.query(f"SELECT * FROM `{project_id}.{dataset_id}.test_30_scaled`").to_dataframe()

X_test = df_test.drop(columns=["membresia_premium"])
y_true = df_test["membresia_premium"]

# Predecir directamente
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print(classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, probs))
