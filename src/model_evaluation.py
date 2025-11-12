import joblib
import os

from sklearn.metrics import classification_report, roc_auc_score
from google.cloud import bigquery
from dotenv import load_dotenv

# ==============
# Cargar modelo entrenado 
# ============== 

model = joblib.load("models/RandomForest.pkl")

# ======================================================
# AUTENTICACIÓN Y CONFIGURACIÓN DE BIGQUERY
# ======================================================

load_dotenv()

# Ruta absoluta al archivo de credenciales
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "..", ".keys", "service_account.json")

if not os.path.exists(CREDENTIALS_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de credenciales en {CREDENTIALS_PATH}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")

bq_client = bigquery.Client(project=PROJECT_ID)

query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.test_30_scaled`"
df_test = bq_client.query(query).to_dataframe()

X_test = df_test.drop(columns=["membresia_premium"])
y_true = df_test["membresia_premium"]

# Predecir directamente
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print(classification_report(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, probs))
