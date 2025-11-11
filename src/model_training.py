# ======================================================
# MODEL TRAINING - Membresía Premium 
# ======================================================
# Se entrenan y evalúan varios modelos con datos desde BigQuery.
# De aquí resulta el mejor modelo y se guardan los artefactos
# para despliegue: modelo y pipeline de features.
# ======================================================

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery

# Scikit-learn y XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ======================================================
# 1️⃣ CONFIGURACIÓN Y CARGA DE DATOS DESDE BIGQUERY
# ======================================================
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
bq_client = bigquery.Client(project=PROJECT_ID)

def load_from_bigquery(table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{table_name}`"
    df = bq_client.query(query).to_dataframe()
    print(f"✅ {table_name} cargada ({df.shape[0]} filas, {df.shape[1]} columnas)")
    return df

# Cargar datos escalados
train_df = load_from_bigquery("train_70_scaled")
test_df = load_from_bigquery("test_30_scaled")

X_train = train_df.drop(columns=["membresia_premium"])
y_train = train_df["membresia_premium"]
X_test = test_df.drop(columns=["membresia_premium"])
y_test = test_df["membresia_premium"]

# Crear carpeta local para guardar resultados
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2️⃣ DEFINICIÓN DE MODELOS
# ======================================================
lr = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, solver="lbfgs")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=3,
    random_state=42
)


models = {
    "LogisticRegression": lr,
    "RandomForest": rf,
    "XGBoost": xgb,
    "GradientBoosting": gb
}

# ======================================================
# 3️⃣ ENTRENAMIENTO Y VALIDACIÓN
# ======================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
saved_models = {}

for name, model in models.items():
    print(f"\n🚀 Entrenando modelo: {name}")

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Probabilidades para ROC-AUC (si aplica)
    if hasattr(model, "predict_proba"):
        proba_test = model.predict_proba(X_test)[:, 1]
    else:
        proba_test = np.zeros_like(y_test)

    rep = classification_report(
        y_test, y_pred_test,
        digits=4, output_dict=True, labels=[0, 1], zero_division=0
    )
    auc = roc_auc_score(y_test, proba_test)

    results.append({
        "Modelo": name,
        "Precision clase 0": round(rep["0"]["precision"], 3),
        "Recall clase 0": round(rep["0"]["recall"], 3),
        "F1 clase 0": round(rep["0"]["f1-score"], 3),
        "Precision clase 1": round(rep["1"]["precision"], 3),
        "Recall clase 1": round(rep["1"]["recall"], 3),
        "F1 clase 1": round(rep["1"]["f1-score"], 3),
        "F1 macro": round(rep["macro avg"]["f1-score"], 3),
        "ROC-AUC": round(auc, 3)
    })

    saved_models[name] = model

# ======================================================
# 4️⃣ RESULTADOS Y SELECCIÓN DEL MEJOR MODELO
# ======================================================
summary = pd.DataFrame(results).sort_values(by=["F1 macro", "ROC-AUC"], ascending=False)
print("\n=== 📊 Resultados comparativos (umbral = 0.5) ===")
print(summary)

summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
best_name = summary.iloc[0]["Modelo"]
best_model = saved_models[best_name]
print(f"\n✅ Mejor modelo seleccionado: {best_name}")

# ======================================================
# 5️⃣ EXPORTACIÓN PARA DESPLIEGUE
# ======================================================
print("\n💾 Guardando artefactos del modelo para despliegue...")

# Guardar el mejor modelo entrenado
model_path = OUTPUT_DIR / f"{best_name}.pkl"
joblib.dump(best_model, model_path)

# 🔹 Simular carga del pipeline de features (si fue guardado desde ft_engineering)
# En producción se usaría:
# feature_pipeline = joblib.load("../models/feature_pipeline.pkl")
# Aquí creamos un placeholder
feature_pipeline = {"columns": list(X_train.columns)}
pipeline_path = OUTPUT_DIR / "feature_pipeline.pkl"
joblib.dump(feature_pipeline, pipeline_path)

# 🔹 Crear metadatos del modelo
metadata = {
    "project_id": PROJECT_ID,
    "dataset": DATASET,
    "best_model": best_name,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "f1_macro": float(summary.iloc[0]["F1 macro"]),
    "roc_auc": float(summary.iloc[0]["ROC-AUC"]),
    "columns_used": list(X_train.columns),
}

metadata_path = OUTPUT_DIR / "model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Modelo guardado en: {model_path}")
print(f"✅ Metadata guardada en: {metadata_path}")
print(f"✅ Pipeline guardado en: {pipeline_path}")

# ======================================================
# 6️⃣ VISUALIZACIÓN
# ======================================================
plt.figure(figsize=(8, 5))
plt.bar(summary["Modelo"], summary["F1 macro"], label="F1 Macro", alpha=0.7)
plt.plot(summary["Modelo"], summary["ROC-AUC"], "o--", color="orange", label="ROC-AUC")
plt.title("Comparativo de desempeño entre modelos (F1 vs ROC-AUC)")
plt.xlabel("Modelo")
plt.ylabel("Puntuación")
plt.legend()
plt.tight_layout()
plt.show()

print("\ Exportación completada.")