# =====================================================
# MODELO HEURÍSTICO - Membresía Premium
# =====================================================
# Lee los datos desde BigQuery (train/test sin escalar)
# Evalúa un modelo basado en reglas heurísticas definidas
# =====================================================

import numpy as np
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


# =====================================================
# 1️⃣ Conexión a BigQuery
# =====================================================
load_dotenv()  # cargar variables del entorno

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
bq_client = bigquery.Client(project=PROJECT_ID)


def load_table_from_bigquery(table_name: str) -> pd.DataFrame:
    """
    Carga una tabla de BigQuery y la retorna como DataFrame de pandas.
    """
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET}.{table_name}`"
    df = bq_client.query(query).to_dataframe()
    print(f"✅ Cargada tabla desde BigQuery: {table_name} ({df.shape[0]} filas, {df.shape[1]} columnas)")
    return df


# =====================================================
# 2️⃣ Definir modelo heurístico
# =====================================================
class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 ingreso_threshold=2_000_000,#1_500_000
                 gasto_threshold=40_000, #30_000
                 frecuencia_threshold=3, #2
                 edad_min_premium=20, #22
                 edad_max_premium=55):#65
        """
        Modelo heurístico simple basado en patrones del EDA:
        ----------------------------------------------------
        - Ingresos altos → mayor probabilidad de membresía.
        - Frecuencia de visita alta → mayor probabilidad.
        - Gasto promedio alto → mayor probabilidad.
        - Edad entre 18 y 60 años → población activa con más probabilidad.
        """
        self.ingreso_threshold = ingreso_threshold
        self.gasto_threshold = gasto_threshold
        self.frecuencia_threshold = frecuencia_threshold
        self.edad_min_premium = edad_min_premium
        self.edad_max_premium = edad_max_premium

    # -------------------------------------------------
    # Método "fit"
    # -------------------------------------------------
    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    # -------------------------------------------------
    # Método "predict" — donde se aplican las reglas
    # -------------------------------------------------
    def predict(self, X):
        preds = []

        for _, row in X.iterrows():
            es_premium = 0  # valor inicial (No Premium)

            # --- Regla 1: ingresos altos ---
            if "numeric__ingresos_mensuales" in X.columns and \
               row["numeric__ingresos_mensuales"] >= self.ingreso_threshold:
                es_premium = 1

            # --- Regla 2: alta frecuencia de visita ---
            if "numeric__frecuencia_visita" in X.columns and \
               row["numeric__frecuencia_visita"] >= self.frecuencia_threshold:
                es_premium = 1

            # --- Regla 3: alto gasto promedio ---
            if "numeric__promedio_gasto_comida" in X.columns and \
               row["numeric__promedio_gasto_comida"] >= self.gasto_threshold:
                es_premium = 1

            # --- Regla 4: rango de edad---
            if "numeric__edad" in X.columns and (
                row["numeric__edad"] < self.edad_min_premium or
                row["numeric__edad"] > self.edad_max_premium
            ):
                es_premium = 0

            # --- Regla 5: consumo de licor --
            if "categoric__consume_licor_Sí" in X.columns and \
               row["categoric__consume_licor_Sí"] == 1:
                es_premium = 1

            # --- Regla 6: estrato socioeconómico ---
            if "categoric_ordinales__estrato_socioeconomico" in X.columns and \
            row["categoric_ordinales__estrato_socioeconomico"] >= 3:
                es_premium = 1

            # --- Penalización por baja frecuencia y bajo gasto ---
            if ("numeric__frecuencia_visita" in X.columns and row["numeric__frecuencia_visita"] < self.frecuencia_threshold) and \
            ("numeric__promedio_gasto_comida" in X.columns and row["numeric__promedio_gasto_comida"] < self.gasto_threshold):
                es_premium = 0
            

            preds.append(es_premium)

        return np.array(preds)


# =====================================================
# 3️⃣ Cargar dataset desde BigQuery
# =====================================================
train_df = load_table_from_bigquery("train_70_raw")
test_df = load_table_from_bigquery("test_30_raw")

X_train = train_df.drop(columns=["membresia_premium"])
y_train = train_df["membresia_premium"]

X_test = test_df.drop(columns=["membresia_premium"])
y_test = test_df["membresia_premium"]

# =====================================================
# 4️⃣ Entrenamiento y evaluación
# =====================================================
model = HeuristicModel()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_metrics = {
    "accuracy": accuracy_score(y_train, y_train_pred),
    "precision": precision_score(y_train, y_train_pred, zero_division=0),
    "recall": recall_score(y_train, y_train_pred, zero_division=0),
    "f1": f1_score(y_train, y_train_pred, zero_division=0)
}

test_metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred),
    "precision": precision_score(y_test, y_test_pred, zero_division=0),
    "recall": recall_score(y_test, y_test_pred, zero_division=0),
    "f1": f1_score(y_test, y_test_pred, zero_division=0)
}

print("\n=== Resultados en Test ===")
for m, v in test_metrics.items():
    print(f"{m}: {v:.3f}")

# =====================================================
# 5️Validación cruzada
# =====================================================
scoring_metrics = ["accuracy", "precision", "recall", "f1"]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

X_full = pd.concat([X_train, X_test], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)

cv_results = {metric: cross_val_score(model, X_full, y_full, cv=kfold, scoring=metric)
              for metric in scoring_metrics}
cv_results_df = pd.DataFrame(cv_results)

print("\n=== Promedios de Validación Cruzada ===")
print(cv_results_df.mean().round(3))

# =====================================================
# 6️⃣ Visualizaciones
# =====================================================
plt.figure(figsize=(8, 6))
x_pos = range(len(scoring_metrics))
plt.bar(x_pos, [train_metrics[m] for m in scoring_metrics], width=0.4, label="Train")
plt.bar([i + 0.4 for i in x_pos], [test_metrics[m] for m in scoring_metrics],
        width=0.4, label="Test")
plt.xticks([i + 0.2 for i in x_pos], scoring_metrics)
plt.ylabel("Score")
plt.title("Métricas Train vs Test - Modelo Heurístico (Membresía Premium)")
plt.legend()
plt.show()

train_sizes, train_scores, test_scores = learning_curve(model, X_full, y_full, cv=kfold, scoring="f1")
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="CV F1")
plt.title("Curva de Aprendizaje - HeuristicModel (Premium)")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("F1 Score")
plt.legend()
plt.show()
