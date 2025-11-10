# ================================================
# 1. Importar librerías y función de carga de datos
# ================================================

import sys
sys.path.append("../src")  

import pandas as pd
import numpy as np
import os


from Carga_datos import load_data_from_bigquery
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from google.cloud import bigquery
from pathlib import Path
from dotenv import load_dotenv

# Configuración de BigQuery
load_dotenv()  # Leer variables del entorno (.env)

project_id = os.getenv("PROJECT_ID")
dataset = os.getenv("DATASET")
bq_client = bigquery.Client(project=project_id)

# ================================================
# 2. Cargar los datos
# ================================================
df = load_data_from_bigquery()
df.head()

def limpia_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza inicial según hallazgos del EDA:
    -------------------------------------------------
    - Elimina columnas irrelevantes.
    - Corrige valores fuera de rango.
    - Elimina frecuencias negativas.
    - Reemplaza ingresos 0 por NaN.
    - Imputa valores faltantes.
    - Convierte variables a categoría.
    - Codifica la variable objetivo (Sí/No → 1/0).
    -------------------------------------------------
    """

    # Eliminar columnas irrelevantes
    cols_irrelevantes = [
        "id_persona", "nombre", "apellido", "telefono_contacto", "correo_electronico"
    ]
    df = df.drop(columns=[c for c in cols_irrelevantes if c in df.columns], errors="ignore")

    # Corregir valores anómalos
    df.loc[~df["edad"].between(16, 100), "edad"] = np.nan
    df.loc[df["frecuencia_visita"] < 0, "frecuencia_visita"] = np.nan
    df["ingresos_mensuales"] = df["ingresos_mensuales"].replace(0, np.nan)

    # Imputación de valores faltantes (numéricos)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Conversión de columnas tipo object a category
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype("category")

    # Codificación binaria de la variable objetivo
    if "membresia_premium" in df.columns:
        df["membresia_premium"] = df["membresia_premium"].replace({
            "Sí": 1, "No": 0
        }).astype(int)

    print("✅ Limpieza completada:")
    return df

# ------------------------------
# CLASIFICACIÓN DE VARIABLES
# ------------------------------
def define_type(X: pd.DataFrame):
    """Clasifica columnas en numéricas, categóricas nominales y ordinales."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    ordinal_cols = [c for c in cat_cols if "nivel" in c or "estrato" in c]
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]

    return num_cols, nominal_cols, ordinal_cols


# ============================================================
# CREACIÓN DE PIPELINES
# ============================================================
def create_pipeline(scale=False):
    """Genera un ColumnTransformer con numéricas, categóricas y categóricas ordinales."""
    def transformer(num_cols, nominal_cols, ordinal_cols):
        transformers = [
            # Numéricas
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),

            # Categóricas nominales
            ("categoric", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), nominal_cols),

            # Categóricas ordinales
            ("categoric_ordinales", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder())
            ]), ordinal_cols)
        ]

        # Agregar escalado si corresponde
        if scale:
            transformers[0][1].steps.append(("scaler", MinMaxScaler()))

        return ColumnTransformer(transformers)

    return transformer

# ============================================================
# TRANSFORMACIÓN Y CARGA A BIGQUERY
# ============================================================
def transform_data(X, y, pipeline):
    """Aplica el pipeline y devuelve un DataFrame transformado."""
    X_transformed = pipeline.fit_transform(X)
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    col_names = pipeline.get_feature_names_out()
    df_final = pd.concat([pd.DataFrame(X_transformed, columns=col_names), y.reset_index(drop=True)], axis=1)
    return df_final


def upload_to_bigquery(df, table_name):
    """Sube un DataFrame a BigQuery."""
    full_id = f"{project_id}.{dataset}.{table_name}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(df, full_id, job_config=job_config).result()
    print(f"📤 Subido a BigQuery → {full_id}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main(target="membresia_premium"):
    print("🚀 Iniciando Feature Engineering con conexión a BigQuery...")

    # 1️⃣ Cargar datos
    df = load_data_from_bigquery()
    df = limpia_data(df)

    # 2️⃣ Separar variables
    X = df.drop(columns=[target])
    y = df[target]

    # 3️⃣ Clasificar columnas
    num_cols, nominal_cols, ordinal_cols = define_type(X)

    # 4️⃣ Crear pipelines
    pipe_scaled = create_pipeline(scale=True)(num_cols, nominal_cols, ordinal_cols)
    pipe_noscale = create_pipeline(scale=False)(num_cols, nominal_cols, ordinal_cols)

    # 5️⃣ Aplicar transformaciones
    df_escalado = transform_data(X, y, pipe_scaled)
    df_sin_escalar = transform_data(X, y, pipe_noscale)

    # 6️⃣ División 70/30 (ambos casos)
    def split(df_in):
        X = df_in.drop(columns=[target])
        y = df_in[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return (pd.concat([X_tr, y_tr], axis=1), pd.concat([X_te, y_te], axis=1))

    train_scaled, test_scaled = split(df_escalado)
    train_raw, test_raw = split(df_sin_escalar)

    # 7️⃣ Subir todos los datasets (6 en total)
    upload_to_bigquery(df_escalado, "df_escalado")
    upload_to_bigquery(df_sin_escalar, "df_sin_escalar")
    upload_to_bigquery(train_scaled, "train_70_scaled")
    upload_to_bigquery(test_scaled, "test_30_scaled")
    upload_to_bigquery(train_raw, "train_70_raw")
    upload_to_bigquery(test_raw, "test_30_raw")

    print("✅ Feature Engineering completado y todos los datasets cargados correctamente.")


if __name__ == "__main__":
    main()