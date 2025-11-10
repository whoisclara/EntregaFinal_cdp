# ======================================================
# Carga_datos.py — Conexión segura a BigQuery
# ======================================================
from google.cloud import bigquery
from dotenv import load_dotenv
import os

# Cargar variables del entorno (.env)
load_dotenv()


def load_data_from_bigquery():
    """
    Conecta a BigQuery y devuelve un DataFrame con los datos solicitados.
    """
    # Leer variables desde .env
    project_id = os.getenv("PROJECT_ID")
    dataset = os.getenv("DATASET")
    table = os.getenv("TABLE")

    # Crear cliente con el proyecto definido
    client = bigquery.Client(project=project_id)

    # Construir la query dinámicamente
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"

    # Ejecutar y retornar el DataFrame
    df = client.query(query).to_dataframe()
    print("✅ Datos cargados correctamente desde BigQuery")
    return df

# def load_data_from_bigquery(
#     project_id="cdp202502",
#     query="SELECT * FROM `cdp202502.proyecto_cp.restaurante_db`"
# ):
#     """
#     Conecta a BigQuery y devuelve un DataFrame con los datos solicitados.
#     """
#     client = bigquery.Client(project=project_id)
#     df = client.query(query).to_dataframe()
#     print("✅ Datos cargados correctamente desde BigQuery")
#     return df
