# ======================================================
# Carga_datos.py — Conexión segura a BigQuery
# ======================================================
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

# Cargar variables del entorno (.env)
load_dotenv()

def load_data_from_bigquery():
    """
    Conecta a BigQuery con credenciales explícitas y devuelve un DataFrame.
    """
    # Leer variables desde .env
    project_id = os.getenv("PROJECT_ID")
    dataset = os.getenv("DATASET")
    table = os.getenv("TABLE")

    # Ruta a las credenciales
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/var/jenkins_home/.keys/service_account.json")

    print(f"🔐 Usando credenciales desde: {credentials_path}")

    # Crear credenciales y cliente
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = bigquery.Client(credentials=credentials, project=project_id)

    # Construir la query
    query = f"SELECT * FROM `{project_id}.{dataset}.{table}`"

    # Ejecutar y retornar DataFrame
    df = client.query(query).to_dataframe()
    print("✅ Datos cargados correctamente desde BigQuery")
    return df

