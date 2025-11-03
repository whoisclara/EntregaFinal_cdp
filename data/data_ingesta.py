from google.cloud import bigquery
 
# Set your GCP project ID
project_id = "my-project-cdp-476119"
 
# Create client
client = bigquery.Client(project=project_id)
 
# SQL Query
query = "SELECT * FROM `my-project-cdp-476119.sbx_proyectoFinal.restaurante_tabla`"
df = client.query(query).to_dataframe()
 
print(df.head())
 