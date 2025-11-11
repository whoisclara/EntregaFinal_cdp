# =========================================================
# DOCKERFILE - Despliegue Membresía Premium (FastAPI)
# =========================================================

# Imagen base: versión estable de Python 3.10
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY ./src ./src
COPY ./models ./models

# Exponer el puerto 8000
EXPOSE 8000

# Comando de ejecución
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
