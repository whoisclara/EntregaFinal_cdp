# ======================================================
# MODEL DEPLOY - Membresía Premium (versión final estable)
# ======================================================

import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from sklearn.exceptions import NotFittedError

# ======================================================
# 1️⃣ CARGA DE ARTEFACTOS
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "src" / "templates"

MODEL_PATH = MODELS_DIR / "RandomForest.pkl"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.pkl"

# Verificar existencia
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el modelo en: {MODEL_PATH}")
if not PIPELINE_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el pipeline en: {PIPELINE_PATH}")

print(f"📁 Modelo: {MODEL_PATH}")
print(f"📁 Pipeline: {PIPELINE_PATH}")

# Cargar modelo y pipeline
model = joblib.load(MODEL_PATH)
loaded_pipeline = joblib.load(PIPELINE_PATH)

# Detectar si viene dentro de un diccionario
if isinstance(loaded_pipeline, dict):
    feature_pipeline = (
        loaded_pipeline.get("pipeline")
        or loaded_pipeline.get("preprocessor")
        or loaded_pipeline
    )
else:
    feature_pipeline = loaded_pipeline

# Validar pipeline
if not hasattr(feature_pipeline, "transform"):
    raise TypeError(
        f"❌ El archivo '{PIPELINE_PATH.name}' cargó un objeto inválido "
        f"({type(feature_pipeline)}). Debe contener un ColumnTransformer o Pipeline válido."
    )

print("✅ Modelo y pipeline cargados correctamente.\n")

# ======================================================
# 2️⃣ CONFIGURAR APP FASTAPI
# ======================================================
app = FastAPI(
    title="Membresía Premium Predictor",
    description="Predice la probabilidad de adquirir una Membresía Premium.",
    version="1.0"
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "src" / "static")), name="static")

# ======================================================
# 🔧 FUNCIÓN DE TRANSFORMACIÓN SEGURA
# ======================================================
def safe_transform(pipeline, df):
    """
    Aplica transform() manejando categorías no vistas por el OrdinalEncoder.
    Si encuentra categorías desconocidas, las codifica como -1.
    """
    try:
        return pipeline.transform(df)
    except ValueError as e:
        if "unknown categories" in str(e):
            print(f"⚠️ Categorías desconocidas detectadas → aplicando 'unknown_value = -1'")
            for name, trans, cols in pipeline.transformers_:
                if hasattr(trans, "named_steps") and "encoder" in trans.named_steps:
                    enc = trans.named_steps["encoder"]
                    if hasattr(enc, "handle_unknown") and enc.handle_unknown == "error":
                        enc.handle_unknown = "use_encoded_value"
                        enc.unknown_value = -1
            return pipeline.transform(df)
        else:
            raise
    except NotFittedError:
        raise RuntimeError("❌ El pipeline no está entrenado correctamente.")

# ======================================================
# 3️⃣ RUTA HOME (FORMULARIO)
# ======================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

# ======================================================
# 4️⃣ ENDPOINT FORMULARIO HTML → RESULTADOS
# ======================================================
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()

    try:
        # ------------------------------
        # 1️⃣ Extraer datos del formulario (sin valores quemados)
        # ------------------------------
        data = {
            "edad": float(form.get("edad")),
            "frecuencia_visita": float(form.get("frecuencia_visita")),
            "promedio_gasto_comida": float(form.get("promedio_gasto_comida")),
            "ingresos_mensuales": float(form.get("ingresos_mensuales")),
            "genero": form.get("genero"),
            "ciudad_residencia": form.get("ciudad_residencia"),
            "consume_licor": form.get("consume_licor"),
            "tipo_de_pago_mas_usado": form.get("tipo_de_pago_mas_usado"),
            "estrato_socioeconomico": form.get("estrato_socioeconomico"),
            "preferencias_alimenticias": form.get("preferencias_alimenticias"),
            "ocio": form.get("ocio")
        }

        df_input = pd.DataFrame([data])

        # ------------------------------
        # 2️⃣ Transformar y predecir con seguridad
        # ------------------------------
        X_transformed = safe_transform(feature_pipeline, df_input)
        prob = float(model.predict_proba(X_transformed)[0][1])
        prediction = int(prob >= 0.5)

        result = {
            "prediction": (
                "✅ Alta probabilidad de Membresía Premium"
                if prediction == 1 else
                "❌ Baja probabilidad de Membresía Premium"
            ),
            "probability": round(prob, 4)
        }

    except Exception as e:
        result = {
            "prediction": "⚠️ Error en la predicción",
            "probability": f"Detalles: {str(e)}"
        }

    return templates.TemplateResponse("results.html", {"request": request, "result": result})

# ======================================================
# 5️⃣ ENDPOINT JSON PARA POSTMAN
# ======================================================
@app.post("/predict", response_class=JSONResponse)
def predict_json(payload: dict):
    try:
        df_input = pd.DataFrame([payload])
        X_transformed = safe_transform(feature_pipeline, df_input)
        prob = float(model.predict_proba(X_transformed)[0][1])
        prediction = int(prob >= 0.5)

        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "message": (
                "Cliente con alta probabilidad de ser Premium"
                if prediction == 1 else
                "Cliente con baja probabilidad de ser Premium"
            )
        }
    except Exception as e:
        return {"error": f"❌ Error al procesar la solicitud: {str(e)}"}

# ======================================================
# 6️⃣ DESCARGAS
# ======================================================
@app.get("/download/model")
def download_model():
    return FileResponse(MODEL_PATH, media_type="application/octet-stream", filename=MODEL_PATH.name)

@app.get("/download/pipeline")
def download_pipeline():
    return FileResponse(PIPELINE_PATH, media_type="application/octet-stream", filename=PIPELINE_PATH.name)

# ======================================================
# 7️⃣ EJECUCIÓN LOCAL
# ======================================================
if __name__ == "__main__":
    uvicorn.run("src.model_deploy:app", host="0.0.0.0", port=8000, reload=True)


