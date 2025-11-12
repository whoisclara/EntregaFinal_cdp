##Proyecto Final – Ciencia de Datos en Producción

Este repositorio contiene el desarrollo completo del proyecto final del curso Ciencia de Datos en Producción, el cual implementa una solución de Machine Learning, integrando procesamiento de datos, modelado, evaluación y despliegue automatizado con Jenkins y Docker.

EntregaFinal_cdp/
│
├── models/                  # Modelos entrenados (.pkl)
│
├── pyops/                   # Scripts de operaciones (Jenkins)
│   ├── jenkins_home/        # Directorio persistente de Jenkins
│   └── check_structure.py   # Verificación de estructura del proyecto
│
├── src/                     # Código fuente principal
│   ├── static/              # Archivos estáticos (web)
│   ├── templates/           # Plantillas HTML (si aplica)
│   ├── Carga_datos.py       # Conexión y carga de datos desde BigQuery
│   ├── ft_engineering.py    # Feature engineering y escalado
│   ├── heuristic_model.py   # Modelo base heurístico
│   ├── model_training.py    # Entrenamiento de modelos ML
│   ├── model_evaluation.py  # Evaluación de performance
│   ├── model_deploy.py      # Despliegue con FastAPI / Vertex / Flask
│   └── EDA.ipynb            # Exploración de datos
│
├── docker-compose.yml       # Orquestación de servicios (Jenkins + app)
├── Dockerfile               # Imagen base del proyecto
├── requirements.txt         # Dependencias de Python
├── .env                     # Variables de entorno (no subir con claves)
├── set_up.bat               # Script de inicialización local
└── README.md                # Este documento

##⚙️ Tecnologías utilizadas

| Componente                        | Descripción                          |
| --------------------------------- | ------------------------------------ |
| **Python 3.10+**                  | Lenguaje principal                   |
| **scikit-learn / pandas / numpy** | Procesamiento, modelado y evaluación |
| **Google BigQuery**               | Fuente de datos en la nube           |
| **FastAPI / Flask**               | Interfaz de despliegue del modelo    |
| **Jenkins**                       | Automatización CI/CD                 |
| **Docker / Docker Compose**       | Contenerización y orquestación       |
| **Vertex AI (opcional)**          | Entrenamiento o evaluación en GCP    |


🚀 Flujo de trabajo

1. Carga y limpieza de datos
    - Carga_datos.py obtiene datos desde BigQuery.
    - Se realiza imputación, normalización y codificación en ft_engineering.py.
2. Entrenamiento y selección de modelos
    - model_training.py entrena múltiples algoritmos (Logistic Regression, RandomForest, etc.)
    - Se aplica cross-validation y selección del mejor modelo según métricas.

3. Evaluación y métricas
    - model_evaluation.py genera reportes de desempeño (accuracy, ROC-AUC, etc.).

4. Despliegue del modelo
    - model_deploy.py expone el modelo mediante API REST (FastAPI/Flask).
    - Se integra con Jenkins para automatizar builds y pruebas.

5.Automatización CI/CD
    - Jenkinsfile y docker-compose.yml coordinan las etapas del pipeline:
        - Build de imagen
        - Test de estructura
        - Entrenamiento / despliegue automatizado

🧠 Autoras:
- Clara Otalvaro 
- Ada Mattos