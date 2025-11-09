@echo off
setlocal EnableDelayedExpansion EnableExtensions

echo === Python Virtual Environment Setup ===
echo.

REM Paso 1: Definir el nombre del proyecto
set "project_code=mlops_pipeline"

REM Paso 2: Navegar al folder src del pipeline
pushd "%~dp0mlops_pipeline\src" || (
    echo ERROR: No se encontró la carpeta "%~dp0mlops_pipeline\src"
    echo Verifica la ruta o el nombre de carpeta.
    exit /b 1
)

echo Proyecto detectado: !project_code!

REM Volver al root del repo
popd
pushd "%~dp0"

REM Paso 3: Crear entorno virtual si no existe
set "VENV_DIR=!project_code!-venv"
if not exist "!VENV_DIR!" (
    echo Creando entorno virtual: !VENV_DIR!
    py -m venv "!VENV_DIR!"
) else (
    echo El entorno virtual ya existe: !VENV_DIR!
)

REM Paso 4: Activar entorno virtual
call "!VENV_DIR!\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: No se pudo activar el entorno virtual.
    exit /b 1
)

REM Paso 5: Instalar librerías del proyecto
if exist requirements.txt (
    echo Instalando librerías desde requirements.txt...
    pip install --no-cache-dir -r requirements.txt
) else (
    echo ADVERTENCIA: No se encontró requirements.txt, se omite instalación.
)

REM Paso 6: Verificar ipykernel
pip show ipykernel >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ipykernel no encontrado, instalando...
    pip install ipykernel
)

REM Paso 7: Registrar kernel en Jupyter
python -m ipykernel install --user --name="!project_code!-venv" --display-name="Python (!project_code!-venv)"
if %ERRORLEVEL% EQU 0 (
    echo Kernel registrado correctamente como "Python (!project_code!-venv)"
) else (
    echo ERROR: Falló el registro del kernel en Jupyter.
)

echo.
echo === Setup finalizado ===
echo Activa tu entorno manualmente con:
echo.
echo     call "!VENV_DIR!\Scripts\activate.bat"
echo.
echo Luego abre VS Code y selecciona el kernel "Python (!project_code!-venv)".
echo =========================

popd
