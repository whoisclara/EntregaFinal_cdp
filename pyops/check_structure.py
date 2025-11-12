import os
import sys

print("🔍 Iniciando validación de estructura del proyecto...\n")

# Ir un nivel arriba desde pyops/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Carpetas requeridas
REQUIRED_FOLDERS = ["src", "models", "pyops"]

# Archivos requeridos
REQUIRED_FILES = ["requirements.txt", "docker-compose.yml"]

missing_folders = []
missing_files = []

print("📁 Verificando carpetas requeridas...")
for folder in REQUIRED_FOLDERS:
    path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(path):
        print(f"❌ Falta la carpeta: {folder}")
        missing_folders.append(folder)
    else:
        print(f"✅ Carpeta encontrada: {folder}")

print("\n📄 Verificando archivos requeridos...")
for file in REQUIRED_FILES:
    path = os.path.join(BASE_DIR, file)
    if not os.path.isfile(path):
        print(f"❌ Falta el archivo: {file}")
        missing_files.append(file)
    else:
        print(f"✅ Archivo encontrado: {file}")

# Resultado final
if not missing_folders and not missing_files:
    print("\n🎉 Estructura completa. Todo está en orden.")
    sys.exit(0)
else:
    print("\n💥 Estructura incompleta.")
    if missing_folders:
        print("🚫 Carpetas faltantes:", ", ".join(missing_folders))
    if missing_files:
        print("🚫 Archivos faltantes:", ", ".join(missing_files))
    sys.exit(1)
