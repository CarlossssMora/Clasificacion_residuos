import json
import os
from PIL import Image
import sys

# Configuración para evitar errores de codificación en Windows
sys.stdout.reconfigure(encoding='utf-8')

# --- RUTAS ---
# Archivo JSON con anotaciones (en la misma carpeta que este script)
RUTA_JSON = "annotations.json"

# Carpeta donde están las imágenes originales
RUTA_IMAGENES_BASE = r"C:\Users\sergi\OneDrive - Instituto Politecnico Nacional\Metodologia de la investigacion\Proyecto Residuos Solidos\Clasificacion_residuos\TACO_desordenado\data"

# Carpeta donde se guardarán los recortes
RUTA_SALIDA = "recortes"

# Asegurar compatibilidad con rutas en Windows
RUTA_IMAGENES_BASE = os.path.normpath(RUTA_IMAGENES_BASE)

# Crear carpeta de salida si no existe
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- CARGAR JSON ---
with open(RUTA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Diccionarios de ayuda
imagenes_dict = {img["id"]: img["file_name"] for img in data["images"]}
categorias_dict = {cat["id"]: cat["name"] for cat in data["categories"]}

# --- VERIFICAR RUTA DE IMÁGENES ---
print("Verificando carpeta de imágenes:")
print(os.path.abspath(RUTA_IMAGENES_BASE))
if os.path.exists(RUTA_IMAGENES_BASE):
    print("✅ Carpeta encontrada correctamente.")
    print("Contiene:", len(os.listdir(RUTA_IMAGENES_BASE)), "subcarpetas (batches)")
else:
    print("❌ No se encontró la carpeta de imágenes, revisa la ruta.")
    sys.exit()

# --- PROCESAR ANOTACIONES ---
for i, ann in enumerate(data["annotations"]):
    image_id = ann["image_id"]
    categoria_id = ann["category_id"]
    bbox = ann["bbox"]  # [x, y, width, height]

    if image_id not in imagenes_dict or categoria_id not in categorias_dict:
        continue

    categoria = categorias_dict[categoria_id].replace("/", "_").replace("\\", "_")
    carpeta_categoria = os.path.join(RUTA_SALIDA, categoria)
    os.makedirs(carpeta_categoria, exist_ok=True)

    # Ruta completa a la imagen original
    nombre_imagen = imagenes_dict[image_id]
    ruta_imagen = os.path.join(RUTA_IMAGENES_BASE, nombre_imagen)
    ruta_imagen = os.path.normpath(ruta_imagen)

    # Intentar buscar con varias extensiones posibles
    if not os.path.exists(ruta_imagen):
        base_sin_ext = os.path.splitext(ruta_imagen)[0]
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            if os.path.exists(base_sin_ext + ext):
                ruta_imagen = base_sin_ext + ext
                break

    if not os.path.exists(ruta_imagen):
        print(f"ADVERTENCIA: No se encontró la imagen -> {ruta_imagen}")
        continue

    try:
        with Image.open(ruta_imagen) as img:
            x, y, w, h = map(int, bbox)
            recorte = img.crop((x, y, x + w, y + h))
            nombre_salida = f"{categoria}_{i}_{os.path.basename(nombre_imagen)}"
            ruta_salida = os.path.join(carpeta_categoria, nombre_salida)
            recorte.save(ruta_salida)
            print(f"Guardado: {ruta_salida}")
    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {e}")

print("\nTodos los recortes se han guardado correctamente por categoria.")
