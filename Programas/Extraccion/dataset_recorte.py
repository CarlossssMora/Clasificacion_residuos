import json
import os
from PIL import Image
import csv
import sys

# Asegurar codificación correcta en Windows
sys.stdout.reconfigure(encoding='utf-8')

# --- RUTAS ---
RUTA_JSON = "annotations.json"
RUTA_IMAGENES_BASE = r"C:\Users\sergi\OneDrive - Instituto Politecnico Nacional\Metodologia de la investigacion\Proyecto Residuos Solidos\Clasificacion_residuos\TACO_desordenado\data"
RUTA_SALIDA = "recortes"
RUTA_CSV = "recortes_dataset.csv"  # nombre del archivo CSV generado

# Asegurar compatibilidad con rutas
RUTA_IMAGENES_BASE = os.path.normpath(RUTA_IMAGENES_BASE)
os.makedirs(RUTA_SALIDA, exist_ok=True)

# --- CARGAR JSON ---
with open(RUTA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

imagenes_dict = {img["id"]: img for img in data["images"]}
categorias_dict = {cat["id"]: cat for cat in data["categories"]}

# --- CSV HEADER ---
encabezados = [
    "img_id", "img_width", "img_height", "img_file",
    "cat_id", "cat_name", "supercategory",
    "ann_id", "x", "y", "width", "height", "area", "recorte_file"
]

filas_csv = []

# --- PROCESAR ANOTACIONES ---
for ann in data["annotations"]:
    image_id = ann["image_id"]
    categoria_id = ann["category_id"]
    bbox = ann["bbox"]  # [x, y, width, height]
    ann_id = ann["id"]

    if image_id not in imagenes_dict or categoria_id not in categorias_dict:
        continue

    img_info = imagenes_dict[image_id]
    cat_info = categorias_dict[categoria_id]

    img_file = img_info["file_name"]
    ruta_imagen = os.path.join(RUTA_IMAGENES_BASE, img_file)
    ruta_imagen = os.path.normpath(ruta_imagen)

    # Buscar extensiones posibles
    if not os.path.exists(ruta_imagen):
        base_sin_ext = os.path.splitext(ruta_imagen)[0]
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            if os.path.exists(base_sin_ext + ext):
                ruta_imagen = base_sin_ext + ext
                break

    if not os.path.exists(ruta_imagen):
        print(f"ADVERTENCIA: No se encontró la imagen -> {ruta_imagen}")
        continue

    # Crear carpeta de categoría
    categoria = cat_info["name"].replace("/", "_").replace("\\", "_")
    carpeta_categoria = os.path.join(RUTA_SALIDA, categoria)
    os.makedirs(carpeta_categoria, exist_ok=True)

    try:
        with Image.open(ruta_imagen) as img:
            x, y, w, h = map(int, bbox)
            recorte = img.crop((x, y, x + w, y + h))
            nombre_salida = f"{categoria}_{ann_id}_{os.path.basename(img_file)}"
            ruta_salida = os.path.join(carpeta_categoria, nombre_salida)
            recorte.save(ruta_salida)

            area = w * h

            filas_csv.append([
                image_id, img_info["width"], img_info["height"], img_file,
                categoria_id, cat_info["name"], cat_info["supercategory"],
                ann_id, x, y, w, h, area, ruta_salida
            ])

    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {e}")

# --- GUARDAR CSV ---
with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(encabezados)
    escritor.writerows(filas_csv)

print(f"\n✅ Se generó el archivo CSV: {RUTA_CSV}")
print(f"Total de recortes procesados: {len(filas_csv)}")
