import os
from PIL import Image
import imagehash
from tqdm import tqdm
import shutil

# === CONFIGURACIÓN ===
RUTA_DATASET = r'D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\TrashNet'
EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png')
UMBRAL_SIMILITUD = 5  # 0 = idénticas, hasta 5 tolera leves diferencias
MOVER_A = os.path.join(RUTA_DATASET, "_duplicados")  # Carpeta donde se guardarán las imágenes eliminadas

# Crear carpeta para duplicados si no existe
os.makedirs(MOVER_A, exist_ok=True)

# === FUNCIONES ===
def obtener_hash(imagen_path):
    try:
        img = Image.open(imagen_path).convert('RGB')
        return imagehash.phash(img)  # phash = perceptual hash robusto
    except Exception as e:
        print(f"Error leyendo {imagen_path}: {e}")
        return None

def eliminar_duplicados_en_carpeta(ruta_base):
    print(f"\nAnalizando carpeta: {ruta_base}")
    hashes = {}
    duplicados = []

    # Recorre todas las subcarpetas
    for root, _, files in os.walk(ruta_base):
        for nombre in tqdm(files, desc=f"Procesando {root}", ncols=80):
            if not nombre.lower().endswith(EXTENSIONES_VALIDAS):
                continue

            ruta_img = os.path.join(root, nombre)
            h = obtener_hash(ruta_img)
            if h is None:
                continue

            # Buscar imágenes similares
            encontrado = False
            for hash_existente, ruta_existente in hashes.items():
                if abs(h - hash_existente) < UMBRAL_SIMILITUD:
                    duplicados.append(ruta_img)
                    encontrado = True
                    break

            if not encontrado:
                hashes[h] = ruta_img

    # Mover duplicados detectados
    for dup in duplicados:
        destino = os.path.join(MOVER_A, os.path.basename(dup))
        try:
            shutil.move(dup, destino)
        except Exception as e:
            print(f"No se pudo mover {dup}: {e}")

    print(f"\n✅ Duplicados eliminados: {len(duplicados)}")
    print(f"📂 Guardados en: {MOVER_A}")

# === EJECUCIÓN ===
if __name__ == "__main__":
    eliminar_duplicados_en_carpeta(RUTA_DATASET)
