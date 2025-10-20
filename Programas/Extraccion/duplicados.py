from PIL import Image
import imagehash
import os

hashes = {}
duplicados = []

ruta_dataset = r'D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\TrashNet'

for clase in os.listdir(ruta_dataset):
    carpeta = os.path.join(ruta_dataset, clase)
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if not nombre.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue
        try:
            h = imagehash.average_hash(Image.open(ruta))
            if h in hashes:
                duplicados.append(ruta)
            else:
                hashes[h] = ruta
        except:
            continue

print(f"Duplicados detectados: {len(duplicados)}")
