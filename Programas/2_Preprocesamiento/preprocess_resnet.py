# ============================================================
# preprocess_resnet.py
# ============================================================
import cv2
import os
import numpy as np
from redimensionar import make_square

def preprocess_for_resnet(dataset_name, input_dir, output_root, size=224):
    """
    Preprocesa un dataset (TACO o TrashNet) para ResNet-18.
    Estructura final:
        data_preprocessed/
        └── dataset_name/
            ├── cardboard/
            ├── glass/
            ├── metal/
            ├── paper/
            ├── plastic/
            └── trash/
    """
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    clases = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    for clase in clases:
        clase_in = os.path.join(input_dir, clase)
        clase_out = os.path.join(output_dir, clase)
        os.makedirs(clase_out, exist_ok=True)

        contador = 0
        for nombre in os.listdir(clase_in):
            if not nombre.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            ruta = os.path.join(clase_in, nombre)
            try:
                img = cv2.imread(ruta)
                if img is None:
                    continue

                img_square = make_square(img, desired_size=size)
                nombre_salida = f"{clase}_{contador:05d}.jpg"
                cv2.imwrite(os.path.join(clase_out, nombre_salida),
                            (img_square).astype(np.uint8))
                contador += 1
            except Exception as e:
                print(f"Error procesando {ruta}: {e}")

        print(f"[{dataset_name}] {clase}: {contador} imágenes procesadas.")
    print(f"\nResNet ({dataset_name}) completado en: {output_dir}")
