# ============================================================
# main.py
# ============================================================
from preprocess_hog_svm import preprocess_for_hogsvm
from preprocess_resnet import preprocess_for_resnet

if __name__ == "__main__":
    datasets = {
        "TACO": r"D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\TACO",
        "TrashNet": r"D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\TrashNet"
    }

    salida_hogsvm = r"D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\data_preprocessed/HOG_SVM"
    salida_resnet = r"D:\Documentos\IPN\6to Semestre\Metodologia_de_la_investigacion\Clasificacion_residuos\data_preprocessed/ResNet18"

    print("=== Procesando para HOG+SVM ===")
    for nombre, ruta in datasets.items():
        preprocess_for_hogsvm(nombre, ruta, salida_hogsvm)

    print("\n=== Procesando para ResNet-18 ===")
    for nombre, ruta in datasets.items():
        preprocess_for_resnet(nombre, ruta, salida_resnet)