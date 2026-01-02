# Comparación de HOG+SVM y ResNet-18 en la clasificación de residuos sólidos utilizando imágenes en escenarios controlados y realistas
Este repositorio contiene el proyecto comparativo entre enfoques de aprendizaje automático clásico y aprendizaje profundo para la clasificación de residuos sólidos a partir de imágenes, empleando los modelos HOG + SVM y ResNet-18, evaluados bajo distintos escenarios controlados y realistas.

---
## Objetivos de Proyecto

### Objetvo General
Implementar y comparar la efectividad del algoritmo Support Vector Machine (SVM) en conjunto con el descriptor de Histograma de Gradientes Orientados (HOG) contra el modelo de red neuronal convolucional ResNet-18 para la clasificación de residuos sólidos para su reciclaje, empleando imágenes que contienen escenarios controlados y realistas.

### Objetivos Específicos
1. Recolectar el conjunto de datos para el entrenamiento de los algoritmos, obteniendo los datos de distintas fuentes públicas.
2. Preprocesar el conjunto de datos mediante técnicas de procesamiento de imágenes.
3. Implementar el algoritmo Support Vector Machine (SVM) en conjunto con el descriptor de Histograma de Gradientes Orientados (HOG) y el modelo de red neuronal convolucional ResNet-18.
4. Evaluar el desempeño de ambos algoritmos por medio de métricas de evaluación clásicas en aprendizaje máquina: accuracy, precisión, recall y f1-score.
5. Comparar las métricas obtenidas de los algoritmos evaluados en cada escenario de prueba.

---
## Dataset utilizado
Para el desarrollo y evaluación del proyecto se emplearon dos conjuntos de datos públicos para la clasificación de residuos sólidos, los cuales representan escenarios visuales contrastantes: uno controlado y otro realista. Esta selección permite analizar el comportamiento de los modelos tanto en condiciones ideales como en situaciones cercanas a aplicaciones del mundo real.

<p align="center">
  <img width="359" height="168" alt="ejemplos_dataset"
       src="https://github.com/user-attachments/assets/25a18da4-783c-4c78-8081-107dd7d94933" />
</p>


### TrashNet (Escenarios controlados)
El dataset TrashNet está compuesto por imágenes de residuos sólidos capturadas bajo condiciones controladas, tales como fondos uniformes, iluminación relativamente constante y objetos bien centrados. Estas características reducen la variabilidad visual y facilitan el aprendizaje de patrones discriminativos básicos. Puede obtenerlo del siguiente enlace: https://www.kaggle.com/datasets/feyzazkefe/trashnet

### TACO (Escenarios realistas)
El dataset TACO (Trash Annotations in Context) está diseñado para representar escenarios realistas, donde los residuos aparecen en contextos reales, con fondos complejos, múltiples objetos, variaciones de escala, oclusiones e iluminación no controlada. Incluye imágenes capturadas en entornos cotidianos (calles, parques, interiores), reflejando condiciones cercanas a aplicaciones prácticas de reciclaje inteligente. Puede obtener el dataset por medio de este enlace: https://www.kaggle.com/datasets/kneroma/tacotrashdataset

---
## Metodología
<p align="center">
  <img width="337" height="469" alt="image" src="https://github.com/user-attachments/assets/ca0e2784-ee6d-4242-9904-4f04355a004e" />
</p>

### 1. Recolección y organización de datos
La metodología inicia con la recolección de datos a partir de dos conjuntos públicos complementarios: TrashNet y TACO. TrashNet aporta imágenes capturadas en escenarios controlados, mientras que TACO proporciona imágenes en contextos realistas con mayor variabilidad visual. Con el fin de asegurar coherencia entre ambos dominios, las clases de los datasets fueron homologadas a un conjunto común de categorías de residuos. Posteriormente, las imágenes se organizaron y dividieron en subconjuntos de entrenamiento, validación y prueba, manteniendo una distribución estratificada que preserva el equilibrio entre clases y escenarios.

### 2. Preprocesamiento de imágenes
En esta etapa, las imágenes fueron adaptadas de manera diferenciada según los requerimientos de cada enfoque. Para el modelo HOG + SVM, las imágenes se transformaron a escala de grises, se redimensionaron a un tamaño uniforme y se prepararon para la extracción de descriptores basados en gradientes. En el caso de ResNet-18, se conservaron los canales RGB, se normalizaron las imágenes y se aplicaron técnicas de aumento de datos con el objetivo de incrementar la diversidad del conjunto de entrenamiento y mejorar la capacidad de generalización, especialmente en escenarios realistas.

### 3. Entrenamiento y configuración de modelos
Una vez preprocesados los datos, se procedió al entrenamiento de los modelos. El enfoque HOG + SVM se entrenó utilizando los vectores de características extraídos de cada imagen, ajustando los parámetros del clasificador para maximizar su desempeño. Por su parte, ResNet-18 se implementó mediante transferencia de aprendizaje, adaptando la capa final de clasificación a las clases del problema y entrenando el modelo bajo un esquema controlado que permite aprovechar los conocimientos previamente aprendidos en grandes conjuntos de datos.

### 4. Experimentación en escenarios de evaluación
Con el propósito de analizar el comportamiento de los modelos en diferentes condiciones, se definieron múltiples escenarios experimentales. Estos incluyeron evaluaciones intradominio, donde el entrenamiento y la prueba se realizan sobre el mismo tipo de escenario (S0 y S1), así como evaluaciones interdominio, en las que los modelos se entrenan en un tipo de escenario y se prueban en otro (S2 y S3). Esta estrategia permitió estudiar la robustez y la capacidad de generalización de cada enfoque frente a cambios en el contexto visual.

### 5. Evaluación y comparación de resultados
Finalmente, el desempeño de los modelos fue evaluado utilizando métricas clásicas de clasificación multiclase, como accuracy, precision, recall y F1-score macro, complementadas con matrices de confusión. Estas métricas permitieron realizar una comparación integral entre el enfoque clásico y el enfoque de aprendizaje profundo, identificando tanto el rendimiento global como los patrones de acierto y confusión entre clases, y facilitando el análisis del impacto del tipo de escenario en los resultados obtenidos.

---
## Orden de ejecución
Para poder realizar la ejecución de los programas en el orden correcto, es importamte contar con los conjuntos de datos utilizados en este proyecto dentro de la carpeta base del proyecto y en carpetas separadas, una para TrashNet y otra para TACO, cada una con subcarpetas que contiene las imagenes clasificadas en las 6 categorías (papel, cartón, vidrio, plástico, metal y basura). El repositorio ya cuenta con los dos conjuntos de datos preprocesados y listos para ser utilizados en la carpeta "data_preprocessed" por lo que los pasos 1 y 2 que se presentan a continuación solo se deben ejecutar en el caso de que se descargue los conjuntos de datos desde los links que se proporcionaron en la parte de Dataset utilizado.

### 1. Extracción
1. Abrir la carpeta 1_Extracción que se encuentra dentro de la parpeta de Programas y ejecutar el primer código llamado 1_recorte, que se encarga de recortar las áreas de interés en las imagenes del conjunto de datos de TACO.
2. Ejecutar el segundo código 2_duplicados que se encarga de identificar si hay imagenes duplicadas en el conjunto de datos.
3. Ejecutar el tercer código 3_eliminar_duplicados, que elimina todas las imagenes que estén duplicadas en el conjunto de datos.

### 2. Preprocesamiento
1. Abrir la carpeta 2_Preprocesamiento en donde se se encontrarán 4 códigos de Python el código "redimensionar" se encarga de redimensionar todas las imagenes a medidas exactas de 224x224 pixeles y se importa en los códigos "preprocess_hog_svm" y "preprocess_resnet" que realizan todo el preprocesamiento que se describe en la metodología. A su vez estos dos códigos se imprtan en el código principal "preprocesar" que se encarga de ejecutar todo y realizar los preprocesamientos completos a los dos conjuntos de datos.

### 3. HOG+SVM
1. Desde Jupyter, abrir la carpeta "3_HOG_SVM" y abrir el primer notebook "S0_S1" y ejecutar los bloques de código, estos bloques corresponden a los escenarios 1 y 2.
2. Abrir el segundo notebook "S2_S3_GD" y ejecutar los bloques de código, estos bloques corresponden a los escenarios 3 y 4 utilizando Grid Search para encontrar los mejores parámteros.

### 4. ResNet
1. De igual forma, desde Jupyter Notebook abrir la carpeta "4_ResNet" y abrir el primer notebook "ResNet_S0" y ajustar la ruta a su ruta local y ejecutar los bloques de codigo.
   ```python
    data_dir = r"ruta_local\Clasificacion_residuos"
   ```
   Este notebook corresponde al primer escenario sin data augmentation y con data augmentation.
2. Abrir el segundo notebook "ResNet_S1" y de igual forma ajustar la ruta a la ruta local y ejecutar los bloques de código.
   ```python
    data_dir = r"ruta_local\Clasificacion_residuos\data_preprocessed\ResNet18\TACO"
   ```
   Este notebook corresponde al escenario 2 sin data augmentation y con data augmentation.
3. Abrir el tercer notebook "ResNet_S2" y de igual forma ajustar la ruta a la ruta local y ejecutar los bloques de código.
   ```python
    trashnet_dir = r"ruta_local\Clasificacion_residuos\data_preprocessed\ResNet18\TrashNet"
    taco_dir     = r"ruta_local\Clasificacion_residuos\data_preprocessed\ResNet18\TACO"
    output_dir = r"ruta_local\Clasificacion_residuos\Programas\ResNet\Modelos_experimentos"
   ```
   En el último bloque ajustar la ruta a la local para el guardado de el hisotirial de exactitud y pérdida
   ```python
    salida = r"ruta_local\Clasificacion_residuos\Programas\ResNet\Historial"
   ```
   Este notebook corresponde al tercer escenario sin data augmentation y con data augmentation.
   
4. Abrir el tercer notebook "ResNet_S3" y de igual forma ajustar la ruta a la ruta local y ejecutar los bloques de código.
   ```python
    trashnet_dir = r"ruta_local\Clasificacion_residuos\data_preprocessed\ResNet18\TrashNet"
    taco_dir     = r"ruta_local\Clasificacion_residuos\data_preprocessed\ResNet18\TACO"
    output_dir = r"ruta_local\Clasificacion_residuos\Programas\ResNet\Modelos_experimentos"
   ```
   En el último bloque ajustar la ruta a la local para el guardado de el hisotirial de exactitud y pérdida
   ```python
    salida = r"ruta_local\Clasificacion_residuos\Programas\ResNet\Historial"
   ```
   Este notebook corresponde al cuarto escenario sin data augmentation y con data augmentation.

   
---
## Autores
* Samuel Soriano Chávez
* Sergio de Jesús Castillo Molano
* Juan Carlos Flores Mora
