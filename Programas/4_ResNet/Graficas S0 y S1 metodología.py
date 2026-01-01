import matplotlib.pyplot as plt
"""
# Datos por época
epocas = list(range(1, 11))
train_loss = [1.3256, 0.8444, 0.7364, 0.6313, 0.5718, 0.5529, 0.5049, 0.4828, 0.4547, 0.4431]
val_loss   = [0.9997, 0.7763, 0.6842, 0.6591, 0.6115, 0.5848, 0.5749, 0.5497, 0.5532, 0.5376]
train_acc  = [0.5106, 0.7284, 0.7530, 0.7885, 0.8102, 0.8130, 0.8285, 0.8422, 0.8588, 0.8496]
val_acc    = [0.6578, 0.7246, 0.7781, 0.7727, 0.7968, 0.8048, 0.8048, 0.8209, 0.8102, 0.8209]

# Crear figura
plt.figure(figsize=(10, 5))

# --- Subgráfico 1: Loss ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Subgráfico 2: Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 con TrashNet y capas convolucionales congeladas', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Datos por época (fine-tuning)
epocas = list(range(1, 11))
train_loss = [0.3850, 0.0582, 0.0153, 0.0076, 0.0049, 0.0042, 0.0024, 0.0021, 0.0019, 0.0012]
val_loss   = [0.3239, 0.2876, 0.2331, 0.2371, 0.2395, 0.2425, 0.2557, 0.2589, 0.2559, 0.2590]
train_acc  = [0.8634, 0.9880, 0.9977, 0.9994, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
val_acc    = [0.8904, 0.9144, 0.9278, 0.9332, 0.9251, 0.9198, 0.9225, 0.9225, 0.9225, 0.9171]

# Crear figura
plt.figure(figsize=(10, 5))

# --- Subgráfico 1: Loss ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida (Fine-tuning)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Subgráfico 2: Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo (Fine-tuning)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (fine-tuning de capas superiores)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""

"""
# Datos por época (S1_split - TACO, capas congeladas)
epocas = list(range(1, 11))
train_loss = [1.3372, 1.0983, 1.0088, 0.9669, 0.9403, 0.9112, 0.9017, 0.8782, 0.8662, 0.8614]
val_loss   = [1.1007, 1.0455, 0.9890, 0.9640, 0.9488, 0.9592, 0.9231, 0.9444, 0.9386, 0.9477]
train_acc  = [0.5051, 0.6076, 0.6377, 0.6527, 0.6576, 0.6746, 0.6681, 0.6707, 0.6828, 0.6841]
val_acc    = [0.6204, 0.6326, 0.6479, 0.6418, 0.6509, 0.6418, 0.6799, 0.6555, 0.6707, 0.6646]

# Crear figura
plt.figure(figsize=(10, 5))

# --- Subgráfico 1: Loss ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Subgráfico 2: Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (dataset TACO, capas convolucionales congeladas)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""

"""
# Datos por época (S1_split_finetuned - TACO)
epocas = list(range(1, 11))
train_loss = [0.8698, 0.2636, 0.0798, 0.0345, 0.0181, 0.0184, 0.0098, 0.0075, 0.0222, 0.0565]
val_loss   = [0.8816, 0.9277, 0.9731, 0.9665, 1.0280, 1.1855, 1.1345, 1.1121, 1.3804, 1.4343]
train_acc  = [0.6776, 0.9242, 0.9860, 0.9958, 0.9980, 0.9977, 0.9993, 0.9993, 0.9954, 0.9817]
val_acc    = [0.6601, 0.6662, 0.6905, 0.7226, 0.7012, 0.6890, 0.6921, 0.7088, 0.6662, 0.6357]

# Crear figura
plt.figure(figsize=(10, 5))

# --- Subgráfico 1: Loss ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida (Fine-tuning)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Subgráfico 2: Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo (Fine-tuning)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (dataset TACO, fine-tuning de capas superiores)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# --- Escenario 1: capas congeladas (S1_split) ---
epocas = list(range(1, 11))
train_loss_freeze = [1.3256, 0.8444, 0.7364, 0.6313, 0.5718, 0.5529, 0.5049, 0.4828, 0.4547, 0.4431]
val_loss_freeze   = [0.9997, 0.7763, 0.6842, 0.6591, 0.6115, 0.5848, 0.5749, 0.5497, 0.5532, 0.5376]
train_acc_freeze  = [0.5106, 0.7284, 0.7530, 0.7885, 0.8102, 0.8130, 0.8285, 0.8422, 0.8588, 0.8496]
val_acc_freeze    = [0.6578, 0.7246, 0.7781, 0.7727, 0.7968, 0.8048, 0.8048, 0.8209, 0.8102, 0.8209]

# --- Escenario 2: fine-tuning (S0_split_finetuned) ---
train_loss_finetune = [0.3850, 0.0582, 0.0153, 0.0076, 0.0049, 0.0042, 0.0024, 0.0021, 0.0019, 0.0012]
val_loss_finetune   = [0.3239, 0.2876, 0.2331, 0.2371, 0.2395, 0.2425, 0.2557, 0.2589, 0.2559, 0.2590]
train_acc_finetune  = [0.8634, 0.9880, 0.9977, 0.9994, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
val_acc_finetune    = [0.8904, 0.9144, 0.9278, 0.9332, 0.9251, 0.9198, 0.9225, 0.9225, 0.9225, 0.9171]

# --- Figura comparativa ---
plt.figure(figsize=(10, 5))

# Subgráfico 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_loss_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas, train_loss_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_loss_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de función de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

# Subgráfico 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_acc_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas, train_acc_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_acc_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

plt.suptitle('Comparación de entrenamiento ResNet-18 (dataset TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
epocas = list(range(1, 11))

# --- Escenario 1: Capas congeladas ---
train_loss_freeze = [1.7244, 1.4068, 1.3307, 1.2425, 1.2730, 1.2320, 1.1917, 1.2138, 1.2293, 1.2133]
val_loss_freeze   = [1.5207, 1.4706, 1.4312, 1.5226, 1.2939, 1.2788, 1.2213, 1.3130, 1.3198, 1.1891]
train_acc_freeze  = [0.2956, 0.4470, 0.4804, 0.5304, 0.5128, 0.5386, 0.5602, 0.5494, 0.5291, 0.5383]
val_acc_freeze    = [0.4074, 0.4119, 0.4273, 0.3997, 0.5253, 0.4992, 0.5681, 0.5115, 0.5161, 0.5681]

# --- Escenario 2: Fine-tuning ---
train_loss_finetune = [0.7803, 0.3670, 0.2325, 0.1510, 0.1045, 0.0739, 0.0574]
val_loss_finetune   = [1.1057, 0.9725, 0.9285, 0.9151, 0.9641, 1.0803, 1.0411]
train_acc_finetune  = [0.7217, 0.8842, 0.9274, 0.9532, 0.9683, 0.9787, 0.9853]
val_acc_finetune    = [0.6080, 0.6708, 0.6861, 0.6922, 0.6922, 0.6891, 0.6830]

# --- Figura comparativa ---
plt.figure(figsize=(10, 5))

# Subgráfico 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_loss_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas, train_loss_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_loss_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de función de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

# Subgráfico 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_acc_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas, train_acc_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_acc_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

plt.suptitle('Comparación de entrenamiento ResNet-18 (dataset TrashNet)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Épocas
epocas = list(range(1, 11))

# --- Escenario 1: Capas congeladas ---
train_loss_freeze = [1.7244, 1.4068, 1.3307, 1.2425, 1.2730, 1.2320, 1.1917, 1.2138, 1.2293, 1.2133]
val_loss_freeze   = [1.5207, 1.4706, 1.4312, 1.5226, 1.2939, 1.2788, 1.2213, 1.3130, 1.3198, 1.1891]
train_acc_freeze  = [0.2956, 0.4470, 0.4804, 0.5304, 0.5128, 0.5386, 0.5602, 0.5494, 0.5291, 0.5383]
val_acc_freeze    = [0.4074, 0.4119, 0.4273, 0.3997, 0.5253, 0.4992, 0.5681, 0.5115, 0.5161, 0.5681]

# --- Escenario 2: Fine-tuning ---
train_loss_finetune = [0.7803, 0.3670, 0.2325, 0.1510, 0.1045, 0.0739, 0.0574]
val_loss_finetune   = [1.1057, 0.9725, 0.9285, 0.9151, 0.9641, 1.0803, 1.0411]
train_acc_finetune  = [0.7217, 0.8842, 0.9274, 0.9532, 0.9683, 0.9787, 0.9853]
val_acc_finetune    = [0.6080, 0.6708, 0.6861, 0.6922, 0.6922, 0.6891, 0.6830]

# --- Figura 1: Capa final (capas congeladas) ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss_freeze, marker='o', label='Train Loss')
plt.plot(epocas, val_loss_freeze, marker='s', label='Val Loss')
plt.title('Pérdida - Capa final (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc_freeze, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc_freeze, marker='s', label='Val Accuracy')
plt.title('Exactitud - Capa final (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (TACO, capas convolucionales congeladas)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Figura 2: Fine-tuning ---
epocas_ft = list(range(1, len(train_loss_finetune)+1))
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epocas_ft, train_loss_finetune, marker='o', label='Train Loss')
plt.plot(epocas_ft, val_loss_finetune, marker='s', label='Val Loss')
plt.title('Pérdida - Fine-tuning')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epocas_ft, train_acc_finetune, marker='o', label='Train Accuracy')
plt.plot(epocas_ft, val_acc_finetune, marker='s', label='Val Accuracy')
plt.title('Exactitud - Fine-tuning')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (TACO, fine-tuning de capas superiores)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Figura 3: Comparativa directa ---
plt.figure(figsize=(10, 5))

# Comparación de Accuracy
plt.subplot(1, 2, 1)
plt.plot(epocas, val_acc_freeze, 's--', label='Val (capas congeladas)')
plt.plot(epocas_ft, val_acc_finetune, 'o-', label='Val (fine-tuning)')
plt.title('Comparación de exactitud en validación')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Comparación de Loss
plt.subplot(1, 2, 2)
plt.plot(epocas, val_loss_freeze, 's--', label='Val (capas congeladas)')
plt.plot(epocas_ft, val_loss_finetune, 'o-', label='Val (fine-tuning)')
plt.title('Comparación de pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Comparación de rendimiento ResNet-18 (TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Épocas
epocas = list(range(1, 11))
epocas_ft = list(range(1, 8))  # fine-tuning terminó en la época 7

# === FASE 1: CAPAS CONGELADAS ===
train_loss_freeze = [1.7244, 1.4068, 1.3307, 1.2425, 1.2730, 1.2320, 1.1917, 1.2138, 1.2293, 1.2133]
val_loss_freeze   = [1.5207, 1.4706, 1.4312, 1.5226, 1.2939, 1.2788, 1.2213, 1.3130, 1.3198, 1.1891]
train_acc_freeze  = [0.2956, 0.4470, 0.4804, 0.5304, 0.5128, 0.5386, 0.5602, 0.5494, 0.5291, 0.5383]
val_acc_freeze    = [0.4074, 0.4119, 0.4273, 0.3997, 0.5253, 0.4992, 0.5681, 0.5115, 0.5161, 0.5681]

# === FASE 2: FINE-TUNING ===
train_loss_finetune = [0.7803, 0.3670, 0.2325, 0.1510, 0.1045, 0.0739, 0.0574]
val_loss_finetune   = [1.1057, 0.9725, 0.9285, 0.9151, 0.9641, 1.0803, 1.0411]
train_acc_finetune  = [0.7217, 0.8842, 0.9274, 0.9532, 0.9683, 0.9787, 0.9853]
val_acc_finetune    = [0.6080, 0.6708, 0.6861, 0.6922, 0.6922, 0.6891, 0.6830]

# === GRÁFICA COMPARATIVA ===
plt.figure(figsize=(10, 5))

# Subgráfico 1: Función de pérdida
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_loss_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas_ft, train_loss_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas_ft, val_loss_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de función de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

# Subgráfico 2: Exactitud
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc_freeze, 'o-', label='Train (congeladas)')
plt.plot(epocas, val_acc_freeze, 's--', label='Val (congeladas)')
plt.plot(epocas_ft, train_acc_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas_ft, val_acc_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

plt.suptitle('Comparación de entrenamiento ResNet-18 (dataset TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Datos
escenarios = ['Escenario 1\n(Fine-tuning TrashNet -> TrashNet)', 'Escenario 2\n(Fine-tuning TACO -> TACO)']
accuracy = [0.9175, 0.70]

# Colores personalizados
colors = ['#4CAF50', '#2196F3']

plt.figure(figsize=(6,5))
bars = plt.bar(escenarios, accuracy, color=colors, edgecolor='black')

# Mostrar valor sobre cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Detalles de la gráfica
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparación de rendimiento entre escenarios de entrenamiento', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()
"""

"""
# Épocas
epocas = list(range(1, 11))

# Datos del entrenamiento (capas congeladas)
train_loss = [1.3439, 1.0961, 1.0103, 0.9677, 0.9363, 0.9106, 0.9030, 0.8799, 0.8673, 0.8612]
val_loss   = [1.1143, 1.0399, 0.9804, 0.9619, 0.9280, 0.9324, 0.9176, 0.9123, 0.9207, 0.9311]
train_acc  = [0.5162, 0.6132, 0.6364, 0.6615, 0.6596, 0.6762, 0.6762, 0.6811, 0.6887, 0.6929]
val_acc    = [0.6189, 0.6433, 0.6570, 0.6570, 0.6768, 0.6555, 0.6784, 0.6707, 0.6738, 0.6799]

# Crear figura
plt.figure(figsize=(10,5))

# --- Subgráfico 1: Pérdida ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Subgráfico 2: Exactitud ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo (capas congeladas)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (fase 1: capa final, dataset TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Épocas
epocas = list(range(1, 11))

# Datos FASE 2: Fine-Tuning
train_loss = [0.8861, 0.2902, 0.0862, 0.0330, 0.0181, 0.0123, 0.0137, 0.0066, 0.0061, 0.0057]
val_loss   = [0.8818, 0.8508, 0.8956, 0.9202, 0.9910, 1.1696, 1.0818, 1.0348, 1.1747, 1.1141]
train_acc  = [0.6798, 0.9111, 0.9840, 0.9967, 0.9987, 0.9990, 0.9984, 0.9997, 0.9997, 1.0000]
val_acc    = [0.7012, 0.7073, 0.7149, 0.7241, 0.7195, 0.7226, 0.7180, 0.7348, 0.7256, 0.7393]

# Gráfica de entrenamiento (Fine-Tuning)
plt.figure(figsize=(10,5))

# --- Pérdida ---
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss, marker='o', label='Train Loss')
plt.plot(epocas, val_loss, marker='s', label='Val Loss')
plt.title('Progreso de la función de pérdida (Fine-Tuning)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- Exactitud ---
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc, marker='o', label='Train Accuracy')
plt.plot(epocas, val_acc, marker='s', label='Val Accuracy')
plt.title('Progreso de la exactitud del modelo (Fine-Tuning)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.suptitle('Entrenamiento ResNet-18 (Fase 2: Fine-Tuning, dataset TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""
"""
# Épocas
epocas = list(range(1, 11))

# === FASE 1: CAPAS CONGELADAS ===
train_loss_freeze = [1.3439, 1.0961, 1.0103, 0.9677, 0.9363, 0.9106, 0.9030, 0.8799, 0.8673, 0.8612]
val_loss_freeze   = [1.1143, 1.0399, 0.9804, 0.9619, 0.9280, 0.9324, 0.9176, 0.9123, 0.9207, 0.9311]
train_acc_freeze  = [0.5162, 0.6132, 0.6364, 0.6615, 0.6596, 0.6762, 0.6762, 0.6811, 0.6887, 0.6929]
val_acc_freeze    = [0.6189, 0.6433, 0.6570, 0.6570, 0.6768, 0.6555, 0.6784, 0.6707, 0.6738, 0.6799]

# === FASE 2: FINE-TUNING ===
train_loss_finetune = [0.8861, 0.2902, 0.0862, 0.0330, 0.0181, 0.0123, 0.0137, 0.0066, 0.0061, 0.0057]
val_loss_finetune   = [0.8818, 0.8508, 0.8956, 0.9202, 0.9910, 1.1696, 1.0818, 1.0348, 1.1747, 1.1141]
train_acc_finetune  = [0.6798, 0.9111, 0.9840, 0.9967, 0.9987, 0.9990, 0.9984, 0.9997, 0.9997, 1.0000]
val_acc_finetune    = [0.7012, 0.7073, 0.7149, 0.7241, 0.7195, 0.7226, 0.7180, 0.7348, 0.7256, 0.7393]

# === GRÁFICA COMPARATIVA ===
plt.figure(figsize=(10, 5))

# Subgráfico 1: Función de pérdida
plt.subplot(1, 2, 1)
plt.plot(epocas, train_loss_freeze, 'o-', label='Train (capas congeladas)')
plt.plot(epocas, val_loss_freeze, 's--', label='Val (capas congeladas)')
plt.plot(epocas, train_loss_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_loss_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de función de pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

# Subgráfico 2: Exactitud
plt.subplot(1, 2, 2)
plt.plot(epocas, train_acc_freeze, 'o-', label='Train (capas congeladas)')
plt.plot(epocas, val_acc_freeze, 's--', label='Val (capas congeladas)')
plt.plot(epocas, train_acc_finetune, 'o-', label='Train (fine-tuning)')
plt.plot(epocas, val_acc_finetune, 's--', label='Val (fine-tuning)')
plt.title('Comparación de exactitud del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=8)

plt.suptitle('Comparación de entrenamiento ResNet-18 (dataset TACO)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Datos
escenarios = ['Escenario 1\n(Fine-tuning TrashNet -> TrashNet)', 'Escenario 2\n(Fine-tuning TACO -> TACO)']
accuracy = [0.9175, 0.756]

# Colores personalizados
colors = ['#4CAF50', '#2196F3']

plt.figure(figsize=(6,5))
bars = plt.bar(escenarios, accuracy, color=colors, edgecolor='black')

# Mostrar valor sobre cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Detalles de la gráfica
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparación de rendimiento entre escenarios de entrenamiento', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()
"""

# Datos solo para países de bajos ingresos
categoria = ['Bajos ingresos']
no_recolectado = [60]
mal_manejo = [93]

x = range(len(categoria))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x, no_recolectado, width, label='No recolectado', color='#42a5f5')
plt.bar([i + width for i in x], mal_manejo, width, label='Manejo inadecuado', color='#ef5350')

# Configuración del gráfico
plt.xticks([i + width/2 for i in x], categoria)
plt.ylabel('Porcentaje (%)')
plt.title('Gestión de residuos en países de bajos ingresos')
plt.legend()
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
