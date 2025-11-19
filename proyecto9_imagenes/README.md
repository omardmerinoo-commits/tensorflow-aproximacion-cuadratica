# Proyecto 9: Clasificador de Imágenes CIFAR-10 con CNN Profunda y Transfer Learning

## 1. Introducción

Este proyecto implementa un clasificador de imágenes para el dataset **CIFAR-10**, reconociendo 10 categorías de objetos:

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

**Características principales**:
- CNN profunda personalizada (5+ capas convolucionales)
- Transfer learning con MobileNetV2 pre-entrenada en ImageNet
- Data augmentation avanzada (rotación, zoom, flip, desplazamiento)
- Técnicas de regularización: Batch normalization, dropout, L2
- Métricas completas: Accuracy, precision, recall, F1-score

---

## 2. Teoría Fundamental

### 2.1 Redes Convolucionales (CNN)

Una CNN es una red neuronal especializada para procesar imágenes, basada en operaciones de convolución:

$$Y[i,j] = \sum_{m,n} K[m,n] \cdot X[i+m, j+n] + b$$

Donde:
- $X$: Entrada (imagen)
- $K$: Kernel (filtro)
- $Y$: Salida (mapa de características)

**Propiedades clave**:
- **Conexiones locales**: Solo vecinos cercanos se conectan
- **Compartición de parámetros**: Mismo filtro en toda la imagen
- **Invariancia traslacional**: Detecta características en cualquier posición

### 2.2 Arquitectura Jerárquica

```
Entrada [32×32×3]
    ↓
[Conv 32→64→128] Características locales (bordes, texturas)
    ↓
[MaxPool] Compresión espacial
    ↓
[Conv] Características de nivel medio (formas)
    ↓
[GlobalAvgPool] Agregación espacial
    ↓
[Dense 256→128→10] Clasificación
    ↓
Salida [10 clases]
```

**Niveles jerárquicos**:
1. **L1** (3-32 filtros): Bordes, esquinas, texturas simples
2. **L2** (32-64 filtros): Formas, patrones
3. **L3** (64-128 filtros): Partes de objetos
4. **L4+** (128+ filtros): Objetos completos

### 2.3 Transfer Learning

El transfer learning reutiliza modelos pre-entrenados:

**ImageNet → CIFAR-10**:
```
ImageNet (1.2M imágenes, 1000 clases)
    ↓
MobileNetV2 pre-entrenado (parámetros congelados)
    ↓
+ Capas personalizadas (256→128→10)
    ↓
Fine-tune en CIFAR-10 (50k imágenes, 10 clases)
```

**Ventajas**:
- Convergencia más rápida
- Mejor generalización con menos datos
- Características reutilizables

### 2.4 Data Augmentation

Aumenta artificialmente el dataset aplicando transformaciones:

$$\hat{X} = T(X; \theta)$$

Donde $\theta$ son parámetros aleatorios:
- Rotación: $\theta_{rot} \sim U(-20°, +20°)$
- Zoom: $\theta_{zoom} \sim U(0.8, 1.2)$
- Desplazamiento: $\theta_{shift} \sim U(-20\%, +20\%)$
- Flip horizontal: $\theta_{flip} \in \{0, 1\}$

**Beneficios**:
- Aumenta tamaño efectivo del dataset
- Mejora invariancia (modelos robustos)
- Reduce overfitting

---

## 3. Arquitecturas

### 3.1 CNN Personalizada

```
Input: [32×32×3]
    ↓
[Bloque 1]
  Conv2D(32, 3×3) + BatchNorm + ReLU
  Conv2D(32, 3×3) + BatchNorm + ReLU
  MaxPool(2×2) + Dropout(0.25)
  →  [16×16×32]
    ↓
[Bloque 2]
  Conv2D(64, 3×3) + BatchNorm + ReLU
  Conv2D(64, 3×3) + BatchNorm + ReLU
  MaxPool(2×2) + Dropout(0.25)
  →  [8×8×64]
    ↓
[Bloque 3]
  Conv2D(128, 3×3) + BatchNorm + ReLU
  Conv2D(128, 3×3) + BatchNorm + ReLU
  MaxPool(2×2) + Dropout(0.25)
  →  [4×4×128]
    ↓
GlobalAveragePooling2D →  [128]
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
Dense(128) + BatchNorm + Dropout(0.3)
Dense(10, softmax)
    ↓
Output: [10]
```

**Parámetros totales**: ~200k

### 3.2 Transfer Learning (MobileNetV2)

```
Input: [32×32×3]
    ↓
MobileNetV2 (frozen weights)
  Depthwise separable convolutions
  Residual blocks
  → [1×1×1280]
    ↓
GlobalAveragePooling2D →  [1280]
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
Dense(128) + BatchNorm + Dropout(0.3)
Dense(10, softmax)
    ↓
Output: [10]
```

**Parámetros nuevos**: ~300k (congelados: ~3.5M)

---

## 4. CIFAR-10 Dataset

**Características**:
- 60,000 imágenes de 32×32 píxeles
- 3 canales RGB
- 10 clases (6,000 imágenes/clase)
- Split típico: 50k train, 10k test

**Clases**:
```
0: airplane    5: dog       
1: automobile  6: frog      
2: bird        7: horse     
3: cat         8: ship      
4: deer        9: truck     
```

**Desafíos**:
- Imágenes pequeñas (32×32)
- Clases variadas y similares
- Datos reales vs. sintéticos

---

## 5. Guía de Uso

### 5.1 Uso Básico

```python
from clasificador_imagenes import GeneradorCIFAR10, ClasificadorImagenes

# 1. Cargar datos
generador = GeneradorCIFAR10(seed=42)
datos = generador.cargar_datos(validacion_split=0.2)

# 2. Entrenar CNN
clf_cnn = ClasificadorImagenes()
clf_cnn.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    epochs=50,
    arquitectura='cnn',
    usar_augmentacion=True
)

# 3. Entrenar Transfer Learning
clf_tl = ClasificadorImagenes()
clf_tl.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    epochs=50,
    arquitectura='transfer',
    usar_augmentacion=True
)

# 4. Evaluar
metricas_cnn = clf_cnn.evaluar(datos.X_test, datos.y_test)
metricas_tl = clf_tl.evaluar(datos.X_test, datos.y_test)

print(f"CNN Accuracy: {metricas_cnn['accuracy']:.4f}")
print(f"Transfer Learning Accuracy: {metricas_tl['accuracy']:.4f}")

# 5. Predecir
clases, probs = clf_cnn.predecir(datos.X_test[:10])
for i, clase in enumerate(clases):
    print(f"{datos.clases[clase]}: {probs[i, clase]:.2%}")

# 6. Guardar
clf_cnn.guardar("mi_clasificador")
```

### 5.2 Data Augmentation

```python
# Crear augmentador
aug = generador.crear_augmentador()

# Entrenar con augmentación
clf.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    usar_augmentacion=True  # Activa ImageDataGenerator
)

# Sin augmentación
clf.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    usar_augmentacion=False  # Fit directo
)
```

---

## 6. Resultados Esperados

### 6.1 Desempeño por Arquitectura

| Arquitectura | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| CNN Personalizada | 0.72-0.75 | 0.73 | 0.72 | 0.72 |
| Transfer Learning | 0.78-0.82 | 0.79 | 0.78 | 0.78 |

### 6.2 Curva de Aprendizaje

```
Epoch 1:  Loss=2.1234, Acc=0.15, Val_Loss=1.9876, Val_Acc=0.25
Epoch 10: Loss=1.2345, Acc=0.55, Val_Loss=1.3210, Val_Acc=0.52
Epoch 30: Loss=0.6789, Acc=0.72, Val_Loss=0.7654, Val_Acc=0.71
Epoch 50: Loss=0.3456, Acc=0.82, Val_Loss=0.4321, Val_Acc=0.80
```

### 6.3 Matriz de Confusión (Extracto)

```
           airplane  car  bird  cat  deer
airplane      85%    5%    3%   2%   5%
car            4%   88%    2%   2%   4%
bird           3%    2%   82%   8%   5%
cat            2%    2%    8%  80%   8%
deer           4%    3%    5%   8%  80%
```

---

## 7. Suite de Pruebas

### 7.1 Cobertura

```
✓ Carga de datos: 6 tests
✓ Data augmentation: 3 tests
✓ Construcción modelos: 4 tests
✓ Entrenamiento: 3 tests
✓ Evaluación: 3 tests
✓ Predicción: 3 tests
✓ Transfer learning: 1 test
✓ Persistencia: 1 test
✓ Clases específicas: 1 test
✓ Edge cases: 2 tests
✓ Rendimiento: 2 tests

Total: 29 tests (>90% cobertura)
```

---

## 8. Optimización Avanzada

### 8.1 Técnicas Implementadas

1. **Batch Normalization**: Estabiliza activaciones
2. **Dropout**: Previene co-adaptación de neuronas
3. **L2 Regularization**: Penaliza pesos grandes
4. **Learning Rate Scheduling**: Reduce LR cuando improvement se detiene
5. **Early Stopping**: Detiene cuando val_loss no mejora

### 8.2 Comparación de Arquitecturas

```
CNN Personalizada:
  Pros: Flexible, personalizable, entrenamiento rápido
  Contras: Menos datos → overfitting potencial

Transfer Learning:
  Pros: Mejor generalización, menos épocas
  Contras: MobileNetV2 congelada puede ser limitante
```

---

## 9. Análisis de Errores

### 9.1 Clases Difíciles

Típicamente confundidas:
- **dog ↔ cat**: Similares morfológicamente
- **bird ↔ airplane**: Ambas en cielo
- **deer ↔ horse**: Cuadrúpedos

### 9.2 Mejoras Posibles

1. Fine-tune de capas base (descongelar gradualmente)
2. Ensemble de múltiples modelos
3. Data augmentation más agresiva
4. Arquitecturas más grandes (ResNet50)
5. Datos adicionales (pseudo-labeling)

---

## 10. Archivos del Proyecto

```
proyecto9_imagenes/
├── clasificador_imagenes.py       # Módulo principal (900+ líneas)
├── test_clasificador_imagenes.py  # Suite de tests (29 tests)
├── run_training.py                # Script de demostración
├── requirements.txt               # Dependencias
├── README.md                      # Documentación (este archivo)
└── LICENSE                        # MIT License
```

---

## 11. Referencias

- **CNN Fundamentals**: LeCun et al. (1998) "LeNet"
- **Modern Architectures**: Krizhevsky et al. (2012) "AlexNet"
- **Transfer Learning**: Yosinski et al. (2014) "How transferable are features?"
- **MobileNetV2**: Sandler et al. (2018) "Inverted Residuals"
- **Data Augmentation**: Cubuk et al. (2019) "AutoAugment"

---

**Última actualización**: 2024
**Autor**: Omar Demerinoo
**Estado**: ✅ Producción
