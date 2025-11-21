# Aplicaciones del Guion - 13 Proyectos TensorFlow

Este directorio contiene todos los codigos de las aplicaciones presentadas en el **Guion Completo de 13 Proyectos TensorFlow** (video completo).

## Estructura

```
aplicaciones_guion/
├── p0_regresion_cuadratica.py         # Predictor de precios de casas
├── p1_regresion_multilineal.py        # Predictor de consumo energia
├── p2_clasificacion_fraude.py         # Detector de fraude
├── p3_multiclase_diagnostico.py       # Clasificador diagnóstico
├── p4_clustering_kmeans.py            # Segmentador clientes
├── p5_compresor_imagenes.py           # Compresor imágenes
├── p6_cnn_digitos.py                  # Reconocedor dígitos MNIST
├── p7_audio_conv1d.py                 # Clasificador ruido (Conv1D)
├── p8_detector_objetos.py             # Detector de objetos
├── p9_segmentador_unet.py             # Segmentador semántico (U-Net)
├── p10_lstm_series.py                 # Predictor series temporales (LSTM)
├── p11_nlp_sentimientos.py            # Clasificador sentimientos (RNN)
├── p12_vae_generador.py               # Generador imágenes (VAE)
├── ejecutar_todos.py                  # Script para ejecutar todos
└── README.md                          # Este archivo
```

## Descripcion de Proyectos

### Grupo 1: Regresion (2 proyectos)

**P0: Regresion Cuadratica**
- Predice precio de casas basado en superficie
- Relacion cuadratica: y = base + coef1*X + coef2*X²
- Tecnica: Minimos cuadrados (ecuacion normal)
- Metricas: MSE, RMSE, MAE, R²

**P1: Regresion Multilineal**
- Predice consumo de energia basado en 4 variables
- Tecnica: Regresion lineal multivariable
- Metricas: MAE, RMSE

### Grupo 2: Clasificacion (3 proyectos)

**P2: Clasificacion Fraude**
- Detecta transacciones fraudulentas (binaria)
- Tecnica: Regresion logistica
- Explicacion: Sigmoid, gradient descent
- Metricas: Accuracy, Precision, Recall, F1, AUC

**P3: Multiclase Diagnostico**
- Clasifica 3 enfermedades basado en sintomas
- Tecnica: Red neuronal densa
- Funciones: BatchNormalization, Softmax
- Metricas: Accuracy, F1-Score

**P6: CNN Digitos**
- Reconoce digitos manuscritos (0-9)
- Tecnica: Convolutional Neural Network
- Arquitectura: Conv2D -> Pool -> Dense
- Metricas: Accuracy, F1-Score

### Grupo 3: Clustering & Dimensionalidad (2 proyectos)

**P4: Clustering K-Means**
- Agrupa clientes en segmentos
- Tecnica: Autoencoder + K-Means
- Metrica: Silhouette Score

**P5: Compresor Imagenes**
- Comprime imagenes 28x28 a espacio latente
- Tecnica: Autoencoder convolucional
- Ratio: 12:1 compresion

### Grupo 4: Audio (1 proyecto)

**P7: Clasificador Ruido**
- Clasifica 3 tipos de ruido (Conv1D)
- Tecnica: Conv1D para espectrogramas
- Metricas: Accuracy, F1-Score

### Grupo 5: Vision Computacional (2 proyectos)

**P8: Detector Objetos**
- Detecta objetos y bounding boxes
- Tecnica: CNN con region proposals
- Salidas: Clasificacion + Regression (bbox)

**P9: Segmentador Semantico**
- Segmentacion pixel-a-pixel (U-Net)
- Tecnica: Arquitectura U-Net con skip connections
- Metricas: IoU, Dice

### Grupo 6: Series Temporales (1 proyecto)

**P10: Predictor LSTM**
- Predice valores futuros de series temporales
- Tecnica: LSTM apilados
- Explicacion: 4 compuertas LSTM
- Metricas: MAE, RMSE

### Grupo 7: NLP (1 proyecto)

**P11: Clasificador Sentimientos**
- Clasifica sentimientos (positivo/negativo/neutral)
- Tecnica: Embedding + LSTM
- Explicacion: Word embeddings, RNN
- Metricas: Accuracy, F1-Score

### Grupo 8: Generacion (1 proyecto)

**P12: Generador Imagenes (VAE)**
- Genera nuevas imagenes artificiales
- Tecnica: Autoencoder Variacional (VAE)
- Explicacion: KL divergence, reparameterization trick
- Capacidad: Generar imagenes nuevas

## Instalacion

```bash
# Instalar dependencias
pip install tensorflow numpy scikit-learn

# O con archivo requirements
pip install -r requirements.txt
```

## Ejecucion

### Ejecutar un proyecto individual
```bash
# Por ejemplo P0
python p0_regresion_cuadratica.py

# O P11
python p11_nlp_sentimientos.py
```

### Ejecutar todos los proyectos
```bash
python ejecutar_todos.py
```

## Salida Esperada

Cada proyecto imprime:

```
============================================================
P0: PREDICTOR DE PRECIOS DE CASAS
============================================================
[+] Modelo P0 entrenado
RMSE: $12,345
MAE: $8,234
R2: 0.8523
============================================================
```

## Archivos de Configuracion Recomendados

Para la carpeta raiz, crear `requirements.txt`:

```
tensorflow>=2.13.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

## Conceptos Clave Explicados en Codigos

### Regresion
- Ecuacion normal: (X^T X)^-1 X^T y
- Minimos cuadrados
- Normalizacion de datos

### Clasificacion
- Sigmoid: 1/(1+exp(-z))
- Softmax: exp(z)/sum(exp(z))
- Cross-entropy loss
- Gradient descent

### Deep Learning
- Capas convolucionales (Conv2D)
- LSTM y compuertas (input, forget, output)
- Embedding y word2vec
- Autoencoder y VAE

### Metricas
- Accuracy: Porcentaje correcto
- Precision: De positivos predichos, cuantos son reales
- Recall: De positivos reales, cuantos encontre
- F1: Balance precision-recall
- AUC: Area bajo curva ROC

## Modificaciones Comunes

### Cambiar tamaño del dataset
```python
generador = GeneradorDatos()
datos = generador.generar_dataset(n_samples=500)  # Default 200
```

### Cambiar hiperparametros
```python
modelo = ClasificadorSentimientos(
    vocab_size=1000,      # Default 500
    max_len=30,           # Default 20
    embedding_dim=32      # Default 16
)
```

### Agregar validacion cruzada
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(modelo, X, y, cv=5)
print(f"Promedio: {scores.mean():.4f}")
```

## Visualizacion de Resultados

Para agregar plots (requiere matplotlib):

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Prediccion')
plt.legend()
plt.savefig('prediccion.png')
```

## Debugging

Si un proyecto falla:

1. Verificar que TensorFlow esta instalado
2. Revisar version de Python (3.8+)
3. Aumentar verbose: `verbose=1` en model.fit()
4. Revisar memoria si hay OOM error
5. Ejecutar individual vs ejecutar_todos.py

## Proximos Pasos

- Aplicar estos codigos a datos reales
- Combinar multiples proyectos (ej: autoencoder + clustering)
- Agregar deployment (Flask, FastAPI)
- Implementar validacion cruzada completa
- Agregar visualizaciones avanzadas

## Referencia Rapida

| Proyecto | Tecnica Principal | Entrada | Salida | Metrica |
|----------|-----------------|---------|--------|---------|
| P0 | Minimos Cuadrados | Numero | Numero | RMSE |
| P1 | Regresion Lineal | Vector | Numero | MAE |
| P2 | Logistica | Vector | 0/1 | AUC |
| P3 | Red Densa | Vector | 0-2 | F1 |
| P4 | K-Means | Vector | Cluster | Silhouette |
| P5 | Autoencoder | Imagen | Imagen | MSE |
| P6 | CNN | Imagen | 0-9 | Accuracy |
| P7 | Conv1D | Serie | 0-2 | F1 |
| P8 | Detector | Imagen | BBox+Class | mAP |
| P9 | U-Net | Imagen | Mascara | IoU |
| P10 | LSTM | Serie | Numero | RMSE |
| P11 | RNN+Embedding | Texto | 0-2 | Accuracy |
| P12 | VAE | Imagen | Imagen | KL+MSE |

## Contacto y Soporte

Para problemas con los codigos, revisar:
- El guion GUION_COMPLETO_13_PROYECTOS.pdf
- Documentacion de TensorFlow: https://www.tensorflow.org/
- Scikit-learn docs: https://scikit-learn.org/

---

**Generado para el video: TensorFlow - Portafolio Completo de 13 Proyectos**
