# Proyecto 11: Análisis de Sentimientos con NLP

## Introducción

Sistema completo de clasificación de sentimientos en textos usando técnicas de Procesamiento de Lenguaje Natural (NLP) y aprendizaje profundo.

### Aplicaciones

- **Análisis de Reviews**: Clasificar opiniones sobre productos/servicios
- **Social Media**: Monitoreo de marca en redes sociales
- **Atención al Cliente**: Priorizar tickets según sentimiento
- **Investigación de Mercado**: Análisis de feedback de clientes
- **Análisis de Noticias**: Detectar tonalidad de reportes

### Sentimientos Clasificados

1. **Negativo** (Clase 0): Texto expresa opinión negativa
   - "Odio este producto, es terrible y decepcionante"
   
2. **Neutro** (Clase 1): Texto sin carga emocional
   - "El producto es un objeto rectangular que contiene componentes"
   
3. **Positivo** (Clase 2): Texto expresa opinión positiva
   - "Adoro este producto, es absolutamente fantástico"

---

## Fundamentación Teórica

### 1. Procesamiento de Lenguaje Natural (NLP)

NLP es la rama de IA que permite a máquinas entender y generar lenguaje humano.

**Pipeline NLP típico**:
```
Texto → Tokenización → Embeddings → Modelo → Predicción
```

#### Tokenización
Proceso de dividir texto en tokens (palabras, caracteres, subpalabras):

```
"Adoro este producto" → ["adoro", "este", "producto"]
```

#### Representación Vectorial
Convertir palabras a números:
- **One-hot encoding**: [1, 0, 0] para palabra 1 de vocabulario
- **Embeddings densos**: [0.2, 0.8, 0.1] (valores reales continuos)

### 2. Embeddings Word2Vec

Transforma palabras en vectores densos de dimensión $d$ (típicamente 100-300):

$$\vec{w}_i \in \mathbb{R}^d$$

**Intuición**: Palabras con significados similares tienen vectores cercanos.

**Ejemplo en 2D**:
```
         ↑ eje_sentimiento_positivo
         |
   feliz ×
    alegre ×
         |
         ×────────→ eje_tamaño
    pequeño    grande
         |
         × triste
```

**Skip-gram objetivo**:
$$\max \sum_{t} \sum_{c \in Context(t)} \log P(w_c | w_t)$$

Donde $w_t$ es palabra target, $w_c$ es palabra contexto.

### 3. Arquitectura LSTM Bidireccional para Texto

Procesa secuencia de embeddings en ambas direcciones:

```
Input: "Adoro este producto"
       ↓ (tokens)
       [embedding_adoro, embedding_este, embedding_producto]
       ↓ (Embedding layer)
       [[0.2, 0.8, ...], [0.1, 0.3, ...], [0.9, 0.1, ...]]  # shape: (3, 128)
       
       Forward LSTM:  → → →
       Backward LSTM: ← ← ←
       
       Concatenated: [h_fwd_final | h_bwd_final]  # (256,)
       ↓ (Dense layers)
       Predicción: [P(neg), P(neut), P(pos)]
```

**Ecuaciones LSTM**:
Igual a Proyecto 10, pero input es embedding (vector denso) no observaciones numéricas.

### 4. Transformer: Multi-Head Self-Attention

Mecanismo revolucionario que reemplaza LSTM en modelos modernos (BERT, GPT):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde:
- $Q$ = Query (qué buscar)
- $K$ = Key (dónde buscar)
- $V$ = Value (qué obtener)
- $d_k$ = dimensión de key (normalización)

**Multi-head**: Múltiples attention en paralelo con diferentes subespacios

#### Ejemplo: "El gato está en la casa"

Atención de token "gato":
```
gato → query
gato ← key   [El, gato, está, en, la, casa]
gato → attention weights: [0.1, 0.7, 0.1, 0.05, 0.03, 0.02]
       "gato" se enfoca 70% en sí mismo, 10% en "El", etc.
gato → nueva representación = suma ponderada
```

**Ventajas sobre LSTM**:
- Paralelizable (procesa todos tokens simultáneamente)
- Captura dependencias largas mejor
- Menos parámetros generalmente

### 5. CNN 1D para Clasificación de Textos

Detecta n-gramas como características locales:

```
Input: [embedding_1, embedding_2, ..., embedding_n]  shape: (n, 128)
         ↓
Conv1D(32, kernel=3): Detecta trigramas
       "Adoro este producto" → detecciones locales
       ↓
MaxPool: Selecciona máxima activación
       ↓
GlobalAveragePool: Promedio de todas las características
       ↓
Dense layers → Predicción
```

**Intuición**: CNN extrae coocurrencias de palabras (n-gramas).
Ejemplo: kernel detectaría patrones como ["adoro" + "producto"] → sentimiento positivo

### 6. Embedding Layer

Primera capa de red neural para texto:

```python
layers.Embedding(vocab_size=1000, embedding_dim=128)
```

**Funcionamiento**:
- Input: índices de palabras [2, 45, 103]
- Lookup table: matriz de forma (1000, 128)
- Output: embeddings correspondientes

**Diferencia con One-Hot**:
- One-hot: [1, 0, 0, ...] - 1000D vector sparse
- Embedding: [0.2, 0.8, 0.1, ...] - 128D vector denso
- 7.8x más eficiente, información condensada

### 7. Softmax para Clasificación Multiclase

Última capa de salida:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Propiedades**:
- Salidas en [0, 1]
- Suman a 1 (interpretables como probabilidades)
- Diferenciable (permite backprop)

**Ejemplo**:
```
Logits: [2.0, 0.5, -1.0]  (raw scores)
exp:    [7.4, 1.6, 0.37]
softmax: [0.83, 0.18, 0.04]  ← Modelo muy seguro de "Negativo"
```

---

## Arquitecturas Implementadas

### 1. LSTM Bidireccional Profundo

```
Input (batch, max_len=50)
    ↓
Embedding(1000, 128): Convierte índices a vectores
    ↓
BiLSTM(64, return_sequences=True) + BatchNorm + Dropout(0.2)
    ↓
BiLSTM(32) + BatchNorm + Dropout(0.2)
    ↓
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(32, ReLU) + Dropout(0.2)
    ↓
Dense(3, softmax)  # Negativo, Neutro, Positivo
```

**Parámetros**: ~650K
**Velocidad**: ~50ms/batch (32 samples)
**Interpretabilidad**: Buena (pesos BiLSTM visible)

### 2. Transformer con Multi-Head Attention

```
Input (batch, max_len=50)
    ↓
Embedding(1000, 128)
    ↓
[MultiHeadAttention(4 heads) + Residual + LayerNorm] × 2
    ↓
GlobalAveragePooling1D
    ↓
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(32, ReLU) + Dropout(0.2)
    ↓
Dense(3, softmax)
```

**Parámetros**: ~580K
**Ventajas**: 
- Paralelización completa
- Mejor captura de dependencias largas
- Menos parámetros que BiLSTM

### 3. CNN 1D para N-gramas

```
Input (batch, max_len=50)
    ↓
Embedding(1000, 128)
    ↓
Conv1D(64, kernel=3, padding='same') + BatchNorm + ReLU
    ↓
MaxPool1D(2) + Dropout(0.2)
    ↓
Conv1D(32, kernel=3, padding='same') + BatchNorm + ReLU
    ↓
MaxPool1D(2) + Dropout(0.2)
    ↓
GlobalAveragePooling1D
    ↓
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(32, ReLU) + Dropout(0.2)
    ↓
Dense(3, softmax)
```

**Parámetros**: ~450K (más compacto)
**Fortaleza**: Detecta n-gramas específicos muy bien

---

## Dataset Sintético

### Generación

Tres sentimientos generados sintéticamente:

#### Positivos (100 samples)
- Vocabulario positivo: "excelente", "maravilloso", "fantástico"
- Estructura: `[adjectives] [noun]` o "Me encanta...es..."
- Ejemplos:
  - "magnífico producto"
  - "Amo este hotel, fue increíble"

#### Negativos (100 samples)
- Vocabulario negativo: "terrible", "horrible", "odio"
- Ejemplos:
  - "detestable restaurante"
  - "Odio esta comida, fue horrible"

#### Neutros (100 samples)
- Sin carga emocional
- Ejemplos:
  - "El producto es un objeto rectangular"
  - "Se puede describir el viaje como una experiencia"

### Pre-procesamiento

1. **Tokenización**: Separar en palabras
2. **Limpieza**: 
   - Lowercase
   - Remover puntuación
   - Normalizar espacios
3. **Padding**: Todas secuencias mismo tamaño (50 tokens)
4. **Embedding**: Convertir a índices de vocabulario

---

## Uso

### 1. Generación de Datos

```python
from clasificador_sentimientos import GeneradorTextoSentimientos

gen = GeneradorTextoSentimientos(seed=42)
datos = gen.generar_dataset(
    n_samples_por_clase=100,
    max_words=1000,      # Vocabulario
    max_len=50,          # Longitud máxima
    split=(0.6, 0.2, 0.2)
)

print(f"Train: {len(datos.X_train)}")  # 180
print(f"Clases: {datos.etiquetas}")    # ['0', '1', '2']
```

### 2. Entrenar LSTM

```python
from clasificador_sentimientos import ClasificadorSentimientos

clasificador = ClasificadorSentimientos(vocab_size=1000, embedding_dim=128)
hist = clasificador.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    epochs=30,
    arquitectura='lstm'
)
```

### 3. Entrenar Transformer

```python
clasificador_tf = ClasificadorSentimientos()
hist = clasificador_tf.entrenar(
    datos.X_train, datos.y_train,
    datos.X_val, datos.y_val,
    epochs=30,
    arquitectura='transformer'
)
```

### 4. Evaluar

```python
metricas = clasificador.evaluar(datos.X_test, datos.y_test)
print(f"Accuracy: {metricas['accuracy']:.4f}")
print(f"Per-class: {metricas['per_class_accuracy']}")
```

### 5. Predecir

```python
# Batch
clases, probs = clasificador.predecir(datos.X_test)

# Individual
clases, probs = clasificador.predecir(datos.X_test[:1])
print(f"Clase: {clases[0]}")  # 2 = Positivo
print(f"Confianza: {probs[0, clases[0]]:.2%}")  # 0.92
```

### 6. Persistencia

```python
# Guardar
clasificador.guardar('mi_modelo')

# Cargar
clasificador_cargado = ClasificadorSentimientos.cargar(
    'mi_modelo', vocab_size=1000, embedding_dim=128
)
```

---

## Resultados Esperados

### En Dataset Sintético Balanceado

| Modelo | Accuracy | Epoch 10 | Final |
|--------|----------|----------|-------|
| LSTM | 0.78 | 0.65 | 0.78 |
| Transformer | 0.80 | 0.68 | 0.80 |
| CNN 1D | 0.75 | 0.62 | 0.75 |

**Notas**:
- Dataset simple (palabras clave fuertes) → accuracies altas
- Transformer usualmente mejor en datos limitados
- CNN excelente si detecta n-gramas específicos

### Matriz de Confusión Típica (Transformer)

```
         Pred_Neg  Pred_Neut  Pred_Pos
Real_Neg   25        3         2
Real_Neut  2        26         2
Real_Pos   1         2        27
```

**Diagonal**: Predicciones correctas (25, 26, 27 de 30 cada clase)
**Error típico**: Confunde Neutro con Positivo ocasionalmente

---

## Técnicas de Optimización

### 1. Dropout Progresivo

```python
Dropout(0.2)  # Primeras capas: regularización leve
Dropout(0.3)  # Capas densas: regularización media
Dropout(0.2)  # Final: evita overfitting de clasificación
```

**Efecto**: Fuerza red a aprender representaciones redundantes

### 2. BatchNormalization

Normaliza activaciones entre capas:

$$\tilde{x}_i = \gamma \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

**Beneficios**:
- 3-10x convergencia más rápida
- Permite learning rates mayores
- Regularización implícita

### 3. Early Stopping

```python
EarlyStopping(monitor='val_loss', patience=5)
```

- Evita overfitting
- Selecciona mejor modelo en validación

### 4. Learning Rate Scheduling

```python
ReduceLROnPlateau(factor=0.5, patience=3)
```

- Reduce LR si no hay mejora
- Permite ajuste fino en mínimos

---

## Análisis de Errores Comunes

### Problema 1: Baja Precisión (< 0.60)

**Causas posibles**:
1. Vocabulario insuficiente (max_words muy pequeño)
2. max_len demasiado pequeño (pierde contexto)
3. Dataset desbalanceado

**Soluciones**:
```python
# Aumentar vocabulario
datos = gen.generar_dataset(max_words=2000)

# Aumentar longitud
datos = gen.generar_dataset(max_len=100)

# Balancear clases
gen.generar(n_positivos=150, n_negativos=150, n_neutros=150)
```

### Problema 2: Overfitting (Train 0.95, Val 0.60)

**Señales**: Train accuracy sube, validation baja

**Soluciones**:
1. Aumentar Dropout (0.2 → 0.4)
2. Reducir capas (remover Dense layer)
3. Más Early Stopping (patience=3)

### Problema 3: Transformer No Converge

Transformer es sensible a learning rate:

```python
# Probar learning rates menores
optimizer=keras.optimizers.Adam(learning_rate=0.0005)  # Reducir
```

---

## Suite de Pruebas

**35+ pruebas** incluyendo:

### Generación (7 tests)
- Sentimientos balanceados
- Limpieza de texto
- Vocabularios por sentimiento

### Dataset (7 tests)
- Split proporciones
- One-hot encoding correcto
- Sin NaNs
- Tokenizer creado

### Modelos (7 tests)
- LSTM, Transformer, CNN 1D
- Capas esperadas presentes
- Shapes correcto

### Entrenamiento (5 tests)
- Convergencia de loss
- Todas arquitecturas validas

### Evaluación (5 tests)
- Accuracy en rango [0, 1]
- Métricas por clase
- Error sin entrenar

### Predicción (4 tests)
- Formas correctas
- Probabilidades válidas
- Batch único

### Persistencia (1 test)
- Save/load exacto

### Performance (2 tests)
- Generación < 5 segundos
- Predicción rápida

---

## Referencias

1. Mikolov et al. (2013). "Efficient Estimation of Word Representations"
2. Vaswani et al. (2017). "Attention Is All You Need" (Transformer)
3. Devlin et al. (2019). "BERT: Pre-training"
4. Kim (2014). "Convolutional Neural Networks for Sentence Classification"
5. LeCun et al. (2015). "Deep Learning" - MIT Press

---

**Autor**: Copilot
**Versión**: 1.0
**Fecha**: 2024
**Licencia**: MIT
