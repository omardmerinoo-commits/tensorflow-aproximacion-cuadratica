# Proyecto 6: Aproximador de Funciones No-Lineales
## Aproximación Universal de Funciones Matemáticas con Redes Neuronales

---

## 🎯 Introducción

El **Aproximador de Funciones No-Lineales** demuestra el **Teorema de Aproximación Universal** de redes neuronales.
Entrena modelos para aprender automáticamente sin(x), cos(x), exp(x), x³, x⁵ y combinaciones.

**Características**:
- Normalización avanzada (StandardScaler + MinMaxScaler)
- Regularización (L1, L2, dropout)
- Batch normalization
- Learning rate scheduling
- Arquitecturas múltiples (MLP, Residual)
- >90% de precisión en aproximación

---

## 📚 Teoría Matemática

### Teorema de Aproximación Universal (UAT)

Toda función continua acotada $f: [a,b] \rightarrow \mathbb{R}$ puede ser aproximada uniformemente
por una red neuronal artificial con una capa oculta:

$$\forall \epsilon > 0, \exists N \in \mathbb{N}, \mathbf{w}, \mathbf{b} : \|\mathbf{f}(\mathbf{x}) - \hat{\mathbf{f}}(\mathbf{x})\|_\infty < \epsilon$$

Donde $\hat{\mathbf{f}}$ es la salida de la red con $N$ unidades y pesos $\mathbf{w}$, sesgos $\mathbf{b}$.

### Funciones Aproximadas

#### 1. Función Sinusoidal: $f(x) = \sin(x)$
- **Dominio**: $[-2\pi, 2\pi]$
- **Rango**: $[-1, 1]$
- **Características**: Periódica, suave, oscilante
- **Desafío**: Capturar periodicidad

$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

#### 2. Función Exponencial: $f(x) = e^x$
- **Dominio**: $[-2, 2]$
- **Rango**: $[e^{-2}, e^2] \approx [0.135, 7.389]$
- **Características**: Crecimiento exponencial, asimétrica
- **Desafío**: Dinámica rápida en rango positivo

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$$

#### 3. Polinomios: $f(x) = x^n$
- **$x^3$**: Dominio $[-2, 2]$, rango $[-8, 8]$
- **$x^5$**: Dominio $[-1.5, 1.5]$, rango $[-7.59, 7.59]$
- **Características**: No acotados, monotónicos fuera origen
- **Desafío**: Capturar comportamiento no-lineal

---

## 🛠️ Tecnologías

| Componente | Versión |
|------------|---------|
| TensorFlow | 2.16.0+ |
| scikit-learn | 1.3.0+ |
| NumPy | 1.24.0+ |

---

## 📦 Instalación

```bash
cd proyecto6_funciones
pip install -r requirements.txt
```

---

## 🏗️ Arquitecturas

### MLP Estándar

```
Entrada (1 característica)
        ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
        ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
        ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.3)
        ↓
Dense(1, Linear)
```

**Parámetros**: ~12K
**Ventajas**: Simple, rápido, convergencia estable

### Red Residual

```
Entrada (1)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2) ──┐
    ↓                                           │
    └───────────── Suma (Skip Connection) ←────┘
    ↓
Dense(1, Linear)
```

**Ventajas**: Mejor para redes profundas, evita vanishing gradients

---

## 📖 Guía de Uso

### Generar Datos

```python
from aproximador_funciones import GeneradorFuncionesNoLineales

generador = GeneradorFuncionesNoLineales()

# sin(x)
datos_sin = generador.generar('sin', n_muestras=500, ruido=0.05)

# exp(x)
datos_exp = generador.generar('exp', n_muestras=500)

# x³
datos_x3 = generador.generar('x3', n_muestras=500)
```

### Entrenar Modelo

```python
from aproximador_funciones import AproximadorFuncion

aprox = AproximadorFuncion()

historial = aprox.entrenar(
    datos_sin.X_train, datos_sin.y_train,
    datos_sin.X_test, datos_sin.y_test,
    epochs=100,
    arquitectura='mlp'  # o 'residual'
)
```

### Evaluar

```python
metricas = aprox.evaluar(datos_sin.X_test, datos_sin.y_test)
print(f"R²: {metricas['r2_score']:.4f}")
print(f"RMSE: {metricas['rmse']:.6f}")
```

### Predecir

```python
import numpy as np

X_nuevo = np.array([0.5, 1.0, 1.5]).reshape(-1, 1)
y_pred = aprox.predecir(X_nuevo)
```

---

## 🧪 Suite de Pruebas

```bash
pytest test_aproximador_funciones.py -v --cov
```

**70+ tests** cubriendo:
- Generación de 6 funciones diferentes
- Normalización (entrada/salida)
- Arquitecturas (MLP, Residual)
- Regularización (L1, L2)
- Entrenamiento y convergencia
- Evaluación con 5 métricas
- Predicción
- Persistencia

---

## 📊 Resultados Esperados

### Precisión por Función

| Función | R² | RMSE |
|---------|-----|------|
| sin(x) | >0.95 | <0.05 |
| cos(x) | >0.95 | <0.05 |
| exp(x) | >0.98 | <0.02 |
| x³ | >0.99 | <0.01 |
| x⁵ | >0.99 | <0.01 |
| sin·cos | >0.90 | <0.10 |

### Convergencia

- **Épocas típicas**: 30-50
- **Tiempo de entrenamiento**: 5-10 segundos
- **Tiempo de predicción**: <1ms/muestra

---

## 🎓 Conclusión

Este proyecto valida matemáticamente la capacidad de redes neuronales para aproximar
cualquier función continua, fundamental en deep learning moderno.

**Impacto**: Base teórica para regression, forecasting, y aproximación universal.

---

**Status**: ✅ Production Ready | **Tests**: >90% | **Docs**: Completa
