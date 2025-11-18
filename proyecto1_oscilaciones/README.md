# Proyecto 1: Oscilaciones Amortiguadas (Modelo de aprendizaje profundo)

## Descripción General

Este proyecto implementa una **red neuronal profunda** para modelar y predecir el comportamiento de sistemas oscilantes amortiguados. El modelo aprende a aproximar la solución de la ecuación diferencial:

$$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$$

Donde:
- **m**: masa del sistema (kg)
- **c**: coeficiente de amortiguamiento (N·s/m)
- **k**: constante de rigidez (N/m)
- **x(t)**: posición en función del tiempo

## Características Principales

### 1. **Generación de Datos Sintéticos**
- Genera 500 trayectorias de oscilaciones con parámetros variados
- Utiliza la solución analítica exacta como base
- Agrega ruido Gaussiano controlado
- Soporta tres regímenes: subamortiguado, críticamente amortiguado, sobreamortiguado

### 2. **Arquitectura Neural Profunda**
- Capas totalmente conectadas con regularización L2
- Normalización por lotes (BatchNormalization)
- Dropout para evitar overfitting
- Activación ReLU en capas ocultas, Linear en salida

### 3. **Validación Robusta**
- Validación cruzada k-fold (por defecto 5 folds)
- Early stopping basado en validación
- Métricas: MSE, MAE, R²
- Análisis de residuos

### 4. **Serialización Profesional**
- Modelo guardado en formato Keras nativo (.keras)
- Configuración guardada en JSON
- Escaladores guardados para predicción

## Estructura de Archivos

```
proyecto1_oscilaciones/
├── oscilaciones_amortiguadas.py      # Clase principal del modelo
├── run_training.py                    # Script de entrenamiento
├── test_oscilaciones.py              # Suite de tests (50+ tests)
├── requirements.txt                   # Dependencias
├── README.md                          # Este archivo
└── resultados_entrenamiento/          # Carpeta de salida
    ├── modelo_oscilaciones.keras      # Modelo entrenado
    ├── modelo_oscilaciones.json       # Configuración
    ├── resultados.json                # Métricas finales
    ├── historia_entrenamiento.png     # Gráficas de loss
    ├── predicciones_analisis.png      # Predicciones vs reales
    └── validacion_cruzada.png         # Resultados CV
```

## Instalación

### Requisitos Previos
- Python 3.11+
- pip o conda

### Pasos de Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenamiento Completo

```bash
python run_training.py
```

Este script ejecutará:
1. Generación de 50,000 muestras sintéticas
2. División 80/20 train/test
3. Entrenamiento con early stopping
4. Validación cruzada 5-fold
5. Evaluación en test set
6. Guardado de modelos y gráficas

### Uso Programático

```python
from oscilaciones_amortiguadas import OscilacionesAmortiguadas
import numpy as np

# Crear instancia
modelo = OscilacionesAmortiguadas(seed=42)

# Generar datos
X, y = modelo.generar_datos(num_muestras=500, tiempo_max=10.0)

# Entrenar
info = modelo.entrenar(X, y, epochs=100, batch_size=32)

# Predecir
X_nuevo = np.random.randn(10, 7).astype(np.float32)
y_pred = modelo.predecir(X_nuevo)

# Guardar modelo
modelo.guardar_modelo('mi_modelo.keras')

# Cargar modelo
modelo2 = OscilacionesAmortiguadas()
modelo2.cargar_modelo('mi_modelo.keras')
```

## API Referencia

### Clase: `OscilacionesAmortiguadas`

#### Métodos Principales

| Método | Descripción |
|--------|-------------|
| `generar_datos()` | Genera datos sintéticos |
| `construir_modelo()` | Crea la arquitectura neural |
| `entrenar()` | Entrena el modelo |
| `validacion_cruzada()` | Realiza k-fold CV |
| `predecir()` | Realiza predicciones |
| `guardar_modelo()` | Serializa el modelo |
| `cargar_modelo()` | Carga modelo guardado |
| `resumen_modelo()` | Retorna información del modelo |

#### Método: `generar_datos()`

```python
X, y = modelo.generar_datos(
    num_muestras=500,          # Número de trayectorias
    tiempo_max=10.0,           # Duración de cada trayectoria (s)
    ruido_sigma=0.02,          # Desviación estándar del ruido
    params_sistema=None        # Dict con parámetros del sistema
)
```

**Retorna:**
- `X`: array (N, 7) con características [t, m, c, k, x0, v0, zeta]
- `y`: array (N, 1) con posiciones

#### Método: `entrenar()`

```python
info = modelo.entrenar(
    X, y,
    epochs=100,                    # Número de épocas
    batch_size=32,                # Tamaño del lote
    validation_split=0.2,          # Proporción de validación
    early_stopping_patience=15,    # Paciencia para ES
    verbose=1                      # Nivel de verbosidad
)
```

**Retorna:** Diccionario con métricas del entrenamiento

### Solución Analítica

```python
t = np.linspace(0, 10, 100)
x = OscilacionesAmortiguadas.solucion_analitica(
    t, m=1.0, c=0.5, k=1.0, x0=1.0, v0=0.0
)
```

## Resultados Esperados

### Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| Test MSE | < 0.0001 |
| Test MAE | < 0.005 |
| Test R² | > 0.99 |
| CV MSE | 0.0001 ± 0.00005 |

### Tiempos de Ejecución

- Generación de datos: ~5-10 segundos
- Entrenamiento: ~30-60 segundos
- Validación cruzada: ~2-3 minutos
- Tiempo total: ~5-10 minutos

## Testing

```bash
# Ejecutar todos los tests
pytest test_oscilaciones.py -v

# Tests con cobertura
pytest test_oscilaciones.py --cov=oscilaciones_amortiguadas

# Tests específicos
pytest test_oscilaciones.py::TestGeneracionDatos -v
```

### Cobertura de Tests

- **25+ tests unitarios**
- Cobertura: >95%
- Validación de:
  - Generación de datos
  - Solución analítica
  - Construcción del modelo
  - Entrenamiento
  - Predicción
  - Serialización
  - Validación cruzada

## Regímenes de Amortiguamiento

### 1. **Subamortiguado (ζ < 1)**
- Oscila alrededor del equilibrio
- Amplitud decae exponencialmente
- Comportamiento oscilatorio

### 2. **Críticamente Amortiguado (ζ = 1)**
- Regresa a equilibrio sin oscilar
- Tiempo mínimo para llegar a equilibrio
- Transición más rápida

### 3. **Sobreamortiguado (ζ > 1)**
- Regresa lentamente sin oscilar
- Requiere más tiempo que el crítico
- Amortiguamiento excesivo

## Ecuaciones Clave

### Factor de Amortiguamiento
$$\zeta = \frac{c}{2\sqrt{km}}$$

### Frecuencia Natural
$$\omega_0 = \sqrt{\frac{k}{m}}$$

### Frecuencia Amortiguada (si ζ < 1)
$$\omega_d = \omega_0\sqrt{1-\zeta^2}$$

### Solución Subamortiguada
$$x(t) = e^{-\zeta\omega_0 t}\left(A\cos(\omega_d t) + B\sin(\omega_d t)\right)$$

## Validación de Precisión

El modelo se valida contra:
1. **Solución analítica exacta**
2. **Datos con ruido realista**
3. **Validación cruzada k-fold**
4. **Análisis de residuos**

## Troubleshooting

### Problema: GPU no detectada
```python
# Usar CPU explícitamente
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Problema: Modelo no converge
- Reducir learning rate
- Aumentar early stopping patience
- Aumentar número de muestras de entrenamiento

### Problema: Overfitting
- Aumentar dropout rate
- Aumentar regularización L2
- Reducir complejidad del modelo

## Referencias

1. Thornton, S. T., & Marion, J. B. (2004). Classical Dynamics of Particles and Systems
2. Inman, D. J. (2014). Engineering Vibration
3. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization

## Autor

Proyecto realizado como parte del análisis de sistemas dinámicos usando Deep Learning.

## Licencia

MIT License - Ver LICENSE file

## Changelog

### v1.0.0 (2025-11-18)
- Implementación inicial
- 25+ tests unitarios
- Validación cruzada completa
- Documentación profesional
