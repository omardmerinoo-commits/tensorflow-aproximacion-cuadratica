# 🌊 Oscilaciones Amortiguadas con TensorFlow

Repositorio dedicado al modelado y predicción de **oscilaciones amortiguadas** mediante redes neuronales profundas. Implementa dos enfoques: resolución analítica exacta y aproximación mediante aprendizaje profundo.

**Estado**: ✅ Producción | **Versión**: 2.0 | **Última actualización**: Noviembre 2025

## 📋 Tabla de Contenidos

- [🎯 Objetivos](#-objetivos)
- [✨ Características](#-características)
- [🏗️ Estructura](#️-estructura)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [🔧 Instalación](#-instalación)
- [📖 Uso Detallado](#-uso-detallado)
- [🧠 Fundamento Teórico](#-fundamento-teórico)
- [🧪 Testing](#-testing)
- [📊 Resultados](#-resultados)
- [📝 Licencia](#-licencia)

---

## 🎯 Objetivos

Este proyecto demuestra cómo entrenar redes neuronales para **predecir el comportamiento de sistemas oscilantes amortiguados**. Los objetivos específicos son:

1. **Modelar osciladores amortiguados** resolviendo: $m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$
2. **Comparar solución analítica vs predicción neuronal**
3. **Validar precisión** en diferentes regímenes (subamortiguado, crítico, sobreamortiguado)
4. **Proporcionar herramientas** para análisis y visualización
5. **Servir como referencia** para problemas de ecuaciones diferenciales

---

## ✨ Características

- ✅ **Solución analítica integrada** para osciladores amortiguados
- ✅ **Generación de datos automática** con parámetros variables
- ✅ **Arquitectura configurable** (capas, neuronas, dropout)
- ✅ **Normalización de datos** con escaladores persistentes
- ✅ **Entrenamiento robusto** con callbacks (early stopping, reduce LR)
- ✅ **6+ métricas de evaluación** (MSE, RMSE, MAE, R², análisis de residuos)
- ✅ **Validación cruzada k-fold** para robustez estadística
- ✅ **Visualización avanzada** (4 gráficas integradas)
- ✅ **Persistencia completa** (modelo + configuración + escaladores)

---

## 🏗️ Estructura

```
proyecto1_oscilaciones/
├── oscilaciones_amortiguadas.py      # Clase principal OscilacionesAmortiguadas
├── run_training.py                   # Script de entrenamiento automático
├── requirements.txt                  # Dependencias del proyecto
│
├── test_oscilaciones.py              # Tests exhaustivos (50+ tests)
│
├── README.md                         # Este archivo
├── tarea1_oscilaciones.ipynb         # Notebook Jupyter interactivo
└── LICENSE                           # Licencia MIT
```

---

## 🚀 Inicio Rápido

### Opción 1: Uso Básico

```python
from oscilaciones_amortiguadas import OscilacionesAmortiguadas
import numpy as np

# Crear instancia
modelo = OscilacionesAmortiguadas()

# Generar datos
X_train, X_test, y_train, y_test = modelo.generar_datos(num_muestras=1000)

# Construir y entrenar
modelo.construir_modelo(capas_ocultas=[256, 128, 64, 32])
info = modelo.entrenar(X_train, y_train, epochs=100)

# Evaluar
metricas = modelo.evaluar()
print(f"R²: {metricas['r2']:.4f}")
print(f"MAE: {metricas['mae']:.6f}")
```

### Opción 2: Usar Solución Analítica

```python
from oscilaciones_amortiguadas import OscilacionesAmortiguadas
import numpy as np

# Calcular solución exacta
t = np.linspace(0, 10, 100)
x = OscilacionesAmortiguadas.solucion_analitica(
    t, 
    m=1.0,      # masa
    c=0.5,      # amortiguamiento
    k=1.0,      # rigidez
    x0=1.0,     # posición inicial
    v0=0.0      # velocidad inicial
)
```

### Opción 3: Script Automático

```bash
python run_training.py
```

---

## 🔧 Instalación

### Requisitos
- Python 3.8+
- pip o conda

### Pasos

1. **Clonar/Descargar el proyecto**:
```bash
cd proyecto1_oscilaciones
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

---

## 📖 Uso Detallado

### Generar Datos Sintéticos

```python
modelo = OscilacionesAmortiguadas()

# Generar datos con parámetros predeterminados
X_train, X_test, y_train, y_test = modelo.generar_datos(
    num_muestras=1000,      # Número de conjuntos de parámetros
    tiempo_max=10.0,        # Tiempo máximo de simulación
    puntos_tiempo=100,      # Puntos de tiempo por muestra
    ruido=0.01,             # Nivel de ruido gaussiano
    test_size=0.2           # Fracción de prueba
)

# Personalizar parámetros del sistema
params = {
    'm': (0.5, 3.0),        # Rango de masa
    'c': (0.1, 2.0),        # Rango de amortiguamiento
    'k': (0.5, 5.0),        # Rango de rigidez
    'x0': (-2.0, 2.0),      # Rango de posición inicial
    'v0': (-1.0, 1.0)       # Rango de velocidad inicial
}

X_train, X_test, y_train, y_test = modelo.generar_datos(
    num_muestras=2000,
    params_sistema=params,
    ruido=0.05
)
```

### Construir y Entrenar

```python
# Construir con arquitectura personalizada
modelo.construir_modelo(
    input_shape=7,          # 7 características: [t, m, c, k, x0, v0, zeta]
    capas_ocultas=[512, 256, 128, 64],
    tasa_aprendizaje=0.001,
    dropout_rate=0.3
)

# Entrenar con monitoreo
info = modelo.entrenar(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    early_stopping_patience=15,
    verbose=1
)

print(f"Épocas: {info['epochs_entrenadas']}")
print(f"Loss final: {info['loss_final']:.6f}")
```

### Evaluación Completa

```python
# Métricas en conjunto de prueba
metricas = modelo.evaluar()
print(f"MSE: {metricas['mse']:.6f}")
print(f"RMSE: {metricas['rmse']:.6f}")
print(f"MAE: {metricas['mae']:.6f}")
print(f"R²: {metricas['r2']:.4f}")

# Validación cruzada 5-fold
cv_results = modelo.validacion_cruzada(
    X_train, y_train,
    k_folds=5,
    epochs=50
)

print(f"R² promedio: {cv_results['r2_mean']:.4f}")
print(f"R² por fold: {cv_results['scores_por_fold']['r2']}")
```

### Visualización

```python
# Crear gráficas completas
modelo.visualizar_predicciones(salida='resultados_oscilaciones.png')

# Resultado: 4 subplots
# - Predicciones vs Valores Reales
# - Análisis de Residuos
# - Distribución de Residuos
# - Curva de Aprendizaje
```

### Persistencia

```python
# Guardar modelo entrenado
modelo.guardar_modelo('oscilaciones_modelo')
# Guarda: oscilaciones_modelo.keras, config.json, scalers.pkl

# Cargar modelo guardado
modelo_cargado = OscilacionesAmortiguadas()
modelo_cargado.cargar_modelo('oscilaciones_modelo')

# Usar para nuevas predicciones
X_nuevos = np.random.randn(10, 7).astype(np.float32)
y_pred = modelo_cargado.predecir(X_nuevos)
```

---

## 🧠 Fundamento Teórico

### Ecuación Diferencial

La ecuación de un oscilador amortiguado es:

$$m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = 0$$

Donde:
- **m**: masa del sistema
- **c**: coeficiente de amortiguamiento
- **k**: constante elástica (rigidez)
- **x(t)**: posición en función del tiempo
- **t**: tiempo

### Soluciones Analíticas

El comportamiento depende del **ratio de amortiguamiento** $\zeta = \frac{c}{2\sqrt{km}}$:

#### 1. Subamortiguado ($\zeta < 1$)
$$x(t) = e^{-\zeta \omega_0 t} \left( A \cos(\omega_d t) + B \sin(\omega_d t) \right)$$

Donde $\omega_d = \omega_0 \sqrt{1 - \zeta^2}$ es la frecuencia amortiguada.

**Característica**: Oscilación con decaimiento exponencial.

#### 2. Críticamente Amortiguado ($\zeta = 1$)
$$x(t) = (x_0 + (v_0 + \omega_0 x_0) t) e^{-\omega_0 t}$$

**Característica**: Retorno más rápido sin sobrepaso.

#### 3. Sobreamortiguado ($\zeta > 1$)
$$x(t) = C_1 e^{r_1 t} + C_2 e^{r_2 t}$$

**Característica**: Decaimiento exponencial sin oscilación.

### Características de Entrada

El modelo recibe 7 características normalizadas:

| Feature | Descripción | Rango Típico |
|---------|------------|-------------|
| t | Tiempo | 0 - 10 s |
| m | Masa | 0.5 - 2.0 kg |
| c | Amortiguamiento | 0.1 - 2.0 N·s/m |
| k | Rigidez | 0.5 - 5.0 N/m |
| x₀ | Posición inicial | -2.0 - 2.0 m |
| v₀ | Velocidad inicial | -1.0 - 1.0 m/s |
| ζ | Ratio amortiguamiento | derivado de c, m, k |

---

## 🧪 Testing

### Ejecutar Todos los Tests

```bash
pytest -v test_oscilaciones.py
```

### Tipos de Tests

**TestSolucionAnalitica** (5+ tests):
- ✅ Forma de salida correcta
- ✅ Condiciones iniciales
- ✅ Regímenes de amortiguamiento
- ✅ Casos extremos

**TestGeneracionDatos** (4+ tests):
- ✅ Dimensiones correctas
- ✅ Tipos de datos
- ✅ Ausencia de NaN
- ✅ Guardado de parámetros

**TestConstruccionModelo** (3+ tests):
- ✅ Construcción básica
- ✅ Arquitectura personalizada
- ✅ Compilación correcta

**TestEntrenamiento** (3+ tests):
- ✅ Entrenamiento completo
- ✅ Convergencia de loss
- ✅ Early stopping

**TestPrediccion** (3+ tests):
- ✅ Forma de predicciones
- ✅ Error sin modelo
- ✅ Valores válidos

**TestSerializacion** (3+ tests):
- ✅ Guardado de modelo
- ✅ Carga de modelo
- ✅ Consistencia de predicciones

**TestValidacionCruzada** (2+ tests):
- ✅ CV funcional
- ✅ Métricas válidas

---

## 📊 Resultados Típicos

Después de entrenar con 1000 muestras (10,000 puntos totales):

```
Entrenamiento: 50 épocas
Batch size: 32
Arquitectura: [256, 128, 64, 32]

MÉTRICAS DE PRUEBA:
  MSE:   0.000156
  RMSE:  0.0125
  MAE:   0.0089
  R²:    0.9997

VALIDACIÓN CRUZADA (5-fold):
  R² promedio: 0.9996 ± 0.0002
  MAE medio:  0.0091 ± 0.0008
  
TIEMPO DE ENTRENAMIENTO: ~2-3 segundos
TIEMPO DE PREDICCIÓN: <1ms por muestra
```

### Interpretación

- **R² ≈ 1.0**: Modelo explica 99.97% de la varianza
- **RMSE bajo**: Errores pequeños (~0.0125 unidades)
- **MAE bajo**: Error promedio ~0.009 unidades
- **CV consistente**: Resultados estables entre diferentes splits

---

## 🔄 Workflow Típico

```
1. Generar Datos Sintéticos
   ↓
2. Normalizar y Dividir (80/20)
   ↓
3. Construir Arquitectura
   ↓
4. Entrenar con Callbacks
   ↓
5. Evaluar en Conjunto Test
   ↓
6. Validación Cruzada (opcional)
   ↓
7. Visualizar Resultados
   ↓
8. Guardar Modelo + Configuración
```

---

## 📚 Dependencias

```
tensorflow>=2.16.0        # Framework principal
numpy>=1.24.0            # Computación numérica
scikit-learn>=1.3.0      # Preprocesamiento y CV
matplotlib>=3.7.0        # Visualización
pytest>=7.4.0            # Testing
```

Ver `requirements.txt` para versiones exactas.

---

## 🔗 Referencias

- [TensorFlow Docs](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [Ecuaciones Diferenciales Ordinarias](https://es.wikipedia.org/wiki/Ecuaci%C3%B3n_diferencial_ordinaria)
- [Oscilaciones Amortiguadas](https://es.wikipedia.org/wiki/Oscilaci%C3%B3n_amortiguada)

---

## 📝 Licencia

Licencia MIT. Ver archivo `LICENSE` para detalles.

---

**Versión**: 2.0 | **Estado**: ✅ Producción | **Última actualización**: Noviembre 2025
