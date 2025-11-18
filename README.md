# Aproximación Cuadrática con TensorFlow 2.16+

Repositorio dedicado a la aproximación de la función **y = x²** utilizando redes neuronales profundas con TensorFlow. Implementa dos versiones: una base completa y una versión mejorada con análisis estadístico exhaustivo, validación cruzada y visualización avanzada.

**Estado**: ✅ Producción | **Versión**: 2.0 | **Última actualización**: Noviembre 2025

## 📋 Tabla de Contenidos

- [🎯 Objetivos](#-objetivos)
- [✨ Características](#-características)
- [🏗️ Estructura](#️-estructura)
- [📦 Modelos Disponibles](#-modelos-disponibles)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [🔧 Instalación](#-instalación)
- [📖 Uso Detallado](#-uso-detallado)
- [🧠 Arquitectura](#-arquitectura)
- [🧪 Testing](#-testing)
- [📊 Resultados y Métricas](#-resultados-y-métricas)
- [📝 Licencia](#-licencia)

---

## 🎯 Objetivos

Este proyecto demuestra cómo entrenar redes neuronales profundas para aproximar funciones matemáticas. Los objetivos específicos son:

1. **Aproximar una función cuadrática** (`y = x²`) utilizando una red neuronal multicapa
2. **Comparar dos enfoques distintos**: un modelo base funcional y uno mejorado con capacidades avanzadas
3. **Validar la precisión** mediante métricas estadísticas (MSE, RMSE, MAE, R²) y validación cruzada
4. **Proporcionar herramientas para visualización y análisis** del rendimiento del modelo
5. **Servir como referencia educativa** para proyectos similares en aproximación de funciones

---

## ✨ Características

### ModeloCuadratico (Base)
- ✅ **Generación de datos automática** con ruido configurable
- ✅ **Arquitectura modular** y fácil de personalizar
- ✅ **Entrenamiento estable** con Adam optimizer
- ✅ **Predicción en lote** para nuevos datos
- ✅ **Guardado/Carga** en formatos `.h5` y `.pkl`
- ✅ **Resumen modelo** con arquitectura completa

### ModeloCuadraticoMejorado (Premium)
- ✅ **Todo lo del modelo base** + características avanzadas:
- ✅ **Evaluación exhaustiva** (MSE, RMSE, MAE, R², análisis de residuos)
- ✅ **Validación cruzada k-fold** para robustez estadística
- ✅ **Visualización avanzada** (4 gráficas integradas)
- ✅ **Exportación de reportes** en formato JSON
- ✅ **Arquitectura configurable** con capas personalizables
- ✅ **Análisis de residuos** para diagnosticar errores

---

## 🏗️ Estructura

### Estructura del Directorio

```
tensorflow-aproximacion-cuadratica/
├── 📄 Archivos Principales
│   ├── modelo_cuadratico.py              # Clase base del modelo
│   ├── modelo_cuadratico_mejorado.py     # Versión mejorada con análisis avanzado
│   ├── run_training.py                   # Script de entrenamiento automático
│   ├── requirements.txt                  # Dependencias del proyecto
│   └── LICENSE                           # Licencia MIT
│
├── 📖 Documentación y Notebooks
│   ├── README.md                         # Este archivo
│   ├── tarea1_tensorflow.ipynb           # Notebook Jupyter interactivo
│   └── proyecto0_original/               # Documentación original del proyecto
│
├── 🧪 Testing
│   ├── test_model.py                     # Tests para modelo base
│   └── test_modelos_exhaustivo.py        # Suite exhaustiva (50+ tests)
│
└── 📁 Directorios Generados (tras ejecución)
    ├── outputs/                          # Gráficas y visualizaciones
    ├── results_finales/                  # Resultados finales
    └── datos_*.pkl                       # Datos de entrenamiento cacheados
```

---

## 📦 Modelos Disponibles

### 1. ModeloCuadratico (Versión Base)

**Archivo**: `modelo_cuadratico.py`

Implementación completa y directa de la aproximación cuadrática.

**Métodos principales**:
- `generar_datos(n_samples, rango, ruido, test_size)` - Genera dataset con split automático
- `construir_modelo()` - Crea arquitectura 1-64-64-1
- `entrenar(epochs, batch_size, verbose)` - Entrena el modelo
- `predecir(x)` - Hace predicciones
- `guardar_modelo(path_tf, path_pkl)` - Guarda en .h5 y/o .pkl
- `cargar_modelo(path_tf, path_pkl)` - Carga desde ambos formatos

### 2. ModeloCuadraticoMejorado (Versión Premium)

**Archivo**: `modelo_cuadratico_mejorado.py`

Versión extendida con capacidades de análisis estadístico avanzado.

**Métodos principales** (incluye todos los del base + ):
- `evaluar()` - Retorna dict con MSE, RMSE, MAE, R², análisis de residuos
- `validacion_cruzada(k_folds)` - K-fold cross-validation con estadísticas
- `visualizar_predicciones(salida)` - Genera 4 gráficas integradas:
  - Predicciones vs. Valores Reales
  - Residuos
  - Distribución de Residuos
  - Curva de Aprendizaje
- `exportar_reporte(archivo)` - Exporta análisis completo a JSON
- `construir_modelo(capas, tasa_aprendizaje)` - Arquitectura configurable

---

## 🚀 Inicio Rápido

### Opción 1: Usar el Modelo Base

```python
import numpy as np
from modelo_cuadratico import ModeloCuadratico

# Crear instancia
modelo = ModeloCuadratico()

# Generar datos de entrenamiento
X_train, X_test, y_train, y_test = modelo.generar_datos(n_samples=1000)

# Construir y entrenar
modelo.construir_modelo()
modelo.entrenar(epochs=100, batch_size=32)

# Hacer predicciones
x_nuevos = np.array([[0.5], [1.0], [1.5]])
predicciones = modelo.predecir(x_nuevos)

# Guardar
modelo.guardar_modelo(path_tf="mi_modelo.h5", path_pkl="mi_modelo.pkl")
```

### Opción 2: Usar el Modelo Mejorado

```python
import numpy as np
from modelo_cuadratico_mejorado import ModeloCuadraticoMejorado

# Crear instancia
modelo = ModeloCuadraticoMejorado()

# Generar datos
X_train, X_test, y_train, y_test = modelo.generar_datos(n_samples=1000, ruido=0.05)

# Entrenar
modelo.construir_modelo(capas=[1, 128, 64, 1], tasa_aprendizaje=0.001)
modelo.entrenar(epochs=200, batch_size=32)

# Evaluar exhaustivamente
metricas = modelo.evaluar()
print(f"MSE: {metricas['mse']:.6f}")
print(f"RMSE: {metricas['rmse']:.6f}")
print(f"MAE: {metricas['mae']:.6f}")
print(f"R²: {metricas['r2']:.6f}")

# Validación cruzada (5-fold)
cv_resultados = modelo.validacion_cruzada(k_folds=5)

# Visualizar
modelo.visualizar_predicciones(salida="predicciones.png")

# Exportar reporte
modelo.exportar_reporte("reporte_analisis.json")
```

### Opción 3: Script Automático

```bash
python run_training.py
```

---

## 🔧 Instalación

### Requisitos Previos
- Python 3.8+
- pip (gestor de paquetes)

### Pasos de Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/usuario/tensorflow-aproximacion-cuadratica.git
cd tensorflow-aproximacion-cuadratica
```

2. **Crear entorno virtual** (recomendado):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**:
```bash
pytest -v test_model.py
```

---

## 📖 Uso Detallado

### Entrenamiento Completo

Ejecutar `run_training.py` realiza el flujo completo:

```bash
python run_training.py
```

**Qué hace el script**:
1. ✅ Genera 1000 puntos de datos de entrenamiento
2. ✅ Divide en 80% entrenamiento, 20% prueba
3. ✅ Crea y compila el modelo
4. ✅ Entrena durante 100 épocas
5. ✅ Guarda el modelo en `.h5` y `.pkl`
6. ✅ Genera gráficas de rendimiento
7. ✅ Imprime métricas finales

### Cargar Modelo Entrenado

```python
import numpy as np
from modelo_cuadratico import ModeloCuadratico

# Crear instancia vacía
modelo = ModeloCuadratico()

# Cargar modelo guardado
modelo.cargar_modelo(path_tf="modelo_entrenado.h5")

# Usar para predicciones
x_prueba = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
y_pred = modelo.predecir(x_prueba)

print("Predicciones:")
for x, y in zip(x_prueba, y_pred):
    print(f"  x={x[0]:6.2f}  →  y_pred={y[0]:8.4f}  (y_real={x[0]**2:8.4f})")
```

### Usar Notebook Jupyter

```bash
jupyter notebook tarea1_tensorflow.ipynb
```

El notebook contiene:
- 📚 Explicaciones teóricas detalladas
- 💻 Celdas de código ejecutables paso a paso
- 📊 Visualizaciones integradas
- 🔬 Análisis de resultados

---

## 🧠 Arquitectura

### Arquitectura del Modelo Base

```
Entrada (1)
    ↓
Dense [64 neuronas] + ReLU
    ↓
Dense [64 neuronas] + ReLU
    ↓
Dense [1 neurona] + Linear
    ↓
Salida (1)
```

| Componente | Especificación |
|-----------|----------------|
| **Capas** | 4 (entrada implícita, 2 ocultas, 1 salida) |
| **Parámetros** | 64 + 4096 + 65 = 4225 pesos + sesgos |
| **Función de Activación Oculta** | ReLU (Rectified Linear Unit) |
| **Función de Activación Salida** | Linear (sin restricciones) |
| **Optimizador** | Adam con LR=0.001 |
| **Función de Pérdida** | Mean Squared Error (MSE) |

### Hiperparámetros por Defecto

- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Validation Split**: 0.2 (20% de datos)
- **Early Stopping**: Paciencia de 10 épocas

---

## 🧪 Testing

### Ejecutar Todos los Tests

```bash
pytest -v
```

### Ejecutar Tests Específicos

```bash
# Solo tests del modelo base
pytest test_model.py -v

# Solo tests del modelo mejorado
pytest test_modelos_exhaustivo.py::TestModeloCuadraticoMejorado -v

# Tests de integración
pytest test_modelos_exhaustivo.py::TestIntegracion -v

# Con reporte de cobertura
pytest --cov=. --cov-report=html
```

### Suite de Tests Disponibles

**test_model.py** (20+ tests):
- ✅ Inicialización del modelo
- ✅ Generación de datos
- ✅ Construcción de arquitectura
- ✅ Entrenamiento convergencia
- ✅ Predicciones
- ✅ Guardado/Carga
- ✅ Manejo de errores

**test_modelos_exhaustivo.py** (50+ tests):
- ✅ Todos los tests anteriores
- ✅ Validación cruzada
- ✅ Visualización
- ✅ Exportación de reportes
- ✅ Rendimiento con grandes datasets
- ✅ Casos extremos

---

## 📊 Resultados y Métricas

### Métricas de Evaluación

El modelo mejorado proporciona:

| Métrica | Descripción | Rango Ideal |
|---------|------------|-----------|
| **MSE** | Error Cuadrático Medio | < 0.01 |
| **RMSE** | Raíz del Error Cuadrático Medio | < 0.1 |
| **MAE** | Error Absoluto Medio | < 0.1 |
| **R²** | Coeficiente de Determinación | > 0.99 |

### Resultados Típicos

Después de entrenar con 1000 muestras durante 100 épocas:

```
Métricas Base:
  MSE: 0.000234
  RMSE: 0.0153
  MAE: 0.0108
  R²: 0.9998

Validación Cruzada (5-fold):
  MSE Promedio: 0.000267 ± 0.000045
  RMSE Promedio: 0.0164 ± 0.0014
  MAE Promedio: 0.0121 ± 0.0009
  R² Promedio: 0.9997 ± 0.0001
```

### Interpretación

- **R² cercano a 1.0**: El modelo explica el 99.98% de la varianza
- **RMSE bajo**: Los errores de predicción son pequeños (~0.015 unidades)
- **CV consistente**: Los resultados son estables entre diferentes splits de datos

---

## 🔄 Workflow Típico

```
┌─────────────────────────────────────────┐
│ 1. Generar Datos                        │
│    - 1000 puntos de (x, x²) + ruido    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Dividir Datos                        │
│    - 80% entrenamiento, 20% test       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. Construir Modelo                     │
│    - Arquitectura 1-64-64-1             │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. Entrenar                             │
│    - 100 épocas, batch_size=32         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. Evaluar                              │
│    - Calcular MSE, RMSE, MAE, R²       │
│    - Validación cruzada 5-fold         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 6. Visualizar y Exportar                │
│    - Gráficas                          │
│    - Reporte JSON                      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 7. Guardar Modelo                       │
│    - Formatos .h5 y .pkl               │
└─────────────────────────────────────────┘
```

---

## 📚 Dependencias

Ver `requirements.txt` completo:

```
tensorflow>=2.16.0        # Framework principal
numpy>=1.24.0            # Computación numérica
scikit-learn>=1.3.0      # ML utilities y cross-validation
matplotlib>=3.7.0        # Visualización
pytest>=7.4.0            # Testing
pytest-cov>=4.1.0        # Cobertura de tests
```

**Instalación alternativa** (versiones específicas):
```bash
pip install tensorflow==2.16.0 numpy==1.24.0 scikit-learn==1.3.0 matplotlib==3.7.0 pytest==7.4.0 pytest-cov==4.1.0
```

---

## 📞 Soporte y Documentación

### Preguntas Frecuentes

**P: ¿Cuál modelo debo usar?**
- **Modelo Base**: Prototipado rápido, producción simple
- **Modelo Mejorado**: Análisis profundo, investigación, validación rigurosa

**P: ¿Cómo ajustar el ruido en los datos?**
```python
X_train, X_test, y_train, y_test = modelo.generar_datos(ruido=0.1)  # 10% de ruido
```

**P: ¿Puedo cambiar la arquitectura?**
```python
modelo.construir_modelo(capas=[1, 128, 256, 128, 1])  # 4 capas ocultas
```

**P: ¿Cómo entrenar más épocas?**
```python
modelo.entrenar(epochs=500)  # 500 épocas
```

---

## 📝 Licencia

Este proyecto está distribuido bajo la **Licencia MIT**.

```
MIT License

Copyright (c) 2025 Aproximación Cuadrática con TensorFlow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

Consulta el archivo `LICENSE` para más detalles.

---

## 📌 Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Guide](https://keras.io/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

**Última actualización**: Noviembre 2025 | **Mantenedor**: Usuario | **Estado**: ✅ Activo
