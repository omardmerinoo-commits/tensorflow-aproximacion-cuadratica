# Proyecto: Aproximación de la Función y = x² con Red Neuronal

Este repositorio contiene una implementación completa en Python y TensorFlow para entrenar una red neuronal que aprende a aproximar la función cuadrática `y = x²`. El proyecto está diseñado para ser modular, reproducible y fácil de usar, sirviendo como un ejemplo práctico del flujo de trabajo en aprendizaje automático.

![Gráfica de Predicciones](prediccion_vs_real.png)

---

## 📜 Tabla de Contenidos

1.  [🚀 Características Principales](#-características-principales)
2.  [📂 Estructura del Proyecto](#-estructura-del-proyecto)
3.  [⚙️ Instalación](#️-instalación)
4.  [▶️ Cómo Ejecutar](#️-cómo-ejecutar)
    *   [Entrenamiento del Modelo](#entrenamiento-del-modelo)
    *   [Uso del Notebook Interactivo](#uso-del-notebook-interactivo)
    *   [Ejecución de Pruebas](#ejecución-de-pruebas)
5.  [🧠 Arquitectura del Modelo](#-arquitectura-del-modelo)
6.  [💾 Uso del Modelo Guardado](#-uso-del-modelo-guardado)
    *   [Cargar desde formato TensorFlow (.h5)](#cargar-desde-formato-tensorflow-h5)
    *   [Cargar desde formato Pickle (.pkl)](#cargar-desde-formato-pickle-pkl)
7.  [📄 Licencia](#-licencia)

---

## 🚀 Características Principales

*   **Clase Modular `ModeloCuadratico`**: Encapsula toda la lógica del modelo (generación de datos, construcción, entrenamiento, predicción, guardado y carga).
*   **Reproducibilidad**: Uso de semillas fijas para garantizar que los resultados sean consistentes entre ejecuciones.
*   **Script de Entrenamiento**: `run_training.py` automatiza todo el proceso, desde la generación de datos hasta el guardado del modelo y las gráficas.
*   **Notebook Interactivo**: `tarea1_tensorflow.ipynb` ofrece una guía paso a paso con explicaciones detalladas y celdas de código ejecutables.
*   **Visualizaciones Claras**: Genera gráficas para comparar las predicciones con los valores reales y para analizar las curvas de aprendizaje (pérdida y MAE).
*   **Doble Formato de Guardado**: El modelo se guarda tanto en el formato nativo de Keras (`.h5`) como en formato `pickle` (`.pkl`) para máxima compatibilidad.
*   **Pruebas Automatizadas**: Incluye una suite de tests con `pytest` para verificar el correcto funcionamiento de cada componente.

---

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera para mantener el código limpio y modular:

```
. (raíz del proyecto)
├── modelo_cuadratico.py      # Clase principal ModeloCuadratico
├── run_training.py           # Script para ejecutar el entrenamiento completo
├── tarea1_tensorflow.ipynb   # Notebook Jupyter con explicación paso a paso
├── test_model.py             # Pruebas automatizadas con pytest
|
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Archivos y directorios a ignorar por Git
├── LICENSE                   # Licencia MIT del proyecto
├── README.md                 # Este archivo
|
└── (Archivos generados tras la ejecución)
    ├── modelo_entrenado.h5       # Modelo guardado en formato TensorFlow
    ├── modelo_entrenado.pkl      # Modelo guardado en formato pickle
    ├── prediccion_vs_real.png    # Gráfica de predicciones vs. valores reales
    └── loss_vs_epochs.png        # Gráfica de curvas de aprendizaje
```

---

## ⚙️ Instalación

Para configurar el entorno y ejecutar este proyecto, sigue estos pasos. Se recomienda usar un entorno virtual para evitar conflictos de dependencias.

1.  **Clonar el repositorio (si aplica)**:
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd <NOMBRE-DEL-REPOSITORIO>
    ```

2.  **Crear y activar un entorno virtual**:
    ```bash
    # Crear el entorno
    python -m venv venv

    # Activar en Windows
    .\venv\Scripts\activate

    # Activar en macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar las dependencias**:
    El archivo `requirements.txt` contiene todas las librerías necesarias. Instálalas con pip:
    ```bash
    pip install -r requirements.txt
    ```

¡Y eso es todo! El entorno está listo para usar.

---

## ▶️ Cómo Ejecutar

### Entrenamiento del Modelo

Para entrenar el modelo desde cero, simplemente ejecuta el script `run_training.py` desde tu terminal. Este script se encargará de todo el proceso:

```bash
python run_training.py
```

El script realizará las siguientes acciones:
1.  Generará 1000 puntos de datos para la función `y = x²` con ruido.
2.  Dividirá los datos en conjuntos de entrenamiento (80%) y prueba (20%).
3.  Construirá el modelo de red neuronal.
4.  Entrenará el modelo usando el 20% de los datos de entrenamiento para validación.
5.  Guardará el modelo entrenado en `modelo_entrenado.h5` y `modelo_entrenado.pkl`.
6.  Generará las gráficas `prediccion_vs_real.png` y `loss_vs_epochs.png`.

### Uso del Notebook Interactivo

Si prefieres una experiencia más guiada y visual, puedes usar el notebook de Jupyter.

1.  **Iniciar Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  **Abrir el notebook**:
    En la interfaz de Jupyter que se abrirá en tu navegador, haz clic en `tarea1_tensorflow.ipynb`.

3.  **Ejecutar las celdas**:
    Puedes ejecutar cada celda en orden para seguir el proceso de creación, entrenamiento y evaluación del modelo, con explicaciones detalladas en cada paso.

### Ejecución de Pruebas

Para verificar que todo funciona como se espera, puedes ejecutar la suite de pruebas automatizadas con `pytest`:

```bash
pytest -v
```

Esto ejecutará todos los tests definidos en `test_model.py`, asegurando que la generación de datos, la construcción del modelo, el entrenamiento, la predicción y el guardado/carga funcionan correctamente.

---

## 🧠 Arquitectura del Modelo

La red neuronal utilizada es un modelo secuencial simple pero efectivo para esta tarea de regresión, implementado con `tf.keras`.

| Capa             | Neuronas | Activación | Propósito                                               |
| ---------------- | :------: | :--------: | ------------------------------------------------------- |
| **Entrada**      |    1     |    N/A     | Recibe el valor de `x`                                  |
| **Oculta 1**     |    64    |   `relu`   | Aprende características no lineales complejas           |
| **Oculta 2**     |    64    |   `relu`   | Refina las características aprendidas por la capa anterior |
| **Salida**       |    1     |  `linear`  | Produce la predicción final de `y` (sin restricciones)    |

*   **Optimizador**: `Adam` (con una tasa de aprendizaje de 0.001).
*   **Función de Pérdida**: `Mean Squared Error (MSE)`, ideal para tareas de regresión.

---

## 💾 Uso del Modelo Guardado

Una vez entrenado, el modelo puede ser cargado y utilizado para hacer nuevas predicciones sin necesidad de reentrenar. A continuación se muestran ejemplos de cómo cargarlo desde ambos formatos.

### Cargar desde formato TensorFlow (.h5)

Este es el método preferido, ya que el formato `.h5` es nativo de Keras y guarda la arquitectura completa, los pesos y la configuración del optimizador.

```python
import numpy as np
from modelo_cuadratico import ModeloCuadratico

# 1. Crear una instancia de la clase
modelo_cargado = ModeloCuadratico()

# 2. Cargar el modelo desde el archivo .h5
modelo_cargado.cargar_modelo(path_tf="modelo_entrenado.h5")

# 3. Realizar nuevas predicciones
x_nuevos = np.array([[0.25], [0.5], [0.75]])
predicciones = modelo_cargado.predecir(x_nuevos)

print("Predicciones:")
for x, y_pred in zip(x_nuevos, predicciones):
    print(f"  x = {x[0]:.2f} -> y_pred = {y_pred[0]:.4f}")
```

### Cargar desde formato Pickle (.pkl)

El formato `pickle` serializa el objeto completo del modelo. Es útil para interoperabilidad, aunque puede ser menos portable entre diferentes versiones de librerías.

```python
import numpy as np
from modelo_cuadratico import ModeloCuadratico

# 1. Crear una instancia de la clase
modelo_cargado = ModeloCuadratico()

# 2. Cargar el modelo desde el archivo .pkl
modelo_cargado.cargar_modelo(path_pkl="modelo_entrenado.pkl")

# 3. Realizar nuevas predicciones
x_nuevos = np.array([[-1.0], [0.0], [1.0]])
predicciones = modelo_cargado.predecir(x_nuevos)

print("Predicciones:")
for x, y_pred in zip(x_nuevos, predicciones):
    print(f"  x = {x[0]:.2f} -> y_pred = {y_pred[0]:.4f}")
```

---

## 📄 Licencia

Este proyecto está distribuido bajo la **Licencia MIT**. Consulta el archivo `LICENSE` para más detalles.
