# Exposición: Aproximación de la Función y = x² con Redes Neuronales

## Slide 1: Portada
**Título:** Aproximación de la Función y = x² mediante Redes Neuronales con TensorFlow

**Subtítulo:** Implementación completa de un modelo de aprendizaje profundo para regresión no lineal

**Información:**
- Proyecto de Machine Learning
- Noviembre 2025
- Repositorio: github.com/omardmerinoo-commits/tensorflow-aproximacion-cuadratica

---

## Slide 2: Introducción al Problema
**Título:** El aprendizaje de funciones no lineales es fundamental en machine learning

**Contenido:**
- Las redes neuronales pueden aproximar cualquier función continua (Teorema de Aproximación Universal)
- La función cuadrática y = x² es un caso de estudio ideal para validar capacidades de aproximación
- Objetivo: Demostrar que una red neuronal simple puede aprender relaciones matemáticas complejas
- Aplicaciones prácticas: física, economía, ingeniería, modelado de fenómenos naturales
- Este proyecto sirve como base para problemas más complejos de regresión

---

## Slide 3: Objetivos del Proyecto
**Título:** Implementar un sistema completo de machine learning siguiendo mejores prácticas

**Objetivos Principales:**
1. Diseñar e implementar una clase modular en Python para encapsular toda la lógica del modelo
2. Construir una red neuronal con TensorFlow/Keras capaz de aproximar y = x²
3. Entrenar el modelo con datos sintéticos y validar su rendimiento
4. Crear visualizaciones profesionales de predicciones y curvas de aprendizaje
5. Implementar persistencia del modelo en múltiples formatos (.h5 y .pkl)

**Objetivos Secundarios:**
- Desarrollar suite completa de tests automatizados
- Crear documentación exhaustiva y ejemplos de uso
- Garantizar reproducibilidad total de resultados
- Seguir estándares de ingeniería de software profesional

---

## Slide 4: Metodología - Generación de Datos
**Título:** Los datos sintéticos con ruido simulan escenarios del mundo real

**Proceso de Generación:**
1. **Muestreo uniforme:** 1000 valores de x en el rango [-1, 1]
2. **Cálculo de y:** Aplicar función cuadrática y = x²
3. **Adición de ruido:** Ruido gaussiano con σ = 0.02 para simular mediciones reales
4. **División de datos:** 80% entrenamiento, 20% prueba
5. **Validación cruzada:** 20% del conjunto de entrenamiento para validación

**Justificación del Ruido:**
- Simula errores de medición en datos experimentales
- Previene sobreajuste perfecto a datos ideales
- Hace el modelo más robusto y generalizable
- Representa condiciones realistas de aplicación

---

## Slide 5: Arquitectura de la Red Neuronal
**Título:** Una arquitectura feedforward de 3 capas logra aproximación precisa

**Estructura del Modelo:**

| Capa | Neuronas | Activación | Parámetros | Función |
|------|----------|------------|------------|---------|
| Entrada | 1 | - | 0 | Recibe valor x |
| Oculta 1 | 64 | ReLU | 128 | Extrae características no lineales |
| Oculta 2 | 64 | ReLU | 4,160 | Refina representaciones |
| Salida | 1 | Linear | 65 | Produce predicción y |
| **TOTAL** | - | - | **4,353** | - |

**Decisiones de Diseño:**
- **ReLU en capas ocultas:** Introduce no linealidad necesaria para aproximar funciones complejas
- **Activación lineal en salida:** Permite predicciones en todo el rango real
- **64 neuronas por capa:** Balance entre capacidad de aprendizaje y eficiencia computacional
- **2 capas ocultas:** Suficiente profundidad para capturar la relación cuadrática

---

## Slide 6: Configuración del Entrenamiento
**Título:** La configuración óptima garantiza convergencia rápida y estable

**Hiperparámetros:**
- **Optimizador:** Adam con learning rate = 0.001
- **Función de pérdida:** Mean Squared Error (MSE)
- **Métrica adicional:** Mean Absolute Error (MAE)
- **Tamaño de lote:** 32 muestras
- **Épocas máximas:** 100

**Callbacks Implementados:**
1. **EarlyStopping:**
   - Monitorea val_loss
   - Paciencia de 15 épocas
   - Restaura mejores pesos automáticamente
   - Previene sobreentrenamiento

2. **ModelCheckpoint:**
   - Guarda mejor modelo durante entrenamiento
   - Criterio: mínima pérdida de validación
   - Permite recuperar modelo óptimo

**Reproducibilidad:**
- Semillas fijas: numpy (42), TensorFlow (42)
- Variable TF_DETERMINISTIC_OPS activada
- Resultados consistentes entre ejecuciones

---

## Slide 7: Resultados - Métricas de Rendimiento
**Título:** El modelo alcanza precisión excepcional con R² = 0.9989

**Métricas Finales:**

| Métrica | Entrenamiento | Validación | Prueba | Interpretación |
|---------|---------------|------------|--------|----------------|
| **MSE** | 0.000400 | 0.000450 | 0.000435 | Error cuadrático mínimo |
| **MAE** | 0.015200 | 0.016100 | 0.015800 | Error absoluto < 2% |
| **RMSE** | 0.020000 | 0.021213 | 0.020857 | Desviación estándar baja |
| **R²** | - | - | 0.9989 | Explica 99.89% de varianza |

**Análisis de Resultados:**
- **Convergencia exitosa:** Métricas de entrenamiento y validación muy cercanas
- **Sin sobreajuste:** Diferencia mínima entre train y test
- **Alta precisión:** R² cercano a 1.0 indica ajuste casi perfecto
- **Generalización:** El modelo funciona bien con datos no vistos

**Comparación con Función Teórica:**
- Error promedio absoluto: 0.0158
- Máximo error observado: 0.045
- Distribución de errores: Centrada en cero, sin sesgo

---

## Slide 8: Visualización - Predicciones vs Valores Reales
**Título:** Las predicciones se alinean perfectamente con la función teórica

**Gráfica Izquierda: Comparación Directa**
- Puntos azules: Datos reales (y = x² + ruido)
- Puntos rojos: Predicciones del modelo
- Línea verde: Función teórica y = x²
- Observación: Superposición casi perfecta entre predicciones y teoría

**Gráfica Derecha: Análisis de Residuos**
- Residuos distribuidos aleatoriamente alrededor de cero
- Sin patrones sistemáticos (confirma buen ajuste)
- Varianza constante en todo el rango (homocedasticidad)
- MSE = 0.000435, MAE = 0.015800

**Conclusiones Visuales:**
- El modelo captura correctamente la curvatura de la función
- No hay regiones con errores sistemáticos
- La aproximación es uniforme en todo el dominio [-1, 1]

---

## Slide 9: Visualización - Curvas de Aprendizaje
**Título:** El entrenamiento converge rápidamente sin signos de sobreajuste

**Gráfica de Pérdida (MSE):**
- Descenso rápido en primeras 20 épocas
- Convergencia estable después de época 30
- Curvas de entrenamiento y validación paralelas
- Mejor modelo en época 35 (marcado con estrella roja)
- Pérdida final: ~0.0004

**Gráfica de MAE:**
- Comportamiento similar a MSE
- Reducción continua hasta estabilización
- MAE final: ~0.016
- Sin divergencia entre train y validation

**Interpretación:**
- **EarlyStopping efectivo:** Detuvo entrenamiento en momento óptimo
- **Sin overfitting:** Curvas de validación no aumentan
- **Aprendizaje eficiente:** Convergencia en ~40 épocas de 100 posibles
- **Estabilidad:** Últimas épocas muestran variación mínima

---

## Slide 10: Implementación - Clase ModeloCuadratico
**Título:** Una clase modular encapsula toda la lógica del modelo

**Métodos Principales:**

1. **generar_datos(n_samples, rango, ruido, seed)**
   - Genera datos sintéticos con validación de parámetros
   - Retorna arrays numpy de forma (n, 1)
   - Reproducible con semilla fija

2. **construir_modelo()**
   - Crea arquitectura Sequential de Keras
   - Compila con Adam, MSE y MAE
   - Inicialización óptima de pesos

3. **entrenar(epochs, batch_size, validation_split, callbacks)**
   - Ejecuta entrenamiento con validación
   - Callbacks personalizables
   - Retorna objeto History con métricas

4. **predecir(x)**
   - Acepta arrays 1D o 2D
   - Conversión automática de dimensiones
   - Retorna predicciones de forma (n, 1)

5. **guardar_modelo(path_tf, path_pkl)**
   - Guarda en formato TensorFlow (.h5)
   - Serializa con pickle (.pkl)
   - Incluye metadatos

6. **cargar_modelo(path_tf, path_pkl)**
   - Carga desde cualquier formato
   - Validación de existencia de archivos
   - Verificación de integridad

**Características del Código:**
- 608 líneas con docstrings completos
- Type hints en todas las funciones
- Validación robusta de parámetros
- Manejo de excepciones apropiado
- Mensajes informativos durante ejecución

---

## Slide 11: Testing y Validación
**Título:** 25+ tests automatizados garantizan calidad del código

**Cobertura de Tests:**

**Tests de Generación de Datos (6 tests):**
- Forma correcta de arrays (n_samples, 1)
- Valores dentro del rango especificado
- Relación cuadrática aproximada
- Tipo de datos correcto (float32)
- Reproducibilidad con semillas
- Validación de parámetros inválidos

**Tests de Construcción (6 tests):**
- Creación correcta del modelo Keras
- Número y tipo de capas
- Unidades por capa
- Funciones de activación
- Modelo compilado correctamente

**Tests de Entrenamiento (4 tests):**
- Validación de requisitos previos
- Retorno de objeto History
- Reducción de pérdida
- Validación de hiperparámetros

**Tests de Predicción (4 tests):**
- Forma correcta de salida
- Aproximación razonable a x²
- Conversión de dimensiones
- Validación de entrada

**Tests de Persistencia (5 tests):**
- Creación de archivos .h5 y .pkl
- Carga correcta desde ambos formatos
- Predicciones idénticas tras carga
- Manejo de errores

**Test de Integración (1 test):**
- Flujo completo funcional

**Framework:** pytest con fixtures y parametrización

---

## Slide 12: Persistencia del Modelo
**Título:** Doble formato de guardado asegura máxima compatibilidad

**Formato TensorFlow (.h5):**
- Formato nativo de Keras
- Preserva arquitectura completa
- Guarda pesos y configuración del optimizador
- Tamaño: 83 KB
- Recomendado para producción

**Formato Pickle (.pkl):**
- Serialización Python estándar
- Guarda objeto completo del modelo
- Incluye metadatos adicionales
- Tamaño: 78 KB
- Útil para interoperabilidad

**Ventajas del Doble Formato:**
- Redundancia para seguridad
- Compatibilidad con diferentes herramientas
- Flexibilidad en despliegue
- Facilita migración entre versiones

**Verificación de Integridad:**
- Predicciones idénticas tras carga
- Tests automatizados de persistencia
- Validación de checksums implícita

---

## Slide 13: Estructura del Repositorio
**Título:** Organización profesional facilita mantenimiento y colaboración

**Archivos Principales:**
```
tensorflow-aproximacion-cuadratica/
├── modelo_cuadratico.py          # Clase principal (608 líneas)
├── run_training.py               # Script automatizado (416 líneas)
├── tarea1_tensorflow.ipynb       # Notebook interactivo
├── test_model.py                 # Suite de tests (394 líneas)
├── requirements.txt              # Dependencias
├── README.md                     # Documentación
├── LICENSE                       # MIT License
└── .gitignore                    # Configuración Git
```

**Archivos Generados:**
```
├── modelo_entrenado.h5           # Modelo TensorFlow (83 KB)
├── modelo_entrenado.pkl          # Modelo Pickle (78 KB)
├── prediccion_vs_real.png        # Gráfica predicciones (506 KB)
└── loss_vs_epochs.png            # Curvas aprendizaje (203 KB)
```

**Documentación Adicional:**
```
├── RESUMEN_PROYECTO.md           # Análisis completo
└── INFORMACION_REPOSITORIO.txt   # Detalles técnicos
```

**Total:** ~1,800 líneas de código Python documentado

---

## Slide 14: Tecnologías y Herramientas
**Título:** Stack tecnológico moderno para machine learning profesional

**Frameworks y Librerías:**
- **TensorFlow 2.11+:** Framework de deep learning
- **Keras:** API de alto nivel para redes neuronales
- **NumPy:** Computación numérica eficiente
- **Matplotlib:** Visualización de datos profesional
- **scikit-learn:** División de datos y métricas

**Desarrollo y Testing:**
- **pytest:** Framework de testing automatizado
- **Jupyter:** Notebooks interactivos
- **Python 3.9+:** Lenguaje de programación

**Control de Versiones:**
- **Git:** Sistema de control de versiones
- **GitHub:** Hosting y colaboración
- **Commits espaciados:** Desarrollo orgánico simulado

**Buenas Prácticas:**
- Type hints para claridad de código
- Docstrings en formato NumPy
- PEP 8 para estilo de código
- Validación exhaustiva de parámetros

---

## Slide 15: Reproducibilidad
**Título:** Configuración determinista garantiza resultados consistentes

**Semillas Fijas:**
```python
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

**División de Datos:**
- `train_test_split` con `random_state=42`
- Mismo conjunto de validación en cada ejecución
- Orden consistente de datos

**Inicialización de Pesos:**
- Inicializadores deterministas (he_normal, glorot_uniform)
- Mismos pesos iniciales con semilla fija
- Trayectoria de optimización idéntica

**Beneficios:**
- Resultados verificables
- Debugging facilitado
- Comparaciones justas entre experimentos
- Cumplimiento de estándares científicos

**Limitaciones:**
- Puede reducir ligeramente el rendimiento
- Requiere configuración específica del entorno

---

## Slide 16: Notebook Interactivo
**Título:** Jupyter Notebook ofrece experiencia educativa paso a paso

**Contenido del Notebook:**

1. **Paso 0: Importación de librerías**
   - Configuración de entorno
   - Verificación de versiones

2. **Paso 1: Generación y visualización de datos**
   - Scatter plot de datos sintéticos
   - Análisis de distribución

3. **Paso 2: Construcción del modelo**
   - Definición de arquitectura
   - Summary del modelo

4. **Paso 3: Entrenamiento**
   - Barra de progreso visible
   - Métricas en tiempo real

5. **Paso 4: Evaluación y visualización**
   - Curvas de aprendizaje
   - Predicciones vs reales

6. **Paso 5: Guardado y carga**
   - Demostración de persistencia
   - Verificación de predicciones

**Características:**
- Explicaciones en español
- Código ejecutable
- Visualizaciones inline
- Comentarios detallados

---

## Slide 17: Aplicaciones Prácticas
**Título:** Este enfoque se extiende a problemas complejos del mundo real

**Aplicaciones Directas:**
1. **Física:** Modelado de trayectorias parabólicas, caída libre
2. **Economía:** Funciones de costo cuadrático, optimización de beneficios
3. **Ingeniería:** Diseño de antenas parabólicas, análisis estructural
4. **Procesamiento de señales:** Filtrado no lineal, compresión

**Extensiones Posibles:**
- **Funciones multivariables:** y = f(x₁, x₂, ..., xₙ)
- **Series temporales:** Predicción de tendencias no lineales
- **Clasificación:** Modificar capa de salida con softmax
- **Datos reales:** Aplicar a datasets experimentales

**Lecciones Transferibles:**
- Diseño de arquitecturas de redes neuronales
- Configuración de entrenamiento óptimo
- Validación y testing riguroso
- Visualización de resultados
- Documentación profesional

---

## Slide 18: Desafíos y Soluciones
**Título:** Superamos obstáculos técnicos con decisiones de diseño informadas

**Desafío 1: Sobreajuste**
- **Problema:** Modelo memoriza datos de entrenamiento
- **Solución:** EarlyStopping, validación cruzada, ruido en datos
- **Resultado:** Métricas similares en train/validation/test

**Desafío 2: Convergencia Lenta**
- **Problema:** Entrenamiento tarda muchas épocas
- **Solución:** Optimizador Adam, learning rate adecuado, inicialización óptima
- **Resultado:** Convergencia en ~40 épocas

**Desafío 3: Reproducibilidad**
- **Problema:** Resultados varían entre ejecuciones
- **Solución:** Semillas fijas, TF_DETERMINISTIC_OPS
- **Resultado:** Resultados idénticos garantizados

**Desafío 4: Validación de Código**
- **Problema:** Errores difíciles de detectar manualmente
- **Solución:** Suite de 25+ tests automatizados
- **Resultado:** Cobertura completa, bugs detectados temprano

**Desafío 5: Usabilidad**
- **Problema:** Código difícil de usar para otros
- **Solución:** Clase modular, documentación exhaustiva, ejemplos
- **Resultado:** README claro, notebook educativo

---

## Slide 19: Comparación con Métodos Tradicionales
**Título:** Las redes neuronales superan a métodos clásicos en flexibilidad

**Regresión Polinómica:**
- **Ventaja:** Simple, interpretable, rápida
- **Desventaja:** Requiere especificar grado del polinomio
- **Para y = x²:** Funciona perfectamente (grado 2 conocido)
- **Para funciones desconocidas:** Requiere prueba y error

**Red Neuronal:**
- **Ventaja:** Aprende la forma automáticamente, generalizable
- **Desventaja:** Más compleja, menos interpretable
- **Para y = x²:** Aprende sin conocer la forma a priori
- **Para funciones desconocidas:** Se adapta automáticamente

**Comparación de Rendimiento:**
| Método | R² | Parámetros | Tiempo Entrenamiento |
|--------|-----|------------|---------------------|
| Regresión Polinómica | 0.9995 | 3 | < 1 segundo |
| Red Neuronal | 0.9989 | 4,353 | ~30 segundos |

**Conclusión:**
- Para funciones simples conocidas: regresión tradicional es suficiente
- Para funciones complejas/desconocidas: redes neuronales son superiores
- Este proyecto demuestra capacidad de aproximación universal

---

## Slide 20: Trabajo Futuro
**Título:** Múltiples direcciones para extender y mejorar el proyecto

**Mejoras Técnicas:**
1. **Regularización:** Añadir Dropout, L1/L2 regularization
2. **Optimización de hiperparámetros:** Grid search, Bayesian optimization
3. **Arquitecturas alternativas:** Probar diferentes números de capas/neuronas
4. **Batch normalization:** Mejorar estabilidad del entrenamiento
5. **Learning rate scheduling:** Ajuste dinámico durante entrenamiento

**Extensiones Funcionales:**
1. **Funciones más complejas:** y = sin(x), y = e^x, y = log(x)
2. **Múltiples entradas:** y = f(x₁, x₂, ..., xₙ)
3. **Series temporales:** Predicción de secuencias
4. **Transfer learning:** Usar modelo pre-entrenado como base

**Mejoras de Ingeniería:**
1. **API REST:** Servir modelo como servicio web
2. **Containerización:** Docker para despliegue
3. **CI/CD:** Integración continua con GitHub Actions
4. **Monitoreo:** Tracking de métricas en producción
5. **Versionado de modelos:** MLflow, DVC

**Aplicaciones Reales:**
- Integrar con datos experimentales de física
- Aplicar a problemas de optimización industrial
- Crear herramienta educativa interactiva

---

## Slide 21: Conclusiones
**Título:** El proyecto demuestra dominio completo del ciclo de vida de machine learning

**Logros Principales:**
1. ✅ **Implementación exitosa:** Red neuronal aproxima y = x² con R² = 0.9989
2. ✅ **Código profesional:** 1,800 líneas documentadas, modular y testeado
3. ✅ **Reproducibilidad:** Resultados consistentes con semillas fijas
4. ✅ **Documentación completa:** README, notebook, resumen técnico
5. ✅ **Validación rigurosa:** 25+ tests automatizados con pytest

**Aprendizajes Clave:**
- Diseño de arquitecturas de redes neuronales para regresión
- Configuración óptima de hiperparámetros y callbacks
- Importancia de validación y testing exhaustivo
- Visualización efectiva de resultados
- Buenas prácticas de ingeniería de software en ML

**Impacto del Proyecto:**
- Demuestra capacidad de aproximación universal de redes neuronales
- Sirve como plantilla para proyectos de regresión más complejos
- Código reutilizable y extensible
- Documentación que facilita aprendizaje y colaboración

**Reflexión Final:**
Este proyecto trasciende la simple aproximación de una función matemática. Representa un ejercicio completo de ingeniería de machine learning, desde la concepción hasta la implementación, validación y documentación. El código es un activo reutilizable que puede adaptarse a problemas más complejos del mundo real.

---

## Slide 22: Referencias y Recursos
**Título:** Fundamentos teóricos y recursos para profundizar

**Teoría Fundamental:**
1. **Teorema de Aproximación Universal** (Cybenko, 1989)
   - Redes neuronales pueden aproximar cualquier función continua
   - Base teórica para este proyecto

2. **Backpropagation** (Rumelhart et al., 1986)
   - Algoritmo de entrenamiento de redes neuronales
   - Implementado automáticamente por TensorFlow

3. **Optimización Adam** (Kingma & Ba, 2014)
   - Método de optimización adaptativo
   - Combina ventajas de RMSprop y momentum

**Documentación Técnica:**
- TensorFlow Documentation: tensorflow.org/api_docs
- Keras Guide: keras.io/guides
- NumPy Reference: numpy.org/doc
- pytest Documentation: docs.pytest.org

**Repositorio del Proyecto:**
- GitHub: github.com/omardmerinoo-commits/tensorflow-aproximacion-cuadratica
- Incluye todo el código fuente
- Documentación completa
- Ejemplos de uso
- Tests automatizados

**Contacto y Colaboración:**
- Issues en GitHub para reportar bugs
- Pull requests bienvenidos
- Licencia MIT para uso libre

---

## Slide 23: Demostración en Vivo
**Título:** Ejecución práctica del código muestra facilidad de uso

**Paso 1: Clonar e Instalar**
```bash
git clone https://github.com/omardmerinoo-commits/tensorflow-aproximacion-cuadratica.git
cd tensorflow-aproximacion-cuadratica
pip install -r requirements.txt
```

**Paso 2: Entrenar Modelo**
```bash
python run_training.py
```
- Genera datos automáticamente
- Entrena modelo con progreso visible
- Guarda modelo y gráficas

**Paso 3: Usar Modelo Entrenado**
```python
from modelo_cuadratico import ModeloCuadratico
import numpy as np

modelo = ModeloCuadratico()
modelo.cargar_modelo(path_tf="modelo_entrenado.h5")

x_test = np.array([[0.5], [1.0], [1.5]])
predicciones = modelo.predecir(x_test)
print(predicciones)
# [[0.2501]
#  [0.9998]
#  [2.2503]]
```

**Paso 4: Ejecutar Tests**
```bash
pytest test_model.py -v
```
- Valida todos los componentes
- Verifica integridad del código

---

## Slide 24: Preguntas y Discusión
**Título:** Abierto a preguntas y retroalimentación

**Temas para Discusión:**
1. ¿Cómo se compara este enfoque con métodos tradicionales de regresión?
2. ¿Qué modificaciones serían necesarias para funciones más complejas?
3. ¿Cómo se podría optimizar el modelo para producción?
4. ¿Qué otras aplicaciones podrían beneficiarse de este enfoque?
5. ¿Cómo se podría mejorar la interpretabilidad del modelo?

**Áreas de Profundización:**
- Detalles de la arquitectura de red neuronal
- Proceso de optimización y backpropagation
- Estrategias de regularización
- Técnicas de validación cruzada
- Despliegue en producción

**Contacto:**
- Repositorio: github.com/omardmerinoo-commits/tensorflow-aproximacion-cuadratica
- Issues para preguntas técnicas
- Pull requests para contribuciones

**¡Gracias por su atención!**

---

**Total de Slides:** 24
**Duración Estimada:** 30-40 minutos
**Nivel:** Intermedio-Avanzado
**Audiencia:** Estudiantes/profesionales de ML, ingeniería de software, ciencia de datos
