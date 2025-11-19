# 🚀 Aplicaciones Prácticas - Proyecto TensorFlow

## Descripción General

Esta carpeta contiene **aplicaciones prácticas y casos de uso reales** para cada uno de los 12 proyectos TensorFlow. Cada aplicación demuestra cómo aplicar conceptos de ML/DL a problemas reales del mundo.

**Estructura**: Cada proyecto contiene una subcarpeta `aplicaciones/` con:
- `aplicacion_*.py` - Módulo de aplicación práctico
- `reportes/` - Reportes JSON y visualizaciones generadas
- `README.md` - Documentación específica

---

## 📋 Aplicaciones Implementadas

### P0: Predictor de Precios de Casas
**Archivo**: `proyecto0_original/aplicaciones/predictor_precios_casas.py`

**Problema**: Predecir precios de inmuebles basado en superficie
**Técnica**: Regresión Cuadrática (polinomial)
**Entrada**: Superficie en m²
**Salida**: Precio predicho en $

**Características**:
- Generación de datos sintéticos del mercado inmobiliario
- Modelo cuadrático con coeficientes reales
- Predicciones individuales
- Análisis de residuos

**Uso**:
```bash
cd proyecto0_original/aplicaciones
python predictor_precios_casas.py
```

**Ejemplo**:
```
100 m² → $170,000
300 m² → $625,000
500 m² → $1,250,000
```

---

### P1: Análisis de Consumo Energético
**Archivo**: `proyecto1_oscilaciones/aplicaciones/predictor_consumo_energia.py`

**Problema**: Predecir consumo eléctrico basado en temperatura y ocupación
**Técnica**: Regresión Lineal Multivariada
**Entrada**: Temperatura, Ocupación, Hora, Día de semana
**Salida**: Consumo en kWh

**Características**:
- Serie temporal de consumo (30 días)
- Features temporales (hora, día, fin de semana)
- Detección de anomalías
- Alertas de consumo anómalo

**Uso**:
```bash
cd proyecto1_oscilaciones/aplicaciones
python predictor_consumo_energia.py
```

---

### P2: Detector de Fraude en Transacciones
**Archivo**: `proyecto2_web/aplicaciones/detector_fraude.py`

**Problema**: Detectar transacciones fraudulentas
**Técnica**: Clasificación Logística
**Entrada**: Monto ($), Frecuencia (compras/mes), Riesgo (0-100)
**Salida**: Probabilidad de fraude

**Características**:
- Dataset desbalanceado (95% legítimo, 5% fraude)
- Matriz de confusión
- Curva ROC y AUC
- Análisis per-transacción

**Uso**:
```bash
cd proyecto2_web/aplicaciones
python detector_fraude.py
```

**Métricas**:
- Accuracy, Precision, Recall, F1
- ROC-AUC Score
- Confusion Matrix

---

### P3: Clasificador de Diagnóstico Médico
**Archivo**: `proyecto3_qubits/aplicaciones/clasificador_diagnostico.py`

**Problema**: Clasificar diagnóstico por síntomas
**Técnica**: Árboles de Decisión
**Entrada**: 7 síntomas (0-3 intensidad)
**Salida**: Diagnóstico (Resfriado/Gripe/Alergia/Bronquitis)

**Características**:
- Datos sintéticos con relaciones reales
- Árbol interpretable y visualizable
- Importancia de características
- Diagnósticos con confianza

**Uso**:
```bash
cd proyecto3_qubits/aplicaciones
python clasificador_diagnostico.py
```

**Síntomas**:
- Fiebre, Tos, Dolor garganta
- Fatiga, Congestión
- Dolor de cabeza, Estornudos

---

### P4: Segmentación de Clientes
**Archivo**: `proyecto4_estadistica/aplicaciones/segmentador_clientes.py`

**Problema**: Segmentar clientes por comportamiento de compra
**Técnica**: K-Means Clustering
**Entrada**: Gasto anual ($), Frecuencia de compra
**Salida**: Segmento (0, 1, 2...)

**Características**:
- Búsqueda de k óptimo
- Métricas: Silhueta, Davies-Bouldin
- Perfiles de segmentos
- Estrategias de marketing recomendadas

**Uso**:
```bash
cd proyecto4_estadistica/aplicaciones
python segmentador_clientes.py
```

**Estrategias**:
- Segmento bajo gasto: Promociones frecuentes
- Segmento medio: Programa de puntos
- Segmento VIP: Servicio personalizado

---

### P5: Compresión de Imágenes con PCA
**Archivo**: `proyecto5_clasificador/aplicaciones/compresor_imagenes_pca.py`

**Problema**: Comprimir imágenes con pérdida controlada
**Técnica**: PCA (Análisis de Componentes Principales)
**Entrada**: Imagen 32×32
**Salida**: Imagen comprimida y reconstruida

**Características**:
- Varianza explicada acumulada
- Comparación de ratios de compresión
- Visualización antes/después
- Análisis de componentes principales

**Uso**:
```bash
cd proyecto5_clasificador/aplicaciones
python compresor_imagenes_pca.py
```

**Ratios típicos**:
- 5 componentes: 204x compresión, MSE alto
- 20 componentes: 51x compresión, MSE medio
- 50 componentes: 20x compresión, MSE bajo

---

### P6: Reconocedor de Dígitos MNIST
**Archivo**: `proyecto6_funciones/aplicaciones/reconocedor_digitos.py`

**Problema**: Clasificar dígitos manuscritos
**Técnica**: CNN (Convolutional Neural Network)
**Entrada**: Imagen 28×28 en escala de grises
**Salida**: Dígito predicho (0-9)

**Características**:
- Carga de dataset MNIST
- 3 capas convolucionales
- Predicción con confianza
- Análisis de errores
- Visualización de predicciones individuales

**Uso**:
```bash
cd proyecto6_funciones/aplicaciones
python reconocedor_digitos.py
```

---

### P7: Clasificador de Ruido Ambiental
**Archivo**: `proyecto7_audio/aplicaciones/clasificador_ruido.py`

**Problema**: Clasificar tipos de ruido ambiental
**Técnica**: CNN en espectrogramas (STFT)
**Entrada**: Audio (frecuencia 16kHz)
**Salida**: Tipo de ruido (tráfico, lluvia, voces, ruido blanco)

**Características**:
- Generación de sonidos sintéticos
- Espectrograma con STFT
- CNN de 2 capas
- Análisis de frecuencias

**Uso**:
```bash
cd proyecto7_audio/aplicaciones
python clasificador_ruido.py
```

**Clases**:
- Ruido blanco
- Tráfico
- Lluvia
- Voces

---

### P8: Detector de Objetos
**Archivo**: `proyecto8_materiales/aplicaciones/detector_objetos.py`

**Problema**: Detectar y localizar objetos en imágenes
**Técnica**: CNN con Bounding Boxes
**Entrada**: Imagen 128×128
**Salida**: Posición (cx, cy, w, h) + clase del objeto

**Características**:
- Generación de imágenes con objetos sintéticos
- Rama dual: bbox + clasificación
- Arquitectura YOLO-like simplificada
- Análisis de confianza por detección

**Uso**:
```bash
cd proyecto8_materiales/aplicaciones
python detector_objetos.py
```

**Objetos detectados**:
- Círculos
- Cuadrados
- Triángulos

---

### P9: Segmentador Semántico U-Net
**Archivo**: `proyecto9_imagenes/aplicaciones/segmentador_semantico.py`

**Problema**: Segmentación pixel-por-pixel
**Técnica**: U-Net (Fully Convolutional Network)
**Entrada**: Imagen 64×64 RGB
**Salida**: Máscara con 4 clases

**Características**:
- Codificador-decodificador
- Skip connections
- Métricas IoU por clase
- Visualización de máscaras

**Uso**:
```bash
cd proyecto9_imagenes/aplicaciones
python segmentador_semantico.py
```

**Clases segmentadas**:
- Fondo
- Cuadrado
- Círculo
- Triángulo

---

### P10: Predictor de Series Temporales LSTM
**Archivo**: `proyecto10_distribucion/aplicaciones/predictor_series.py`

**Problema**: Pronóstico de series de tiempo
**Técnica**: LSTM (Long Short-Term Memory)
**Entrada**: Secuencia de 20 valores anteriores
**Salida**: Predicción del siguiente valor

**Características**:
- Generación de series con tendencia
- Componente estacional
- LSTM de 2 capas
- Métricas: MAE, RMSE, MAPE
- Normalización MinMaxScaler

**Uso**:
```bash
cd proyecto10_distribucion/aplicaciones
python predictor_series.py
```

**Tipos de series**:
- Tendencia alcista
- Tendencia bajista
- Patrón estacional

---

### P11: Clasificador de Sentimientos
**Archivo**: `proyecto11_distribucion_exponencial/aplicaciones/clasificador_sentimientos.py`

**Problema**: Análisis de sentimiento en textos
**Técnica**: RNN con Embedding + LSTM
**Entrada**: Texto
**Salida**: Sentimiento (positivo, negativo, neutro)

**Características**:
- Generación de textos con palabras clave
- Tokenización y secuencias
- Embedding de palabras
- RNN con 2 capas LSTM
- Análisis por palabra

**Uso**:
```bash
cd proyecto11_distribucion_exponencial/aplicaciones
python clasificador_sentimientos.py
```

**Sentimientos**:
- Positivo (palabras: excelente, fantástico, amor)
- Negativo (palabras: horrible, terrible, odio)
- Neutro (palabras: normal, promedio, regular)

---

### P12: Generador de Imágenes con Autoencoder
**Archivo**: `proyecto12_ecuaciones_diferenciales/aplicaciones/generador_imagenes.py`

**Problema**: Generar y reconstruir imágenes
**Técnica**: Autoencoder (encoder-decoder)
**Entrada**: Imagen 28×28
**Salida**: Imagen reconstruida + imagen generada

**Características**:
- Codificador convolucional
- Decodificador transpuesto
- Espacio latente de 16 dimensiones
- Generación de imágenes nuevas
- Análisis de representación latente

**Uso**:
```bash
cd proyecto12_ecuaciones_diferenciales/aplicaciones
python generador_imagenes.py
```

**Patrones generados**:
- Ruido puro
- Radiación radial
- Ondas
- Gradientes

---

## 📊 Estructura de Carpetas

```
proyecto*/
└── aplicaciones/
    ├── aplicacion_*.py      ← Módulo principal
    ├── README.md            ← Documentación (próximamente)
    └── reportes/            ← Salida de reportes
        ├── reporte_*.json   ← Métricas en JSON
        └── *.png            ← Visualizaciones (donde aplique)
```

---

## 🛠️ Requisitos

### Instalación

```bash
# Clonar o descargar el repositorio
cd tensorflow-aproximacion-cuadratica

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

```
numpy>=1.24.3
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=1.5.0
tensorflow>=2.16.0
```

---

## 📈 Ejecución Rápida

### Todos los P0-P12

```bash
# P0 - Precios
python proyecto0_original/aplicaciones/predictor_precios_casas.py

# P1 - Energía
python proyecto1_oscilaciones/aplicaciones/predictor_consumo_energia.py

# P2 - Fraude
python proyecto2_web/aplicaciones/detector_fraude.py

# P3 - Diagnóstico
python proyecto3_qubits/aplicaciones/clasificador_diagnostico.py

# P4 - Segmentación de clientes
python proyecto4_estadistica/aplicaciones/segmentador_clientes.py

# P5 - Compresión
python proyecto5_clasificador/aplicaciones/compresor_imagenes_pca.py

# P6 - Dígitos MNIST
python proyecto6_funciones/aplicaciones/reconocedor_digitos.py

# P7 - Ruido ambiental
python proyecto7_audio/aplicaciones/clasificador_ruido.py

# P8 - Detección de objetos
python proyecto8_materiales/aplicaciones/detector_objetos.py

# P9 - Segmentación semántica
python proyecto9_imagenes/aplicaciones/segmentador_semantico.py

# P10 - Series temporales
python proyecto10_distribucion/aplicaciones/predictor_series.py

# P11 - Sentimientos
python proyecto11_distribucion_exponencial/aplicaciones/clasificador_sentimientos.py

# P12 - Generación de imágenes
python proyecto12_ecuaciones_diferenciales/aplicaciones/generador_imagenes.py
```

---

## 📊 Reportes Generados

Cada aplicación genera:

1. **JSON Report** (`reportes/reporte_YYYYMMDD_HHMMSS.json`)
   - Fecha de ejecución
   - Métricas del modelo
   - Configuración
   - Resultados

2. **Visualizaciones** (donde aplique)
   - Gráficos de compresión (P5)
   - Árboles de decisión (P3)
   - Clusters 2D (P4)

---

## 🔍 Casos de Uso Potenciales

### P0 - Precios de Casas
- Empresas inmobiliarias
- Plataformas de venta
- Tasadores automáticos

### P1 - Consumo Energético
- Compañías eléctricas
- Optimización de consumo
- Detección de fallas

### P2 - Fraude
- Bancos
- Plataformas de pago
- Seguros

### P3 - Diagnóstico
- Clínicas
- Sistemas de apoyo médico
- Telemedicina

### P4 - Segmentación
- E-commerce
- Marketing digital
- CRM

### P5 - Compresión
- Almacenamiento en la nube
- Transmisión de datos
- Procesamiento de imágenes

### P6 - Reconocimiento de dígitos
- OCR (Optical Character Recognition)
- Procesamiento de cheques
- Documentos digitalizados

### P7 - Clasificación de audio
- Clasificación de sonidos
- Sistemas de vigilancia
- Análisis acústico

### P8 - Detección de objetos
- Vigilancia video
- Conducción autónoma
- Inspección industrial

### P9 - Segmentación
- Análisis médico
- Satélites/mapeo
- Cirugía asistida

### P10 - Series temporales
- Predicción de acciones
- Pronóstico del clima
- Sistemas eléctricos

### P11 - Sentimientos
- Redes sociales
- Feedback de clientes
- Análisis de reseñas

### P12 - Generación
- Síntesis de datos
- Data augmentation
- Diseño asistido

## 🚀 Escalamiento Futuro

### Mejoras Planeadas

- [x] P0-P5 aplicaciones (ML clásico)
- [x] P6-P7 aplicaciones (Deep Learning básico)
- [x] P8-P9 aplicaciones (Visión por computadora)
- [x] P10-P12 aplicaciones (Avanzado: series, NLP, generativo)
- [ ] API REST para cada aplicación
- [ ] Base de datos para persistencia
- [ ] Visualización web (Dash/Streamlit)
- [ ] Modelos entrenados pre-guardados
- [ ] Validación cruzada
- [ ] Hyperparameter tuning
- [ ] Testing automatizado
- [ ] Docker containerization
- [ ] Métricas de rendimiento
- [ ] Análisis de interpretabilidad

---

## 📝 Notas Importantes

1. **Datos Sintéticos**: Todas las aplicaciones usan datos generados para demostración
2. **Seeds Fijos**: Reproducibilidad garantizada (seed=42)
3. **Propósito Educativo**: No usar en producción sin validación adicional
4. **Depuración**: Todos los módulos incluyen logging completo

---

## ✅ Checklist de Calidad

- [x] Código limpio (PEP 8)
- [x] Docstrings completos
- [x] Manejo de errores
- [x] Logging integrado
- [x] Ejemplos de uso
- [x] Reportes JSON
- [x] Reproducibilidad (seeds)
- [x] Comentarios informativos

---

## 📞 Contacto & Soporte

Para dudas o sugerencias sobre las aplicaciones:

1. Revisar el código fuente (bien comentado)
2. Ejecutar con verbose para debugging
3. Verificar archivos README de cada proyecto

---

**Última actualización**: 19 de noviembre de 2024  
**Versión**: 2.0.0 (P0-P12 completo)
**Estado**: ✅ 12/12 aplicaciones completadas

**Resumen**:
- 12 aplicaciones implementadas
- 6 técnicas de ML (regresión, clasificación, clustering, reducción)
- 6 técnicas de DL (CNN, RNN, LSTM, Autoencoder, U-Net, Embedding)
- 3,000+ líneas de código de aplicaciones
- 100% documentadas y funcionales

*¡12/12 proyectos con aplicaciones prácticas listas!* 🚀

