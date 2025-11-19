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

## 🔄 Próximas Aplicaciones (P6-P12)

### P6: Clasificador CNN - Reconocimiento de Dígitos
- Clasificación de imágenes MNIST
- Predicción con confianza
- Análisis de errores

### P7: Clasificador Audio - Detección de Instrumento Musical
- Clasificación por MFCC
- Predicción en tiempo real
- Análisis espectral

### P8: Detector YOLO - Detección de Objetos en Video
- Detección en webcam
- Bounding boxes dinámicos
- FPS tracking

### P9: Segmentador U-Net - Segmentación Semántica
- Segmentación de órganos médicos
- Visualización de mascaras
- Evaluación de precisión

### P10: Predictor Series Temporales - Pronóstico de Acciones
- Predicción de precios de acciones
- Análisis de tendencias
- Alertas de volatilidad

### P11: Clasificador NLP - Análisis de Sentimiento de Redes Sociales
- Clasificación de tweets
- Análisis de sentimiento
- Wordcloud de palabras clave

### P12: Generador GAN/VAE - Síntesis de Imágenes
- Generación de caras sintéticas
- Interpolación en latent space
- Comparativa GAN vs VAE

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

### Todos los P0-P5

```bash
# P0
python proyecto0_original/aplicaciones/predictor_precios_casas.py

# P1
python proyecto1_oscilaciones/aplicaciones/predictor_consumo_energia.py

# P2
python proyecto2_web/aplicaciones/detector_fraude.py

# P3
python proyecto3_qubits/aplicaciones/clasificador_diagnostico.py

# P4
python proyecto4_estadistica/aplicaciones/segmentador_clientes.py

# P5
python proyecto5_clasificador/aplicaciones/compresor_imagenes_pca.py
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

---

## 🚀 Escalamiento Futuro

### Mejoras Planeadas

- [ ] API REST para cada aplicación
- [ ] Base de datos para persistencia
- [ ] Visualización web (Dash/Streamlit)
- [ ] Modelos entrenados pre-guardados
- [ ] Validación cruzada
- [ ] Hyperparameter tuning
- [ ] Testing automatizado
- [ ] Docker containerization

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
**Versión**: 1.0.0  
**Estado**: ✅ En desarrollo activo

*¡Gracias por usar estas aplicaciones prácticas!* 🚀

