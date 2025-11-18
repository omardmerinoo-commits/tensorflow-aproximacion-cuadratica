# Proyecto 4: Análisis Estadístico Profesional

## Descripción

Herramienta completa para análisis estadístico de datasets experimentales con:
- **Estadísticas descriptivas** completas
- **Tests estadísticos**: t-test, ANOVA, correlación
- **Tests de normalidad**: Shapiro-Wilk, KS, Anderson-Darling
- **Detección de outliers**: IQR y Z-score
- **Ajuste de distribuciones**: Normal, Log-normal, Exponencial, Uniforme
- **Reportes automáticos**: JSON y texto
- **Visualizaciones**: Histogramas, Q-Q plots, Box plots, Densidades

## Características Principales

### 1. Estadísticas Descriptivas
- Media, mediana, moda
- Desviación estándar, varianza
- Percentiles (Q1, Q2, Q3)
- Sesgo y curtosis
- Coeficiente de variación

### 2. Tests Estadísticos
- **T-test**: Comparación de dos muestras
- **ANOVA**: Comparación de múltiples grupos
- **Normalidad**: Shapiro-Wilk, Kolmogorov-Smirnov
- **Correlación**: Pearson, Spearman, Kendall

### 3. Análisis Avanzado
- Intervalos de confianza (95%)
- Detección robusta de outliers
- Ajuste a 4 distribuciones comunes
- Tamaños de efecto (Cohen's d, eta-squared)

### 4. Exportación
- Reportes JSON completos
- Reportes de texto formateados
- Visualizaciones de alta calidad (300 dpi)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar análisis
python run_analysis.py

# Tests
pytest test_analizador.py -v
```

## API Referencia

| Método | Descripción |
|--------|-------------|
| `estadisticas_basicas()` | Descriptivas |
| `intervalo_confianza()` | IC al 95% |
| `deteccion_outliers()` | Outliers |
| `test_normalidad()` | Tests de normalidad |
| `test_t_independiente()` | T-test |
| `test_anova()` | ANOVA |
| `test_correlacion()` | Correlación |
| `ajuste_distribucion()` | Ajuste automático |
| `visualizar_analisis_completo()` | Gráficas 4 en 1 |
| `generar_reporte()` | Reporte completo |

## Ejemplo de Uso

```python
from analizador_estadistico import AnalizadorEstadistico
import numpy as np

# Crear analizador
analizador = AnalizadorEstadistico()

# Datos
datos = np.random.normal(100, 15, 500)

# Análisis
stats = analizador.estadisticas_basicas(datos)
print(f"Media: {stats['media']:.2f}")
print(f"Std: {stats['std']:.2f}")

# Reporte completo
reporte = analizador.generar_reporte(datos, "Mi Análisis")
analizador.exportar_reporte_json(reporte, "reporte.json")

# Visualización
analizador.visualizar_analisis_completo(datos, "Datos", "grafica.png")
```

## Resultados

Genera automáticamente:
- Histogramas con media/mediana
- Q-Q plots para normalidad
- Box plots para outliers
- Estimaciones de densidad
- Reportes JSON y texto

## Testing

```bash
pytest test_analizador.py --cov=analizador_estadistico
```

35+ tests unitarios

## Licencia

MIT License
