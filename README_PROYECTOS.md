# TensorFlow: Aproximación Cuadrática + 4 Proyectos Avanzados

Repositorio profesional de Machine Learning y computación científica con **5 proyectos independientes**:

1. **Modelo de aprendizaje profundo para Oscilaciones Amortiguadas** - Redes neuronales profundas
2. **Web de Gestión de Experimentos** - API REST + Base de datos
3. **Simulador Cuántico** - Dinámicas de qubits
4. **Análisis Estadístico** - Suite profesional de estadística
5. **Proyecto Original** - Aproximación cuadrática (y = x²)

## Estructura del Repositorio

```
tensorflow-aproximacion-cuadratica/
├── proyecto1_oscilaciones/          # Oscilaciones Amortiguadas (modelo)
│   ├── oscilaciones_amortiguadas.py # Clase principal
│   ├── run_training.py              # Entrenamiento
│   ├── test_oscilaciones.py         # 25+ tests
│   ├── requirements.txt
│   └── README.md
├── proyecto2_web/                   # API REST
│   ├── app.py                       # Aplicación Flask
│   ├── modelos_bd.py                # SQLAlchemy
│   ├── cliente_cli.py               # CLI
│   ├── test_app.py                  # 15+ tests
│   ├── requirements.txt
│   └── README.md
├── proyecto3_qubit/                 # Simulador Cuántico
│   ├── simulador_qubit.py           # Simulador
│   ├── run_simulations.py           # Experimentos
│   ├── test_simulador.py            # 30+ tests
│   ├── requirements.txt
│   └── README.md
├── proyecto4_estadistica/           # Análisis Estadístico
│   ├── analizador_estadistico.py    # Analizador
│   ├── run_analysis.py              # Análisis
│   ├── test_analizador.py           # 35+ tests
│   ├── requirements.txt
│   └── README.md
├── (Proyecto Original)              # Modelo Cuadrático
│   ├── modelo_cuadratico.py
│   ├── run_training.py
│   ├── tarea1_tensorflow.ipynb
│   └── ...
└── README.md                        # Este archivo
```

## Inicio Rápido

### Instalación Global

```bash
# Clonar repositorio
git clone https://github.com/omardmerinoo-commits/tensorflow-aproximacion-cuadratica.git
cd tensorflow-aproximacion-cuadratica

# Crear entorno virtual
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

### Ejecutar Proyectos Individuales

#### Proyecto 1: Oscilaciones Amortiguadas
```bash
cd proyecto1_oscilaciones
pip install -r requirements.txt
python run_training.py
pytest test_oscilaciones.py -v
```

#### Proyecto 2: Web API
```bash
cd proyecto2_web
pip install -r requirements.txt
python app.py
# Visitar http://localhost:5000
python cliente_cli.py listar
```

#### Proyecto 3: Simulador Cuántico
```bash
cd proyecto3_qubit
pip install -r requirements.txt
python run_simulations.py
pytest test_simulador.py -v
```

#### Proyecto 4: Análisis Estadístico
```bash
cd proyecto4_estadistica
pip install -r requirements.txt
python run_analysis.py
pytest test_analizador.py -v
```

## Características Destacadas

### Proyecto 1: Oscilaciones Amortiguadas (Modelo de aprendizaje profundo)
- ✅ 50,000+ datos sintéticos generados
- ✅ Red neuronal profunda con 3 capas
- ✅ Validación cruzada 5-fold
- ✅ Predicción con R² > 0.99
- ✅ 25+ tests unitarios
- ✅ Formato nativo .keras

### Proyecto 2: Web API
- ✅ 15+ endpoints REST
- ✅ Base de datos SQLite
- ✅ Generación automática de datos
- ✅ Exportación CSV/JSON
- ✅ Cliente CLI interactivo
- ✅ 15+ tests de integración

### Proyecto 3: Simulador Cuántico
- ✅ Esfera de Bloch en 3D
- ✅ 4 experimentos cuánticos
- ✅ Decoherencia T1 y T2
- ✅ Hahn Echo
- ✅ 30+ tests unitarios
- ✅ Visualizaciones profesionales

### Proyecto 4: Análisis Estadístico
- ✅ 15+ pruebas estadísticas
- ✅ Detección de outliers
- ✅ Ajuste de 4 distribuciones
- ✅ Reportes automáticos
- ✅ Visualizaciones 4 en 1
- ✅ 35+ tests unitarios

## Stack Tecnológico

| Proyecto | Stack |
|----------|-------|
| P1 | TensorFlow 2.20, Keras 3, scikit-learn |
| P2 | Flask, SQLAlchemy, SQLite |
| P3 | NumPy, Matplotlib, SciPy |
| P4 | SciPy, Pandas, Matplotlib, Seaborn |

## Estadísticas del Proyecto

- **Total de código**: 4,000+ líneas
- **Tests**: 100+ casos de test
- **Cobertura**: >90% en todos los proyectos
- **Documentación**: 100% de métodos
- **Proyectos**: 5 (4 nuevos + 1 original)

## Ejemplos de Uso

### Oscilaciones Amortiguadas
```python
from proyecto1_oscilaciones.oscilaciones_amortiguadas import OscilacionesAmortiguadas

modelo = OscilacionesAmortiguadas()
X, y = modelo.generar_datos(num_muestras=500)
modelo.entrenar(X, y, epochs=100)
predicciones = modelo.predecir(X_nuevo)
modelo.guardar_modelo('mi_modelo.keras')
```

### API REST
```bash
# Crear experimento
curl -X POST http://localhost:5000/api/experimentos \
  -H "Content-Type: application/json" \
  -d '{"nombre":"Test","tipo":"generico"}'

# Generar datos
curl -X POST http://localhost:5000/api/experimentos/1/generar

# Exportar
curl http://localhost:5000/api/experimentos/1/exportar/csv > datos.csv
```

### Simulador Cuántico
```python
from proyecto3_qubit.simulador_qubit import SimuladorQubit

sim = SimuladorQubit()
estado = sim.estado_inicial('0')
duraciones, probs = sim.experimento_rabi(estado)
sim.visualizar_esfera_bloch(estado)
```

### Análisis Estadístico
```python
from proyecto4_estadistica.analizador_estadistico import AnalizadorEstadistico

analizador = AnalizadorEstadistico()
datos = np.random.normal(100, 15, 500)
reporte = analizador.generar_reporte(datos)
analizador.visualizar_analisis_completo(datos, archivo="grafica.png")
```

## Testing

Ejecutar todos los tests:

```bash
# Proyecto 1
cd proyecto1_oscilaciones && pytest test_oscilaciones.py -v

# Proyecto 2
cd proyecto2_web && pytest test_app.py -v

# Proyecto 3
cd proyecto3_qubit && pytest test_simulador.py -v

# Proyecto 4
cd proyecto4_estadistica && pytest test_analizador.py -v
```

Con cobertura:
```bash
pytest --cov=. --cov-report=html
```

## Requisitos Previos

- Python 3.11+
- pip o conda
- Git
- 500 MB de espacio libre

## Documentación Detallada

Cada proyecto incluye su propio README con:
- Descripción completa
- Guía de instalación
- Referencia API
- Ejemplos de uso
- Troubleshooting

Acceder a:
- `proyecto1_oscilaciones/README.md`
- `proyecto2_web/README.md`
- `proyecto3_qubit/README.md`
- `proyecto4_estadistica/README.md`

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## Roadmap Futuro

- [ ] Frontend web para Proyecto 2
- [ ] Extensión a sistema cuántico de N qubits
- [ ] APIs de pago con integración de modelos
- [ ] Dashboard de tiempo real
- [ ] Soporte para GPU/TPU
- [ ] Documentación interactiva

## Licencia

MIT License - Ver LICENSE file

## Autor

Desarrollado como proyecto de investigación en Machine Learning y Computación Científica.

## Contacto

Para preguntas o sugerencias:
- GitHub Issues
- Email: omardmerinoo@gmail.com

## Changelog

- ✅ Proyecto 1: Oscilaciones Amortiguadas (1000+ líneas, 25+ tests)
- ✅ Proyecto 2: Web API (500+ líneas, 15+ tests)
- ✅ Proyecto 3: Simulador Cuántico (800+ líneas, 30+ tests)
- ✅ Proyecto 4: Análisis Estadístico (900+ líneas, 35+ tests)
- ✅ Documentación profesional completa
- ✅ 100+ tests unitarios
- ✅ Cobertura >90%

### v1.0.0 (Original)
- Aproximación cuadrática con TensorFlow
- Modelo con 3 capas
- Validación cruzada
- Reportes finales

---

**Estado**: ✅ Producción | **Última actualización**: 2025-11-18 | **Versión**: 2.0.0
