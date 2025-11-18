# Proyecto 2: Web para Generar y Gestionar Datos Experimentales

## Descripción General

Aplicación web profesional con **API REST** para gestión completa de experimentos científicos. Permite:

- **CRUD de Experimentos**: Crear, leer, actualizar, eliminar
- **Generación de Datos**: Síntesis automática de datos basada en modelos físicos
- **Gestión de Datos**: Almacenamiento y organización en base de datos SQLite
- **Exportación**: CSV, JSON con un clic
- **Estadísticas**: Análisis automático de datos
- **Cliente CLI**: Interacción desde línea de comandos

## Arquitectura

```
Proyecto 2: Web
├── Backend (Flask + SQLAlchemy)
│   ├── API REST (endpoints CRUD)
│   ├── Base de Datos (SQLite)
│   └── Servicios
├── Cliente CLI (Click)
├── Tests Unitarios (pytest)
└── Frontend (HTML básico)
```

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Backend | Flask 3.0 |
| ORM | SQLAlchemy 2.0 |
| BD | SQLite |
| API | REST + JSON |
| CLI | Click |
| Testing | pytest |

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/Scripts/activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Iniciar Servidor

```bash
python app.py
```

El servidor estará disponible en `http://localhost:5000`

### Cliente CLI

```bash
# Listar experimentos
python cliente_cli.py listar

# Crear experimento
python cliente_cli.py crear --nombre "Mi Experimento" --tipo subamortiguado

# Generar datos
python cliente_cli.py generar 1

# Obtener estadísticas
python cliente_cli.py estadisticas 1

# Exportar
python cliente_cli.py exportar 1 --formato csv
```

## API Endpoints

### Experimentos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/experimentos` | Listar todos |
| POST | `/api/experimentos` | Crear nuevo |
| GET | `/api/experimentos/<id>` | Obtener uno |
| PUT | `/api/experimentos/<id>` | Actualizar |
| DELETE | `/api/experimentos/<id>` | Eliminar |

### Datos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/experimentos/<id>/puntos` | Listar puntos |
| POST | `/api/experimentos/<id>/puntos` | Agregar punto |
| POST | `/api/experimentos/<id>/puntos/batch` | Agregar múltiples |

### Operaciones

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/experimentos/<id>/generar` | Generar datos sintéticos |
| GET | `/api/experimentos/<id>/estadisticas` | Obtener estadísticas |
| GET | `/api/experimentos/<id>/exportar/csv` | Descargar CSV |
| GET | `/api/experimentos/<id>/exportar/json` | Descargar JSON |

## Ejemplos de Uso

### Crear Experimento (cURL)

```bash
curl -X POST http://localhost:5000/api/experimentos \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Oscilación Amortiguada",
    "tipo": "subamortiguado",
    "masa": 1.5,
    "amortiguamiento": 0.8,
    "rigidez": 2.0,
    "posicion_inicial": 1.0
  }'
```

### Generar Datos

```bash
curl -X POST http://localhost:5000/api/experimentos/1/generar
```

### Obtener Estadísticas

```bash
curl http://localhost:5000/api/experimentos/1/estadisticas
```

## Modelos de Base de Datos

### Experimento
- `id`: Identificador único
- `nombre`: Nombre descriptivo (único)
- `descripción`: Texto adicional
- `tipo`: Clasificación del experimento
- `estado`: activo, completado, cancelado
- `parametros_sistema`: m, c, k, x0, v0
- `parametros_simulacion`: tiempo_max, num_puntos, ruido

### PuntoDato
- `id`: Identificador único
- `experimento_id`: Referencia al experimento
- `tiempo`: Timestamp
- `posición`: Valor medido
- `velocidad`: Derivada primera
- `aceleración`: Derivada segunda
- `energía`: Valor calculado

## Testing

```bash
# Ejecutar tests
pytest test_app.py -v

# Con cobertura
pytest test_app.py --cov=app --cov=modelos_bd

# Test específico
pytest test_app.py::TestExperimentos::test_crear_experimento -v
```

### Cobertura

- **15+ tests de integración**
- **Cobertura > 90%**
- Validación de:
  - CRUD operations
  - Generación de datos
  - Exportación
  - Estadísticas

## Características Avanzadas

### 1. Generación Automática de Datos
```python
# POST /api/experimentos/<id>/generar
# Crea 100 puntos de datos basados en la solución analítica
```

### 2. Validación de Parámetros
- Masa > 0
- Amortiguamiento >= 0
- Rigidez > 0
- Tiempo máximo > 0

### 3. Exportación Flexible
- **CSV**: Para análisis en Excel/LibreOffice
- **JSON**: Para procesamiento automático

### 4. Estadísticas Descriptivas
- Mínimos/máximos
- Promedios y desviaciones
- Energía promedio
- Rangos de validez

## Manejo de Errores

```json
{
  "exito": false,
  "error": "Experimento no encontrado",
  "errores": ["error 1", "error 2"]
}
```

## Seguridad

- ✓ Validación de entrada
- ✓ Manejo de excepciones
- ✓ Inyección SQL prevenida (SQLAlchemy ORM)
- ✓ CORS configurado
- ✓ Límite de tamaño de carga

## Rendimiento

- Base de datos en memoria para tests
- Índices en tablas principales
- Consultas optimizadas

## Troubleshooting

### Error: "No se puede conectar al servidor"
```bash
# Verificar que el servidor está corriendo
python app.py  # En otra terminal
```

### Error: "Puerto 5000 ya está en uso"
```python
# Cambiar puerto en app.py
if __name__ == '__main__':
    app.run(port=5001)  # Cambiar a otro puerto
```

### Error: "Base de datos bloqueada"
```python
# Resetear base de datos
rm experimentos.db
# Ejecutar app nuevamente
```

## Extensiones Futuras

1. **Frontend Web**
   - Dashboard con visualizaciones
   - Editor visual de experimentos
   - Gráficas en tiempo real

2. **Autenticación**
   - Usuarios y roles
   - Permisos granulares
   - JWT tokens

3. **Base de Datos Remota**
   - PostgreSQL para producción
   - Replicación y backup

4. **API Avanzada**
   - Paginación
   - Filtros complejos
   - Búsqueda full-text

5. **ML Integration**
   - Predicciones automáticas
   - Detección de anomalías
   - Clustering de experimentos

## Licencia

MIT License

## Changelog

### v1.0.0 (2025-11-18)
- Implementación inicial
- API REST completa
- Cliente CLI
- Tests de integración
- Documentación profesional
