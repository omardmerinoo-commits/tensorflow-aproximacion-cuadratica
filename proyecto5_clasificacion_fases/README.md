# Proyecto 5: Clasificación de Fases

Red neuronal profunda para clasificar fases de la materia (sólido, líquido, gas) basada en características físicas como temperatura, presión y densidad.

## Características

- Generador de datos sintéticos con 3 fases balanceadas
- Red neuronal MLP con 4 capas densas
- Normalización de datos y validación cruzada
- Tests unitarios completos

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```python
python run_fases.py
```

## Archivos

- `generador_datos_fases.py`: Generador de datos físicos
- `modelo_clasificador_fases.py`: Modelo de clasificación
- `run_fases.py`: Script principal
- `test_fases.py`: Tests unitarios

## Resultados

- Accuracy en test: ~95-98%
- Modelo guardado: `modelo_fases.keras`
- Reporte: `REPORTE_FASES.json`
