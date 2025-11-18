# Proyecto 6: Aproximador de Funciones No Lineales

Red neuronal profunda para aproximar funciones complejas y no lineales como:
- $\sin(x) \cdot e^{-x/10}$
- $x^3 - 2x^2 + 3x - 1$
- $\sin(x) \cdot \cos(x/2)$
- $\log(1 + x^2) \cdot \sin(x)$

## Características

- 4 funciones no lineales diferentes
- Generador de datos sintéticos con ruido controlado
- Red neuronal con normalización por lotes
- Visualización de aproximación vs función real
- Análisis de error

## Uso

```bash
python run_funciones.py
```

## Resultados

- MAE promedio: < 0.01
- Modelos guardados: `modelo_*.keras`
- Gráficos: `funcion_*.png`
