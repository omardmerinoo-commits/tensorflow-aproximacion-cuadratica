# Proyecto 9: Conteo Automático de Objetos

CNN para contar automáticamente objetos en imágenes usando visión computacional.

## Características

- Generador de imágenes sintéticas con círculos aleatorios
- Red neuronal convolucional (CNN) de 3 bloques
- Predicción de conteo de 0-15 objetos
- Imágenes de 64x64 píxeles

## Arquitectura CNN

- Bloque 1: Conv2D(32) → BatchNorm → MaxPool
- Bloque 2: Conv2D(64) → BatchNorm → MaxPool
- Bloque 3: Conv2D(128) → BatchNorm → MaxPool
- Capas Densas: 64 → 32 → 1

## Uso

```bash
python run_vision.py
```

## Resultados

- MAE en test: < 0.5 objetos
- Precisión: ±1 objeto
- Modelo: `modelo_contador.keras`
