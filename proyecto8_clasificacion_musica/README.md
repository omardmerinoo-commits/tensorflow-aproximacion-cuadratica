# Proyecto 8: Clasificador de Géneros Musicales

Red neuronal para clasificar géneros musicales usando características de audio extraídas de señales sintéticas.

## Características

- Generador de audio sintético para 3 géneros: Rock, Clásica, Pop
- Extracción de características: MFCC, energía, ZCR, spectral centroid
- Clasificación con red neuronal profunda
- 300 muestras de entrenamiento

## Características Extraídas

- MFCC: 13 coeficientes + desviación estándar
- Energía melespectral
- Zero Crossing Rate (ZCR)
- Spectral Centroid

## Uso

```bash
python run_musica.py
```

## Resultados

- Accuracy esperado: > 90%
- Modelo: `modelo_musica.keras`
