"""
P10: PREDICTOR DE SERIES TEMPORALES CON LSTM
Aplicación para predicción de series temporales usando redes neuronales recurrentes LSTM.

Técnica: LSTM (Long Short-Term Memory)
Dataset: Series temporales sintéticas con patrones
Métrica: MAE, RMSE, MAPE
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from pathlib import Path


class GeneradorSeriesTiempo:
    """Generador de series temporales sintéticas"""
    
    @staticmethod
    def generar_dataset(n_samples=500, tipo='uptrend', seed=42):
        """
        Genera serie temporal sintética
        
        Args:
            n_samples: número de muestras
            tipo: 'uptrend', 'downtrend', 'seasonal', 'random'
            seed: semilla para reproducibilidad
        
        Returns:
            np.ndarray: Serie temporal de forma (n_samples, 1)
        """
        np.random.seed(seed)
        t = np.arange(n_samples)
        
        if tipo == 'uptrend':
            serie = 0.1 * t + np.random.normal(0, 5, n_samples)
        elif tipo == 'downtrend':
            serie = -0.05 * t + np.random.normal(0, 5, n_samples)
        elif tipo == 'seasonal':
            serie = 50 + 20 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 3, n_samples)
        else:  # random walk
            serie = np.cumsum(np.random.normal(0, 1, n_samples))
        
        return serie.reshape(-1, 1).astype(np.float32)
    
    @staticmethod
    def crear_secuencias(data, lookback=20):
        """Crea secuencias para LSTM"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)


class PredictorSeries:
    """Predictor de series temporales con LSTM"""
    
    def __init__(self, lookback=20):
        self.modelo = None
        self.lookback = lookback
        self.historial = None
        
    def construir_modelo(self):
        """Construye modelo LSTM"""
        self.modelo = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(self.lookback, 1), 
                       return_sequences=True, name='lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False, name='lstm_2'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dense(1, activation='linear', name='salida')
        ], name='PredictorSeriesTiempo')
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def entrenar(self, X_train, y_train, epochs=50, batch_size=32):
        """Entrena el modelo"""
        self.historial = self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def predecir(self, X):
        """Realiza predicciones"""
        return self.modelo.predict(X, verbose=0)


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(" "*10 + "P10: PREDICTOR DE SERIES TEMPORALES CON LSTM")
    print("="*70 + "\n")
    
    # Generar datos
    print("[1/7] Generando serie temporal...")
    generador = GeneradorSeriesTiempo()
    serie = generador.generar_dataset(n_samples=500, tipo='seasonal', seed=42)
    print(f"     Serie de {len(serie)} puntos generada")
    
    # Normalizar
    print("[2/7] Normalizando datos...")
    media = np.mean(serie)
    std = np.std(serie)
    serie_norm = (serie - media) / std
    print(f"     Media: {media:.4f}, Std: {std:.4f}")
    
    # Crear secuencias
    print("[3/7] Creando secuencias...")
    predictor = PredictorSeries(lookback=20)
    X, y = GeneradorSeriesTiempo.crear_secuencias(serie_norm, lookback=20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"     Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Construir modelo
    print("[4/7] Construyendo modelo LSTM...")
    predictor.construir_modelo()
    predictor.modelo.summary()
    
    # Entrenar
    print("[5/7] Entrenando modelo...")
    predictor.entrenar(X_train, y_train, epochs=30, batch_size=16)
    print("     Entrenamiento completado")
    
    # Evaluar
    print("[6/7] Evaluando modelo...")
    y_pred_train = predictor.predecir(X_train).flatten()
    y_pred_test = predictor.predecir(X_test).flatten()
    
    mae_train = np.mean(np.abs(y_train - y_pred_train))
    mae_test = np.mean(np.abs(y_test - y_pred_test))
    rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
    
    print(f"     MAE Train: {mae_train:.6f}")
    print(f"     MAE Test:  {mae_test:.6f}")
    print(f"     RMSE Test: {rmse_test:.6f}")
    
    # Ejemplo de predicción
    print("[7/7] Realizando predicciones de ejemplo...")
    if len(X_test) > 0:
        pred_ejemplo = predictor.predecir(X_test[:5])
        for i in range(min(5, len(X_test))):
            print(f"     Valor real: {y_test[i]:.4f}, Predicción: {pred_ejemplo[i,0]:.4f}")
    
    # Guardar resultados
    reporte = {
        "proyecto": "P10 - Predictor de Series Temporales",
        "tecnica": "LSTM",
        "fecha": datetime.now().isoformat(),
        "configuracion": {
            "lookback": predictor.lookback,
            "epochs": 30,
            "batch_size": 16,
            "split": "80/20"
        },
        "metricas": {
            "mae_train": float(mae_train),
            "mae_test": float(mae_test),
            "rmse_test": float(rmse_test),
            "muestras_train": len(X_train),
            "muestras_test": len(X_test)
        },
        "modelo": {
            "parametros": int(predictor.modelo.count_params()),
            "capas": "LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1)"
        }
    }
    
    # Guardar reporte
    os.makedirs('reportes', exist_ok=True)
    with open('reportes/reporte_p10.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "="*70)
    print("[OK] Aplicación P10 completada correctamente")
    print(f"[REPORTE] Guardado en: reportes/reporte_p10.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
