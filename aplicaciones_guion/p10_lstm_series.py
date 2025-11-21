#!/usr/bin/env python3
"""
P10: Predictor de Series Temporales - LSTM
Predecir valores futuros basado en historial pasado
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class GeneradorSeriesTemporales:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_timesteps=500, seq_length=30):
        """Generar series temporales sinteticas"""
        # Crear serie con tendencia + estacionalidad + ruido
        t = np.arange(n_timesteps)
        tendencia = 0.1 * t
        estacionalidad = 10 * np.sin(2 * np.pi * t / 50)
        ruido = np.random.normal(0, 2, n_timesteps)
        serie = tendencia + estacionalidad + ruido
        
        # Crear secuencias (ventana deslizante)
        X, y = [], []
        for i in range(n_timesteps - seq_length):
            X.append(serie[i:i+seq_length])
            y.append(serie[i+seq_length])
        
        return {
            'X': np.array(X).reshape(-1, seq_length, 1).astype('float32'),
            'y': np.array(y).astype('float32')
        }


class PredictorLSTM:
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
    
    def construir_modelo(self, seq_length=30):
        """Modelo LSTM para prediccion de series"""
        self.modelo = keras.Sequential([
            keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss='mse',
            metrics=['mae']
        )
        print("[+] Modelo LSTM P10 construido")
    
    def entrenar(self, X_train, y_train, epochs=20):
        self.modelo.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=16)
    
    def evaluar(self, X_test, y_test):
        y_pred = self.modelo.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {'mae': float(mae), 'rmse': float(rmse)}


def main():
    print("\n" + "="*60)
    print("P10: PREDICTOR DE SERIES TEMPORALES (LSTM)")
    print("="*60)
    
    generador = GeneradorSeriesTemporales()
    datos = generador.generar_dataset(500, seq_length=30)
    
    n_train = int(0.8 * len(datos['X']))
    X_train, X_test = datos['X'][:n_train], datos['X'][n_train:]
    y_train, y_test = datos['y'][:n_train], datos['y'][n_train:]
    
    modelo = PredictorLSTM()
    modelo.construir_modelo(seq_length=30)
    modelo.entrenar(X_train, y_train, epochs=20)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"MAE: {metricas['mae']:.4f}")
    print(f"RMSE: {metricas['rmse']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
