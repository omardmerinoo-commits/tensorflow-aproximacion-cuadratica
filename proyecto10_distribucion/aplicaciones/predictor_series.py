"""
Aplicación: Predictor de Series Temporales
==========================================

Caso de uso real: Predicción de series de tiempo (acciones, clima, tráfico)

Características:
- Generación de series temporales sinéticas
- LSTM para predicción secuencial
- Análisis de tendencias
- Métricas de pronóstico

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json


class GeneradorSeriesTemporales:
    """Generador de series temporales."""
    
    @staticmethod
    def generar_tendencia_alcista(longitud=200, volatilidad=0.05):
        """Genera serie con tendencia alcista."""
        t = np.arange(longitud)
        tendencia = 100 + 0.5 * t
        ruido = np.random.randn(longitud) * volatilidad * tendencia
        return tendencia + ruido
    
    @staticmethod
    def generar_tendencia_bajista(longitud=200, volatilidad=0.05):
        """Genera serie con tendencia bajista."""
        t = np.arange(longitud)
        tendencia = 150 - 0.3 * t
        ruido = np.random.randn(longitud) * volatilidad * tendencia
        return tendencia + ruido
    
    @staticmethod
    def generar_estacional(longitud=200, periodo=30):
        """Genera serie con componente estacional."""
        t = np.arange(longitud)
        estacional = 100 + 20 * np.sin(2 * np.pi * t / periodo)
        ruido = np.random.randn(longitud) * 5
        return estacional + ruido
    
    @staticmethod
    def generar_dataset_series(n_series=100, longitud=200, look_back=20):
        """Genera dataset de series temporales."""
        X = []
        y = []
        
        generadores = [
            GeneradorSeriesTemporales.generar_tendencia_alcista,
            GeneradorSeriesTemporales.generar_tendencia_bajista,
            GeneradorSeriesTemporales.generar_estacional
        ]
        
        for _ in range(n_series):
            gen = np.random.choice(generadores)
            serie = gen(longitud + look_back)
            
            # Crear secuencias (look_back -> siguiente valor)
            for i in range(len(serie) - look_back):
                X.append(serie[i:i+look_back])
                y.append(serie[i+look_back])
        
        return {
            'X': np.array(X).astype(np.float32),
            'y': np.array(y).astype(np.float32),
            'look_back': look_back
        }


class PredictorSeriesTemporales:
    """Predictor LSTM para series temporales."""
    
    def __init__(self, seed=42):
        """Inicializa el predictor."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.metricas = {}
    
    def construir_modelo(self, look_back=20):
        """Construye LSTM."""
        self.modelo = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(look_back, 1), return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ LSTM construido para predicción de series")
    
    def normalizar_datos(self, X_train, X_test):
        """Normaliza los datos."""
        X_train_flat = X_train.reshape(-1, 1)
        X_test_flat = X_test.reshape(-1, 1)
        
        X_train_norm = self.scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_norm = self.scaler.transform(X_test_flat).reshape(X_test.shape)
        
        return X_train_norm, X_test_norm
    
    def entrenar(self, X_train, y_train, epochs=15):
        """Entrena el modelo."""
        # Reshape para LSTM: (n_samples, look_back, 1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(-1, 1)
        
        self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo."""
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_test = y_test.reshape(-1, 1)
        
        pérdida, mae = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.modelo.predict(X_test, verbose=0)
        
        # Desnormalizar
        y_test_original = self.scaler.inverse_transform(y_test)
        y_pred_original = self.scaler.inverse_transform(y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        # RMSE
        rmse = np.sqrt(np.mean((y_test_original - y_pred_original) ** 2))
        
        self.metricas = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
        
        print(f"\n📊 Métricas:")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.4f}%")
        
        return self.metricas
    
    def predecir_siguiente(self, secuencia):
        """Predice el siguiente valor."""
        secuencia = np.array(secuencia).reshape(1, -1, 1).astype(np.float32)
        prediccion = self.modelo.predict(secuencia, verbose=0)[0, 0]
        prediccion_original = self.scaler.inverse_transform([[prediccion]])[0, 0]
        return float(prediccion_original)


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("📈 PREDICTOR DE SERIES TEMPORALES - LSTM")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando series temporales...")
    look_back = 20
    datos = GeneradorSeriesTemporales.generar_dataset_series(
        n_series=100,
        longitud=200,
        look_back=look_back
    )
    
    X = datos['X']
    y = datos['y']
    
    print(f"✅ Dataset generado: {X.shape}")
    print(f"   Muestras: {len(X)}")
    print(f"   Look-back: {look_back}")
    print(f"   Rango valores: [{np.min(X):.2f}, {np.max(X):.2f}]")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalizar
    predictor = PredictorSeriesTemporales()
    X_train, X_test = predictor.normalizar_datos(X_train, X_test)
    y_train = predictor.scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = predictor.scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Paso 3: Construir
    print("\n[3] Construyendo LSTM...")
    predictor.construir_modelo(look_back=look_back)
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando...")
    predictor.entrenar(X_train, y_train, epochs=15)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    predictor.evaluar(X_test, y_test)
    
    # Paso 6: Predicciones
    print("\n[6] Realizando predicciones:")
    for i in range(min(5, len(X_test))):
        prediccion = predictor.predecir_siguiente(X_test[i])
        valor_real = predictor.scaler.inverse_transform([[y_test[i]]])[0, 0]
        error = abs(prediccion - valor_real)
        
        print(f"\n   Secuencia {i+1}:")
        print(f"     Predicción: {prediccion:.4f}")
        print(f"     Real: {valor_real:.4f}")
        print(f"     Error: {error:.4f}")
    
    # Paso 7: Reporte
    print("\n[7] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'LSTM Predictor de Series Temporales',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'look_back': look_back,
        'metricas': predictor.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
