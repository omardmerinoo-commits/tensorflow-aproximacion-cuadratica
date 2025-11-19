"""
Proyecto 10: Análisis y Pronóstico de Series Temporales
=======================================================

Sistema para predicción de series temporales multivariadas usando
ARIMA (univariado) y LSTM (multivariado).

Aplicaciones:
- Predicción de precios de acciones
- Pronóstico de temperatura
- Demanda de energía
- Datos de sensores

Características:
- Generación sintética realista de series temporales
- Descomposición: tendencia, estacionalidad, ruido
- Modelos ARIMA(p,d,q) para series estacionarias
- LSTM bidireccional para capturar dependencias largas
- Validación temporal sin shuffle
- Métricas: MAE, RMSE, MAPE, R²

Teórico:
ARIMA modela dependencias lineales hasta el pasado cercano.
LSTM captura patrones no-lineales y dependencias largas.
Hybrid ARIMA-LSTM combina ambos: ARIMA residuos + LSTM.

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosSeriesTemporales:
    """Contenedor para datos de series temporales"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    serie_original: np.ndarray
    nombres_features: List[str]
    
    def info(self) -> str:
        return (f"Series Temporales: Train {self.X_train.shape}, "
                f"Val {self.X_val.shape}, Test {self.X_test.shape}")


class GeneradorSeriesTemporales:
    """Generador de series temporales sintéticas realistas"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def _tendencia(self, t: np.ndarray, tipo: str = 'lineal') -> np.ndarray:
        """Genera componente de tendencia"""
        if tipo == 'lineal':
            return 0.1 * t
        elif tipo == 'cuadratica':
            return 0.001 * t**2
        elif tipo == 'exponencial':
            return 0.01 * np.exp(0.001 * t)
        else:
            return np.zeros_like(t)
    
    def _estacionalidad(self, t: np.ndarray, periodo: int = 12) -> np.ndarray:
        """Genera componente de estacionalidad"""
        amplitud = 2.0
        return amplitud * np.sin(2 * np.pi * t / periodo)
    
    def _generar_arima(self, n_puntos: int, p: int = 2, d: int = 1, 
                       q: int = 1) -> np.ndarray:
        """Genera serie con proceso ARIMA(p,d,q)"""
        # Coeficientes AR y MA
        phi = np.random.uniform(0.3, 0.7, p)  # AR
        theta = np.random.uniform(-0.5, 0.5, q)  # MA
        
        # Inicialización
        y = np.zeros(n_puntos)
        epsilon = np.random.normal(0, 1, n_puntos)
        
        for t in range(max(p, q), n_puntos):
            # Componente AR
            ar_term = sum(phi[i] * y[t-1-i] for i in range(min(p, t)))
            # Componente MA
            ma_term = sum(theta[i] * epsilon[t-1-i] for i in range(min(q, t)))
            # Innovación
            y[t] = ar_term + ma_term + epsilon[t]
        
        return y
    
    def generar(self, n_puntos: int = 500, n_series: int = 1,
               tipo: str = 'tendencia_estacional') -> Tuple[np.ndarray, List[str]]:
        """
        Genera series temporales multivariadas
        
        Args:
            n_puntos: Número de puntos temporales
            n_series: Número de series (variables)
            tipo: 'tendencia_estacional', 'ruido_blanco', 'arima'
        
        Returns:
            Serie [n_puntos, n_series] y nombres
        """
        t = np.arange(n_puntos)
        series_list = []
        nombres = []
        
        for i in range(n_series):
            if tipo == 'tendencia_estacional':
                # Componente base
                serie = self._tendencia(t, 'lineal')
                serie += self._estacionalidad(t, periodo=20 + i*5)
                # Ruido
                serie += np.random.normal(0, 0.5, n_puntos)
                nombres.append(f'Variable_{i+1}_TendEst')
                
            elif tipo == 'arima':
                serie = self._generar_arima(n_puntos)
                nombres.append(f'ARIMA_{i+1}')
                
            elif tipo == 'ruido_blanco':
                serie = np.random.normal(0, 1, n_puntos)
                nombres.append(f'Ruido_{i+1}')
            
            series_list.append(serie)
        
        return np.column_stack(series_list), nombres
    
    def generar_dataset(self, n_puntos: int = 500, n_series: int = 2,
                       ventana: int = 10,
                       split: Tuple[float, float, float] = (0.6, 0.2, 0.2)
                       ) -> DatosSeriesTemporales:
        """
        Genera dataset con ventanas deslizantes
        
        Args:
            n_puntos: Longitud serie
            n_series: Número de variables
            ventana: Tamaño de ventana (lags)
            split: (train_ratio, val_ratio, test_ratio)
        
        Returns:
            DatosSeriesTemporales
        """
        serie, nombres = self.generar(n_puntos, n_series)
        
        # Crear ventanas deslizantes
        X, y = [], []
        for t in range(len(serie) - ventana):
            X.append(serie[t:t+ventana])
            y.append(serie[t+ventana])
        
        X, y = np.array(X), np.array(y)
        
        # Split temporal (sin shuffle)
        train_ratio, val_ratio, test_ratio = split
        n_train = int(len(X) * train_ratio)
        n_val = int(len(X) * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        return DatosSeriesTemporales(
            X_train, y_train, X_val, y_val, X_test, y_test,
            serie, nombres
        )


class PronostadorSeriesTemporales:
    """Pronóstico con LSTM bidireccional"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.entrenado = False
    
    def _normalizar(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza a [0, 1]"""
        shape_original = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        if fit:
            X_norm = self.scaler.fit_transform(X_flat)
        else:
            X_norm = self.scaler.transform(X_flat)
        
        return X_norm.reshape(shape_original)
    
    def _desnormalizar(self, X_norm: np.ndarray) -> np.ndarray:
        """Invierte normalización"""
        shape_original = X_norm.shape
        X_flat = X_norm.reshape(X_norm.shape[0], -1)
        X = self.scaler.inverse_transform(X_flat)
        return X.reshape(shape_original)
    
    def construir_lstm(self, input_shape: Tuple[int, ...],
                      output_shape: int = 1) -> models.Model:
        """
        Construye LSTM bidireccional para series
        
        Input: [ventana, n_series]
        Output: [n_series] (predicción siguiente)
        """
        modelo = models.Sequential([
            layers.Input(shape=input_shape),
            
            # LSTM bidireccional 1
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # LSTM bidireccional 2
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Salida
            layers.Dense(output_shape)
        ])
        
        return modelo
    
    def construir_cnn_lstm(self, input_shape: Tuple[int, ...],
                          output_shape: int = 1) -> models.Model:
        """
        Construye híbrido CNN-LSTM
        Extrae características locales con CNN, luego LSTM
        """
        modelo = models.Sequential([
            layers.Input(shape=input_shape),
            
            # CNN 1D para características locales
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # LSTM bidireccional
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2)
            ),
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # Dense
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(output_shape)
        ])
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 50, arquitectura: str = 'lstm',
                 verbose: int = 1) -> Dict:
        """Entrena el modelo"""
        X_train_norm = self._normalizar(X_train, fit=True)
        X_val_norm = self._normalizar(X_val)
        y_train_norm = self._normalizar(y_train, fit=True)
        y_val_norm = self._normalizar(y_val)
        
        input_shape = X_train_norm.shape[1:]
        output_shape = y_train_norm.shape[1] if y_train_norm.ndim > 1 else 1
        
        if arquitectura == 'lstm':
            self.modelo = self.construir_lstm(input_shape, output_shape)
        elif arquitectura == 'cnn_lstm':
            self.modelo = self.construir_cnn_lstm(input_shape, output_shape)
        else:
            raise ValueError(f"Arquitectura inválida: {arquitectura}")
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
            )
        ]
        
        hist = self.modelo.fit(
            X_train_norm, y_train_norm,
            validation_data=(X_val_norm, y_val_norm),
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            batch_size=16
        )
        
        self.entrenado = True
        return hist.history
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evalúa el modelo"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        X_test_norm = self._normalizar(X_test)
        y_test_norm = self._normalizar(y_test)
        
        y_pred_norm = self.modelo.predict(X_test_norm, verbose=0)
        y_pred = self._desnormalizar(y_pred_norm)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred, multioutput='raw_values' if y_test.ndim > 1 else 'uniform_average')
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'predicciones': y_pred,
            'residuos': y_test - y_pred
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        X_norm = self._normalizar(X)
        y_pred_norm = self.modelo.predict(X_norm, verbose=0)
        return self._desnormalizar(y_pred_norm)
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(f"{ruta}_modelo.h5")
        import pickle
        with open(f"{ruta}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @staticmethod
    def cargar(ruta: str) -> 'PronostadorSeriesTemporales':
        """Carga un modelo guardado"""
        import pickle
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.modelo = keras.models.load_model(f"{ruta}_modelo.h5")
        with open(f"{ruta}_scaler.pkl", 'rb') as f:
            pronosticador.scaler = pickle.load(f)
        pronosticador.entrenado = True
        return pronosticador


def demo():
    """Demostración completa"""
    print("="*70)
    print("PRONOSTICADOR DE SERIES TEMPORALES - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Generar datos
    print("\n[1] Generando series temporales...")
    generador = GeneradorSeriesTemporales()
    datos = generador.generar_dataset(n_puntos=500, n_series=2, ventana=10)
    print(f"✓ {datos.info()}")
    print(f"  Series: {', '.join(datos.nombres_features)}")
    
    # 2. Entrenar LSTM
    print("\n[2] Entrenando LSTM bidireccional...")
    pronosticador = PronostadorSeriesTemporales()
    pronosticador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20, arquitectura='lstm', verbose=0
    )
    metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
    print(f"✓ RMSE: {metricas['rmse']:.4f}")
    print(f"  MAE: {metricas['mae']:.4f}")
    print(f"  MAPE: {metricas['mape']:.2f}%")
    
    # 3. CNN-LSTM
    print("\n[3] Entrenando híbrido CNN-LSTM...")
    pronosticador_cnn = PronostadorSeriesTemporales()
    pronosticador_cnn.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20, arquitectura='cnn_lstm', verbose=0
    )
    metricas_cnn = pronosticador_cnn.evaluar(datos.X_test, datos.y_test)
    print(f"✓ CNN-LSTM RMSE: {metricas_cnn['rmse']:.4f}")
    
    # 4. Predicciones
    print("\n[4] Predicciones futuras:")
    y_pred = pronosticador.predecir(datos.X_test[:5])
    print(f"Primeras 5 predicciones:")
    for i in range(5):
        real = datos.y_test[i]
        pred = y_pred[i]
        error = np.abs(real - pred)
        print(f"  {i}: Real={real:.2f}, Pred={pred:.2f}, Error={error:.4f}")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
