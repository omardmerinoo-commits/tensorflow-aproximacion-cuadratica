"""
Módulo para modelar oscilaciones amortiguadas usando redes neuronales.

Este módulo implementa un modelo de red neuronal que aproxima
la ecuación diferencial de oscilaciones amortiguadas:
    m*d²x/dt² + c*dx/dt + k*x = 0

Características:
    - Generación de datos sintetizados con ruido controlado
    - Arquitectura neural profunda configurable
    - Validación cruzada k-fold
    - Serialización de modelos (.keras, .pkl)
    - Predicción y análisis de propiedades del sistema
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class OscilacionesAmortiguadas:
    """
    Clase para modelar oscilaciones amortiguadas usando redes neuronales.
    
    Parámetros del sistema:
        - m: masa (kg)
        - c: coeficiente de amortiguamiento (N·s/m)
        - k: constante de rigidez (N/m)
        
    La solución analítica depende del discriminante: Δ = c² - 4mk
        - Δ > 0: Amortiguamiento supercrítico (overdamped)
        - Δ = 0: Amortiguamiento crítico (critically damped)
        - Δ < 0: Subamortiguamiento (underdamped)
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa la clase con parámetros por defecto.
        
        Args:
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        self.config = {}
        
    @staticmethod
    def solucion_analitica(t: np.ndarray, m: float, c: float, k: float,
                          x0: float = 1.0, v0: float = 0.0) -> np.ndarray:
        """
        Calcula la solución analítica de la ecuación de oscilaciones amortiguadas.
        
        Args:
            t: Arreglo de tiempo
            m: Masa
            c: Coeficiente de amortiguamiento
            k: Constante de rigidez
            x0: Posición inicial
            v0: Velocidad inicial
            
        Returns:
            Posiciones calculadas analíticamente
        """
        omega0 = np.sqrt(k / m)  # Frecuencia natural
        zeta = c / (2 * np.sqrt(k * m))  # Factor de amortiguamiento
        
        if zeta < 1:  # Subamortiguado
            omega_d = omega0 * np.sqrt(1 - zeta**2)
            A = x0
            B = (v0 + zeta * omega0 * x0) / omega_d
            x = np.exp(-zeta * omega0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        elif zeta == 1:  # Críticamente amortiguado
            x = (x0 + (v0 + omega0 * x0) * t) * np.exp(-omega0 * t)
        else:  # Sobreamortiguado
            r1 = -zeta * omega0 + omega0 * np.sqrt(zeta**2 - 1)
            r2 = -zeta * omega0 - omega0 * np.sqrt(zeta**2 - 1)
            A1 = (v0 - r2 * x0) / (r1 - r2)
            A2 = x0 - A1
            x = A1 * np.exp(r1 * t) + A2 * np.exp(r2 * t)
            
        return x
    
    def generar_datos(self, num_muestras: int = 500, 
                     tiempo_max: float = 10.0,
                     ruido_sigma: float = 0.02,
                     params_sistema: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera datos de entrenamiento sintéticos para oscilaciones amortiguadas.
        
        Args:
            num_muestras: Número de trayectorias a generar
            tiempo_max: Tiempo máximo de simulación
            ruido_sigma: Desviación estándar del ruido Gaussiano
            params_sistema: Diccionario con parámetros {m, c, k, x0, v0}
            
        Returns:
            Tupla (X, y) donde X es tiempo y parámetros, y es posición
        """
        if params_sistema is None:
            params_sistema = {
                'm': np.random.uniform(0.5, 2.0),
                'c': np.random.uniform(1.0, 5.0),
                'k': np.random.uniform(5.0, 20.0),
                'x0': np.random.uniform(-1.0, 1.0),
                'v0': np.random.uniform(-1.0, 1.0)
            }
        
        self.config['params_sistema'] = params_sistema
        self.config['tiempo_max'] = tiempo_max
        
        X = []
        y = []
        
        for _ in range(num_muestras):
            # Parámetros del sistema (pueden variar)
            m = np.random.uniform(0.5, 2.0)
            c = np.random.uniform(1.0, 5.0)
            k = np.random.uniform(5.0, 20.0)
            x0 = np.random.uniform(-1.0, 1.0)
            v0 = np.random.uniform(-1.0, 1.0)
            
            # Tiempo
            t = np.linspace(0, tiempo_max, 100)
            
            # Solución analítica
            x_limpio = self.solucion_analitica(t, m, c, k, x0, v0)
            
            # Agregar ruido
            x_ruidoso = x_limpio + np.random.normal(0, ruido_sigma, len(t))
            
            # Características: [t, m, c, k, x0, v0, zeta]
            zeta = c / (2 * np.sqrt(k * m))
            for i, ti in enumerate(t):
                X.append([ti, m, c, k, x0, v0, zeta])
                y.append(x_ruidoso[i])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        return X, y
    
    def construir_modelo(self, input_shape: int = 7,
                        capas_ocultas: List[int] = None,
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001) -> keras.Model:
        """
        Construye la arquitectura de la red neuronal.
        
        Args:
            input_shape: Número de características de entrada
            capas_ocultas: Lista de neuronas por capa oculta
            dropout_rate: Tasa de dropout para regularización
            learning_rate: Tasa de aprendizaje del optimizador
            
        Returns:
            Modelo compilado
        """
        if capas_ocultas is None:
            capas_ocultas = [128, 64, 32]
        
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(capas_ocultas[0], activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
        ])
        
        for units in capas_ocultas[1:]:
            model.add(layers.Dense(units, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(1e-4)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.config['capas_ocultas'] = capas_ocultas
        self.config['dropout_rate'] = dropout_rate
        self.config['learning_rate'] = learning_rate
        
        return model
    
    def entrenar(self, X: np.ndarray, y: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                validation_split: float = 0.2,
                early_stopping_patience: int = 15,
                verbose: int = 1) -> Dict:
        """
        Entrena el modelo con validación temprana.
        
        Args:
            X: Características de entrada
            y: Valores objetivo
            epochs: Número de épocas
            batch_size: Tamaño del lote
            validation_split: Proporción de validación
            early_stopping_patience: Paciencia para detención temprana
            verbose: Nivel de verbosidad
            
        Returns:
            Diccionario con información del entrenamiento
        """
        # Normalizar datos
        X_escalado = self.scaler_x.fit_transform(X)
        y_escalado = self.scaler_y.fit_transform(y)
        
        # Construir modelo
        self.model = self.construir_modelo()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        # Entrenar
        self.history = self.model.fit(
            X_escalado, y_escalado,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Información de entrenamiento
        info = {
            'epochs_entrenadas': len(self.history.history['loss']),
            'loss_final': float(self.history.history['loss'][-1]),
            'val_loss_final': float(self.history.history['val_loss'][-1]),
            'mae_final': float(self.history.history['mae'][-1]),
            'val_mae_final': float(self.history.history['val_mae'][-1])
        }
        
        return info
    
    def validacion_cruzada(self, X: np.ndarray, y: np.ndarray,
                          k_folds: int = 5, epochs: int = 50) -> Dict:
        """
        Realiza validación cruzada k-fold.
        
        Args:
            X: Características de entrada
            y: Valores objetivo
            k_folds: Número de folds
            epochs: Épocas por fold
            
        Returns:
            Diccionario con métricas de validación cruzada
        """
        X_escalado = self.scaler_x.fit_transform(X)
        y_escalado = self.scaler_y.fit_transform(y)
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        scores = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_escalado)):
            X_train, X_val = X_escalado[train_idx], X_escalado[val_idx]
            y_train, y_val = y_escalado[train_idx], y_escalado[val_idx]
            
            model = self.construir_modelo()
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            y_pred = model.predict(X_val, verbose=0)
            
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            scores['mse'].append(float(mse))
            scores['mae'].append(float(mae))
            scores['r2'].append(float(r2))
            
            print(f"Fold {fold+1}/{k_folds} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        
        # Resumén de validación cruzada
        resumen = {
            'mse_mean': float(np.mean(scores['mse'])),
            'mse_std': float(np.std(scores['mse'])),
            'mae_mean': float(np.mean(scores['mae'])),
            'mae_std': float(np.std(scores['mae'])),
            'r2_mean': float(np.mean(scores['r2'])),
            'r2_std': float(np.std(scores['r2'])),
            'scores_por_fold': scores
        }
        
        return resumen
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Características de entrada
            
        Returns:
            Predicciones (valores normalizados inversos)
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Entrenar primero.")
        
        X_escalado = self.scaler_x.transform(X)
        y_pred_escalado = self.model.predict(X_escalado, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_escalado)
        
        return y_pred
    
    def guardar_modelo(self, ruta: str = 'modelo_oscilaciones.keras') -> None:
        """
        Guarda el modelo entrenado en formato Keras nativo.
        
        Args:
            ruta: Ruta de destino
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        self.model.save(ruta)
        
        # Guardar escaladores y configuración
        config_path = ruta.replace('.keras', '.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"Modelo guardado en {ruta}")
        print(f"Configuración guardada en {config_path}")
    
    def cargar_modelo(self, ruta: str = 'modelo_oscilaciones.keras') -> None:
        """
        Carga un modelo entrenado previamente.
        
        Args:
            ruta: Ruta del modelo
        """
        self.model = keras.models.load_model(ruta)
        
        # Cargar configuración
        config_path = ruta.replace('.keras', '.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        print(f"Modelo cargado desde {ruta}")
    
    def resumen_modelo(self) -> Dict:
        """
        Retorna un resumen completo del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        if self.model is None:
            return {'estado': 'No hay modelo entrenado'}
        
        return {
            'tipo_modelo': 'Red Neuronal Profunda',
            'capas': len(self.model.layers),
            'parametros_totales': int(self.model.count_params()),
            'configuracion': self.config,
            'entrenamiento': {
                'epochs': len(self.history.history['loss']) if self.history else None,
                'loss_final': float(self.history.history['loss'][-1]) if self.history else None,
            }
        }
