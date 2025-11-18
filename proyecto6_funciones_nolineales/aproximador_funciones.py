"""
Aproximador de funciones no lineales complejas usando redes neuronales.

Este módulo implementa un modelo para aprender funciones complejas como
sin(x)*exp(-x/10), cos(x²), y otras funciones no lineales.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Callable, Tuple, Optional, Dict, Any
import json


class GeneradorFuncionesNoLineales:
    """Generador de datos para funciones no lineales."""
    
    @staticmethod
    def funcion_exponencial_amortiguada(x: np.ndarray) -> np.ndarray:
        """sin(x) * exp(-x/10)"""
        return np.sin(x) * np.exp(-x / 10.0)
    
    @staticmethod
    def funcion_polinomica_compleja(x: np.ndarray) -> np.ndarray:
        """x³ - 2x² + 3x - 1"""
        return x**3 - 2*x**2 + 3*x - 1
    
    @staticmethod
    def funcion_trigonometrica(x: np.ndarray) -> np.ndarray:
        """sin(x) * cos(x/2)"""
        return np.sin(x) * np.cos(x / 2.0)
    
    @staticmethod
    def funcion_logaritmica(x: np.ndarray) -> np.ndarray:
        """log(1 + x²) * sin(x)"""
        return np.log(1.0 + x**2) * np.sin(x)
    
    @staticmethod
    def generar_datos(funcion: Callable,
                     x_min: float = -10.0,
                     x_max: float = 10.0,
                     n_samples: int = 500,
                     ruido: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar datos de entrenamiento.
        
        Parameters
        ----------
        funcion : Callable
            Función a aproximar
        x_min : float
            Límite inferior
        x_max : float
            Límite superior
        n_samples : int
            Número de muestras
        ruido : float
            Desviación estándar del ruido Gaussiano
            
        Returns
        -------
        X : np.ndarray
            Valores de entrada
        y : np.ndarray
            Valores de salida
        """
        X = np.random.uniform(x_min, x_max, n_samples).reshape(-1, 1).astype(np.float32)
        y = funcion(X.flatten()).reshape(-1, 1).astype(np.float32)
        
        if ruido > 0:
            y += np.random.normal(0, ruido, y.shape)
        
        # Ordenar por X
        idx = np.argsort(X.flatten())
        X = X[idx]
        y = y[idx]
        
        return X, y


class AproximadorFuncionesNoLineales:
    """Red neuronal para aproximar funciones no lineales."""
    
    def __init__(self, seed: int = 42):
        """
        Inicializar aproximador.
        
        Parameters
        ----------
        seed : int
            Semilla para reproducibilidad
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.history = None
        self.scaler_x = None
        self.scaler_y = None
    
    def construir_modelo(self, capas: list = None) -> keras.Model:
        """
        Construir arquitectura de red.
        
        Parameters
        ----------
        capas : list
            Número de neuronas por capa
            
        Returns
        -------
        modelo : keras.Model
            Modelo compilado
        """
        if capas is None:
            capas = [128, 128, 64, 32, 16]
        
        modelo = keras.Sequential()
        modelo.add(layers.Input(shape=(1,)))
        
        for capa in capas[:-1]:
            modelo.add(layers.Dense(capa, activation='relu'))
            modelo.add(layers.BatchNormalization())
            modelo.add(layers.Dropout(0.2))
        
        modelo.add(layers.Dense(capas[-1], activation='relu'))
        modelo.add(layers.Dense(1, activation='linear'))
        
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.modelo = modelo
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 epochs: int = 200,
                 batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, Any]:
        """
        Entrenar el modelo.
        
        Parameters
        ----------
        X_train : np.ndarray
            Datos de entrenamiento
        y_train : np.ndarray
            Valores objetivo
        X_val : np.ndarray, optional
            Datos de validación
        y_val : np.ndarray, optional
            Valores objetivo de validación
        epochs : int
            Número de épocas
        batch_size : int
            Tamaño de lote
        verbose : int
            Nivel de verbosidad
            
        Returns
        -------
        history : Dict
            Historial de entrenamiento
        """
        if self.modelo is None:
            self.construir_modelo()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=30, restore_best_weights=True),
            ModelCheckpoint('modelo_funciones.keras', save_best_only=True, monitor='loss')
        ]
        
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        
        self.history = self.modelo.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return {
            'epochs': len(self.history.history['loss']),
            'loss_final': float(self.history.history['loss'][-1]),
            'mae_final': float(self.history.history['mae'][-1])
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realizar predicciones.
        
        Parameters
        ----------
        X : np.ndarray
            Datos para predecir
            
        Returns
        -------
        predicciones : np.ndarray
            Valores predichos
        """
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        
        return self.modelo.predict(X, verbose=0)
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluar en datos de test.
        
        Parameters
        ----------
        X_test : np.ndarray
            Datos de prueba
        y_test : np.ndarray
            Valores objetivo
            
        Returns
        -------
        metricas : Dict
            Loss y MAE
        """
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        
        loss, mae = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': float(loss),
            'mae': float(mae)
        }
    
    def guardar_modelo(self, ruta: str = 'modelo_funciones.keras') -> None:
        """Guardar modelo."""
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        self.modelo.save(ruta)
    
    def cargar_modelo(self, ruta: str = 'modelo_funciones.keras') -> None:
        """Cargar modelo."""
        self.modelo = keras.models.load_model(ruta)
