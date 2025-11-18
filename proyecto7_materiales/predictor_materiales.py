"""
Predictor de propiedades de materiales usando red neuronal.

Predice propiedades físicas (dureza, conductividad, módulo elástico)
a partir de la composición química.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict, Any
import json


class GeneradorDatosMateriales:
    """Generador de datos sintéticos de materiales."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def generar_composiciones(self, n_muestras: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar composiciones químicas (Fe, Ni, Cu, C, Cr).
        
        Parameters
        ----------
        n_muestras : int
            Número de muestras
            
        Returns
        -------
        X : np.ndarray
            Fracciones en masa (5 elementos)
        y : np.ndarray
            Propiedades (dureza, conductividad, modulo_elastico)
        """
        # Generar 5 elementos que suman 1
        X = np.random.dirichlet([1, 1, 1, 1, 1], n_muestras).astype(np.float32)
        
        # Simular propiedades basadas en composición
        Fe_frac, Ni_frac, Cu_frac, C_frac, Cr_frac = X.T
        
        dureza = (7.0 * Fe_frac + 4.0 * Ni_frac + 3.0 * Cu_frac + 
                 9.0 * C_frac + 8.0 * Cr_frac + np.random.normal(0, 0.5, n_muestras))
        
        conductividad = (10.0 * Cu_frac + 18.0 * Fe_frac + 12.0 * Ni_frac + 
                        1.0 * C_frac + 7.0 * Cr_frac + np.random.normal(0, 1, n_muestras))
        
        modulo = (200.0 * Fe_frac + 220.0 * Ni_frac + 130.0 * Cu_frac + 
                 400.0 * C_frac + 230.0 * Cr_frac + np.random.normal(0, 10, n_muestras))
        
        y = np.column_stack([dureza, conductividad, modulo]).astype(np.float32)
        
        return X, y


class PredictorMateriales:
    """Red neuronal para predicción de propiedades de materiales."""
    
    def __init__(self, input_dim: int = 5, output_dim: int = 3, seed: int = 42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modelo = None
        self.history = None
    
    def construir_modelo(self) -> keras.Model:
        """Construir arquitectura."""
        modelo = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.output_dim, activation='linear')
        ])
        
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
                 epochs: int = 100, batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, Any]:
        """Entrenar modelo."""
        if self.modelo is None:
            self.construir_modelo()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
            ModelCheckpoint('modelo_materiales.keras', save_best_only=True)
        ]
        
        val_data = (X_val, y_val) if X_val is not None else None
        
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
            'loss_final': float(self.history.history['loss'][-1])
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predicción."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        return self.modelo.predict(X, verbose=0)
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluar."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        loss, mae = self.modelo.evaluate(X_test, y_test, verbose=0)
        return {'loss': float(loss), 'mae': float(mae)}
    
    def guardar_modelo(self, ruta: str = 'modelo_materiales.keras') -> None:
        """Guardar modelo."""
        if self.modelo is None:
            raise ValueError("No hay modelo")
        self.modelo.save(ruta)
