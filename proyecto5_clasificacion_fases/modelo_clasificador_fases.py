"""
Red neuronal para clasificación de fases de la materia.

Implementa un modelo de red neuronal profunda (MLP) para clasificar
datos físicos en tres categorías: sólido, líquido o gas.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict, Any
import json
import pickle


class ModeloClasificadorFases:
    """
    Red neuronal para clasificación de fases.
    
    Modelo MLP con capas densas, normalización y regularización
    para clasificar datos físicos en tres fases del material.
    """
    
    def __init__(self, input_dim: int = 5, num_classes: int = 3, seed: int = 42):
        """
        Inicializar modelo de clasificación.
        
        Parameters
        ----------
        input_dim : int
            Dimensión de entrada (características)
        num_classes : int
            Número de clases (3: sólido, líquido, gas)
        seed : int
            Semilla para reproducibilidad
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.modelo = None
        self.history = None
        self.etiquetas_clases = ['Sólido', 'Líquido', 'Gas']
    
    def construir_modelo(self) -> keras.Model:
        """
        Construir arquitectura de red neuronal.
        
        Returns
        -------
        modelo : keras.Model
            Modelo compilado y listo para entrenar
        """
        modelo = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # Capa 1
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Capa 2
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Capa 3
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Capa 4
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Capa de salida
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.modelo = modelo
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None,
                 epochs: int = 100, batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, Any]:
        """
        Entrenar el modelo.
        
        Parameters
        ----------
        X_train : np.ndarray
            Datos de entrenamiento
        y_train : np.ndarray
            Etiquetas de entrenamiento
        X_val : np.ndarray, optional
            Datos de validación
        y_val : np.ndarray, optional
            Etiquetas de validación
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
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
            ModelCheckpoint('modelo_fases.keras', save_best_only=True, monitor='loss')
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
            'accuracy_final': float(self.history.history['accuracy'][-1])
        }
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluar modelo en datos de prueba.
        
        Parameters
        ----------
        X_test : np.ndarray
            Datos de prueba
        y_test : np.ndarray
            Etiquetas de prueba
            
        Returns
        -------
        metricas : Dict
            Pérdida y precisión
        """
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        
        loss, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realizar predicciones.
        
        Parameters
        ----------
        X : np.ndarray
            Datos para predecir
            
        Returns
        -------
        predicciones : np.ndarray
            Clases predichas
        probabilidades : np.ndarray
            Probabilidades por clase
        """
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        
        probabilidades = self.modelo.predict(X, verbose=0)
        predicciones = np.argmax(probabilidades, axis=1)
        
        return predicciones, probabilidades
    
    def guardar_modelo(self, ruta: str = 'modelo_fases.keras') -> None:
        """
        Guardar modelo entrenado.
        
        Parameters
        ----------
        ruta : str
            Ruta para guardar el modelo
        """
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        self.modelo.save(ruta)
    
    def cargar_modelo(self, ruta: str = 'modelo_fases.keras') -> None:
        """
        Cargar modelo guardado.
        
        Parameters
        ----------
        ruta : str
            Ruta del modelo guardado
        """
        self.modelo = keras.models.load_model(ruta)
    
    def guardar_config(self, ruta: str = 'config_fases.json') -> None:
        """
        Guardar configuración del modelo.
        
        Parameters
        ----------
        ruta : str
            Ruta para guardar configuración
        """
        config = {
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'etiquetas_clases': self.etiquetas_clases
        }
        
        with open(ruta, 'w') as f:
            json.dump(config, f, indent=2)
