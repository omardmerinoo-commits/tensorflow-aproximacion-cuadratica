"""
Conteo automático de objetos usando visión computacional.

Detecta y cuenta objetos en imágenes usando CNN y procesamiento
de imagen con OpenCV.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict, Any
import json


class GeneradorImagenesSinteticas:
    """Generador de imágenes sintéticas con objetos."""
    
    @staticmethod
    def generar_imagen_con_objetos(num_objetos: int = 5,
                                   tamanio: int = 64,
                                   tamanio_objeto: int = 8) -> Tuple[np.ndarray, int]:
        """
        Generar imagen sintética con círculos.
        
        Parameters
        ----------
        num_objetos : int
            Número de objetos a dibujar
        tamanio : int
            Tamaño de la imagen
        tamanio_objeto : int
            Radio de los círculos
            
        Returns
        -------
        imagen : np.ndarray
            Imagen generada
        count : int
            Número de objetos
        """
        imagen = np.ones((tamanio, tamanio, 3), dtype=np.uint8) * 255
        
        actual_count = 0
        for _ in range(num_objetos):
            x = np.random.randint(tamanio_objeto, tamanio - tamanio_objeto)
            y = np.random.randint(tamanio_objeto, tamanio - tamanio_objeto)
            color = (np.random.randint(0, 256), np.random.randint(0, 256),
                    np.random.randint(0, 256))
            cv2.circle(imagen, (x, y), tamanio_objeto, color, -1)
            actual_count += 1
        
        # Agregar ruido
        ruido = np.random.normal(0, 10, imagen.shape)
        imagen = np.clip(imagen.astype(float) + ruido, 0, 255).astype(np.uint8)
        
        return imagen.astype(np.float32) / 255.0, actual_count
    
    @staticmethod
    def generar_dataset(num_muestras: int = 500,
                       tamanio: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar dataset completo.
        
        Parameters
        ----------
        num_muestras : int
            Número de imágenes
        tamanio : int
            Tamaño de cada imagen
            
        Returns
        -------
        imagenes : np.ndarray
            Array de imágenes
        conteos : np.ndarray
            Conteo para cada imagen
        """
        imagenes = []
        conteos = []
        
        for _ in range(num_muestras):
            num_obj = np.random.randint(0, 15)
            img, count = GeneradorImagenesSinteticas.generar_imagen_con_objetos(
                num_objetos=num_obj, tamanio=tamanio
            )
            imagenes.append(img)
            conteos.append(count)
        
        return np.array(imagenes, dtype=np.float32), np.array(conteos, dtype=np.float32)


class ContadorObjetos:
    """Red neuronal para contar objetos en imágenes."""
    
    def __init__(self, tamanio_imagen: int = 64, seed: int = 42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.tamanio_imagen = tamanio_imagen
        self.modelo = None
        self.history = None
    
    def construir_modelo(self) -> keras.Model:
        """Construir arquitectura CNN."""
        modelo = keras.Sequential([
            layers.Input(shape=(self.tamanio_imagen, self.tamanio_imagen, 3)),
            
            # Bloque 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Bloque 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Bloque 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Flatten
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
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
                 epochs: int = 100, verbose: int = 1) -> Dict[str, Any]:
        """Entrenar."""
        if self.modelo is None:
            self.construir_modelo()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('modelo_contador.keras', save_best_only=True)
        ]
        
        val_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.modelo.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return {
            'epochs': len(self.history.history['loss']),
            'loss_final': float(self.history.history['loss'][-1])
        }
    
    def predecir(self, imagenes: np.ndarray) -> np.ndarray:
        """Predicción."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        predicciones = self.modelo.predict(imagenes, verbose=0)
        return np.round(np.clip(predicciones.flatten(), 0, 15)).astype(int)
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluar."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        loss, mae = self.modelo.evaluate(X_test, y_test, verbose=0)
        return {'loss': float(loss), 'mae': float(mae)}
    
    def guardar_modelo(self, ruta: str = 'modelo_contador.keras') -> None:
        """Guardar."""
        if self.modelo is None:
            raise ValueError("No hay modelo")
        self.modelo.save(ruta)
