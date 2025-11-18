"""
Clasificador de géneros musicales usando características de audio.

Extrae características como MFCC, espectrograma y energía para
clasificar música en géneros.
"""

import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict, Any
import json


class ExtractorCaracteristicasAudio:
    """Extrae características de audio."""
    
    @staticmethod
    def generar_audio_sintetico(duracion: float = 1.0, sr: int = 22050,
                               genero: str = 'rock') -> np.ndarray:
        """
        Generar audio sintético para un género.
        
        Parameters
        ----------
        duracion : float
            Duración en segundos
        sr : int
            Sample rate
        genero : str
            Género musical
            
        Returns
        -------
        y : np.ndarray
            Señal de audio
        """
        n_samples = int(sr * duracion)
        t = np.linspace(0, duracion, n_samples)
        
        if genero == 'rock':
            # Frecuencias bajas y altas
            y = 0.5 * np.sin(2 * np.pi * 100 * t)
            y += 0.3 * np.sin(2 * np.pi * 200 * t)
            y += 0.2 * np.random.randn(n_samples)
            
        elif genero == 'clasica':
            # Más suave, múltiples componentes
            y = 0.3 * np.sin(2 * np.pi * 261 * t)  # Do
            y += 0.3 * np.sin(2 * np.pi * 329 * t)  # Mi
            y += 0.3 * np.sin(2 * np.pi * 392 * t)  # Sol
            y += 0.1 * np.random.randn(n_samples)
            
        else:  # pop
            # Más energía en frecuencias medias
            y = 0.4 * np.sin(2 * np.pi * 150 * t)
            y += 0.4 * np.sin(2 * np.pi * 300 * t)
            y += 0.2 * np.random.randn(n_samples)
        
        return y.astype(np.float32)
    
    @staticmethod
    def extraer_mfcc(y: np.ndarray, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
        """Extraer MFCC."""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    
    @staticmethod
    def extraer_caracteristicas(y: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extraer características completas.
        
        Parameters
        ----------
        y : np.ndarray
            Señal de audio
        sr : int
            Sample rate
            
        Returns
        -------
        caracteristicas : np.ndarray
            Vector de características (20 elementos)
        """
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Energía
        energia = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=1)
        energia_mean = np.mean(energia)
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_mean = np.mean(spec_centroid)
        
        # Combinar todas
        caracteristicas = np.concatenate([
            mfcc_mean, mfcc_std, [energia_mean], [zcr_mean, zcr_std, spec_mean]
        ])
        
        return caracteristicas.astype(np.float32)


class ClasificadorMusica:
    """Red neuronal para clasificación de géneros musicales."""
    
    def __init__(self, input_dim: int = 20, num_generos: int = 3, seed: int = 42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.input_dim = input_dim
        self.num_generos = num_generos
        self.modelo = None
        self.generos = ['Rock', 'Clásica', 'Pop']
    
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
            layers.Dense(self.num_generos, activation='softmax')
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
                 epochs: int = 50, verbose: int = 1) -> Dict[str, Any]:
        """Entrenar."""
        if self.modelo is None:
            self.construir_modelo()
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('modelo_musica.keras', save_best_only=True)
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
            'accuracy_final': float(self.history.history['accuracy'][-1])
        }
    
    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicción."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        probs = self.modelo.predict(X, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluar."""
        if self.modelo is None:
            raise ValueError("Modelo no construido")
        loss, acc = self.modelo.evaluate(X_test, y_test, verbose=0)
        return {'loss': float(loss), 'accuracy': float(acc)}
    
    def guardar_modelo(self, ruta: str = 'modelo_musica.keras') -> None:
        """Guardar."""
        if self.modelo is None:
            raise ValueError("No hay modelo")
        self.modelo.save(ruta)
