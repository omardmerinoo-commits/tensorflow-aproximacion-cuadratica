#!/usr/bin/env python3
"""
P5: Compresor de Imagenes - PCA + Autoencoder Convolucional
Comprimir imagenes 28x28 a espacio latente pequeno
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


class GeneradorImagenes:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=200):
        """Generar imagenes sinteticas 28x28"""
        # Generar patrones simples
        imagenes = []
        for i in range(n_samples):
            img = np.random.randn(28, 28) * 0.3 + 0.5
            # Agregar formas
            y, x = np.ogrid[:28, :28]
            mask = (x - 14)**2 + (y - 14)**2 <= (5 + i % 10)**2
            img[mask] = 0.9
            imagenes.append(img)
        
        return np.array(imagenes).reshape(n_samples, 28, 28, 1).astype('float32')


class AutoencoderConvolucional:
    def __init__(self):
        self.encoder = None
        self.autoencoder = None
    
    def construir_modelo(self):
        """Autoencoder convolucional"""
        # Encoder
        entrada = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(entrada)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.autoencoder = keras.Model(entrada, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        self.encoder = keras.Model(entrada, encoded)
        print("[+] Autoencoder convolucional P5 construido")
    
    def entrenar(self, X, epochs=15):
        self.autoencoder.fit(X, X, epochs=epochs, verbose=0)
    
    def evaluar(self, X):
        reconstruidas = self.autoencoder.predict(X, verbose=0)
        mse = np.mean((X - reconstruidas)**2)
        ratio_compresion = (28*28*1) / (7*7*64)
        return {'mse': float(mse), 'ratio_compresion': float(ratio_compresion)}


def main():
    print("\n" + "="*60)
    print("P5: COMPRESOR DE IMAGENES (PCA + Autoencoder)")
    print("="*60)
    
    generador = GeneradorImagenes()
    X = generador.generar_dataset(200)
    
    modelo = AutoencoderConvolucional()
    modelo.construir_modelo()
    modelo.entrenar(X, epochs=15)
    metricas = modelo.evaluar(X)
    
    print(f"MSE reconstruccion: {metricas['mse']:.6f}")
    print(f"Ratio compresion: {metricas['ratio_compresion']:.2f}x")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
