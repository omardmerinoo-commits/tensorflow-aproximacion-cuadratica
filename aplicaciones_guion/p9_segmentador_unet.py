#!/usr/bin/env python3
"""
P9: Segmentador Semantico - U-Net
Segmentacion pixel-a-pixel de imagenes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class GeneradorMascaras:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=100):
        """Generar imagenes y mascaras de segmentacion"""
        imagenes = []
        mascaras = []
        
        for _ in range(n_samples):
            img = np.random.rand(64, 64, 3) * 0.3
            mascara = np.zeros((64, 64, 1))
            
            # Crear 2-3 regiones para segmentar
            n_regiones = np.random.randint(2, 4)
            for region_id in range(1, n_regiones + 1):
                x = np.random.randint(5, 50)
                y = np.random.randint(5, 50)
                w = np.random.randint(5, 25)
                h = np.random.randint(5, 25)
                
                img[y:y+h, x:x+w, :] = 0.7 + region_id * 0.1
                mascara[y:y+h, x:x+w, 0] = region_id / n_regiones
            
            imagenes.append(img)
            mascaras.append(mascara)
        
        return {
            'imagenes': np.array(imagenes).astype('float32'),
            'mascaras': np.array(mascaras).astype('float32')
        }


class SegmentadorUNet:
    def __init__(self):
        self.modelo = None
    
    def construir_modelo(self):
        """Arquitectura U-Net para segmentacion"""
        entrada = keras.Input(shape=(64, 64, 3))
        
        # Encoder
        enc1 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(entrada)
        pool1 = keras.layers.MaxPooling2D((2, 2))(enc1)
        
        enc2 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
        pool2 = keras.layers.MaxPooling2D((2, 2))(enc2)
        
        # Bottom
        bottom = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
        
        # Decoder con skip connections
        up1 = keras.layers.UpSampling2D((2, 2))(bottom)
        concat1 = keras.layers.Concatenate()([up1, enc2])
        dec1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
        
        up2 = keras.layers.UpSampling2D((2, 2))(dec1)
        concat2 = keras.layers.Concatenate()([up2, enc1])
        dec2 = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(concat2)
        
        # Output
        salida = keras.layers.Conv2D(1, 1, activation='sigmoid')(dec2)
        
        self.modelo = keras.Model(entrada, salida)
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss='mse',
            metrics=['mae']
        )
        print("[+] Modelo U-Net P9 construido")
    
    def entrenar(self, X, mascaras, epochs=15):
        self.modelo.fit(X, mascaras, epochs=epochs, verbose=0, batch_size=8)


def main():
    print("\n" + "="*60)
    print("P9: SEGMENTADOR SEMANTICO (U-Net)")
    print("="*60)
    
    generador = GeneradorMascaras()
    datos = generador.generar_dataset(100)
    
    n_train = int(0.8 * len(datos['imagenes']))
    X_train = datos['imagenes'][:n_train]
    mascaras_train = datos['mascaras'][:n_train]
    
    modelo = SegmentadorUNet()
    modelo.construir_modelo()
    modelo.entrenar(X_train, mascaras_train, epochs=15)
    
    print("Segmentador semantico entrenado exitosamente")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
