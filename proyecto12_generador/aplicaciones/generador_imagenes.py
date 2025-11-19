"""
P12: GENERADOR DE IMÁGENES CON AUTOENCODER
Aplicación para generación y reconstrucción de imágenes usando autoencoders.

Técnica: Autoencoder (Convolucional)
Dataset: Dígitos sintéticos
Métrica: MSE de reconstrucción
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime


class GeneradorDigitos:
    """Generador de imágenes de dígitos sintéticos"""
    
    @staticmethod
    def generar_dataset(n_samples=1000, seed=42):
        """
        Genera imágenes de dígitos sintéticos
        
        Args:
            n_samples: número de muestras
            seed: semilla para reproducibilidad
        
        Returns:
            np.ndarray: Imágenes de forma (n_samples, 28, 28, 1)
        """
        np.random.seed(seed)
        
        # Generar imágenes de 28x28 con patrones aleatorios
        imagenes = np.zeros((n_samples, 28, 28, 1), dtype=np.float32)
        
        for i in range(n_samples):
            # Crear patrón aleatorio
            imagen = np.random.rand(28, 28)
            
            # Añadir formas geométricas
            x_centro = np.random.randint(5, 23)
            y_centro = np.random.randint(5, 23)
            
            # Círculo
            for x in range(max(0, x_centro-5), min(28, x_centro+5)):
                for y in range(max(0, y_centro-5), min(28, y_centro+5)):
                    dist = np.sqrt((x-x_centro)**2 + (y-y_centro)**2)
                    if dist < 5:
                        imagen[y, x] *= 0.5
            
            imagenes[i, :, :, 0] = imagen
        
        # Normalizar
        imagenes = (imagenes - imagenes.min()) / (imagenes.max() - imagenes.min())
        return imagenes


class Autoencoder:
    """Autoencoder para generación de imágenes"""
    
    def __init__(self, latent_dim=16):
        self.modelo = None
        self.encoder = None
        self.decoder = None
        self.latent_dim = latent_dim
        self.historial = None
        
    def construir_modelo(self):
        """Construye el autoencoder"""
        # Entrada
        entrada = layers.Input(shape=(28, 28, 1))
        
        # Encoder
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(entrada)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        latente = layers.Dense(self.latent_dim, activation='relu', name='latente')(x)
        
        self.encoder = keras.Model(entrada, latente, name='Encoder')
        
        # Decoder
        entrada_latente = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64 * 3 * 3, activation='relu')(entrada_latente)
        x = layers.Reshape((3, 3, 64))(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        salida = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.decoder = keras.Model(entrada_latente, salida, name='Decoder')
        
        # Autoencoder completo
        encoded = self.encoder(entrada)
        decoded = self.decoder(encoded)
        
        self.modelo = keras.Model(entrada, decoded, name='Autoencoder')
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def entrenar(self, X_train, epochs=20, batch_size=32):
        """Entrena el autoencoder"""
        self.historial = self.modelo.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def reconstruir(self, X):
        """Reconstruye imágenes"""
        return self.modelo.predict(X, verbose=0)
    
    def generar_latente(self):
        """Genera vector latente aleatorio"""
        return np.random.randn(1, self.latent_dim).astype(np.float32)
    
    def generar_imagen(self):
        """Genera una nueva imagen desde vector latente"""
        latente = self.generar_latente()
        imagen = self.decoder.predict(latente, verbose=0)
        return imagen[0]


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(" "*10 + "P12: GENERADOR DE IMÁGENES CON AUTOENCODER")
    print("="*70 + "\n")
    
    # Generar datos
    print("[1/7] Generando imágenes sintéticas...")
    generador = GeneradorDigitos()
    imagenes = generador.generar_dataset(n_samples=500, seed=42)
    print(f"     {len(imagenes)} imágenes de 28x28 generadas")
    
    # Split
    print("[2/7] Dividiendo dataset...")
    split = int(0.8 * len(imagenes))
    X_train = imagenes[:split]
    X_test = imagenes[split:]
    print(f"     Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Crear autoencoder
    print("[3/7] Construyendo autoencoder...")
    autoencoder = Autoencoder(latent_dim=16)
    autoencoder.construir_modelo()
    autoencoder.modelo.summary()
    
    # Entrenar
    print("[4/7] Entrenando autoencoder...")
    autoencoder.entrenar(X_train, epochs=20, batch_size=32)
    print("     Entrenamiento completado")
    
    # Evaluar reconstrucción
    print("[5/7] Evaluando reconstrucción...")
    X_recon_train = autoencoder.reconstruir(X_train)
    X_recon_test = autoencoder.reconstruir(X_test)
    
    mse_train = np.mean((X_train - X_recon_train)**2)
    mse_test = np.mean((X_test - X_recon_test)**2)
    
    print(f"     MSE Train: {mse_train:.6f}")
    print(f"     MSE Test:  {mse_test:.6f}")
    
    # Generar nuevas imágenes
    print("[6/7] Generando imágenes nuevas...")
    imagenes_generadas = []
    for i in range(5):
        img = autoencoder.generar_imagen()
        imagenes_generadas.append(img)
    print(f"     {len(imagenes_generadas)} imágenes generadas")
    
    # Información del modelo
    print("[7/7] Información del modelo...")
    print(f"     Encoder params: {autoencoder.encoder.count_params():,}")
    print(f"     Decoder params: {autoencoder.decoder.count_params():,}")
    print(f"     Total params:   {autoencoder.modelo.count_params():,}")
    
    # Guardar resultados
    reporte = {
        "proyecto": "P12 - Generador de Imágenes",
        "tecnica": "Autoencoder Convolucional",
        "fecha": datetime.now().isoformat(),
        "configuracion": {
            "latent_dim": 16,
            "tamanio_imagen": [28, 28, 1],
            "epochs": 20,
            "batch_size": 32
        },
        "metricas": {
            "mse_train": float(mse_train),
            "mse_test": float(mse_test),
            "muestras_train": len(X_train),
            "muestras_test": len(X_test),
            "imagenes_generadas": len(imagenes_generadas)
        },
        "modelo": {
            "encoder_params": int(autoencoder.encoder.count_params()),
            "decoder_params": int(autoencoder.decoder.count_params()),
            "total_params": int(autoencoder.modelo.count_params()),
            "arquitectura": "Conv2D -> MaxPool -> Flatten -> Dense -> DeConv2D -> UpSampling"
        }
    }
    
    os.makedirs('reportes', exist_ok=True)
    with open('reportes/reporte_p12.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "="*70)
    print("[OK] Aplicación P12 completada correctamente")
    print(f"[REPORTE] Guardado en: reportes/reporte_p12.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
