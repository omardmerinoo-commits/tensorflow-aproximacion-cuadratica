"""
Aplicación: Generador de Imágenes
===============================

Caso de uso real: Generación de imágenes sinéticas con Autoencoders

Características:
- Autoencoder para generación
- Compresión y reconstrucción
- Análisis de representación latente
- Generación de nuevas imágenes

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json


class GeneradorImagenesSinteticas:
    """Generador de imágenes sintéticas."""
    
    @staticmethod
    def generar_imagen_ruido(tamaño=28):
        """Genera imagen de ruido puro."""
        return np.random.rand(tamaño, tamaño)
    
    @staticmethod
    def generar_imagen_patron_radial(tamaño=28):
        """Genera imagen con patrón radial."""
        centro = tamaño / 2
        imagen = np.zeros((tamaño, tamaño))
        
        for i in range(tamaño):
            for j in range(tamaño):
                distancia = np.sqrt((i - centro)**2 + (j - centro)**2)
                imagen[i, j] = np.sin(distancia / 5)
        
        return (imagen + 1) / 2  # Normalizar a [0, 1]
    
    @staticmethod
    def generar_imagen_ondas(tamaño=28):
        """Genera imagen con patrón de ondas."""
        imagen = np.zeros((tamaño, tamaño))
        
        for i in range(tamaño):
            for j in range(tamaño):
                imagen[i, j] = np.sin(i / 5) * np.cos(j / 5)
        
        return (imagen + 1) / 2
    
    @staticmethod
    def generar_imagen_gradiente(tamaño=28):
        """Genera imagen con gradiente."""
        imagen = np.zeros((tamaño, tamaño))
        
        for i in range(tamaño):
            for j in range(tamaño):
                imagen[i, j] = (i + j) / (2 * tamaño)
        
        return imagen
    
    @staticmethod
    def generar_dataset(n_samples=500, tamaño=28):
        """Genera dataset de imágenes."""
        generadores = [
            GeneradorImagenesSinteticas.generar_imagen_ruido,
            GeneradorImagenesSinteticas.generar_imagen_patron_radial,
            GeneradorImagenesSinteticas.generar_imagen_ondas,
            GeneradorImagenesSinteticas.generar_imagen_gradiente
        ]
        
        X = []
        
        for _ in range(n_samples):
            gen = np.random.choice(generadores)
            imagen = gen(tamaño)
            X.append(imagen)
        
        return {
            'X': np.array(X).astype(np.float32)
        }


class GeneradorAutoencoder:
    """Autoencoder para generación de imágenes."""
    
    def __init__(self, dim_latente=16, seed=42):
        """Inicializa el generador."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.dim_latente = dim_latente
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.metricas = {}
    
    def construir_modelo(self, tamaño_imagen=28):
        """Construye Autoencoder."""
        # Encoder
        inputs = keras.Input(shape=(tamaño_imagen, tamaño_imagen, 1))
        
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        z = layers.Dense(self.dim_latente, activation='relu', name='latente')(x)
        
        self.encoder = keras.Model(inputs, z, name='encoder')
        
        # Decoder
        latente_input = keras.Input(shape=(self.dim_latente,))
        
        x = layers.Dense(7 * 7 * 64, activation='relu')(latente_input)
        x = layers.Reshape((7, 7, 64))(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        self.decoder = keras.Model(latente_input, outputs, name='decoder')
        
        # Autoencoder completo
        outputs = self.decoder(self.encoder(inputs))
        self.autoencoder = keras.Model(inputs, outputs, name='autoencoder')
        
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse'
        )
        
        print(f"✅ Autoencoder construido con dimensión latente {self.dim_latente}")
    
    def entrenar(self, X_train, epochs=20):
        """Entrena el autoencoder."""
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
        
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test):
        """Evalúa el autoencoder."""
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], 1)
        
        pérdida = self.autoencoder.evaluate(X_test, X_test, verbose=0)
        
        # MSE por imagen
        X_reconstructed = self.autoencoder.predict(X_test, verbose=0)
        mse_por_imagen = np.mean((X_test - X_reconstructed) ** 2, axis=(1, 2, 3))
        
        self.metricas = {
            'mse_total': float(pérdida),
            'mse_promedio': float(np.mean(mse_por_imagen)),
            'mse_min': float(np.min(mse_por_imagen)),
            'mse_max': float(np.max(mse_por_imagen))
        }
        
        print(f"\n📊 Métricas:")
        print(f"   MSE Total: {pérdida:.6f}")
        print(f"   MSE Promedio por imagen: {np.mean(mse_por_imagen):.6f}")
        print(f"   Rango MSE: [{np.min(mse_por_imagen):.6f}, {np.max(mse_por_imagen):.6f}]")
        
        return self.metricas
    
    def reconstruir(self, imagen):
        """Reconstruye una imagen."""
        imagen_batch = imagen.reshape(1, imagen.shape[0], imagen.shape[1], 1)
        imagen_reconstructed = self.autoencoder.predict(imagen_batch, verbose=0)[0, :, :, 0]
        return imagen_reconstructed
    
    def generar_imagen(self):
        """Genera una imagen nueva desde ruido latente."""
        z = np.random.randn(1, self.dim_latente)
        imagen_generada = self.decoder.predict(z, verbose=0)[0, :, :, 0]
        return imagen_generada
    
    def obtener_representacion_latente(self, imagen):
        """Obtiene la representación latente de una imagen."""
        imagen_batch = imagen.reshape(1, imagen.shape[0], imagen.shape[1], 1)
        z = self.encoder.predict(imagen_batch, verbose=0)[0]
        return z


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🎨 GENERADOR DE IMÁGENES - AUTOENCODER")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando imágenes sintéticas...")
    datos = GeneradorImagenesSinteticas.generar_dataset(n_samples=500, tamaño=28)
    X = datos['X']
    
    print(f"✅ Dataset generado: {X.shape}")
    print(f"   Rango valores: [{np.min(X):.4f}, {np.max(X):.4f}]")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Paso 3: Construir
    print("\n[3] Construyendo autoencoder...")
    generador = GeneradorAutoencoder(dim_latente=16)
    generador.construir_modelo(tamaño_imagen=28)
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando...")
    generador.entrenar(X_train, epochs=20)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    generador.evaluar(X_test)
    
    # Paso 6: Reconstruir
    print("\n[6] Reconstruyendo imágenes:")
    for i in range(3):
        imagen_original = X_test[i]
        imagen_reconstructed = generador.reconstruir(imagen_original)
        
        mse = np.mean((imagen_original - imagen_reconstructed) ** 2)
        print(f"\n   Imagen {i+1}:")
        print(f"     MSE reconstrucción: {mse:.6f}")
    
    # Paso 7: Generar imágenes nuevas
    print("\n[7] Generando imágenes nuevas:")
    for i in range(3):
        imagen_nueva = generador.generar_imagen()
        print(f"\n   Imagen generada {i+1}:")
        print(f"     Rango: [{np.min(imagen_nueva):.4f}, {np.max(imagen_nueva):.4f}]")
        print(f"     Media: {np.mean(imagen_nueva):.4f}")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'Autoencoder Generador de Imágenes',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'dim_latente': 16,
        'tamaño_imagen': 28,
        'metricas': generador.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
