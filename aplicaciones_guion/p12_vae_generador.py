#!/usr/bin/env python3
"""
P12: Generador de Imagenes - VAE (Autoencoder Variacional)
Generar nuevas imagenes aprendiendo distribucion latente
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GeneradorImagenesVAE:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=200):
        """Generar imagenes sinteticas"""
        imagenes = []
        for i in range(n_samples):
            img = np.random.randn(28, 28) * 0.3 + 0.5
            # Agregar formas
            y, x = np.ogrid[:28, :28]
            mask = (x - 14)**2 + (y - 14)**2 <= (5 + i % 10)**2
            img[mask] = 0.9
            imagenes.append(img)
        
        return np.array(imagenes).reshape(n_samples, 28, 28, 1).astype('float32')


class VAE(keras.Model):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(16, activation='relu'),
        ])
        
        self.mu_layer = layers.Dense(latent_dim)
        self.logvar_layer = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Reshape((1, 1, 16)),
            layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
            layers.UpSampling2D(2),
            layers.Conv2D(1, 3, activation='sigmoid', padding='same'),
        ])
    
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        z = mu + eps * tf.exp(0.5 * logvar)
        return z
    
    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def compute_loss(self, x):
        reconstructed, mu, logvar = self(x)
        
        reconstruction_loss = keras.losses.mse(x, reconstructed)
        reconstruction_loss *= 28 * 28
        
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        
        return tf.reduce_mean(reconstruction_loss + kl_loss)


class GeneradorVAE:
    def __init__(self, latent_dim=16):
        self.vae = VAE(latent_dim)
        self.optimizer = keras.optimizers.Adam(1e-3)
    
    def entrenar(self, X_train, epochs=20):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                loss = self.vae.compute_loss(x)
            
            gradients = tape.gradient(loss, self.vae.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.vae.trainable_weights))
            return loss
        
        for epoch in range(epochs):
            n_batches = len(X_train) // 16
            for i in range(n_batches):
                batch = X_train[i*16:(i+1)*16]
                loss = train_step(batch)
        
        print("[+] VAE P12 entrenado")


def main():
    print("\n" + "="*60)
    print("P12: GENERADOR DE IMAGENES (VAE)")
    print("="*60)
    
    generador_img = GeneradorImagenesVAE()
    X = generador_img.generar_dataset(200)
    
    generador_vae = GeneradorVAE(latent_dim=16)
    generador_vae.entrenar(X, epochs=20)
    
    # Generar imagenes nuevas
    z_nuevo = np.random.randn(10, 16).astype('float32')
    imagenes_generadas = generador_vae.vae.decoder(z_nuevo)
    
    print(f"Imagenes generadas: {imagenes_generadas.shape}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
