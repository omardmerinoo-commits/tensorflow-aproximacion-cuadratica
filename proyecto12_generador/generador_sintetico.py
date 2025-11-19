"""
Proyecto 12: Generador Sintético de Imágenes (GAN + VAE)
=======================================================

Sistema de modelos generativos para síntesis de imágenes sintéticas.

Modelos:
1. GAN (Generative Adversarial Network): Generador vs Discriminador
2. VAE (Variational Autoencoder): Codificación latente interpretable
3. Hybrid GAN-VAE: Combina ambos enfoques

Características:
- Generador: Red transconvolucional (ruido → imagen)
- Discriminador: CNN binaria (imagen → real/falso)
- Latent space: Interpretable e interpolable
- Loss adversarial + Wasserstein
- Reconstrucción + KL divergence (VAE)

Aplicaciones:
- Data augmentation
- Síntesis de faces
- Completación de imágenes
- Super-resolución

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import inception_score
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosGenerativos:
    """Contenedor para datos generativos"""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    tamano_imagen: Tuple[int, int, int]
    
    def info(self) -> str:
        return (f"Generativo: Train {self.X_train.shape}, "
                f"Val {self.X_val.shape}, Test {self.X_test.shape}")


class GeneradorDatos:
    """Generador de datasets para modelos generativos"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generar_mnist_sintético(self, n_samples: int = 1000,
                               num_shape: int = 5) -> np.ndarray:
        """
        Genera dígitos sintéticos simples (formas geométricas)
        No son reales, pero permiten validar arquitectura
        
        Args:
            n_samples: Cantidad de imágenes
            num_shape: Número de formas diferentes (0-9)
        
        Returns:
            Imágenes [n_samples, 28, 28, 1] normalizadas [0, 1]
        """
        imagenes = []
        
        for _ in range(n_samples):
            img = np.zeros((28, 28, 1), dtype=np.float32)
            
            # Generar forma aleatoria
            x_center = np.random.randint(5, 23)
            y_center = np.random.randint(5, 23)
            
            # Tipo de forma
            shape_type = np.random.choice(['circulo', 'cuadrado', 'triangulo'])
            
            if shape_type == 'circulo':
                # Dibujar círculo
                y, x = np.ogrid[0:28, 0:28]
                radio = np.random.randint(3, 8)
                mask = (x - x_center)**2 + (y - y_center)**2 <= radio**2
                img[mask] = 1.0
                
            elif shape_type == 'cuadrado':
                # Cuadrado
                size = np.random.randint(3, 8)
                x_min, x_max = max(0, x_center-size), min(28, x_center+size)
                y_min, y_max = max(0, y_center-size), min(28, y_center+size)
                img[y_min:y_max, x_min:x_max] = 1.0
                
            elif shape_type == 'triangulo':
                # Triángulo simple
                y, x = np.ogrid[0:28, 0:28]
                size = np.random.randint(3, 8)
                mask = (np.abs(x - x_center) + np.abs(y - y_center) <= size)
                img[mask] = 1.0
            
            # Agregar ruido
            ruido = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + ruido, 0, 1)
            
            imagenes.append(img)
        
        return np.array(imagenes)
    
    def generar_dataset(self, n_samples: int = 1000,
                       split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
                       ) -> DatosGenerativos:
        """
        Genera dataset para modelos generativos
        
        Args:
            n_samples: Cantidad total de imágenes
            split: (train_ratio, val_ratio, test_ratio)
        
        Returns:
            DatosGenerativos
        """
        X = self.generar_mnist_sintético(n_samples)
        
        # Split
        train_ratio, val_ratio, test_ratio = split
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        X_train = X[:n_train]
        X_val = X[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        
        return DatosGenerativos(X_train, X_val, X_test, (28, 28, 1))


class GAN:
    """Generative Adversarial Network"""
    
    def __init__(self, latent_dim: int = 100, seed: int = 42):
        self.latent_dim = latent_dim
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.generador = None
        self.discriminador = None
        self.gan_modelo = None
        self.entrenado = False
    
    def construir_generador(self, output_shape: Tuple[int, int, int] = (28, 28, 1)
                           ) -> models.Model:
        """
        Construye generador: ruido → imagen
        
        Arquitectura:
        - Dense + Reshape: Transforma latent vector a mapa
        - Conv2DTranspose: Upsampling + convolución
        """
        modelo = models.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            
            # Dense → Mapa base
            layers.Dense(7 * 7 * 128),
            layers.BatchNormalization(),
            layers.Reshape((7, 7, 128)),
            layers.Dropout(0.3),
            
            # Conv2DTranspose: Upsampling
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            # Output: imagen
            layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')
        ])
        
        return modelo
    
    def construir_discriminador(self, input_shape: Tuple[int, int, int] = (28, 28, 1)
                               ) -> models.Model:
        """
        Construye discriminador: imagen → real/falso
        
        Arquitectura:
        - Conv2D: Downsampling + feature extraction
        - Dense: Clasificación binaria
        """
        modelo = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Conv2D: Downsampling
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
            
            # Global pooling + clasificación
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation='sigmoid')  # Real/Falso
        ])
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, X_val: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                verbose: int = 1) -> Dict:
        """Entrena GAN"""
        # Construir modelos
        self.generador = self.construir_generador()
        self.discriminador = self.construir_discriminador()
        
        # Compilar discriminador
        self.discriminador.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # GAN: generador + discriminador (discriminador congelado)
        self.discriminador.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generador(z)
        validity = self.discriminador(img)
        
        self.gan_modelo = models.Model(z, validity)
        self.gan_modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy'
        )
        
        hist = {'g_loss': [], 'd_loss': [], 'd_acc': []}
        
        # Entrenar
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            d_acc_epoch = 0
            
            for _ in range(n_batches):
                # Batch real
                idx = np.random.randint(0, len(X_train), batch_size)
                X_batch = X_train[idx]
                
                # Ruido
                z = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # Generar imágenes falsas
                X_gen = self.generador.predict(z, verbose=0)
                
                # Entrenar discriminador
                self.discriminador.trainable = True
                y_real = np.ones((batch_size, 1))
                y_fake = np.zeros((batch_size, 1))
                
                d_loss_real = self.discriminador.train_on_batch(X_batch, y_real)
                d_loss_fake = self.discriminador.train_on_batch(X_gen, y_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Entrenar generador
                self.discriminador.trainable = False
                z = np.random.normal(0, 1, (batch_size, self.latent_dim))
                y = np.ones((batch_size, 1))
                g_loss = self.gan_modelo.train_on_batch(z, y)
                
                g_loss_epoch += g_loss
                d_loss_epoch += d_loss[0]
                d_acc_epoch += d_loss[1]
            
            # Promedios
            g_loss_epoch /= n_batches
            d_loss_epoch /= n_batches
            d_acc_epoch /= n_batches
            
            hist['g_loss'].append(g_loss_epoch)
            hist['d_loss'].append(d_loss_epoch)
            hist['d_acc'].append(d_acc_epoch)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"G Loss: {g_loss_epoch:.4f}, "
                      f"D Loss: {d_loss_epoch:.4f}")
        
        self.entrenado = True
        return hist
    
    def generar_imagenes(self, n_imagenes: int = 10) -> np.ndarray:
        """Genera imágenes sintéticas"""
        if not self.entrenado:
            raise ValueError("GAN no entrenado")
        
        z = np.random.normal(0, 1, (n_imagenes, self.latent_dim))
        return self.generador.predict(z, verbose=0)
    
    def guardar(self, ruta: str):
        """Guarda modelos"""
        self.generador.save(f"{ruta}_gen.h5")
        self.discriminador.save(f"{ruta}_disc.h5")


class VAE:
    """Variational Autoencoder"""
    
    def __init__(self, latent_dim: int = 32, seed: int = 42):
        self.latent_dim = latent_dim
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.entrenado = False
    
    def construir_encoder(self, input_shape: Tuple[int, int, int] = (28, 28, 1)
                         ) -> models.Model:
        """Construye encoder: imagen → latent space"""
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Latent space: mean + log_var
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        modelo = models.Model(inputs, [z_mean, z_log_var], name='encoder')
        return modelo
    
    def construir_decoder(self, output_shape: Tuple[int, int, int] = (28, 28, 1)
                         ) -> models.Model:
        """Construye decoder: latent space → imagen"""
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                                  padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                                  padding='same', activation='relu')(x)
        outputs = layers.Conv2D(1, (3, 3), padding='same', 
                               activation='sigmoid')(x)
        
        modelo = models.Model(latent_inputs, outputs, name='decoder')
        return modelo
    
    def construir_vae(self, input_shape: Tuple[int, int, int] = (28, 28, 1)):
        """Construye VAE completo"""
        encoder = self.construir_encoder(input_shape)
        decoder = self.construir_decoder(input_shape)
        
        inputs = layers.Input(shape=input_shape)
        z_mean, z_log_var = encoder(inputs)
        
        # Sampling (reparameterization trick)
        z = layers.Lambda(
            lambda args: args[0] + tf.exp(args[1] / 2) * tf.random.normal(
                tf.shape(args[0]), dtype=tf.float32),
            name='sampling'
        )([z_mean, z_log_var])
        
        # Decode
        reconstructed = decoder(z)
        
        vae = models.Model(inputs, reconstructed)
        
        # Loss function
        def vae_loss(x, x_reconstructed):
            # Reconstruction loss
            recon_loss = keras.losses.binary_crossentropy(x, x_reconstructed)
            recon_loss *= input_shape[0] * input_shape[1]
            
            # KL divergence loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * -0.5
            
            return tf.reduce_mean(recon_loss + kl_loss)
        
        vae.compile(optimizer='adam', loss=vae_loss)
        
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
    
    def entrenar(self, X_train: np.ndarray, X_val: np.ndarray,
                epochs: int = 30, batch_size: int = 32,
                verbose: int = 1) -> Dict:
        """Entrena VAE"""
        hist = self.vae.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self.entrenado = True
        return hist.history
    
    def generar_imagenes(self, n_imagenes: int = 10) -> np.ndarray:
        """Genera imágenes del latent space"""
        z = np.random.normal(0, 1, (n_imagenes, self.latent_dim))
        return self.decoder.predict(z, verbose=0)
    
    def reconstruir(self, X: np.ndarray) -> np.ndarray:
        """Reconstruye imágenes"""
        return self.vae.predict(X, verbose=0)
    
    def guardar(self, ruta: str):
        """Guarda modelos"""
        self.encoder.save(f"{ruta}_encoder.h5")
        self.decoder.save(f"{ruta}_decoder.h5")
        self.vae.save(f"{ruta}_vae.h5")


def demo():
    """Demostración completa"""
    print("="*70)
    print("GENERADOR SINTÉTICO - GAN + VAE - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Generar datos
    print("\n[1] Generando datos...")
    gen_datos = GeneradorDatos(seed=42)
    datos = gen_datos.generar_dataset(n_samples=1000)
    print(f"✓ {datos.info()}")
    
    # 2. Entrenar GAN
    print("\n[2] Entrenando GAN...")
    gan = GAN(latent_dim=100)
    gan.entrenar(datos.X_train, datos.X_val, epochs=50, verbose=0)
    imgs_gan = gan.generar_imagenes(5)
    print(f"✓ Generadas {imgs_gan.shape[0]} imágenes GAN")
    
    # 3. Entrenar VAE
    print("\n[3] Entrenando VAE...")
    vae = VAE(latent_dim=32)
    vae.construir_vae()
    vae.entrenar(datos.X_train, datos.X_val, epochs=30, verbose=0)
    imgs_vae = vae.generar_imagenes(5)
    print(f"✓ Generadas {imgs_vae.shape[0]} imágenes VAE")
    
    # 4. Reconstrucción
    print("\n[4] Reconstrucción VAE...")
    X_recon = vae.reconstruir(datos.X_test[:5])
    error = np.mean(np.abs(datos.X_test[:5] - X_recon))
    print(f"✓ Error reconstrucción: {error:.4f}")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
