#!/usr/bin/env python3
"""
P4: Segmentador de Clientes - K-Means + Autoencoder
Agrupar clientes similares en segmentos
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class GeneradorDatosClientes:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=300):
        """Generar comportamiento de clientes"""
        # 8 caracteristicas: gasto, frecuencia, antiguedad, etc
        X = np.random.rand(n_samples, 8) * 100
        
        # Crear 3 clusters naturales
        n_por_cluster = n_samples // 3
        X[:n_por_cluster, :] += np.array([50, 30, 20, 10, 5, 15, 25, 35])
        X[n_por_cluster:2*n_por_cluster, :] += np.array([20, 50, 10, 30, 25, 5, 15, 10])
        X[2*n_por_cluster:, :] += np.array([10, 15, 50, 20, 15, 30, 10, 20])
        
        return {'X': X, 'features': [f'Feature_{i}' for i in range(8)]}


class AutoencoderClientes:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = None
        self.autoencoder = None
    
    def construir_modelo(self, input_dim=8, latent_dim=3):
        """Autoencoder para extraccion de features"""
        entrada = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(16, activation='relu')(entrada)
        encoded = keras.layers.Dense(latent_dim, activation='relu')(encoded)
        
        decoded = keras.layers.Dense(16, activation='relu')(encoded)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = keras.Model(entrada, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        self.encoder = keras.Model(entrada, encoded)
        print("[+] Autoencoder P4 construido")
    
    def entrenar(self, X, epochs=20):
        X_scaled = self.scaler.fit_transform(X)
        self.autoencoder.fit(X_scaled, X_scaled, epochs=epochs, verbose=0)
    
    def extraer_features(self, X):
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)


class SegmentadorClientes:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.score = 0
    
    def ajustar(self, X):
        self.kmeans.fit(X)
        self.score = silhouette_score(X, self.kmeans.labels_)
    
    def predecir(self, X):
        return self.kmeans.predict(X)


def main():
    print("\n" + "="*60)
    print("P4: SEGMENTADOR DE CLIENTES (K-Means + Autoencoder)")
    print("="*60)
    
    generador = GeneradorDatosClientes()
    datos = generador.generar_dataset(300)
    X = datos['X']
    
    # Autoencoder
    ae = AutoencoderClientes()
    ae.construir_modelo(input_dim=8, latent_dim=3)
    ae.entrenar(X, epochs=20)
    X_latent = ae.extraer_features(X)
    
    # K-Means
    segmentador = SegmentadorClientes(n_clusters=3)
    segmentador.ajustar(X_latent)
    
    print(f"Silhouette Score: {segmentador.score:.4f}")
    print(f"Clusters identificados: {segmentador.n_clusters}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
