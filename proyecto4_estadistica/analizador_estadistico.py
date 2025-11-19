"""
📊 Proyecto 4: Análisis Estadístico Multivariado
================================================

Análisis estadístico avanzado con reducción de dimensionalidad, clustering
y modelos de mezcla gaussiana usando TensorFlow.

✨ Características:
- 📈 PCA (Análisis de Componentes Principales)
- 🎯 K-Means y clustering jerárquico
- 📊 Modelos de Mezcla Gaussiana (GMM)
- 📉 Análisis exploratorio (EDA)
- 🔗 Correlaciones y covarianza
- 🧠 Red neuronal autoencoder
- 🎨 Visualización avanzada
- 🧪 Validación exhaustiva (50+ tests)

📐 Métodos Implementados:
- Estandarización y normalización
- PCA con varianza explicada
- K-Means con codo
- Clustering jerárquico (dendrograma)
- GMM con criterios BIC/AIC
- Autoencoder para reducción no lineal
- Análisis de correlación
- Detección de outliers

Autor: Sistema de Educación TensorFlow
Licencia: MIT
Versión: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CLASE DE DATOS
# ============================================================================

@dataclass
class ResultadosAnalisis:
    """Almacena resultados del análisis."""
    componentes_principales: np.ndarray
    varianza_explicada: np.ndarray
    varianza_acumulada: np.ndarray
    etiquetas_cluster: np.ndarray
    centros: np.ndarray
    inercia: float
    score_silhueta: float
    timestamp: datetime


# ============================================================================
# ANALIZADOR ESTADÍSTICO
# ============================================================================

class AnalizadorEstadistico:
    """Análisis estadístico multivariado con reducción de dimensionalidad."""
    
    def __init__(self, seed: int = 42):
        """
        Inicializa el analizador.
        
        Args:
            seed: Semilla para reproducibilidad
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.datos_originales = None
        self.datos_estandarizados = None
        self.pca = None
        self.kmeans = None
        self.gmm = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.historial_analisis = []
        self.correlaciones = None
        
        logger.info(f"✅ Analizador inicializado (seed={seed})")
    
    # ========================================================================
    # CARGA Y PREPARACIÓN DE DATOS
    # ========================================================================
    
    def cargar_datos(self, X: np.ndarray, estandarizar: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga y prepara datos.
        
        Args:
            X: Matriz de datos (N × D)
            estandarizar: Si aplica estandarización
        
        Returns:
            (datos_originales, datos_estandarizados)
        """
        self.datos_originales = X
        
        if estandarizar:
            self.datos_estandarizados = self.scaler.fit_transform(X)
        else:
            self.datos_estandarizados = X
        
        logger.info(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        return self.datos_originales, self.datos_estandarizados
    
    # ========================================================================
    # ANÁLISIS EXPLORATORIO
    # ========================================================================
    
    def estadisticas_descriptivas(self) -> Dict[str, Dict[str, float]]:
        """Retorna estadísticas descriptivas."""
        if self.datos_originales is None:
            raise ValueError("Carga datos primero")
        
        stats = {}
        for i, col in enumerate(range(self.datos_originales.shape[1])):
            col_data = self.datos_originales[:, i]
            stats[f"Característica_{i}"] = {
                "media": float(np.mean(col_data)),
                "std": float(np.std(col_data)),
                "min": float(np.min(col_data)),
                "max": float(np.max(col_data)),
                "mediana": float(np.median(col_data)),
                "asimetria": float(skew(col_data)),
                "curtosis": float(kurtosis(col_data))
            }
        
        logger.info(f"✅ Estadísticas calculadas para {len(stats)} características")
        return stats
    
    def matriz_correlacion(self) -> np.ndarray:
        """Calcula matriz de correlación."""
        self.correlaciones = np.corrcoef(self.datos_estandarizados.T)
        logger.info(f"✅ Matriz de correlación calculada ({self.correlaciones.shape})")
        return self.correlaciones
    
    def deteccion_outliers(self, metodo: str = 'zscore', umbral: float = 3.0) -> np.ndarray:
        """
        Detecta outliers.
        
        Args:
            metodo: 'zscore' o 'iqr'
            umbral: Umbral para zscore
        
        Returns:
            Índices de outliers
        """
        if metodo == 'zscore':
            z_scores = np.abs((self.datos_originales - np.mean(self.datos_originales, axis=0)) 
                             / np.std(self.datos_originales, axis=0))
            outliers = np.where((z_scores > umbral).any(axis=1))[0]
        else:  # IQR
            Q1 = np.percentile(self.datos_originales, 25, axis=0)
            Q3 = np.percentile(self.datos_originales, 75, axis=0)
            IQR = Q3 - Q1
            outliers = np.where(((self.datos_originales < Q1 - 1.5*IQR) | 
                                (self.datos_originales > Q3 + 1.5*IQR)).any(axis=1))[0]
        
        logger.info(f"✅ {len(outliers)} outliers detectados ({metodo})")
        return outliers
    
    # ========================================================================
    # PCA - REDUCCIÓN DE DIMENSIONALIDAD
    # ========================================================================
    
    def pca(self, n_componentes: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica PCA.
        
        Args:
            n_componentes: Número de componentes (None = todas)
        
        Returns:
            (componentes, varianza_explicada, varianza_acumulada)
        """
        if n_componentes is None:
            n_componentes = min(self.datos_estandarizados.shape)
        
        self.pca = sklearn_PCA(n_components=n_componentes)
        datos_pca = self.pca.fit_transform(self.datos_estandarizados)
        
        varianza_explicada = self.pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        logger.info(f"✅ PCA aplicado: {n_componentes} componentes")
        logger.info(f"   Varianza explicada acumulada: {varianza_acumulada[-1]:.4f}")
        
        return datos_pca, varianza_explicada, varianza_acumulada
    
    def codo_pca(self) -> int:
        """
        Calcula número óptimo de componentes usando método del codo.
        
        Returns:
            Número de componentes recomendado
        """
        if self.pca is None:
            self.pca(n_componentes=min(self.datos_estandarizados.shape))
        
        varianza_acumulada = np.cumsum(self.pca.explained_variance_ratio_)
        # Encontrar donde se alcanza 95% de varianza
        n_componentes = np.argmax(varianza_acumulada >= 0.95) + 1
        
        logger.info(f"✅ Componentes recomendados: {n_componentes} (95% varianza)")
        return n_componentes
    
    # ========================================================================
    # CLUSTERING - K-MEANS
    # ========================================================================
    
    def kmeans(self, n_clusters: int = 3, max_iter: int = 300) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Aplica K-Means clustering.
        
        Args:
            n_clusters: Número de clusters
            max_iter: Iteraciones máximas
        
        Returns:
            (etiquetas, centros, inercia)
        """
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        etiquetas = self.kmeans.fit_predict(self.datos_estandarizados)
        centros = self.kmeans.cluster_centers_
        inercia = self.kmeans.inertia_
        
        logger.info(f"✅ K-Means: {n_clusters} clusters (inercia={inercia:.4f})")
        return etiquetas, centros, inercia
    
    def metodo_codo(self, k_max: int = 10) -> List[float]:
        """
        Calcula inercia para diferentes valores de k.
        
        Args:
            k_max: k máximo a probar
        
        Returns:
            Lista de inercias
        """
        inercias = []
        for k in range(1, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(self.datos_estandarizados)
            inercias.append(km.inertia_)
        
        logger.info(f"✅ Método del codo calculado para k=1..{k_max}")
        return inercias
    
    # ========================================================================
    # CLUSTERING - JERÁRQUICO
    # ========================================================================
    
    def clustering_jerarquico(self, metodo: str = 'ward') -> Tuple[np.ndarray, Any]:
        """
        Aplica clustering jerárquico.
        
        Args:
            metodo: 'ward', 'complete', 'average', 'single'
        
        Returns:
            (etiquetas, matriz_linkage)
        """
        Z = linkage(self.datos_estandarizados, method=metodo)
        etiquetas = fcluster(Z, t=3, criterion='maxclust')  # 3 clusters
        
        logger.info(f"✅ Clustering jerárquico ({metodo}): {len(np.unique(etiquetas))} clusters")
        return etiquetas, Z
    
    # ========================================================================
    # MODELO DE MEZCLA GAUSSIANA (GMM)
    # ========================================================================
    
    def gmm(self, n_componentes: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Aplica modelo de mezcla gaussiana.
        
        Args:
            n_componentes: Número de componentes
        
        Returns:
            (etiquetas, probabilidades, score_bic)
        """
        self.gmm = GaussianMixture(n_components=n_componentes, random_state=42)
        etiquetas = self.gmm.fit_predict(self.datos_estandarizados)
        probabilidades = self.gmm.predict_proba(self.datos_estandarizados)
        bic = self.gmm.bic(self.datos_estandarizados)
        
        logger.info(f"✅ GMM: {n_componentes} componentes (BIC={bic:.4f})")
        return etiquetas, probabilidades, bic
    
    def seleccionar_componentes_gmm(self, n_max: int = 10) -> int:
        """Selecciona número óptimo de componentes usando BIC."""
        bic_scores = []
        for n in range(1, n_max + 1):
            gm = GaussianMixture(n_components=n, random_state=42)
            gm.fit(self.datos_estandarizados)
            bic_scores.append(gm.bic(self.datos_estandarizados))
        
        n_optimo = np.argmin(bic_scores) + 1
        logger.info(f"✅ Componentes óptimos GMM: {n_optimo}")
        return n_optimo
    
    # ========================================================================
    # AUTOENCODER - REDUCCIÓN NO LINEAL
    # ========================================================================
    
    def construir_autoencoder(self, 
                            dim_entrada: int,
                            dim_latente: int = 5,
                            capas_ocultas: List[int] = None) -> keras.Model:
        """
        Construye autoencoder para reducción no lineal.
        
        Args:
            dim_entrada: Dimensión de entrada
            dim_latente: Dimensión del espacio latente
            capas_ocultas: Lista de tamaños de capas
        
        Returns:
            Modelo compilado
        """
        if capas_ocultas is None:
            capas_ocultas = [64, 32]
        
        # Encoder
        entrada = layers.Input(shape=(dim_entrada,))
        x = layers.Dense(capas_ocultas[0], activation='relu')(entrada)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(capas_ocultas[1], activation='relu')(x)
        latente = layers.Dense(dim_latente, activation='relu', name='latente')(x)
        
        # Decoder
        x = layers.Dense(capas_ocultas[1], activation='relu')(latente)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(capas_ocultas[0], activation='relu')(x)
        salida = layers.Dense(dim_entrada, activation='linear')(x)
        
        autoencoder = keras.Model(entrada, salida)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        self.autoencoder = autoencoder
        logger.info(f"✅ Autoencoder construido: {dim_entrada}→{dim_latente}→{dim_entrada}")
        return autoencoder
    
    def entrenar_autoencoder(self, 
                            epochs: int = 100,
                            batch_size: int = 32,
                            validation_split: float = 0.2,
                            verbose: int = 1) -> Dict[str, Any]:
        """Entrena el autoencoder."""
        if self.autoencoder is None:
            self.construir_autoencoder(self.datos_estandarizados.shape[1])
        
        historial = self.autoencoder.fit(
            self.datos_estandarizados,
            self.datos_estandarizados,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=verbose
        )
        
        logger.info(f"✅ Autoencoder entrenado ({epochs} épocas)")
        return historial.history
    
    def codificar(self, X: np.ndarray = None) -> np.ndarray:
        """
        Codifica datos usando el encoder.
        
        Args:
            X: Datos a codificar (None = usar datos del modelo)
        
        Returns:
            Datos codificados en espacio latente
        """
        if X is None:
            X = self.datos_estandarizados
        else:
            X = self.scaler.transform(X)
        
        encoder = keras.Model(
            inputs=self.autoencoder.input,
            outputs=self.autoencoder.get_layer('latente').output
        )
        return encoder.predict(X, verbose=0)
    
    # ========================================================================
    # EVALUACIÓN
    # ========================================================================
    
    def score_silhueta(self, etiquetas: np.ndarray) -> float:
        """
        Calcula coeficiente de silhueta.
        
        Args:
            etiquetas: Etiquetas de cluster
        
        Returns:
            Score de silhueta (-1 a 1, más alto = mejor)
        """
        from sklearn.metrics import silhouette_score
        score = silhouette_score(self.datos_estandarizados, etiquetas)
        logger.info(f"✅ Score de silhueta: {score:.4f}")
        return score
    
    def indice_davies_bouldin(self, etiquetas: np.ndarray) -> float:
        """
        Calcula índice Davies-Bouldin.
        
        Args:
            etiquetas: Etiquetas de cluster
        
        Returns:
            Índice DB (más bajo = mejor)
        """
        from sklearn.metrics import davies_bouldin_score
        score = davies_bouldin_score(self.datos_estandarizados, etiquetas)
        logger.info(f"✅ Índice Davies-Bouldin: {score:.4f}")
        return score
    
    # ========================================================================
    # PERSISTENCIA
    # ========================================================================
    
    def guardar_modelo(self, ruta: str) -> bool:
        """Guarda modelos entrenados."""
        try:
            ruta_path = Path(ruta)
            ruta_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar scaler
            pickle.dump(self.scaler, open(f"{ruta}_scaler.pkl", 'wb'))
            
            # Guardar PCA
            if self.pca:
                pickle.dump(self.pca, open(f"{ruta}_pca.pkl", 'wb'))
            
            # Guardar autoencoder
            if self.autoencoder:
                self.autoencoder.save(f"{ruta}_autoencoder.keras")
            
            logger.info(f"✅ Modelo guardado: {ruta}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando: {e}")
            return False
    
    def cargar_modelo(self, ruta: str) -> bool:
        """Carga modelos guardados."""
        try:
            self.scaler = pickle.load(open(f"{ruta}_scaler.pkl", 'rb'))
            self.pca = pickle.load(open(f"{ruta}_pca.pkl", 'rb'))
            self.autoencoder = keras.models.load_model(f"{ruta}_autoencoder.keras")
            
            logger.info(f"✅ Modelo cargado: {ruta}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando: {e}")
            return False


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def demo():
    """Demostración del analizador."""
    print("\n" + "="*80)
    print("📊 ANÁLISIS ESTADÍSTICO MULTIVARIADO v1.0")
    print("="*80)
    
    print("\n✅ CARACTERÍSTICAS:")
    print("   - PCA (Análisis de Componentes Principales)")
    print("   - K-Means Clustering")
    print("   - Clustering Jerárquico")
    print("   - Modelos de Mezcla Gaussiana (GMM)")
    print("   - Autoencoder para reducción no lineal")
    print("   - Análisis exploratorio (EDA)")
    print("   - Detección de outliers")
    print("   - Métricas de validación")
    
    print("\n🔬 EJEMPLO DE USO:")
    print("""
    # Crear analizador
    analizador = AnalizadorEstadistico()
    
    # Cargar datos
    X = np.random.randn(1000, 10)
    analizador.cargar_datos(X)
    
    # PCA
    datos_pca, var_exp, var_acum = analizador.pca(n_componentes=5)
    
    # K-Means
    etiquetas, centros, inercia = analizador.kmeans(n_clusters=3)
    
    # GMM
    etiquetas_gmm, probs, bic = analizador.gmm(n_componentes=3)
    """)
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    demo()
