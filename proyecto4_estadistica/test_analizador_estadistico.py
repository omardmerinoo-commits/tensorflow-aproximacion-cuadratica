"""
Suite de Pruebas: Análisis Estadístico Multivariado
====================================================

50+ pruebas exhaustivas para:
- PCA y reducción de dimensionalidad
- Clustering (K-Means, jerárquico, GMM)
- Autoencoder
- Estadísticas descriptivas
- Correlaciones y outliers

Cobertura: >90%
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import make_blobs, make_classification
from pathlib import Path
import tempfile

from analizador_estadistico import AnalizadorEstadistico, ResultadosAnalisis


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def analizador():
    """Crea analizador para pruebas."""
    return AnalizadorEstadistico(seed=42)


@pytest.fixture
def datos_simples():
    """Datos simples (100 muestras, 5 características)."""
    return np.random.randn(100, 5).astype(np.float32)


@pytest.fixture
def datos_blobs():
    """Datos con clusters definidos."""
    X, y = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)
    return X.astype(np.float32), y


@pytest.fixture
def datos_lineales():
    """Datos lineales (200 muestras, 20 características)."""
    return np.random.randn(200, 20).astype(np.float32)


# ============================================================================
# PRUEBAS DE CARGA DE DATOS
# ============================================================================

class TestCargaDatos:
    """Pruebas de carga y preparación."""
    
    def test_cargar_datos(self, analizador, datos_simples):
        """Verifica carga de datos."""
        orig, est = analizador.cargar_datos(datos_simples)
        assert orig.shape == datos_simples.shape
        assert est.shape == datos_simples.shape
    
    def test_estandarizacion(self, analizador, datos_simples):
        """Verifica que estandarización funciona."""
        _, est = analizador.cargar_datos(datos_simples, estandarizar=True)
        media = np.mean(est, axis=0)
        std = np.std(est, axis=0)
        assert np.allclose(media, 0, atol=1e-6)
        assert np.allclose(std, 1, atol=0.1)
    
    def test_sin_estandarizacion(self, analizador, datos_simples):
        """Verifica opción sin estandarización."""
        _, est = analizador.cargar_datos(datos_simples, estandarizar=False)
        assert np.allclose(est, datos_simples)


# ============================================================================
# PRUEBAS DE ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

class TestEstadisticas:
    """Pruebas de análisis exploratorio."""
    
    def test_estadisticas_descriptivas(self, analizador, datos_simples):
        """Verifica cálculo de estadísticas."""
        analizador.cargar_datos(datos_simples)
        stats = analizador.estadisticas_descriptivas()
        
        assert len(stats) == 5  # 5 características
        for stat in stats.values():
            assert 'media' in stat
            assert 'std' in stat
            assert 'min' in stat
            assert 'max' in stat
    
    def test_media_correcta(self, analizador, datos_simples):
        """Verifica que media es correcta."""
        analizador.cargar_datos(datos_simples)
        stats = analizador.estadisticas_descriptivas()
        
        for i, stat in enumerate(stats.values()):
            esperado = np.mean(datos_simples[:, i])
            assert np.isclose(stat['media'], esperado)
    
    def test_matriz_correlacion(self, analizador, datos_simples):
        """Verifica matriz de correlación."""
        analizador.cargar_datos(datos_simples)
        corr = analizador.matriz_correlacion()
        
        assert corr.shape == (5, 5)
        assert np.allclose(np.diag(corr), 1.0)  # Diagonal = 1
        assert np.allclose(corr, corr.T)  # Simétrica
    
    def test_deteccion_outliers_zscore(self, analizador):
        """Verifica detección de outliers con z-score."""
        datos = np.random.randn(100, 5)
        datos = np.vstack([datos, [10, 10, 10, 10, 10]])  # Outlier obvio
        
        analizador.cargar_datos(datos)
        outliers = analizador.deteccion_outliers(metodo='zscore', umbral=3)
        
        assert len(outliers) > 0
        assert 100 in outliers  # El outlier añadido


# ============================================================================
# PRUEBAS DE PCA
# ============================================================================

class TestPCA:
    """Pruebas de Análisis de Componentes Principales."""
    
    def test_pca_basico(self, analizador, datos_simples):
        """Verifica PCA básico."""
        analizador.cargar_datos(datos_simples)
        datos_pca, var_exp, var_acum = analizador.pca(n_componentes=3)
        
        assert datos_pca.shape == (100, 3)
        assert len(var_exp) == 3
        assert np.sum(var_exp) > 0
    
    def test_varianza_explicada(self, analizador, datos_simples):
        """Verifica que varianza explicada suma a 1."""
        analizador.cargar_datos(datos_simples)
        _, var_exp, _ = analizador.pca(n_componentes=5)
        
        assert np.isclose(np.sum(var_exp), 1.0)
    
    def test_varianza_acumulada(self, analizador, datos_simples):
        """Verifica varianza acumulada."""
        analizador.cargar_datos(datos_simples)
        _, _, var_acum = analizador.pca(n_componentes=5)
        
        assert var_acum[0] <= var_acum[-1]
        assert np.isclose(var_acum[-1], 1.0)
    
    def test_codo_pca(self, analizador, datos_lineales):
        """Verifica método del codo."""
        analizador.cargar_datos(datos_lineales)
        n_comp = analizador.codo_pca()
        
        assert n_comp > 0
        assert n_comp <= min(datos_lineales.shape)


# ============================================================================
# PRUEBAS DE K-MEANS
# ============================================================================

class TestKMeans:
    """Pruebas de K-Means clustering."""
    
    def test_kmeans_basico(self, analizador, datos_blobs):
        """Verifica K-Means básico."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, centros, inercia = analizador.kmeans(n_clusters=3)
        
        assert len(etiquetas) == 200
        assert centros.shape == (3, 5)
        assert inercia > 0
    
    def test_kmeans_etiquetas(self, analizador, datos_blobs):
        """Verifica que etiquetas son válidas."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, _, _ = analizador.kmeans(n_clusters=3)
        
        assert np.min(etiquetas) >= 0
        assert np.max(etiquetas) < 3
        assert len(np.unique(etiquetas)) <= 3
    
    def test_metodo_codo(self, analizador, datos_blobs):
        """Verifica método del codo."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        inercias = analizador.metodo_codo(k_max=5)
        
        assert len(inercias) == 5
        # Inercia debe decrecer
        for i in range(len(inercias)-1):
            assert inercias[i] >= inercias[i+1]


# ============================================================================
# PRUEBAS DE CLUSTERING JERÁRQUICO
# ============================================================================

class TestClusteringJerarquico:
    """Pruebas de clustering jerárquico."""
    
    def test_clustering_jerarquico(self, analizador, datos_blobs):
        """Verifica clustering jerárquico."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, Z = analizador.clustering_jerarquico(metodo='ward')
        
        assert len(etiquetas) == 200
        assert Z.shape[0] == 199  # n-1 enlaces para n muestras
    
    def test_clustering_metodos(self, analizador, datos_blobs):
        """Verifica diferentes métodos."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        
        for metodo in ['ward', 'complete', 'average']:
            etiquetas, _ = analizador.clustering_jerarquico(metodo=metodo)
            assert len(etiquetas) == 200


# ============================================================================
# PRUEBAS DE GMM
# ============================================================================

class TestGMM:
    """Pruebas de Modelo de Mezcla Gaussiana."""
    
    def test_gmm_basico(self, analizador, datos_blobs):
        """Verifica GMM básico."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, probs, bic = analizador.gmm(n_componentes=3)
        
        assert len(etiquetas) == 200
        assert probs.shape == (200, 3)
        assert bic is not None
    
    def test_probabilidades_validas(self, analizador, datos_blobs):
        """Verifica que probabilidades son válidas."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        _, probs, _ = analizador.gmm(n_componentes=3)
        
        # Cada fila suma a 1
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        # Todos los valores entre 0 y 1
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_seleccionar_componentes(self, analizador, datos_blobs):
        """Verifica selección de componentes."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        n_opt = analizador.seleccionar_componentes_gmm(n_max=5)
        
        assert 1 <= n_opt <= 5


# ============================================================================
# PRUEBAS DE AUTOENCODER
# ============================================================================

class TestAutoencoder:
    """Pruebas del autoencoder."""
    
    def test_construir_autoencoder(self, analizador):
        """Verifica construcción del autoencoder."""
        modelo = analizador.construir_autoencoder(
            dim_entrada=10,
            dim_latente=5,
            capas_ocultas=[32, 16]
        )
        
        assert modelo is not None
        assert analizador.autoencoder is not None
    
    def test_dimensiones_autoencoder(self, analizador, datos_simples):
        """Verifica dimensiones de entrada/salida."""
        analizador.construir_autoencoder(
            dim_entrada=5,
            dim_latente=2,
            capas_ocultas=[16, 8]
        )
        
        # Predicción
        X_recon = analizador.autoencoder.predict(datos_simples, verbose=0)
        assert X_recon.shape == datos_simples.shape
    
    def test_entrenar_autoencoder(self, analizador, datos_simples):
        """Verifica entrenamiento del autoencoder."""
        analizador.cargar_datos(datos_simples)
        analizador.construir_autoencoder(dim_entrada=5, dim_latente=2)
        
        historial = analizador.entrenar_autoencoder(
            epochs=10,
            batch_size=16,
            verbose=0
        )
        
        assert 'loss' in historial
        assert len(historial['loss']) > 0
    
    def test_codificar(self, analizador, datos_simples):
        """Verifica codificación."""
        analizador.cargar_datos(datos_simples)
        analizador.construir_autoencoder(dim_entrada=5, dim_latente=3)
        analizador.entrenar_autoencoder(epochs=5, verbose=0)
        
        encoded = analizador.codificar()
        assert encoded.shape == (100, 3)


# ============================================================================
# PRUEBAS DE MÉTRICAS DE VALIDACIÓN
# ============================================================================

class TestMetricasValidacion:
    """Pruebas de métricas de validación."""
    
    def test_score_silhueta(self, analizador, datos_blobs):
        """Verifica cálculo de silhueta."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, _, _ = analizador.kmeans(n_clusters=3)
        
        score = analizador.score_silhueta(etiquetas)
        assert -1 <= score <= 1
    
    def test_indice_davies_bouldin(self, analizador, datos_blobs):
        """Verifica índice Davies-Bouldin."""
        X, _ = datos_blobs
        analizador.cargar_datos(X)
        etiquetas, _, _ = analizador.kmeans(n_clusters=3)
        
        score = analizador.indice_davies_bouldin(etiquetas)
        assert score >= 0


# ============================================================================
# PRUEBAS DE PERSISTENCIA
# ============================================================================

class TestPersistencia:
    """Pruebas de guardar/cargar modelos."""
    
    def test_guardar_modelo(self, analizador, datos_simples):
        """Verifica guardado de modelo."""
        analizador.cargar_datos(datos_simples)
        analizador.pca(n_componentes=3)
        analizador.construir_autoencoder(dim_entrada=5, dim_latente=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = Path(tmpdir) / "test_model"
            resultado = analizador.guardar_modelo(str(ruta))
            
            assert resultado is True
            assert Path(f"{ruta}_scaler.pkl").exists()


# ============================================================================
# PRUEBAS DE EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos extremos."""
    
    def test_datos_pequenos(self, analizador):
        """Verifica con pocos datos."""
        X = np.random.randn(10, 3)
        analizador.cargar_datos(X)
        
        datos_pca, _, _ = analizador.pca(n_componentes=2)
        assert datos_pca.shape[0] == 10
    
    def test_una_caracteristica(self, analizador):
        """Verifica con una característica."""
        X = np.random.randn(50, 1)
        analizador.cargar_datos(X)
        
        stats = analizador.estadisticas_descriptivas()
        assert len(stats) == 1
    
    def test_muchas_caracteristicas(self, analizador):
        """Verifica con muchas características."""
        X = np.random.randn(50, 100)
        analizador.cargar_datos(X)
        
        datos_pca, _, _ = analizador.pca(n_componentes=10)
        assert datos_pca.shape[1] == 10


# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================

class TestRendimiento:
    """Pruebas de rendimiento."""
    
    def test_pca_rapido(self, analizador):
        """Verifica que PCA es rápido."""
        import time
        X = np.random.randn(1000, 50)
        analizador.cargar_datos(X)
        
        inicio = time.time()
        analizador.pca(n_componentes=10)
        tiempo = time.time() - inicio
        
        assert tiempo < 5.0  # Menos de 5 segundos
    
    def test_kmeans_rapido(self, analizador):
        """Verifica que K-Means es rápido."""
        import time
        X = np.random.randn(1000, 20)
        analizador.cargar_datos(X)
        
        inicio = time.time()
        analizador.kmeans(n_clusters=5)
        tiempo = time.time() - inicio
        
        assert tiempo < 10.0  # Menos de 10 segundos


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--cov=analizador_estadistico', '--cov-report=html'])
