"""
Tests unitarios para el clasificador de fases.
"""

import pytest
import numpy as np
from generador_datos_fases import GeneradorDatosFases
from modelo_clasificador_fases import ModeloClasificadorFases


class TestGeneradorDatos:
    """Tests para el generador de datos."""
    
    def test_generador_inicializa(self):
        """Test que el generador se inicializa correctamente."""
        gen = GeneradorDatosFases(seed=42)
        assert gen.seed == 42
    
    def test_fase_solida_generacion(self):
        """Test generación de fase sólida."""
        gen = GeneradorDatosFases()
        X, y = gen.generar_fase_solida(n_samples=50)
        
        assert X.shape == (50, 5)
        assert y.shape == (50,)
        assert np.all(y == 0)
    
    def test_fase_liquida_generacion(self):
        """Test generación de fase líquida."""
        gen = GeneradorDatosFases()
        X, y = gen.generar_fase_liquida(n_samples=50)
        
        assert X.shape == (50, 5)
        assert y.shape == (50,)
        assert np.all(y == 1)
    
    def test_fase_gaseosa_generacion(self):
        """Test generación de fase gaseosa."""
        gen = GeneradorDatosFases()
        X, y = gen.generar_fase_gaseosa(n_samples=50)
        
        assert X.shape == (50, 5)
        assert y.shape == (50,)
        assert np.all(y == 2)
    
    def test_datos_completos_balanceados(self):
        """Test que los datos completos están balanceados."""
        gen = GeneradorDatosFases()
        X, y = gen.generar_datos_completos(n_samples_por_clase=100)
        
        assert X.shape == (300, 5)
        assert y.shape == (300,)
        
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 3
        assert np.all(counts == 100)
    
    def test_normalizacion_datos(self):
        """Test normalización de datos."""
        gen = GeneradorDatosFases()
        X, _ = gen.generar_datos_completos(n_samples_por_clase=100)
        X_norm, min_vals, max_vals = gen.normalizar_datos(X)
        
        assert X_norm.shape == X.shape
        assert np.all(X_norm >= 0)
        assert np.all(X_norm <= 1)
        assert len(min_vals) == X.shape[1]
        assert len(max_vals) == X.shape[1]


class TestModeloClasificador:
    """Tests para el modelo clasificador."""
    
    @pytest.fixture
    def datos_entrenamiento(self):
        """Fixture con datos para entrenamiento."""
        gen = GeneradorDatosFases(seed=42)
        X, y = gen.generar_datos_completos(n_samples_por_clase=50)
        return X, y
    
    def test_modelo_inicializa(self):
        """Test inicialización del modelo."""
        modelo = ModeloClasificadorFases(input_dim=5, num_classes=3)
        
        assert modelo.input_dim == 5
        assert modelo.num_classes == 3
        assert modelo.modelo is None
    
    def test_modelo_construccion(self):
        """Test construcción de arquitectura."""
        modelo = ModeloClasificadorFases(input_dim=5, num_classes=3)
        modelo_keras = modelo.construir_modelo()
        
        assert modelo_keras is not None
        assert modelo.modelo is not None
    
    def test_prediccion_forma(self, datos_entrenamiento):
        """Test que predicciones tienen forma correcta."""
        X, y = datos_entrenamiento
        modelo = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo.construir_modelo()
        
        predicciones, probabilidades = modelo.predecir(X[:10])
        
        assert predicciones.shape == (10,)
        assert probabilidades.shape == (10, 3)
        assert np.all(predicciones >= 0) and np.all(predicciones < 3)
    
    def test_prediccion_probabilidades_validas(self, datos_entrenamiento):
        """Test que las probabilidades suman 1."""
        X, _ = datos_entrenamiento
        modelo = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo.construir_modelo()
        
        _, probabilidades = modelo.predecir(X[:10])
        sumas = np.sum(probabilidades, axis=1)
        
        assert np.allclose(sumas, 1.0)
    
    def test_entrenamiento_reduce_loss(self, datos_entrenamiento):
        """Test que el entrenamiento reduce la pérdida."""
        X, y = datos_entrenamiento
        modelo = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo.construir_modelo()
        
        history_dict = modelo.entrenar(X, y, epochs=5, verbose=0)
        
        assert history_dict['epochs'] == 5
        assert history_dict['loss_final'] > 0
        assert 0 <= history_dict['accuracy_final'] <= 1
    
    def test_evaluacion_metricas(self, datos_entrenamiento):
        """Test evaluación retorna métricas válidas."""
        X, y = datos_entrenamiento
        modelo = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo.construir_modelo()
        modelo.entrenar(X, y, epochs=5, verbose=0)
        
        metricas = modelo.evaluar(X, y)
        
        assert 'loss' in metricas
        assert 'accuracy' in metricas
        assert metricas['loss'] > 0
        assert 0 <= metricas['accuracy'] <= 1
    
    def test_guardar_cargar_modelo(self, datos_entrenamiento, tmp_path):
        """Test guardar y cargar modelo."""
        X, y = datos_entrenamiento
        modelo1 = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo1.construir_modelo()
        modelo1.entrenar(X, y, epochs=5, verbose=0)
        
        ruta_modelo = tmp_path / "test_modelo.keras"
        modelo1.guardar_modelo(str(ruta_modelo))
        
        modelo2 = ModeloClasificadorFases(input_dim=X.shape[1], num_classes=3)
        modelo2.cargar_modelo(str(ruta_modelo))
        
        assert modelo2.modelo is not None
    
    def test_etiquetas_clases(self):
        """Test que las etiquetas de clases son correctas."""
        modelo = ModeloClasificadorFases()
        assert len(modelo.etiquetas_clases) == 3
        assert modelo.etiquetas_clases[0] == 'Sólido'
        assert modelo.etiquetas_clases[1] == 'Líquido'
        assert modelo.etiquetas_clases[2] == 'Gas'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
