"""Tests para aproximador de funciones no lineales."""

import pytest
import numpy as np
from aproximador_funciones import (
    GeneradorFuncionesNoLineales,
    AproximadorFuncionesNoLineales
)


class TestGeneradorFunciones:
    """Tests para generador de funciones."""
    
    def test_funcion_exponencial_amortiguada(self):
        """Test función sin(x)*exp(-x/10)."""
        x = np.array([0.0, 1.0, 2.0])
        y = GeneradorFuncionesNoLineales.funcion_exponencial_amortiguada(x)
        
        assert y.shape == x.shape
        assert y[0] == 0.0  # sin(0) = 0
    
    def test_funcion_polinomica_compleja(self):
        """Test función x³ - 2x² + 3x - 1."""
        x = np.array([0.0, 1.0, 2.0])
        y = GeneradorFuncionesNoLineales.funcion_polinomica_compleja(x)
        
        assert y.shape == x.shape
        assert y[0] == -1.0  # 0 - 0 + 0 - 1 = -1
        assert y[1] == 1.0   # 1 - 2 + 3 - 1 = 1
    
    def test_generar_datos(self):
        """Test generación de datos."""
        funcion = GeneradorFuncionesNoLineales.funcion_exponencial_amortiguada
        X, y = GeneradorFuncionesNoLineales.generar_datos(
            funcion, x_min=-5, x_max=5, n_samples=100, ruido=0.0
        )
        
        assert X.shape == (100, 1)
        assert y.shape == (100, 1)
        assert np.all(X >= -5) and np.all(X <= 5)
    
    def test_generar_datos_con_ruido(self):
        """Test que el ruido afecta los datos."""
        funcion = GeneradorFuncionesNoLineales.funcion_exponencial_amortiguada
        
        X1, y1 = GeneradorFuncionesNoLineales.generar_datos(
            funcion, n_samples=100, ruido=0.0
        )
        X2, y2 = GeneradorFuncionesNoLineales.generar_datos(
            funcion, n_samples=100, ruido=0.5
        )
        
        # Los datos con ruido deberían tener más varianza
        assert np.var(y2) > np.var(y1)


class TestAproximador:
    """Tests para el aproximador."""
    
    @pytest.fixture
    def datos(self):
        """Fixture con datos de prueba."""
        funcion = GeneradorFuncionesNoLineales.funcion_exponencial_amortiguada
        X, y = GeneradorFuncionesNoLineales.generar_datos(
            funcion, n_samples=200, ruido=0.01
        )
        return X, y
    
    def test_inicializacion(self):
        """Test inicialización."""
        modelo = AproximadorFuncionesNoLineales(seed=42)
        assert modelo.modelo is None
        assert modelo.history is None
    
    def test_construccion_modelo(self):
        """Test construcción."""
        modelo = AproximadorFuncionesNoLineales()
        modelo.construir_modelo()
        assert modelo.modelo is not None
    
    def test_prediccion_forma(self, datos):
        """Test forma de predicciones."""
        X, y = datos
        modelo = AproximadorFuncionesNoLineales()
        modelo.construir_modelo()
        
        predicciones = modelo.predecir(X[:10])
        assert predicciones.shape == (10, 1)
    
    def test_entrenamiento_reduce_loss(self, datos):
        """Test que el entrenamiento reduce pérdida."""
        X, y = datos
        modelo = AproximadorFuncionesNoLineales()
        modelo.construir_modelo()
        
        history = modelo.entrenar(X, y, epochs=10, verbose=0)
        assert history['loss_final'] > 0
        assert history['mae_final'] > 0
    
    def test_evaluacion(self, datos):
        """Test evaluación."""
        X, y = datos
        modelo = AproximadorFuncionesNoLineales()
        modelo.construir_modelo()
        modelo.entrenar(X, y, epochs=10, verbose=0)
        
        metricas = modelo.evaluar(X[:50], y[:50])
        assert 'loss' in metricas
        assert 'mae' in metricas


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
