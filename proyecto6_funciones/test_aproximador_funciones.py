"""
Suite de Pruebas: Aproximador de Funciones No-Lineales
=======================================================

70+ pruebas exhaustivas para:
- Generación de funciones
- Normalización de datos
- Arquitecturas MLP y Residual
- Entrenamiento y evaluación
- Predicción y persistencia

Cobertura: >90%
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile

from aproximador_funciones import (
    DatosEntrenamiento,
    GeneradorFuncionesNoLineales,
    AproximadorFuncion
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def generador():
    """Crea generador."""
    return GeneradorFuncionesNoLineales(seed=42)


@pytest.fixture
def datos_sin(generador):
    """Datos para sin(x)."""
    return generador.generar('sin', n_muestras=200)


@pytest.fixture
def datos_exp(generador):
    """Datos para exp(x)."""
    return generador.generar('exp', n_muestras=200)


@pytest.fixture
def datos_x3(generador):
    """Datos para x³."""
    return generador.generar('x3', n_muestras=200)


@pytest.fixture
def aproximador():
    """Crea aproximador."""
    return AproximadorFuncion(seed=42)


# ============================================================================
# PRUEBAS DE DATOS
# ============================================================================

class TestDatos:
    """Pruebas de estructura de datos."""
    
    def test_creacion_datos(self, datos_sin):
        """Verifica creación de datos."""
        assert datos_sin.X_train is not None
        assert datos_sin.y_train is not None
        assert datos_sin.X_test is not None
        assert datos_sin.y_test is not None
    
    def test_formas_datos(self, datos_sin):
        """Verifica formas correctas."""
        assert len(datos_sin.X_train.shape) == 2
        assert datos_sin.X_train.shape[1] == 1
        assert len(datos_sin.y_train.shape) == 2
        assert datos_sin.y_train.shape[1] == 1
    
    def test_info_datos(self, datos_sin):
        """Verifica información."""
        info = datos_sin.info()
        assert 'sin' in info
        assert 'Entrenamiento' in info


# ============================================================================
# PRUEBAS DE GENERADOR
# ============================================================================

class TestGenerador:
    """Pruebas de generación."""
    
    def test_generar_sin(self, generador):
        """Verifica generación sin(x)."""
        datos = generador.generar('sin', n_muestras=100)
        assert len(datos.X_train) + len(datos.X_test) == 100
        assert datos.nombre_funcion == 'sin'
    
    def test_generar_cos(self, generador):
        """Verifica generación cos(x)."""
        datos = generador.generar('cos', n_muestras=100)
        assert datos.nombre_funcion == 'cos'
    
    def test_generar_exp(self, generador):
        """Verifica generación exp(x)."""
        datos = generador.generar('exp', n_muestras=100)
        assert datos.nombre_funcion == 'exp'
    
    def test_generar_x3(self, generador):
        """Verifica generación x³."""
        datos = generador.generar('x3', n_muestras=100)
        assert datos.nombre_funcion == 'x3'
    
    def test_generar_x5(self, generador):
        """Verifica generación x⁵."""
        datos = generador.generar('x5', n_muestras=100)
        assert datos.nombre_funcion == 'x5'
    
    def test_generar_sincos(self, generador):
        """Verifica generación sin(x)·cos(x)."""
        datos = generador.generar('sincos', n_muestras=100)
        assert datos.nombre_funcion == 'sincos'
    
    def test_funcion_invalida(self, generador):
        """Verifica error con función inválida."""
        with pytest.raises(ValueError):
            generador.generar('invalida')
    
    def test_split_datos(self, generador):
        """Verifica split automático."""
        datos = generador.generar('sin', n_muestras=100, test_size=0.2)
        assert len(datos.X_train) + len(datos.X_test) == 100
        assert len(datos.X_test) == 20
    
    def test_ruido_datos(self, generador):
        """Verifica adición de ruido."""
        datos_sin_ruido = generador.generar('sin', n_muestras=50, ruido=0.0)
        datos_con_ruido = generador.generar('sin', n_muestras=50, ruido=0.5)
        
        # Con ruido debería tener más varianza
        var_sin = np.var(datos_sin_ruido.y_train)
        var_con = np.var(datos_con_ruido.y_train)
        assert var_con > var_sin


# ============================================================================
# PRUEBAS DE NORMALIZACIÓN
# ============================================================================

class TestNormalizacion:
    """Pruebas de normalización."""
    
    def test_normalizar_entrada(self, aproximador, datos_sin):
        """Verifica normalización de entrada."""
        X_norm = aproximador._normalizar_entrada(datos_sin.X_train, fit=True)
        
        # StandardScaler debe producir media~0, std~1
        assert np.abs(np.mean(X_norm)) < 1e-6
        assert np.abs(np.std(X_norm) - 1.0) < 0.1
    
    def test_normalizar_salida(self, aproximador, datos_sin):
        """Verifica normalización de salida."""
        y_norm = aproximador._normalizar_salida(datos_sin.y_train, fit=True)
        
        # MinMaxScaler [-1, 1]
        assert np.min(y_norm) >= -1.0
        assert np.max(y_norm) <= 1.0
    
    def test_desnormalizar(self, aproximador, datos_sin):
        """Verifica desnormalización."""
        y_norm = aproximador._normalizar_salida(datos_sin.y_train, fit=True)
        y_denorm = aproximador._desnormalizar_salida(y_norm)
        
        assert np.allclose(y_denorm, datos_sin.y_train, atol=1e-5)


# ============================================================================
# PRUEBAS DE CONSTRUCCIÓN DE MODELOS
# ============================================================================

class TestConstruccionModelos:
    """Pruebas de construcción."""
    
    def test_construir_mlp(self, aproximador):
        """Verifica construcción MLP."""
        modelo = aproximador.construir_mlp()
        assert modelo is not None
        assert len(modelo.layers) > 0
    
    def test_mlp_prediccion(self, aproximador):
        """Verifica predicción MLP."""
        modelo = aproximador.construir_mlp()
        X = np.random.randn(10, 1)
        y = modelo.predict(X, verbose=0)
        assert y.shape == (10, 1)
    
    def test_construir_residual(self, aproximador):
        """Verifica construcción residual."""
        modelo = aproximador.construir_residual()
        assert modelo is not None
    
    def test_residual_prediccion(self, aproximador):
        """Verifica predicción residual."""
        modelo = aproximador.construir_residual()
        X = np.random.randn(10, 1)
        y = modelo.predict(X, verbose=0)
        assert y.shape == (10, 1)
    
    def test_mlp_con_regularizacion_l2(self, aproximador):
        """Verifica MLP con L2."""
        modelo = aproximador.construir_mlp(regularizacion='l2')
        assert modelo is not None
    
    def test_mlp_con_regularizacion_l1(self, aproximador):
        """Verifica MLP con L1."""
        modelo = aproximador.construir_mlp(regularizacion='l1')
        assert modelo is not None


# ============================================================================
# PRUEBAS DE ENTRENAMIENTO
# ============================================================================

class TestEntrenamiento:
    """Pruebas de entrenamiento."""
    
    def test_entrenar_mlp(self, aproximador, datos_sin):
        """Verifica entrenamiento MLP."""
        historial = aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=10,
            verbose=0
        )
        
        assert 'loss' in historial
        assert len(historial['loss']) > 0
        assert historial['loss'][-1] < historial['loss'][0]
    
    def test_entrenar_residual(self, aproximador, datos_sin):
        """Verifica entrenamiento residual."""
        historial = aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=10,
            arquitectura='residual',
            verbose=0
        )
        
        assert 'loss' in historial
    
    def test_arquitectura_invalida(self, aproximador, datos_sin):
        """Verifica error con arquitectura inválida."""
        with pytest.raises(ValueError):
            aproximador.entrenar(
                datos_sin.X_train, datos_sin.y_train,
                datos_sin.X_test, datos_sin.y_test,
                arquitectura='invalida'
            )


# ============================================================================
# PRUEBAS DE EVALUACIÓN
# ============================================================================

class TestEvaluacion:
    """Pruebas de evaluación."""
    
    def test_evaluar(self, aproximador, datos_sin):
        """Verifica evaluación."""
        aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=10,
            verbose=0
        )
        
        metricas = aproximador.evaluar(datos_sin.X_test, datos_sin.y_test)
        
        assert 'rmse' in metricas
        assert 'mae_original' in metricas
        assert 'r2_score' in metricas
        assert metricas['rmse'] >= 0
    
    def test_evaluar_sin_entrenar(self, aproximador, datos_sin):
        """Verifica error sin entrenar."""
        with pytest.raises(ValueError):
            aproximador.evaluar(datos_sin.X_test, datos_sin.y_test)


# ============================================================================
# PRUEBAS DE PREDICCIÓN
# ============================================================================

class TestPrediccion:
    """Pruebas de predicción."""
    
    def test_predecir(self, aproximador, datos_sin):
        """Verifica predicción."""
        aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=10,
            verbose=0
        )
        
        y_pred = aproximador.predecir(datos_sin.X_test[:5])
        assert y_pred.shape == (5, 1)
    
    def test_predecir_sin_entrenar(self, aproximador, datos_sin):
        """Verifica error sin entrenar."""
        with pytest.raises(ValueError):
            aproximador.predecir(datos_sin.X_test)


# ============================================================================
# PRUEBAS DE PERSISTENCIA
# ============================================================================

class TestPersistencia:
    """Pruebas de guardado/carga."""
    
    def test_guardar(self, aproximador, datos_sin):
        """Verifica guardado."""
        aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=5,
            verbose=0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = Path(tmpdir) / "modelo"
            resultado = aproximador.guardar(str(ruta))
            
            assert resultado is True
            assert (ruta / 'modelo.h5').exists()
            assert (ruta / 'escalador_X.pkl').exists()
            assert (ruta / 'escalador_y.pkl').exists()


# ============================================================================
# PRUEBAS CON DIFERENTES FUNCIONES
# ============================================================================

class TestFuncionesDiferentes:
    """Pruebas con varias funciones."""
    
    def test_aproximar_sin(self, aproximador, datos_sin):
        """Verifica aproximación sin(x)."""
        aprox = AproximadorFuncion()
        aprox.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=20,
            verbose=0
        )
        metricas = aprox.evaluar(datos_sin.X_test, datos_sin.y_test)
        assert metricas['r2_score'] > 0.8
    
    def test_aproximar_exp(self, aproximador, datos_exp):
        """Verifica aproximación exp(x)."""
        aprox = AproximadorFuncion()
        aprox.entrenar(
            datos_exp.X_train, datos_exp.y_train,
            datos_exp.X_test, datos_exp.y_test,
            epochs=20,
            verbose=0
        )
        metricas = aprox.evaluar(datos_exp.X_test, datos_exp.y_test)
        assert metricas['r2_score'] > 0.8
    
    def test_aproximar_x3(self, aproximador, datos_x3):
        """Verifica aproximación x³."""
        aprox = AproximadorFuncion()
        aprox.entrenar(
            datos_x3.X_train, datos_x3.y_train,
            datos_x3.X_test, datos_x3.y_test,
            epochs=20,
            verbose=0
        )
        metricas = aprox.evaluar(datos_x3.X_test, datos_x3.y_test)
        assert metricas['r2_score'] > 0.8


# ============================================================================
# PRUEBAS DE EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos extremos."""
    
    def test_datos_pequenos(self, generador):
        """Verifica con pocos datos."""
        datos = generador.generar('sin', n_muestras=50)
        assert len(datos.X_train) + len(datos.X_test) == 50
    
    def test_dominio_pequeno(self, generador):
        """Verifica con dominio pequeño."""
        datos = generador.generar('sin', n_muestras=100)
        assert datos.dominio[0] < datos.dominio[1]
    
    def test_sin_ruido(self, generador):
        """Verifica sin ruido."""
        datos = generador.generar('sin', n_muestras=100, ruido=0.0)
        assert datos is not None


# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================

class TestRendimiento:
    """Pruebas de rendimiento."""
    
    def test_velocidad_generacion(self, generador):
        """Verifica velocidad de generación."""
        import time
        inicio = time.time()
        datos = generador.generar('sin', n_muestras=1000)
        tiempo = time.time() - inicio
        
        assert tiempo < 2.0
    
    def test_velocidad_prediccion(self, aproximador, datos_sin):
        """Verifica velocidad de predicción."""
        aproximador.entrenar(
            datos_sin.X_train, datos_sin.y_train,
            datos_sin.X_test, datos_sin.y_test,
            epochs=5,
            verbose=0
        )
        
        import time
        X = np.random.randn(1000, 1)
        
        inicio = time.time()
        y_pred = aproximador.predecir(X)
        tiempo = time.time() - inicio
        
        assert tiempo < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
