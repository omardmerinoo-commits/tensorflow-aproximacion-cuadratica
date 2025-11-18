"""
Suite de tests exhaustiva para los modelos cuadráticos.

Incluye pruebas unitarias, de integración y de regresión
para la clase ModeloCuadratico y ModeloCuadraticoMejorado.

Pytest es el framework de testing usado.
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import tempfile
from pathlib import Path

# Importar los modelos
from modelo_cuadratico import ModeloCuadratico
from modelo_cuadratico_mejorado import ModeloCuadraticoMejorado


class TestModeloCuadratico:
    """Tests para la clase ModeloCuadratico base."""
    
    @pytest.fixture
    def modelo(self):
        """Fixture que proporciona una instancia limpia del modelo."""
        return ModeloCuadratico()
    
    # =============== Tests de Inicialización ===============
    
    def test_inicializacion(self, modelo):
        """Test que el modelo se inicializa correctamente."""
        assert modelo.modelo is None
        assert modelo.x_train is None
        assert modelo.y_train is None
        assert modelo.history is None
    
    # =============== Tests de Generación de Datos ===============
    
    def test_generar_datos_basico(self, modelo):
        """Test generación básica de datos."""
        x, y = modelo.generar_datos(n_samples=100)
        assert x.shape == (100, 1)
        assert y.shape == (100, 1)
        assert modelo.x_train.shape == (100, 1)
        assert modelo.y_train.shape == (100, 1)
    
    def test_generar_datos_rango_personalizado(self, modelo):
        """Test con rango personalizado."""
        x, y = modelo.generar_datos(n_samples=50, rango=(-2, 2))
        assert np.all(x >= -2) and np.all(x <= 2)
    
    def test_generar_datos_sin_ruido(self, modelo):
        """Test sin ruido."""
        x, y = modelo.generar_datos(n_samples=10, rango=(0, 1), ruido=0.0)
        # y debería estar muy cerca de x²
        y_teorica = x ** 2
        assert np.allclose(y, y_teorica, atol=1e-6)
    
    def test_generar_datos_reproducibilidad(self, modelo):
        """Test reproducibilidad con seed."""
        modelo1 = ModeloCuadratico()
        modelo2 = ModeloCuadratico()
        
        x1, y1 = modelo1.generar_datos(n_samples=100, seed=42)
        x2, y2 = modelo2.generar_datos(n_samples=100, seed=42)
        
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
    
    def test_generar_datos_error_n_samples_negativo(self, modelo):
        """Test error con n_samples negativo."""
        with pytest.raises(ValueError):
            modelo.generar_datos(n_samples=-1)
    
    def test_generar_datos_error_rango_invalido(self, modelo):
        """Test error con rango inválido."""
        with pytest.raises(ValueError):
            modelo.generar_datos(rango=(2, 1))
    
    def test_generar_datos_error_ruido_negativo(self, modelo):
        """Test error con ruido negativo."""
        with pytest.raises(ValueError):
            modelo.generar_datos(ruido=-0.1)
    
    # =============== Tests de Construcción del Modelo ===============
    
    def test_construir_modelo(self, modelo):
        """Test construcción del modelo."""
        modelo.construir_modelo()
        assert modelo.modelo is not None
        assert isinstance(modelo.modelo, tf.keras.Model)
        assert len(modelo.modelo.layers) == 3  # 2 ocultas + 1 salida
    
    def test_construir_modelo_parametros(self, modelo):
        """Test que el modelo tiene el número correcto de parámetros."""
        modelo.construir_modelo()
        # (1*64 + 64) + (64*64 + 64) + (64*1 + 1) = 64+64 + 4096+64 + 64+1 = 4353
        assert modelo.modelo.count_params() == 4353
    
    # =============== Tests de Entrenamiento ===============
    
    def test_entrenar_error_sin_modelo(self, modelo):
        """Test error al entrenar sin modelo construido."""
        modelo.generar_datos(n_samples=10)
        with pytest.raises(RuntimeError):
            modelo.entrenar()
    
    def test_entrenar_error_sin_datos(self, modelo):
        """Test error al entrenar sin datos."""
        modelo.construir_modelo()
        with pytest.raises(RuntimeError):
            modelo.entrenar()
    
    def test_entrenar_basico(self, modelo):
        """Test entrenamiento básico."""
        modelo.generar_datos(n_samples=50)
        modelo.construir_modelo()
        history = modelo.entrenar(epochs=10, batch_size=16)
        
        assert modelo.history is not None
        assert 'loss' in historia.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) > 0
    
    def test_entrenar_loss_disminuye(self, modelo):
        """Test que loss disminuye durante el entrenamiento."""
        modelo.generar_datos(n_samples=100)
        modelo.construir_modelo()
        history = modelo.entrenar(epochs=20, batch_size=32, verbose=0)
        
        # El loss final debería ser menor que el inicial (con alta probabilidad)
        loss_inicial = history.history['loss'][0]
        loss_final = history.history['loss'][-1]
        assert loss_final < loss_inicial
    
    # =============== Tests de Predicción ===============
    
    def test_predecir_error_sin_modelo(self, modelo):
        """Test error al predecir sin modelo."""
        x = np.array([[0.5]])
        with pytest.raises(RuntimeError):
            modelo.predecir(x)
    
    def test_predecir_basico(self, modelo):
        """Test predicción básica."""
        modelo.generar_datos(n_samples=50)
        modelo.construir_modelo()
        modelo.entrenar(epochs=10, verbose=0)
        
        x_test = np.array([[0.5]])
        prediccion = modelo.predecir(x_test)
        
        assert prediccion.shape == (1, 1)
        assert isinstance(prediccion, np.ndarray)
    
    def test_predecir_dimension_1d(self, modelo):
        """Test predicción con entrada 1D."""
        modelo.generar_datos(n_samples=50)
        modelo.construir_modelo()
        modelo.entrenar(epochs=10, verbose=0)
        
        x_test = np.array([0.5, 1.0])
        prediccion = modelo.predecir(x_test)
        
        assert prediccion.shape == (2, 1)
    
    def test_predecir_valores_razonables(self, modelo):
        """Test que las predicciones son razonables."""
        modelo.generar_datos(n_samples=100, ruido=0.01)
        modelo.construir_modelo()
        modelo.entrenar(epochs=50, verbose=0)
        
        # Probar con valores conocidos
        x_test = np.array([[0.0], [0.5], [1.0]])
        predicciones = modelo.predecir(x_test)
        
        # Deberían estar cerca de 0, 0.25, 1 respectivamente
        assert abs(predicciones[0][0] - 0.0) < 0.1
        assert abs(predicciones[1][0] - 0.25) < 0.15
        assert abs(predicciones[2][0] - 1.0) < 0.2
    
    # =============== Tests de Persistencia ===============
    
    def test_guardar_modelo(self, modelo):
        """Test guardado de modelo."""
        modelo.generar_datos(n_samples=50)
        modelo.construir_modelo()
        modelo.entrenar(epochs=5, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_keras = os.path.join(tmpdir, 'modelo.keras')
            ruta_pkl = os.path.join(tmpdir, 'modelo.pkl')
            
            modelo.guardar_modelo(ruta_keras, ruta_pkl)
            
            assert os.path.exists(ruta_keras)
            assert os.path.exists(ruta_pkl)
    
    def test_cargar_modelo(self, modelo):
        """Test carga de modelo."""
        # Crear y guardar modelo original
        modelo1 = ModeloCuadratico()
        modelo1.generar_datos(n_samples=50)
        modelo1.construir_modelo()
        modelo1.entrenar(epochs=5, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_keras = os.path.join(tmpdir, 'modelo.keras')
            modelo1.guardar_modelo(ruta_keras)
            
            # Cargar en nuevo modelo
            modelo2 = ModeloCuadratico()
            modelo2.cargar_modelo(ruta_keras)
            
            assert modelo2.modelo is not None
            assert modelo1.modelo.count_params() == modelo2.modelo.count_params()
    
    # =============== Tests de Métodos Auxiliares ===============
    
    def test_resumen(self, modelo, capsys):
        """Test que resumen() imprime información."""
        modelo.construir_modelo()
        modelo.resumen()
        
        capturado = capsys.readouterr()
        assert 'RESUMEN' in capturado.out
        assert 'Arquitectura' in capturado.out


class TestModeloCuadraticoMejorado:
    """Tests para la clase mejorada."""
    
    @pytest.fixture
    def modelo(self):
        """Fixture para el modelo mejorado."""
        return ModeloCuadraticoMejorado()
    
    def test_inicializacion(self, modelo):
        """Test inicialización del modelo mejorado."""
        assert modelo.modelo is None
        assert modelo.x_train is None
        assert modelo.x_test is None
    
    def test_generar_datos_split(self, modelo):
        """Test generación con split train/test."""
        (x_train, y_train), (x_test, y_test) = modelo.generar_datos(
            n_samples=100, test_size=0.2
        )
        
        assert len(x_train) == 80
        assert len(x_test) == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
    
    def test_evaluar(self, modelo):
        """Test evaluación del modelo."""
        modelo.generar_datos(n_samples=100)
        modelo.construir_modelo()
        modelo.entrenar(epochs=20, verbose=0)
        
        metricas = modelo.evaluar()
        
        assert 'mse' in metricas
        assert 'rmse' in metricas
        assert 'mae' in metricas
        assert 'r2' in metricas
        assert metricas['rmse'] == np.sqrt(metricas['mse'])
    
    def test_visualizar_predicciones(self, modelo):
        """Test generación de visualización."""
        modelo.generar_datos(n_samples=100)
        modelo.construir_modelo()
        modelo.entrenar(epochs=20, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'predicciones.png')
            modelo.visualizar_predicciones(ruta)
            
            assert os.path.exists(ruta)
            assert os.path.getsize(ruta) > 0
    
    def test_exportar_reporte(self, modelo):
        """Test exportación de reporte."""
        modelo.generar_datos(n_samples=50)
        modelo.construir_modelo()
        modelo.entrenar(epochs=10, verbose=0)
        modelo.evaluar()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'reporte.json')
            modelo.exportar_reporte(ruta)
            
            assert os.path.exists(ruta)
            
            import json
            with open(ruta) as f:
                reporte = json.load(f)
            
            assert 'configuracion' in reporte
            assert 'metricas' in reporte
            assert 'arquitectura' in reporte
    
    def test_validacion_cruzada(self, modelo):
        """Test validación cruzada."""
        modelo.generar_datos(n_samples=100)
        
        cv_stats = modelo.validacion_cruzada(k_folds=3)
        
        assert 'mse_mean' in cv_stats
        assert 'mae_mean' in cv_stats
        assert 'r2_mean' in cv_stats
        assert cv_stats['mse_mean'] > 0
    
    def test_predicciones_mejoradas(self, modelo):
        """Test predicciones del modelo mejorado."""
        modelo.generar_datos(n_samples=100)
        modelo.construir_modelo(capas=[32, 32])
        modelo.entrenar(epochs=30, verbose=0)
        
        x_test = np.array([[0.5], [1.0]])
        predicciones = modelo.predecir(x_test)
        
        assert predicciones.shape == (2, 1)


class TestIntegracion:
    """Tests de integración entre ambos modelos."""
    
    def test_comparacion_predicciones(self):
        """Test que ambos modelos generan predicciones similares."""
        # Usar mismos datos
        modelo1 = ModeloCuadratico()
        modelo2 = ModeloCuadraticoMejorado()
        
        # Generar datos idénticos
        x1, y1 = modelo1.generar_datos(n_samples=50, seed=42)
        x2, y2 = modelo2.generar_datos(n_samples=50, seed=42)
        
        assert np.allclose(x1, x2)
        assert np.allclose(y1, y2)
        
        # Construir y entrenar ambos
        modelo1.construir_modelo()
        modelo1.entrenar(epochs=5, verbose=0)
        
        modelo2.construir_modelo()
        modelo2.entrenar(epochs=5, verbose=0)
        
        # Predicciones deberían ser similares
        x_test = np.array([[0.5]])
        pred1 = modelo1.predecir(x_test)
        pred2 = modelo2.predecir(x_test)
        
        # Deberían estar cercanas (pero no necesariamente idénticas por aleatoriedades)
        assert abs(pred1[0][0] - pred2[0][0]) < 1.0  # Tolerancia razonable


# Tests de Rendimiento
class TestRendimiento:
    """Tests de rendimiento y escalabilidad."""
    
    def test_escalabilidad_n_samples_grande(self):
        """Test con muchas muestras."""
        modelo = ModeloCuadratico()
        x, y = modelo.generar_datos(n_samples=10000)
        
        assert x.shape[0] == 10000
        assert y.shape[0] == 10000
    
    def test_rango_grande(self):
        """Test con rango numérico grande."""
        modelo = ModeloCuadratico()
        x, y = modelo.generar_datos(rango=(-100, 100))
        
        assert x.min() >= -100
        assert x.max() <= 100


if __name__ == "__main__":
    # Ejecutar todos los tests
    pytest.main([__file__, '-v', '--tb=short'])
