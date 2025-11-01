"""
Tests automatizados para el modelo de aproximación cuadrática.

Este módulo contiene pruebas unitarias para verificar el correcto
funcionamiento de la clase ModeloCuadratico y sus métodos.

Uso:
    pytest test_model.py -v
    
O alternativamente:
    python -m pytest test_model.py -v

Autor: Proyecto TensorFlow
Fecha: Noviembre 2025
"""

import pytest
import numpy as np
import tensorflow as tf
import os
import tempfile
from modelo_cuadratico import ModeloCuadratico


class TestModeloCuadratico:
    """Suite de tests para la clase ModeloCuadratico."""
    
    @pytest.fixture
    def modelo(self):
        """
        Fixture que proporciona una instancia fresca de ModeloCuadratico.
        
        Yields
        ------
        ModeloCuadratico
            Instancia nueva del modelo para cada test.
        """
        return ModeloCuadratico()
    
    @pytest.fixture
    def modelo_entrenado(self):
        """
        Fixture que proporciona un modelo entrenado.
        
        Yields
        ------
        ModeloCuadratico
            Instancia del modelo ya entrenada.
        """
        modelo = ModeloCuadratico()
        modelo.generar_datos(n_samples=100, rango=(-1, 1), ruido=0.01, seed=42)
        modelo.construir_modelo()
        modelo.entrenar(epochs=10, batch_size=16, validation_split=0.2)
        return modelo
    
    # ==================== Tests de generación de datos ====================
    
    def test_generar_datos_forma_correcta(self, modelo):
        """Verifica que los datos generados tengan la forma correcta."""
        n_samples = 500
        x, y = modelo.generar_datos(n_samples=n_samples, seed=42)
        
        assert x.shape == (n_samples, 1), f"Forma de x incorrecta: {x.shape}"
        assert y.shape == (n_samples, 1), f"Forma de y incorrecta: {y.shape}"
    
    def test_generar_datos_rango_correcto(self, modelo):
        """Verifica que los datos estén en el rango especificado."""
        rango = (-2, 3)
        x, y = modelo.generar_datos(n_samples=1000, rango=rango, seed=42)
        
        assert x.min() >= rango[0], f"x mínimo fuera de rango: {x.min()}"
        assert x.max() <= rango[1], f"x máximo fuera de rango: {x.max()}"
    
    def test_generar_datos_relacion_cuadratica(self, modelo):
        """Verifica que y ≈ x² (dentro de tolerancia por ruido)."""
        x, y = modelo.generar_datos(n_samples=1000, rango=(-1, 1), ruido=0.01, seed=42)
        
        y_esperado = x ** 2
        diferencia_media = np.mean(np.abs(y - y_esperado))
        
        # La diferencia media debe ser pequeña (cercana al ruido)
        assert diferencia_media < 0.05, f"Diferencia muy grande: {diferencia_media}"
    
    def test_generar_datos_tipo_correcto(self, modelo):
        """Verifica que los datos sean arrays de numpy con tipo float32."""
        x, y = modelo.generar_datos(seed=42)
        
        assert isinstance(x, np.ndarray), "x no es numpy array"
        assert isinstance(y, np.ndarray), "y no es numpy array"
        assert x.dtype == np.float32, f"Tipo de x incorrecto: {x.dtype}"
        assert y.dtype == np.float32, f"Tipo de y incorrecto: {y.dtype}"
    
    def test_generar_datos_reproducibilidad(self, modelo):
        """Verifica que usar la misma semilla produzca los mismos datos."""
        x1, y1 = modelo.generar_datos(n_samples=100, seed=42)
        x2, y2 = modelo.generar_datos(n_samples=100, seed=42)
        
        np.testing.assert_array_equal(x1, x2, err_msg="x no es reproducible")
        np.testing.assert_array_equal(y1, y2, err_msg="y no es reproducible")
    
    def test_generar_datos_validacion_parametros(self, modelo):
        """Verifica que se validen correctamente los parámetros."""
        with pytest.raises(ValueError):
            modelo.generar_datos(n_samples=-10)
        
        with pytest.raises(ValueError):
            modelo.generar_datos(rango=(5, 1))  # min > max
        
        with pytest.raises(ValueError):
            modelo.generar_datos(ruido=-0.1)
    
    # ==================== Tests de construcción del modelo ====================
    
    def test_construir_modelo_crea_modelo(self, modelo):
        """Verifica que se cree un modelo de Keras."""
        modelo.construir_modelo()
        
        assert modelo.modelo is not None, "Modelo no fue creado"
        assert isinstance(modelo.modelo, tf.keras.Model), "No es un modelo de Keras"
    
    def test_construir_modelo_arquitectura_correcta(self, modelo):
        """Verifica que la arquitectura tenga el número correcto de capas."""
        modelo.construir_modelo()
        
        # Debe tener 3 capas: 2 ocultas + 1 salida
        assert len(modelo.modelo.layers) == 3, f"Número de capas incorrecto: {len(modelo.modelo.layers)}"
    
    def test_construir_modelo_capas_densas(self, modelo):
        """Verifica que todas las capas sean Dense."""
        modelo.construir_modelo()
        
        for capa in modelo.modelo.layers:
            assert isinstance(capa, tf.keras.layers.Dense), f"Capa no es Dense: {type(capa)}"
    
    def test_construir_modelo_unidades_correctas(self, modelo):
        """Verifica que las capas tengan el número correcto de neuronas."""
        modelo.construir_modelo()
        
        assert modelo.modelo.layers[0].units == 64, "Capa 1 debe tener 64 neuronas"
        assert modelo.modelo.layers[1].units == 64, "Capa 2 debe tener 64 neuronas"
        assert modelo.modelo.layers[2].units == 1, "Capa de salida debe tener 1 neurona"
    
    def test_construir_modelo_activaciones_correctas(self, modelo):
        """Verifica que las activaciones sean correctas."""
        modelo.construir_modelo()
        
        # Capas ocultas deben usar ReLU
        assert modelo.modelo.layers[0].activation.__name__ == 'relu'
        assert modelo.modelo.layers[1].activation.__name__ == 'relu'
        
        # Capa de salida debe usar activación lineal
        assert modelo.modelo.layers[2].activation.__name__ == 'linear'
    
    def test_construir_modelo_compilado(self, modelo):
        """Verifica que el modelo esté compilado."""
        modelo.construir_modelo()
        
        # Un modelo compilado tiene optimizer configurado
        assert modelo.modelo.optimizer is not None, "Modelo no está compilado"
    
    # ==================== Tests de entrenamiento ====================
    
    def test_entrenar_sin_modelo_falla(self, modelo):
        """Verifica que entrenar sin construir el modelo lance error."""
        modelo.generar_datos(n_samples=100, seed=42)
        
        with pytest.raises(RuntimeError):
            modelo.entrenar(epochs=1)
    
    def test_entrenar_sin_datos_falla(self, modelo):
        """Verifica que entrenar sin datos lance error."""
        modelo.construir_modelo()
        
        with pytest.raises(RuntimeError):
            modelo.entrenar(epochs=1)
    
    def test_entrenar_retorna_history(self, modelo):
        """Verifica que entrenar retorne un objeto History."""
        modelo.generar_datos(n_samples=100, seed=42)
        modelo.construir_modelo()
        
        history = modelo.entrenar(epochs=5, batch_size=16, validation_split=0.2)
        
        assert history is not None, "No se retornó History"
        assert hasattr(history, 'history'), "History no tiene atributo 'history'"
    
    def test_entrenar_reduce_perdida(self, modelo):
        """Verifica que el entrenamiento reduzca la pérdida."""
        modelo.generar_datos(n_samples=200, seed=42)
        modelo.construir_modelo()
        
        history = modelo.entrenar(epochs=20, batch_size=16, validation_split=0.2)
        
        loss_inicial = history.history['loss'][0]
        loss_final = history.history['loss'][-1]
        
        assert loss_final < loss_inicial, "La pérdida no disminuyó durante el entrenamiento"
    
    def test_entrenar_validacion_parametros(self, modelo):
        """Verifica validación de parámetros de entrenamiento."""
        modelo.generar_datos(n_samples=100, seed=42)
        modelo.construir_modelo()
        
        with pytest.raises(ValueError):
            modelo.entrenar(epochs=-5)
        
        with pytest.raises(ValueError):
            modelo.entrenar(batch_size=0)
        
        with pytest.raises(ValueError):
            modelo.entrenar(validation_split=1.5)
    
    # ==================== Tests de predicción ====================
    
    def test_predecir_forma_correcta(self, modelo_entrenado):
        """Verifica que las predicciones tengan la forma correcta."""
        x_test = np.array([[0.5], [1.0], [1.5]])
        predicciones = modelo_entrenado.predecir(x_test)
        
        assert predicciones.shape == (3, 1), f"Forma incorrecta: {predicciones.shape}"
    
    def test_predecir_sin_modelo_falla(self, modelo):
        """Verifica que predecir sin modelo lance error."""
        x_test = np.array([[0.5]])
        
        with pytest.raises(RuntimeError):
            modelo.predecir(x_test)
    
    def test_predecir_aproximacion_razonable(self, modelo_entrenado):
        """Verifica que las predicciones sean razonablemente cercanas a x²."""
        x_test = np.array([[0.0], [0.5], [1.0]])
        predicciones = modelo_entrenado.predecir(x_test)
        
        y_esperado = x_test ** 2
        
        # Permitir un margen de error razonable
        diferencia = np.abs(predicciones - y_esperado)
        assert np.all(diferencia < 0.1), f"Predicciones muy alejadas de x²: {diferencia}"
    
    def test_predecir_acepta_1d(self, modelo_entrenado):
        """Verifica que predecir acepte arrays 1D y los convierta correctamente."""
        x_test_1d = np.array([0.5, 1.0, 1.5])
        predicciones = modelo_entrenado.predecir(x_test_1d)
        
        assert predicciones.shape == (3, 1), "No convirtió correctamente de 1D a 2D"
    
    def test_predecir_validacion_entrada(self, modelo_entrenado):
        """Verifica validación de entrada en predicción."""
        with pytest.raises(ValueError):
            modelo_entrenado.predecir("no es un array")
        
        with pytest.raises(ValueError):
            # Array con más de 1 columna
            modelo_entrenado.predecir(np.array([[1, 2], [3, 4]]))
    
    # ==================== Tests de guardado y carga ====================
    
    def test_guardar_modelo_crea_archivos(self, modelo_entrenado):
        """Verifica que guardar el modelo cree los archivos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_h5 = os.path.join(tmpdir, "test_modelo.h5")
            path_pkl = os.path.join(tmpdir, "test_modelo.pkl")
            
            modelo_entrenado.guardar_modelo(path_h5, path_pkl)
            
            assert os.path.exists(path_h5), "Archivo .h5 no fue creado"
            assert os.path.exists(path_pkl), "Archivo .pkl no fue creado"
    
    def test_guardar_sin_modelo_falla(self, modelo):
        """Verifica que guardar sin modelo lance error."""
        with pytest.raises(RuntimeError):
            modelo.guardar_modelo("test.h5", "test.pkl")
    
    def test_cargar_modelo_h5_funciona(self, modelo_entrenado):
        """Verifica que cargar desde .h5 funcione correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_h5 = os.path.join(tmpdir, "test_modelo.h5")
            
            # Guardar
            modelo_entrenado.guardar_modelo(path_h5, path_h5.replace('.h5', '.pkl'))
            
            # Cargar en nuevo modelo
            modelo_nuevo = ModeloCuadratico()
            modelo_nuevo.cargar_modelo(path_tf=path_h5)
            
            assert modelo_nuevo.modelo is not None, "Modelo no fue cargado"
    
    def test_cargar_modelo_pkl_funciona(self, modelo_entrenado):
        """Verifica que cargar desde .pkl funcione correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_pkl = os.path.join(tmpdir, "test_modelo.pkl")
            
            # Guardar
            modelo_entrenado.guardar_modelo(path_pkl.replace('.pkl', '.h5'), path_pkl)
            
            # Cargar en nuevo modelo
            modelo_nuevo = ModeloCuadratico()
            modelo_nuevo.cargar_modelo(path_pkl=path_pkl)
            
            assert modelo_nuevo.modelo is not None, "Modelo no fue cargado"
    
    def test_cargar_modelo_predicciones_identicas(self, modelo_entrenado):
        """Verifica que el modelo cargado produzca predicciones idénticas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_h5 = os.path.join(tmpdir, "test_modelo.h5")
            path_pkl = os.path.join(tmpdir, "test_modelo.pkl")
            
            # Guardar
            modelo_entrenado.guardar_modelo(path_h5, path_pkl)
            
            # Predicciones originales
            x_test = np.array([[0.5], [1.0]])
            pred_original = modelo_entrenado.predecir(x_test)
            
            # Cargar y predecir
            modelo_cargado = ModeloCuadratico()
            modelo_cargado.cargar_modelo(path_tf=path_h5)
            pred_cargado = modelo_cargado.predecir(x_test)
            
            np.testing.assert_array_almost_equal(
                pred_original, pred_cargado,
                decimal=6,
                err_msg="Las predicciones difieren después de cargar"
            )
    
    def test_cargar_sin_ruta_falla(self, modelo):
        """Verifica que cargar sin proporcionar ruta lance error."""
        with pytest.raises(ValueError):
            modelo.cargar_modelo()
    
    def test_cargar_archivo_inexistente_falla(self, modelo):
        """Verifica que cargar archivo inexistente lance error."""
        with pytest.raises(ValueError):
            modelo.cargar_modelo(path_tf="archivo_que_no_existe.h5")
    
    # ==================== Tests de integración ====================
    
    def test_flujo_completo(self):
        """Test de integración del flujo completo."""
        # 1. Crear modelo
        modelo = ModeloCuadratico()
        
        # 2. Generar datos
        x, y = modelo.generar_datos(n_samples=200, rango=(-1, 1), seed=42)
        assert x.shape == (200, 1)
        
        # 3. Construir
        modelo.construir_modelo()
        assert modelo.modelo is not None
        
        # 4. Entrenar
        history = modelo.entrenar(epochs=10, batch_size=16)
        assert len(history.history['loss']) > 0
        
        # 5. Predecir
        predicciones = modelo.predecir(np.array([[0.5]]))
        assert predicciones.shape == (1, 1)
        
        # 6. Guardar y cargar
        with tempfile.TemporaryDirectory() as tmpdir:
            path_h5 = os.path.join(tmpdir, "modelo.h5")
            path_pkl = os.path.join(tmpdir, "modelo.pkl")
            
            modelo.guardar_modelo(path_h5, path_pkl)
            
            modelo_nuevo = ModeloCuadratico()
            modelo_nuevo.cargar_modelo(path_tf=path_h5)
            
            pred_nuevo = modelo_nuevo.predecir(np.array([[0.5]]))
            np.testing.assert_array_almost_equal(predicciones, pred_nuevo)


# ==================== Tests adicionales ====================

def test_version_tensorflow():
    """Verifica que TensorFlow esté instalado y sea versión 2.x."""
    assert tf.__version__.startswith('2.'), f"TensorFlow debe ser 2.x, se encontró: {tf.__version__}"


def test_version_numpy():
    """Verifica que numpy esté instalado."""
    assert np.__version__ is not None, "NumPy no está instalado correctamente"


# ==================== Ejecución directa ====================

if __name__ == "__main__":
    """
    Permite ejecutar los tests directamente sin pytest.
    """
    print("Para ejecutar los tests, use:")
    print("  pytest test_model.py -v")
    print("\nO instale pytest con:")
    print("  pip install pytest")
