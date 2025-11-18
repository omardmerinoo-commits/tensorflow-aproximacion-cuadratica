"""
Suite de tests para el módulo de oscilaciones amortiguadas.

Tests incluyen:
- Validación de generación de datos
- Validación de solución analítica
- Validación de arquitectura del modelo
- Validación de entrenamiento
- Validación de serialización
"""

import pytest
import numpy as np
from pathlib import Path
import os
from oscilaciones_amortiguadas import OscilacionesAmortiguadas


class TestSolucionAnalitica:
    """Tests para la solución analítica."""
    
    def test_solucion_analitica_regresa_array(self):
        """Verifica que la solución analítica retorna un array."""
        t = np.linspace(0, 10, 100)
        x = OscilacionesAmortiguadas.solucion_analitica(t, m=1.0, c=1.0, k=1.0)
        
        assert isinstance(x, np.ndarray)
        assert len(x) == len(t)
    
    def test_solucion_analitica_condicion_inicial_x0(self):
        """Verifica que la posición inicial es correcta."""
        t = np.linspace(0, 10, 100)
        x0 = 2.5
        x = OscilacionesAmortiguadas.solucion_analitica(
            t, m=1.0, c=0.5, k=1.0, x0=x0, v0=0.0
        )
        
        # En t=0, x debe ser aproximadamente x0
        assert np.isclose(x[0], x0, rtol=1e-5)
    
    def test_solucion_analitica_amortiguamiento_criterico(self):
        """Verifica el caso de amortiguamiento crítico."""
        t = np.linspace(0, 10, 100)
        m, k = 1.0, 1.0
        c = 2 * np.sqrt(k * m)  # Amortiguamiento crítico
        
        x = OscilacionesAmortiguadas.solucion_analitica(
            t, m=m, c=c, k=k, x0=1.0, v0=0.0
        )
        
        # Debe ser positiva y decreciente
        assert np.all(x >= 0)
        assert x[-1] < x[0]


class TestGeneracionDatos:
    """Tests para la generación de datos."""
    
    def test_generacion_datos_shape(self):
        """Verifica las dimensiones de los datos generados."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=100, tiempo_max=5.0)
        
        assert X.shape[0] == y.shape[0]  # Mismo número de muestras
        assert X.shape[1] == 7  # 7 características: [t, m, c, k, x0, v0, zeta]
        assert y.shape[1] == 1
    
    def test_generacion_datos_tipos(self):
        """Verifica que los datos son float32."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=50)
        
        assert X.dtype == np.float32
        assert y.dtype == np.float32
    
    def test_generacion_datos_sin_nans(self):
        """Verifica que no hay valores NaN en los datos."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=100)
        
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))
    
    def test_generacion_datos_parametros_salvos(self):
        """Verifica que la configuración se guarda."""
        modelo = OscilacionesAmortiguadas()
        params = {'m': 1.5, 'c': 2.0, 'k': 10.0, 'x0': 0.5, 'v0': -0.5}
        X, y = modelo.generar_datos(num_muestras=50, params_sistema=params)
        
        assert 'params_sistema' in modelo.config
        assert modelo.config['tiempo_max'] > 0


class TestConstruccionModelo:
    """Tests para la construcción del modelo."""
    
    def test_construccion_modelo_basico(self):
        """Verifica la construcción de un modelo básico."""
        modelo = OscilacionesAmortiguadas()
        red = modelo.construir_modelo(input_shape=7)
        
        assert red is not None
        assert len(red.layers) > 0
    
    def test_construccion_modelo_arquitectura_personalizada(self):
        """Verifica modelo con arquitectura personalizada."""
        modelo = OscilacionesAmortiguadas()
        capas_custom = [256, 128, 64, 32]
        red = modelo.construir_modelo(input_shape=7, capas_ocultas=capas_custom)
        
        assert len(red.layers) > 0
        # Verificar que la configuración fue guardada
        assert modelo.config['capas_ocultas'] == capas_custom
    
    def test_modelo_compilacion(self):
        """Verifica que el modelo está compilado."""
        modelo = OscilacionesAmortiguadas()
        red = modelo.construir_modelo()
        
        assert red.optimizer is not None
        assert red.loss is not None


class TestEntrenamiento:
    """Tests para el entrenamiento."""
    
    @pytest.fixture
    def datos_entrenamiento(self):
        """Fixture con datos para entrenamiento."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=100, tiempo_max=5.0)
        return X, y
    
    def test_entrenamiento_basico(self, datos_entrenamiento):
        """Verifica que el entrenamiento funciona."""
        X, y = datos_entrenamiento
        modelo = OscilacionesAmortiguadas()
        
        info = modelo.entrenar(X, y, epochs=5, batch_size=32, verbose=0)
        
        assert 'epochs_entrenadas' in info
        assert info['epochs_entrenadas'] > 0
        assert 'loss_final' in info
    
    def test_entrenamiento_loss_decrece(self, datos_entrenamiento):
        """Verifica que el loss disminuye durante el entrenamiento."""
        X, y = datos_entrenamiento
        modelo = OscilacionesAmortiguadas()
        
        modelo.entrenar(X, y, epochs=10, batch_size=32, verbose=0)
        
        loss_inicial = modelo.history.history['loss'][0]
        loss_final = modelo.history.history['loss'][-1]
        
        # El loss final debe ser menor que el inicial (en general)
        assert loss_final < loss_inicial or np.isclose(loss_final, loss_inicial, rtol=0.1)
    
    def test_entrenamiento_early_stopping(self, datos_entrenamiento):
        """Verifica que early stopping funciona."""
        X, y = datos_entrenamiento
        modelo = OscilacionesAmortiguadas()
        
        info = modelo.entrenar(
            X, y, epochs=100, batch_size=32,
            early_stopping_patience=5, verbose=0
        )
        
        # No debe entrenar todas las épocas
        assert info['epochs_entrenadas'] < 100


class TestPrediccion:
    """Tests para predicción."""
    
    @pytest.fixture
    def modelo_entrenado(self):
        """Fixture con modelo entrenado."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=100, tiempo_max=5.0)
        modelo.entrenar(X, y, epochs=5, batch_size=32, verbose=0)
        return modelo, X, y
    
    def test_prediccion_shape(self, modelo_entrenado):
        """Verifica que las predicciones tienen la forma correcta."""
        modelo, X, _ = modelo_entrenado
        
        y_pred = modelo.predecir(X)
        
        assert y_pred.shape[0] == X.shape[0]
        assert y_pred.shape[1] == 1
    
    def test_prediccion_sin_modelo_falla(self):
        """Verifica que predecir sin modelo genera error."""
        modelo = OscilacionesAmortiguadas()
        X = np.random.randn(10, 7).astype(np.float32)
        
        with pytest.raises(ValueError):
            modelo.predecir(X)
    
    def test_prediccion_valores_validos(self, modelo_entrenado):
        """Verifica que las predicciones son valores válidos."""
        modelo, X, _ = modelo_entrenado
        
        y_pred = modelo.predecir(X)
        
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))


class TestSerializacion:
    """Tests para guardado y carga de modelos."""
    
    @pytest.fixture
    def modelo_entrenado(self):
        """Fixture con modelo entrenado."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=50, tiempo_max=5.0)
        modelo.entrenar(X, y, epochs=3, batch_size=32, verbose=0)
        return modelo
    
    def test_guardado_modelo(self, modelo_entrenado, tmp_path):
        """Verifica que el modelo se guarda correctamente."""
        ruta_modelo = str(tmp_path / 'test_model.keras')
        
        modelo_entrenado.guardar_modelo(ruta_modelo)
        
        assert Path(ruta_modelo).exists()
        assert Path(ruta_modelo.replace('.keras', '.json')).exists()
    
    def test_carga_modelo(self, modelo_entrenado, tmp_path):
        """Verifica que un modelo guardado se puede cargar."""
        ruta_modelo = str(tmp_path / 'test_model.keras')
        
        modelo_entrenado.guardar_modelo(ruta_modelo)
        
        # Crear nuevo modelo e intentar cargar
        modelo2 = OscilacionesAmortiguadas()
        modelo2.cargar_modelo(ruta_modelo)
        
        assert modelo2.model is not None
    
    def test_consistencia_predicciones_guardadas(self, modelo_entrenado, tmp_path):
        """Verifica que las predicciones son consistentes después de guardar/cargar."""
        ruta_modelo = str(tmp_path / 'test_model.keras')
        
        X_test = np.random.randn(10, 7).astype(np.float32)
        y_pred_original = modelo_entrenado.predecir(X_test)
        
        modelo_entrenado.guardar_modelo(ruta_modelo)
        
        # Cargar y predecir
        modelo_cargado = OscilacionesAmortiguadas()
        modelo_cargado.cargar_modelo(ruta_modelo)
        y_pred_cargado = modelo_cargado.predecir(X_test)
        
        # Deben ser aproximadamente iguales
        assert np.allclose(y_pred_original, y_pred_cargado, rtol=1e-5)


class TestResumenModelo:
    """Tests para el resumen del modelo."""
    
    def test_resumen_sin_modelo(self):
        """Verifica resumen cuando no hay modelo entrenado."""
        modelo = OscilacionesAmortiguadas()
        resumen = modelo.resumen_modelo()
        
        assert 'estado' in resumen
    
    @pytest.fixture
    def modelo_entrenado(self):
        """Fixture con modelo entrenado."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=50)
        modelo.entrenar(X, y, epochs=3, batch_size=32, verbose=0)
        return modelo
    
    def test_resumen_con_modelo(self, modelo_entrenado):
        """Verifica resumen con modelo entrenado."""
        resumen = modelo_entrenado.resumen_modelo()
        
        assert 'tipo_modelo' in resumen
        assert 'capas' in resumen
        assert 'parametros_totales' in resumen
        assert resumen['parametros_totales'] > 0


class TestValidacionCruzada:
    """Tests para validación cruzada."""
    
    def test_validacion_cruzada_completa(self):
        """Verifica que la validación cruzada funciona."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=100, tiempo_max=5.0)
        
        cv_results = modelo.validacion_cruzada(X, y, k_folds=3, epochs=5)
        
        assert 'mse_mean' in cv_results
        assert 'mae_mean' in cv_results
        assert 'r2_mean' in cv_results
        assert len(cv_results['scores_por_fold']['mse']) == 3
    
    def test_validacion_cruzada_valores_positivos(self):
        """Verifica que las métricas son positivas."""
        modelo = OscilacionesAmortiguadas()
        X, y = modelo.generar_datos(num_muestras=50)
        
        cv_results = modelo.validacion_cruzada(X, y, k_folds=2, epochs=3)
        
        assert cv_results['mse_mean'] > 0
        assert cv_results['mae_mean'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
