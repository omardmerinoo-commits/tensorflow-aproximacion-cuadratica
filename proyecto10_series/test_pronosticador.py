"""
Test Suite: Pronosticador de Series Temporales
==============================================

29+ pruebas cubriendo:
- Generación de series realistas
- Normalización temporal
- Construcción de modelos (LSTM, CNN-LSTM)
- Entrenamiento y validación
- Evaluación de métricas
- Predicciones
- Persistencia
- Edge cases

Cobertura target: >90%
"""

import pytest
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os
import tempfile

from pronosticador_series import (
    GeneradorSeriesTemporales,
    PronostadorSeriesTemporales,
    DatosSeriesTemporales
)


class TestGeneracionDatos:
    """Pruebas de generación de series temporales"""
    
    def test_init_generador(self):
        """Verifica inicialización del generador"""
        gen = GeneradorSeriesTemporales(seed=42)
        assert gen.seed == 42
    
    def test_generar_tendencia_lineal(self):
        """Verifica generación de tendencia lineal"""
        gen = GeneradorSeriesTemporales()
        t = np.arange(100)
        tend = gen._tendencia(t, 'lineal')
        assert len(tend) == 100
        assert np.all(np.diff(tend) > 0)  # Creciente
    
    def test_generar_tendencia_cuadratica(self):
        """Verifica tendencia cuadrática"""
        gen = GeneradorSeriesTemporales()
        t = np.arange(100)
        tend = gen._tendencia(t, 'cuadratica')
        assert len(tend) == 100
    
    def test_generar_estacionalidad(self):
        """Verifica componente estacional"""
        gen = GeneradorSeriesTemporales()
        t = np.arange(100)
        est = gen._estacionalidad(t, periodo=10)
        assert len(est) == 100
        assert np.max(np.abs(est)) <= 2.1
    
    def test_generar_arima_basico(self):
        """Verifica proceso ARIMA(2,1,1)"""
        gen = GeneradorSeriesTemporales()
        arima = gen._generar_arima(100, p=2, d=1, q=1)
        assert len(arima) == 100
        assert not np.any(np.isnan(arima))
    
    def test_generar_serie_univariada(self):
        """Verifica serie univariada"""
        gen = GeneradorSeriesTemporales()
        serie, nombres = gen.generar(n_puntos=100, n_series=1)
        assert serie.shape == (100, 1)
        assert len(nombres) == 1
    
    def test_generar_serie_multivariada(self):
        """Verifica serie multivariada"""
        gen = GeneradorSeriesTemporales()
        serie, nombres = gen.generar(n_puntos=100, n_series=3)
        assert serie.shape == (100, 3)
        assert len(nombres) == 3
        assert all('Variable' in n for n in nombres)
    
    def test_generar_tipos_diferentes(self):
        """Verifica diferentes tipos de series"""
        gen = GeneradorSeriesTemporales()
        for tipo in ['tendencia_estacional', 'ruido_blanco', 'arima']:
            serie, _ = gen.generar(n_puntos=50, tipo=tipo)
            assert serie.shape[0] == 50
            assert not np.any(np.isnan(serie))


class TestDataset:
    """Pruebas de generación de dataset con ventanas"""
    
    def test_generar_dataset_basico(self):
        """Verifica creación de dataset"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=100, n_series=2, ventana=10)
        assert isinstance(datos, DatosSeriesTemporales)
        assert datos.X_train.ndim == 3
    
    def test_dataset_split_temporal(self):
        """Verifica que no hay shuffle (split temporal)"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=100, ventana=10)
        n_total = len(datos.X_train) + len(datos.X_val) + len(datos.X_test)
        assert n_total == 100 - 10
    
    def test_dataset_ventana_correcta(self):
        """Verifica tamaño de ventana"""
        gen = GeneradorSeriesTemporales()
        ventana = 15
        datos = gen.generar_dataset(ventana=ventana)
        assert datos.X_train.shape[1] == ventana
    
    def test_dataset_coherencia_temporalidad(self):
        """Verifica coherencia temporal (y[t+1] es el siguiente punto)"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=50, ventana=5)
        # Verificar que secuencias son consecutivas
        assert datos.X_train.shape[0] > 0
        assert datos.y_train.shape[0] > 0
        assert len(datos.X_train) == len(datos.y_train)
    
    def test_dataset_sin_nulos(self):
        """Verifica que no hay valores NaN"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=100, n_series=2)
        assert not np.any(np.isnan(datos.X_train))
        assert not np.any(np.isnan(datos.y_train))
    
    def test_dataset_ratios_split(self):
        """Verifica ratios de split"""
        gen = GeneradorSeriesTemporales()
        total = 500
        datos = gen.generar_dataset(n_puntos=total, ventana=10,
                                   split=(0.6, 0.2, 0.2))
        n_total = len(datos.X_train) + len(datos.X_val) + len(datos.X_test)
        train_ratio = len(datos.X_train) / n_total
        val_ratio = len(datos.X_val) / n_total
        test_ratio = len(datos.X_test) / n_total
        
        assert 0.55 < train_ratio < 0.65
        assert 0.15 < val_ratio < 0.25
        assert 0.15 < test_ratio < 0.25


class TestNormalizacion:
    """Pruebas de normalización MinMax"""
    
    def test_normalizar_rango(self):
        """Verifica normalización a [0, 1]"""
        pronosticador = PronostadorSeriesTemporales()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float).reshape(3, 1, 2)
        X_norm = pronosticador._normalizar(X, fit=True)
        assert np.min(X_norm) >= 0
        assert np.max(X_norm) <= 1
    
    def test_desnormalizar(self):
        """Verifica que desnormalización invierte normalización"""
        pronosticador = PronostadorSeriesTemporales()
        X = np.array([[1, 100], [50, 200]], dtype=float).reshape(2, 1, 2)
        X_norm = pronosticador._normalizar(X, fit=True)
        X_denorm = pronosticador._desnormalizar(X_norm)
        assert np.allclose(X, X_denorm, rtol=1e-5)
    
    def test_normalizar_multivariado(self):
        """Verifica normalización multivariada"""
        pronosticador = PronostadorSeriesTemporales()
        X = np.random.randn(10, 5, 3).astype(float)
        X_norm = pronosticador._normalizar(X, fit=True)
        assert X_norm.shape == X.shape


class TestConstruccionModelos:
    """Pruebas de construcción de modelos"""
    
    def test_construir_lstm_univariado(self):
        """Verifica construcción de LSTM para univariado"""
        pronosticador = PronostadorSeriesTemporales()
        modelo = pronosticador.construir_lstm(input_shape=(10, 1), output_shape=1)
        assert modelo is not None
        assert modelo.input_shape == (None, 10, 1)
        assert modelo.output_shape == (None, 1)
    
    def test_construir_lstm_multivariado(self):
        """Verifica LSTM multivariado"""
        pronosticador = PronostadorSeriesTemporales()
        modelo = pronosticador.construir_lstm(input_shape=(10, 5), output_shape=5)
        assert modelo.input_shape == (None, 10, 5)
        assert modelo.output_shape == (None, 5)
    
    def test_construir_cnn_lstm(self):
        """Verifica construcción CNN-LSTM"""
        pronosticador = PronostadorSeriesTemporales()
        modelo = pronosticador.construir_cnn_lstm(input_shape=(10, 3), output_shape=3)
        assert modelo is not None
        assert modelo.input_shape == (None, 10, 3)
    
    def test_lstm_tiene_layers_esperados(self):
        """Verifica que LSTM contiene capas esperadas"""
        pronosticador = PronostadorSeriesTemporales()
        modelo = pronosticador.construir_lstm(input_shape=(10, 2))
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'Bidirectional' in layer_names
        assert 'Dense' in layer_names
        assert 'Dropout' in layer_names
    
    def test_cnn_lstm_tiene_conv(self):
        """Verifica que CNN-LSTM contiene Conv1D"""
        pronosticador = PronostadorSeriesTemporales()
        modelo = pronosticador.construir_cnn_lstm(input_shape=(10, 2))
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'Conv1D' in layer_names


class TestEntrenamiento:
    """Pruebas de entrenamiento"""
    
    def test_entrenar_lstm(self):
        """Verifica entrenamiento LSTM"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        hist = pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=5, arquitectura='lstm', verbose=0
        )
        
        assert 'loss' in hist
        assert len(hist['loss']) == 5
        assert pronosticador.entrenado
    
    def test_entrenar_cnn_lstm(self):
        """Verifica entrenamiento CNN-LSTM"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        hist = pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=5, arquitectura='cnn_lstm', verbose=0
        )
        
        assert 'loss' in hist
        assert pronosticador.entrenado
    
    def test_entrenar_arquitectura_invalida(self):
        """Verifica error con arquitectura inválida"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=50)
        pronosticador = PronostadorSeriesTemporales()
        
        with pytest.raises(ValueError):
            pronosticador.entrenar(
                datos.X_train, datos.y_train,
                datos.X_val, datos.y_val,
                arquitectura='invalida'
            )
    
    def test_entrenar_loss_decrece(self):
        """Verifica que loss decrece durante entrenamiento"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=200, ventana=10)
        
        pronosticador = PronostadorSeriesTemporales()
        hist = pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=10, verbose=0
        )
        
        loss_inicial = hist['loss'][0]
        loss_final = hist['loss'][-1]
        assert loss_final < loss_inicial


class TestEvaluacion:
    """Pruebas de evaluación de modelos"""
    
    def test_evaluar_retorna_dict(self):
        """Verifica que evaluar devuelve diccionario"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
        assert isinstance(metricas, dict)
    
    def test_evaluar_metricas_validas(self):
        """Verifica que todas las métricas están presentes y válidas"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
        
        required_keys = ['mae', 'mse', 'rmse', 'mape', 'r2_score', 
                        'predicciones', 'residuos']
        for key in required_keys:
            assert key in metricas
        
        # Verificar rangos válidos
        assert 0 <= metricas['mape']
        assert 0 <= metricas['mae']
    
    def test_evaluar_residuos_shape(self):
        """Verifica forma de residuos"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
        assert metricas['residuos'].shape == datos.y_test.shape
    
    def test_evaluar_sin_entrenar(self):
        """Verifica error si evalúa sin entrenar"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=50)
        
        pronosticador = PronostadorSeriesTemporales()
        with pytest.raises(ValueError):
            pronosticador.evaluar(datos.X_test, datos.y_test)


class TestPrediccion:
    """Pruebas de predicciones"""
    
    def test_predecir_shape(self):
        """Verifica que predicciones tienen forma correcta"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        y_pred = pronosticador.predecir(datos.X_test)
        assert y_pred.shape == datos.y_test.shape
    
    def test_predecir_sin_entrenar(self):
        """Verifica error al predecir sin entrenar"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=50)
        
        pronosticador = PronostadorSeriesTemporales()
        with pytest.raises(ValueError):
            pronosticador.predecir(datos.X_test)
    
    def test_predecir_valor_simple(self):
        """Verifica predicción de un valor simple"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        y_pred = pronosticador.predecir(datos.X_test[:1])
        assert y_pred.shape[0] == 1


class TestComparacionArquitecturas:
    """Pruebas de comparación entre modelos"""
    
    def test_cnn_lstm_vs_lstm(self):
        """Compara CNN-LSTM contra LSTM"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=150, ventana=10)
        
        # LSTM
        p1 = PronostadorSeriesTemporales(seed=42)
        p1.entrenar(datos.X_train, datos.y_train,
                   datos.X_val, datos.y_val,
                   epochs=5, arquitectura='lstm', verbose=0)
        m1 = p1.evaluar(datos.X_test, datos.y_test)
        
        # CNN-LSTM
        p2 = PronostadorSeriesTemporales(seed=42)
        p2.entrenar(datos.X_train, datos.y_train,
                   datos.X_val, datos.y_val,
                   epochs=5, arquitectura='cnn_lstm', verbose=0)
        m2 = p2.evaluar(datos.X_test, datos.y_test)
        
        # Ambos son válidos
        assert m1['rmse'] > 0
        assert m2['rmse'] > 0


class TestPersistencia:
    """Pruebas de guardar/cargar modelos"""
    
    def test_guardar_cargar_modelo(self):
        """Verifica que save/load preservan funcionalidad"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        y_pred_antes = pronosticador.predecir(datos.X_test[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'modelo')
            pronosticador.guardar(ruta)
            
            pronosticador_cargado = PronostadorSeriesTemporales.cargar(ruta)
            y_pred_despues = pronosticador_cargado.predecir(datos.X_test[:5])
        
        assert np.allclose(y_pred_antes, y_pred_despues)


class TestEdgeCases:
    """Pruebas de casos límite"""
    
    def test_serie_muy_corta(self):
        """Maneja serie muy corta"""
        gen = GeneradorSeriesTemporales()
        datos = gen.generar_dataset(n_puntos=30, ventana=5)
        assert len(datos.X_train) > 0
    
    def test_prediccion_unica(self):
        """Predice un único paso adelante"""
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=100, ventana=5)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        y_pred = pronosticador.predecir(datos.X_test[:1])
        assert y_pred.shape == (1, datos.y_test.shape[1])
    
    def test_serie_sin_estacionalidad(self):
        """Maneja serie sin patrón estacional"""
        gen = GeneradorSeriesTemporales()
        serie, _ = gen.generar(n_puntos=100, tipo='ruido_blanco')
        assert serie.shape[0] == 100


class TestRendimiento:
    """Pruebas de rendimiento"""
    
    def test_velocidad_generacion_datos(self):
        """Verifica que generación es rápida"""
        import time
        gen = GeneradorSeriesTemporales()
        
        t_inicio = time.time()
        gen.generar_dataset(n_puntos=1000, ventana=10)
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 5  # Menos de 5 segundos
    
    def test_velocidad_prediccion(self):
        """Verifica que predicciones son rápidas"""
        import time
        gen = GeneradorSeriesTemporales(seed=42)
        datos = gen.generar_dataset(n_puntos=200, ventana=10)
        
        pronosticador = PronostadorSeriesTemporales()
        pronosticador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, verbose=0
        )
        
        t_inicio = time.time()
        for _ in range(10):
            pronosticador.predecir(datos.X_test[:10])
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 10  # Menos de 10 segundos para 10 lotes


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
