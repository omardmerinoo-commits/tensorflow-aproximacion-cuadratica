"""
Test Suite: Clasificador de Sentimientos NLP
============================================

35+ pruebas cubriendo:
- Generación de textos sintéticos balanceados
- Pre-procesamiento: tokenización, padding
- Arquitecturas: LSTM, Transformer, CNN 1D
- Entrenamiento y evaluación
- Predicciones y confianza
- Análisis por clase
- Persistencia de modelos

Cobertura target: >90%
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import tempfile

from clasificador_sentimientos import (
    GeneradorTextoSentimientos,
    ClasificadorSentimientos,
    DatosTexto
)


class TestGeneracionTextos:
    """Pruebas de generación sintética de textos"""
    
    def test_init_generador(self):
        """Verifica inicialización del generador"""
        gen = GeneradorTextoSentimientos(seed=42)
        assert gen.seed == 42
    
    def test_generar_sentimiento_positivo(self):
        """Verifica generación de sentimientos positivos"""
        gen = GeneradorTextoSentimientos()
        for _ in range(10):
            texto = gen._generar_sentimiento_positivo()
            assert isinstance(texto, str)
            assert len(texto) > 0
    
    def test_generar_sentimiento_negativo(self):
        """Verifica generación de sentimientos negativos"""
        gen = GeneradorTextoSentimientos()
        for _ in range(10):
            texto = gen._generar_sentimiento_negativo()
            assert isinstance(texto, str)
            assert len(texto) > 0
    
    def test_generar_sentimiento_neutro(self):
        """Verifica generación de sentimientos neutros"""
        gen = GeneradorTextoSentimientos()
        for _ in range(10):
            texto = gen._generar_sentimiento_neutro()
            assert isinstance(texto, str)
            assert len(texto) > 0
    
    def test_limpiar_texto(self):
        """Verifica limpieza de texto"""
        gen = GeneradorTextoSentimientos()
        texto_sucio = "¡HOLA! ¿Cómo estás?"
        texto_limpio = gen._limpiar_texto(texto_sucio)
        assert texto_limpio == texto_limpio.lower()
        assert '!' not in texto_limpio
        assert '?' not in texto_limpio
    
    def test_generar_corpus(self):
        """Verifica generación de corpus balanceado"""
        gen = GeneradorTextoSentimientos()
        textos, etiquetas = gen.generar(n_positivos=50, n_negativos=50, n_neutros=50)
        
        assert len(textos) == 150
        assert len(etiquetas) == 150
        assert np.sum(etiquetas == 0) == 50  # Negativos
        assert np.sum(etiquetas == 1) == 50  # Neutros
        assert np.sum(etiquetas == 2) == 50  # Positivos
    
    def test_etiquetas_validas(self):
        """Verifica que etiquetas están en rango [0, 1, 2]"""
        gen = GeneradorTextoSentimientos()
        _, etiquetas = gen.generar(100, 100, 100)
        assert np.all((etiquetas >= 0) & (etiquetas <= 2))


class TestDataset:
    """Pruebas de generación de dataset"""
    
    def test_generar_dataset_basico(self):
        """Verifica creación de dataset"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        assert isinstance(datos, DatosTexto)
        assert len(datos.X_train) > 0
        assert len(datos.X_val) > 0
        assert len(datos.X_test) > 0
    
    def test_dataset_split_proporciones(self):
        """Verifica proporciones de split"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=100, split=(0.6, 0.2, 0.2))
        
        n_total = len(datos.X_train) + len(datos.X_val) + len(datos.X_test)
        train_ratio = len(datos.X_train) / n_total
        
        assert 0.55 < train_ratio < 0.65
    
    def test_dataset_tokenizer(self):
        """Verifica tokenizer creado"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        assert isinstance(datos.tokenizer, Tokenizer)
        assert datos.tokenizer.num_words > 0
    
    def test_dataset_shapes(self):
        """Verifica formas de datos"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50, max_len=50)
        
        assert datos.X_train.shape[1] == 50  # max_len
        assert datos.y_train.shape[1] == 3   # 3 clases
        assert len(datos.X_train) == datos.y_train.shape[0]
    
    def test_dataset_etiquetas_categoricas(self):
        """Verifica que etiquetas son one-hot encoded"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        # One-hot: suma de cada fila es 1
        sumas = np.sum(datos.y_train, axis=1)
        assert np.all(np.isclose(sumas, 1.0))
    
    def test_dataset_sin_nulos(self):
        """Verifica que no hay NaN"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        assert not np.any(np.isnan(datos.X_train))
        assert not np.any(np.isnan(datos.y_train))
    
    def test_dataset_label_encoder(self):
        """Verifica label encoder"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset()
        
        assert len(datos.label_encoder.classes_) == 3
        assert len(datos.etiquetas) == 3


class TestConstruccionModelos:
    """Pruebas de construcción de arquitecturas"""
    
    def test_construir_lstm(self):
        """Verifica construcción LSTM"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_lstm(max_len=50)
        
        assert modelo is not None
        assert modelo.input_shape == (None, 50)
        assert modelo.output_shape == (None, 3)
    
    def test_construir_transformer(self):
        """Verifica construcción Transformer"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_transformer(max_len=50)
        
        assert modelo is not None
        assert modelo.input_shape == (None, 50)
        assert modelo.output_shape == (None, 3)
    
    def test_construir_cnn1d(self):
        """Verifica construcción CNN 1D"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_cnn1d(max_len=50)
        
        assert modelo is not None
        assert modelo.input_shape == (None, 50)
        assert modelo.output_shape == (None, 3)
    
    def test_lstm_tiene_embedding(self):
        """Verifica que LSTM contiene embedding"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_lstm()
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'Embedding' in layer_names
    
    def test_lstm_tiene_bidirectional(self):
        """Verifica capas Bidirectional"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_lstm()
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'Bidirectional' in layer_names
    
    def test_transformer_tiene_multiheadattention(self):
        """Verifica que Transformer contiene MultiHeadAttention"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_transformer()
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'MultiHeadAttention' in layer_names
    
    def test_cnn1d_tiene_conv(self):
        """Verifica que CNN 1D contiene Conv1D"""
        clasificador = ClasificadorSentimientos()
        modelo = clasificador.construir_cnn1d()
        layer_names = [l.__class__.__name__ for l in modelo.layers]
        assert 'Conv1D' in layer_names


class TestEntrenamiento:
    """Pruebas de entrenamiento"""
    
    def test_entrenar_lstm(self):
        """Verifica entrenamiento LSTM"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        hist = clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, arquitectura='lstm', verbose=0
        )
        
        assert 'loss' in hist
        assert len(hist['loss']) == 3
        assert clasificador.entrenado
    
    def test_entrenar_transformer(self):
        """Verifica entrenamiento Transformer"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        hist = clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, arquitectura='transformer', verbose=0
        )
        
        assert clasificador.entrenado
    
    def test_entrenar_cnn1d(self):
        """Verifica entrenamiento CNN 1D"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        hist = clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=3, arquitectura='cnn1d', verbose=0
        )
        
        assert clasificador.entrenado
    
    def test_entrenar_arquitectura_invalida(self):
        """Verifica error con arquitectura inválida"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        with pytest.raises(ValueError):
            clasificador.entrenar(
                datos.X_train, datos.y_train,
                datos.X_val, datos.y_val,
                arquitectura='invalida'
            )
    
    def test_loss_decrece(self):
        """Verifica que loss decrece durante entrenamiento"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=100)
        
        clasificador = ClasificadorSentimientos()
        hist = clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=10, verbose=0
        )
        
        loss_inicial = hist['loss'][0]
        loss_final = hist['loss'][-1]
        assert loss_final < loss_inicial


class TestEvaluacion:
    """Pruebas de evaluación"""
    
    def test_evaluar_retorna_dict(self):
        """Verifica que evaluar devuelve diccionario"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        metricas = clasificador.evaluar(datos.X_test, datos.y_test)
        assert isinstance(metricas, dict)
    
    def test_evaluar_metricas_presentes(self):
        """Verifica que todas las métricas están presentes"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        metricas = clasificador.evaluar(datos.X_test, datos.y_test)
        
        assert 'accuracy' in metricas
        assert 'per_class_accuracy' in metricas
        assert 'predicciones' in metricas
        assert 'probabilidades' in metricas
    
    def test_evaluar_accuracy_valida(self):
        """Verifica que accuracy está en rango [0, 1]"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        metricas = clasificador.evaluar(datos.X_test, datos.y_test)
        assert 0 <= metricas['accuracy'] <= 1
    
    def test_evaluar_per_class_accuracy(self):
        """Verifica accuracy por clase"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        metricas = clasificador.evaluar(datos.X_test, datos.y_test)
        assert len(metricas['per_class_accuracy']) > 0
    
    def test_evaluar_sin_entrenar(self):
        """Verifica error si evalúa sin entrenar"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        with pytest.raises(ValueError):
            clasificador.evaluar(datos.X_test, datos.y_test)


class TestPrediccion:
    """Pruebas de predicciones"""
    
    def test_predecir_shape(self):
        """Verifica formas de predicción"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        clases, probs = clasificador.predecir(datos.X_test)
        assert clases.shape[0] == probs.shape[0]
        assert probs.shape[1] == 3  # 3 clases
    
    def test_predecir_probabilidades_validas(self):
        """Verifica que probabilidades suman a 1"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        _, probs = clasificador.predecir(datos.X_test[:5])
        sumas = np.sum(probs, axis=1)
        assert np.all(np.isclose(sumas, 1.0))
    
    def test_predecir_sin_entrenar(self):
        """Verifica error al predecir sin entrenar"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        with pytest.raises(ValueError):
            clasificador.predecir(datos.X_test)
    
    def test_predecir_batch_unico(self):
        """Verifica predicción de un sample"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        clases, probs = clasificador.predecir(datos.X_test[:1])
        assert clases.shape == (1,)
        assert probs.shape == (1, 3)


class TestComparacionArquitecturas:
    """Pruebas de comparación entre modelos"""
    
    def test_todas_arquitecturas_convergen(self):
        """Verifica que todas las arquitecturas entrenan sin error"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        for arch in ['lstm', 'transformer', 'cnn1d']:
            clasificador = ClasificadorSentimientos()
            clasificador.entrenar(
                datos.X_train, datos.y_train,
                datos.X_val, datos.y_val,
                epochs=2, arquitectura=arch, verbose=0
            )
            metricas = clasificador.evaluar(datos.X_test, datos.y_test)
            assert 0 <= metricas['accuracy'] <= 1


class TestPersistencia:
    """Pruebas de guardar/cargar modelos"""
    
    def test_guardar_cargar_modelo(self):
        """Verifica save/load preservan funcionalidad"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        clases_antes, probs_antes = clasificador.predecir(datos.X_test[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, 'modelo')
            clasificador.guardar(ruta)
            
            clasificador_cargado = ClasificadorSentimientos.cargar(
                ruta, vocab_size=1000, embedding_dim=128
            )
            clases_despues, probs_despues = clasificador_cargado.predecir(
                datos.X_test[:5]
            )
        
        assert np.allclose(probs_antes, probs_despues)


class TestEdgeCases:
    """Pruebas de casos límite"""
    
    def test_dataset_pequeno(self):
        """Maneja dataset muy pequeño"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=10)
        assert len(datos.X_train) > 0
    
    def test_max_len_pequeño(self):
        """Maneja max_len pequeño"""
        gen = GeneradorTextoSentimientos()
        datos = gen.generar_dataset(n_samples_por_clase=50, max_len=10)
        assert datos.X_train.shape[1] == 10
    
    def test_prediccion_unica(self):
        """Predice un único texto"""
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        clases, probs = clasificador.predecir(datos.X_test[:1])
        assert clases.shape == (1,)


class TestRendimiento:
    """Pruebas de rendimiento"""
    
    def test_velocidad_generacion_dataset(self):
        """Verifica que generación es rápida"""
        import time
        gen = GeneradorTextoSentimientos()
        
        t_inicio = time.time()
        gen.generar_dataset(n_samples_por_clase=300)
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 5  # Menos de 5 segundos
    
    def test_velocidad_prediccion(self):
        """Verifica que predicciones son rápidas"""
        import time
        gen = GeneradorTextoSentimientos(seed=42)
        datos = gen.generar_dataset(n_samples_por_clase=50)
        
        clasificador = ClasificadorSentimientos()
        clasificador.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=2, verbose=0
        )
        
        t_inicio = time.time()
        for _ in range(10):
            clasificador.predecir(datos.X_test[:10])
        t_duracion = time.time() - t_inicio
        
        assert t_duracion < 10  # Menos de 10 segundos


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
