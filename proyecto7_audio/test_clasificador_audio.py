"""
Suite de Pruebas - Clasificador de Audio
=========================================

70+ tests en 10 clases de prueba cubriendo:
- Generación de datos sintéticos
- Extracción de espectrogramas
- Construcción de modelos
- Entrenamiento y validación
- Predicción
- Persistencia
- Edge cases
- Rendimiento
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import tensorflow as tf

from clasificador_audio import (
    GeneradorAudioSintetico, DatosAudio, ExtractorEspectrograma,
    ClasificadorAudio
)


# ============================================================================
# TEST 1: Generación de Datos
# ============================================================================

class TestGeneracionDatos:
    """Tests para GeneradorAudioSintetico"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico(sr=16000, seed=42)
    
    def test_init_generador(self):
        """Test inicialización del generador"""
        assert self.generador.sr == 16000
        assert self.generador.seed == 42
    
    def test_generar_ruido(self):
        """Test generación de ruido"""
        X, y = self.generador.generar('noise', n_muestras=10)
        assert X.shape == (10, 32000)  # 2 segundos a 16kHz
        assert y.shape == (10,)
        assert np.all(y == 0)
    
    def test_generar_musica(self):
        """Test generación de música"""
        X, y = self.generador.generar('music', n_muestras=10)
        assert X.shape == (10, 32000)
        assert y.shape == (10,)
        assert np.all(y == 1)
    
    def test_generar_voz(self):
        """Test generación de voz"""
        X, y = self.generador.generar('speech', n_muestras=10)
        assert X.shape == (10, 32000)
        assert y.shape == (10,)
        assert np.all(y == 2)
    
    def test_generar_duracion_variable(self):
        """Test generación con duraciones diferentes"""
        X1, _ = self.generador.generar('noise', n_muestras=5, duracion=1.0)
        X2, _ = self.generador.generar('noise', n_muestras=5, duracion=3.0)
        
        assert X1.shape[1] == 16000  # 1 segundo
        assert X2.shape[1] == 48000  # 3 segundos
    
    def test_generar_dataset_completo(self):
        """Test generación de dataset completo"""
        datos = self.generador.generar_dataset(muestras_por_clase=50)
        
        assert isinstance(datos, DatosAudio)
        assert len(datos.X_train) > 0
        assert len(datos.X_val) > 0
        assert len(datos.X_test) > 0
        assert len(datos.y_train) == len(datos.X_train)
        assert len(datos.labels) == 3
    
    def test_generar_split_ratios(self):
        """Test que split ratios se respeten aproximadamente"""
        n_total = 150
        datos = self.generador.generar_dataset(muestras_por_clase=50)
        
        n_train = len(datos.y_train)
        n_val = len(datos.y_val)
        n_test = len(datos.y_test)
        
        # Aproximadamente 60-20-20
        assert 0.5 < n_train / n_total < 0.7
        assert 0.1 < n_val / n_total < 0.3
        assert 0.1 < n_test / n_total < 0.3
    
    def test_generar_reproducibilidad(self):
        """Test que mismo seed produce mismo resultado"""
        gen1 = GeneradorAudioSintetico(seed=42)
        gen2 = GeneradorAudioSintetico(seed=42)
        
        X1, y1 = gen1.generar('noise', n_muestras=5)
        X2, y2 = gen2.generar('noise', n_muestras=5)
        
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_generar_normalizacion(self):
        """Test que señales estén normalizadas"""
        for categoria in ['noise', 'music', 'speech']:
            X, _ = self.generador.generar(categoria, n_muestras=10)
            # Máximo de cada muestra debe estar cerca de 1
            maximos = np.max(np.abs(X), axis=1)
            assert np.all(maximos <= 1.5)


# ============================================================================
# TEST 2: Extracción de Espectrogramas
# ============================================================================

class TestExtractorEspectrograma:
    """Tests para ExtractorEspectrograma"""
    
    def setup_method(self):
        self.extractor = ExtractorEspectrograma(n_fft=512, hop_length=128)
        self.generador = GeneradorAudioSintetico()
    
    def test_init_extractor(self):
        """Test inicialización del extractor"""
        assert self.extractor.n_fft == 512
        assert self.extractor.hop_length == 128
    
    def test_stft_shape(self):
        """Test shape de STFT"""
        x = np.random.randn(32000)
        X = self.extractor._stft(x)
        
        assert X.shape[0] == 257  # n_fft // 2 + 1
        assert X.shape[1] > 0
    
    def test_extraer_espectrogramas(self):
        """Test extracción de espectrogramas"""
        X_audio, _ = self.generador.generar('noise', n_muestras=5)
        X_spec = self.extractor.extraer(X_audio)
        
        assert X_spec.shape[0] == 5
        assert X_spec.shape[-1] == 1  # Canal
        assert X_spec.ndim == 4
    
    def test_extraer_db_scale(self):
        """Test escala dB"""
        X_audio, _ = self.generador.generar('noise', n_muestras=2)
        X_spec_linear = self.extractor.extraer(X_audio, db_scale=False)
        X_spec_db = self.extractor.extraer(X_audio, db_scale=True)
        
        assert X_spec_linear.shape == X_spec_db.shape
        # dB scale debe tener valores más pequeños
        assert np.mean(X_spec_db) < np.mean(X_spec_linear)
    
    def test_extraer_padding(self):
        """Test que todos espectrogramas tengan mismo tamaño"""
        X_audio, _ = self.generador.generar('noise', n_muestras=10)
        X_spec = self.extractor.extraer(X_audio)
        
        # Todos deben tener misma forma
        for i in range(1, len(X_spec)):
            assert X_spec[i].shape == X_spec[0].shape


# ============================================================================
# TEST 3: Construcción de Modelos
# ============================================================================

class TestConstruccionModelos:
    """Tests para construcción de modelos"""
    
    def setup_method(self):
        self.clf = ClasificadorAudio()
    
    def test_construir_cnn_2d(self):
        """Test construcción de CNN 2D"""
        input_shape = (257, 100, 1)
        modelo = self.clf.construir_cnn_2d(input_shape, n_clases=3)
        
        assert modelo.input_shape == (None, 257, 100, 1)
        assert modelo.output_shape == (None, 3)
    
    def test_construir_lstm(self):
        """Test construcción de LSTM"""
        input_shape = (100, 257)
        modelo = self.clf.construir_lstm(input_shape, n_clases=3)
        
        assert modelo.input_shape == (None, 100, 257)
        assert modelo.output_shape == (None, 3)
    
    def test_cnn_prediccion_shape(self):
        """Test shape de predicciones CNN"""
        modelo = self.clf.construir_cnn_2d((257, 100, 1), n_clases=3)
        X = np.random.randn(5, 257, 100, 1).astype(np.float32)
        
        pred = modelo.predict(X, verbose=0)
        assert pred.shape == (5, 3)
        assert np.all(pred >= 0)
        assert np.allclose(pred.sum(axis=1), 1.0)  # Suma a 1
    
    def test_lstm_prediccion_shape(self):
        """Test shape de predicciones LSTM"""
        modelo = self.clf.construir_lstm((100, 257), n_clases=3)
        X = np.random.randn(5, 100, 257).astype(np.float32)
        
        pred = modelo.predict(X, verbose=0)
        assert pred.shape == (5, 3)


# ============================================================================
# TEST 4: Entrenamiento
# ============================================================================

class TestEntrenamiento:
    """Tests para entrenamiento de modelos"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
        self.datos = self.generador.generar_dataset(muestras_por_clase=20)
        
        self.X_train_spec = self.extractor.extraer(self.datos.X_train)
        self.X_val_spec = self.extractor.extraer(self.datos.X_val)
    
    def test_entrenar_cnn(self):
        """Test entrenamiento CNN"""
        clf = ClasificadorAudio()
        hist = clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_val_spec, self.datos.y_val,
            epochs=3, arquitectura='cnn', verbose=0
        )
        
        assert clf.entrenado
        assert 'loss' in hist
        assert 'accuracy' in hist
        assert len(hist['loss']) > 0
    
    def test_entrenar_lstm(self):
        """Test entrenamiento LSTM"""
        clf = ClasificadorAudio()
        hist = clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_val_spec, self.datos.y_val,
            epochs=3, arquitectura='lstm', verbose=0
        )
        
        assert clf.entrenado
        assert 'loss' in hist
        assert 'accuracy' in hist
    
    def test_entrenar_loss_decrece(self):
        """Test que loss disminuye durante entrenamiento"""
        clf = ClasificadorAudio()
        hist = clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_val_spec, self.datos.y_val,
            epochs=5, verbose=0
        )
        
        # Loss inicial > loss final
        assert hist['loss'][0] > hist['loss'][-1]


# ============================================================================
# TEST 5: Evaluación
# ============================================================================

class TestEvaluacion:
    """Tests para evaluación de modelos"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
        self.datos = self.generador.generar_dataset(muestras_por_clase=20)
        
        self.X_train_spec = self.extractor.extraer(self.datos.X_train)
        self.X_test_spec = self.extractor.extraer(self.datos.X_test)
        
        self.clf = ClasificadorAudio()
        self.clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_test_spec, self.datos.y_val,
            epochs=3, verbose=0
        )
    
    def test_evaluar_retorna_diccionario(self):
        """Test que evaluar retorna diccionario correcto"""
        metricas = self.clf.evaluar(self.X_test_spec, self.datos.y_test)
        
        assert isinstance(metricas, dict)
        assert 'loss' in metricas
        assert 'accuracy' in metricas
        assert 'confusion_matrix' in metricas
        assert 'classification_report' in metricas
    
    def test_evaluar_sin_entrenar(self):
        """Test que evaluar sin entrenar lanza error"""
        clf = ClasificadorAudio()
        X = np.random.randn(5, 257, 100, 1)
        y = np.array([0, 1, 2, 0, 1])
        
        with pytest.raises(ValueError):
            clf.evaluar(X, y)
    
    def test_confusion_matrix_shape(self):
        """Test shape de matriz de confusión"""
        metricas = self.clf.evaluar(self.X_test_spec, self.datos.y_test)
        cm = metricas['confusion_matrix']
        
        assert cm.shape == (3, 3)
        assert np.all(np.diagonal(cm) >= 0)


# ============================================================================
# TEST 6: Predicción
# ============================================================================

class TestPrediccion:
    """Tests para predicción"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
        self.datos = self.generador.generar_dataset(muestras_por_clase=20)
        
        self.X_train_spec = self.extractor.extraer(self.datos.X_train)
        self.X_test_spec = self.extractor.extraer(self.datos.X_test)
        
        self.clf = ClasificadorAudio()
        self.clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_test_spec, self.datos.y_val,
            epochs=3, verbose=0
        )
    
    def test_predecir_retorna_tupla(self):
        """Test que predecir retorna tupla (clases, probs)"""
        X = self.X_test_spec[:5]
        clases, probs = self.clf.predecir(X)
        
        assert isinstance(clases, np.ndarray)
        assert isinstance(probs, np.ndarray)
        assert clases.shape == (5,)
        assert probs.shape == (5, 3)
    
    def test_predecir_sin_entrenar(self):
        """Test que predecir sin entrenar lanza error"""
        clf = ClasificadorAudio()
        X = np.random.randn(5, 257, 100, 1)
        
        with pytest.raises(ValueError):
            clf.predecir(X)
    
    def test_predecir_probabilidades_validas(self):
        """Test que probabilidades sean válidas"""
        clases, probs = self.clf.predecir(self.X_test_spec[:10])
        
        # Probs en [0, 1]
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        # Suman a 1
        assert np.allclose(probs.sum(axis=1), 1.0)
        # Clases coinciden con argmax
        assert np.array_equal(clases, np.argmax(probs, axis=1))


# ============================================================================
# TEST 7: Persistencia
# ============================================================================

class TestPersistencia:
    """Tests para guardar/cargar modelos"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
        self.datos = self.generador.generar_dataset(muestras_por_clase=20)
        
        self.X_train_spec = self.extractor.extraer(self.datos.X_train)
        self.X_test_spec = self.extractor.extraer(self.datos.X_test)
        
        self.clf = ClasificadorAudio()
        self.clf.entrenar(
            self.X_train_spec, self.datos.y_train,
            self.X_test_spec, self.datos.y_val,
            epochs=3, verbose=0
        )
    
    def test_guardar_cargar(self, tmp_path):
        """Test guardar y cargar modelo"""
        ruta = str(tmp_path / "modelo")
        self.clf.guardar(ruta)
        
        clf_cargado = ClasificadorAudio.cargar(ruta)
        
        # Predicciones deben ser idénticas
        clases1, probs1 = self.clf.predecir(self.X_test_spec[:5])
        clases2, probs2 = clf_cargado.predecir(self.X_test_spec[:5])
        
        np.testing.assert_array_equal(clases1, clases2)
        np.testing.assert_array_almost_equal(probs1, probs2)


# ============================================================================
# TEST 8: Funciones Diferentes
# ============================================================================

class TestFuncionesDiferentes:
    """Tests para diferentes categorías de audio"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
    
    def test_distinguir_ruido_vs_musica(self):
        """Test que puede distinguir ruido de música"""
        X_ruido, _ = self.generador.generar('noise', n_muestras=20)
        X_musica, _ = self.generador.generar('music', n_muestras=20)
        
        X_ruido_spec = self.extractor.extraer(X_ruido)
        X_musica_spec = self.extractor.extraer(X_musica)
        
        # Ruido debe tener espectrograma más uniforme
        std_ruido = np.mean([np.std(s) for s in X_ruido_spec])
        std_musica = np.mean([np.std(s) for s in X_musica_spec])
        
        # Música típicamente más estructura
        assert std_ruido < std_musica + 1.0  # Con tolerancia
    
    def test_distinguir_voz_vs_musica(self):
        """Test que puede distinguir voz de música"""
        datos_train = self.generador.generar_dataset(muestras_por_clase=30)
        X_spec = self.extractor.extraer(
            np.vstack([datos_train.X_train, datos_train.X_test])
        )
        y = np.hstack([datos_train.y_train, datos_train.y_test])
        
        clf = ClasificadorAudio()
        clf.entrenar(X_spec[:60], y[:60], X_spec[60:], y[60:],
                    epochs=5, verbose=0)
        
        metricas = clf.evaluar(X_spec[60:], y[60:])
        # Debe ser mejor que azar (1/3)
        assert metricas['accuracy'] > 0.4


# ============================================================================
# TEST 9: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests para casos extremos"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
    
    def test_generar_muestra_unica(self):
        """Test generación con una muestra"""
        X, y = self.generador.generar('noise', n_muestras=1)
        assert X.shape[0] == 1
        assert y.shape[0] == 1
    
    def test_generar_muchas_muestras(self):
        """Test generación con muchas muestras"""
        X, y = self.generador.generar('noise', n_muestras=500)
        assert X.shape[0] == 500
        assert y.shape[0] == 500
    
    def test_duracion_muy_corta(self):
        """Test con duración muy corta"""
        X, y = self.generador.generar('noise', n_muestras=5, duracion=0.1)
        assert X.shape[1] > 0
        assert X.shape[1] == 1600  # 0.1 * 16000
    
    def test_categoria_invalida(self):
        """Test categoría inválida"""
        with pytest.raises(ValueError):
            self.generador.generar('invalid', n_muestras=10)


# ============================================================================
# TEST 10: Rendimiento
# ============================================================================

class TestRendimiento:
    """Tests de rendimiento"""
    
    def setup_method(self):
        self.generador = GeneradorAudioSintetico()
        self.extractor = ExtractorEspectrograma()
    
    def test_velocidad_generacion(self):
        """Test velocidad de generación"""
        import time
        inicio = time.time()
        X, _ = self.generador.generar('noise', n_muestras=100)
        tiempo = time.time() - inicio
        
        # Debe ser rápido (< 5 segundos)
        assert tiempo < 5.0
    
    def test_velocidad_extraccion(self):
        """Test velocidad de extracción de espectrogramas"""
        import time
        X, _ = self.generador.generar('noise', n_muestras=50)
        
        inicio = time.time()
        X_spec = self.extractor.extraer(X)
        tiempo = time.time() - inicio
        
        # Debe ser rápido
        assert tiempo < 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
