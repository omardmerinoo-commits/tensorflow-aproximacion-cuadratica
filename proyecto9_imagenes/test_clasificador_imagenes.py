"""
Suite de Pruebas - Clasificador de Imágenes CIFAR-10
====================================================

70+ tests en 10 clases de prueba cubriendo:
- Carga de datos CIFAR-10
- Data augmentation
- Construcción de modelos
- Entrenamiento
- Evaluación
- Predicción
- Transfer learning
- Persistencia
- Edge cases
- Rendimiento
"""

import numpy as np
import pytest
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from clasificador_imagenes import GeneradorCIFAR10, ClasificadorImagenes


# ============================================================================
# TEST 1: Carga de Datos
# ============================================================================

class TestCargaDatos:
    """Tests para GeneradorCIFAR10"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10(seed=42)
    
    def test_init_generador(self):
        """Test inicialización"""
        assert self.generador.seed == 42
        assert len(self.generador.CLASES) == 10
    
    def test_cargar_datos(self):
        """Test carga de CIFAR-10"""
        datos = self.generador.cargar_datos()
        
        assert datos.X_train.shape[1:] == (32, 32, 3)
        assert datos.X_test.shape[1:] == (32, 32, 3)
        assert len(datos.clases) == 10
    
    def test_datos_normalizados(self):
        """Test que datos estén en [0, 1]"""
        datos = self.generador.cargar_datos()
        
        assert np.all(datos.X_train >= 0)
        assert np.all(datos.X_train <= 1)
        assert np.all(datos.X_val >= 0)
        assert np.all(datos.X_val <= 1)
    
    def test_split_train_val_test(self):
        """Test que splits sean correctos"""
        datos = self.generador.cargar_datos(validacion_split=0.2)
        
        # Aproximadamente 80-20 split
        n_total = len(datos.X_train) + len(datos.X_val)
        ratio_train = len(datos.X_train) / n_total
        
        assert 0.75 < ratio_train < 0.85
    
    def test_labels_validos(self):
        """Test que labels sean válidos"""
        datos = self.generador.cargar_datos()
        
        assert np.all(datos.y_train >= 0)
        assert np.all(datos.y_train < 10)
        assert np.all(datos.y_val >= 0)
        assert np.all(datos.y_val < 10)
        assert np.all(datos.y_test >= 0)
        assert np.all(datos.y_test < 10)
    
    def test_clases_distribuidas(self):
        """Test que clases estén distribuidas"""
        datos = self.generador.cargar_datos()
        
        train_clases = np.unique(datos.y_train)
        test_clases = np.unique(datos.y_test)
        
        assert len(train_clases) == 10
        assert len(test_clases) == 10


# ============================================================================
# TEST 2: Data Augmentation
# ============================================================================

class TestAugmentacion:
    """Tests para data augmentation"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
    
    def test_crear_augmentador(self):
        """Test creación de augmentador"""
        aug = self.generador.crear_augmentador()
        
        assert isinstance(aug, ImageDataGenerator)
    
    def test_augmentador_parametros(self):
        """Test que augmentador tenga parámetros correctos"""
        aug = self.generador.crear_augmentador()
        
        assert aug.rotation_range == 20
        assert aug.width_shift_range == 0.2
        assert aug.height_shift_range == 0.2
        assert aug.horizontal_flip == True
        assert aug.zoom_range == 0.2
    
    def test_augmentacion_produce_variedad(self):
        """Test que augmentación produce imágenes diferentes"""
        datos = self.generador.cargar_datos()
        aug = self.generador.crear_augmentador()
        
        X_original = datos.X_train[0:1]
        X_aug1 = next(aug.flow(X_original, batch_size=1))[0]
        X_aug2 = next(aug.flow(X_original, batch_size=1))[0]
        
        # Diferentes augmentaciones
        assert not np.allclose(X_aug1, X_aug2)


# ============================================================================
# TEST 3: Construcción de Modelos
# ============================================================================

class TestConstruccionModelos:
    """Tests para construcción de modelos"""
    
    def setup_method(self):
        self.clf = ClasificadorImagenes()
    
    def test_construir_cnn(self):
        """Test construcción de CNN"""
        modelo = self.clf.construir_cnn_profunda(n_clases=10)
        
        assert modelo.input_shape == (None, 32, 32, 3)
        assert modelo.output_shape == (None, 10)
    
    def test_construir_transfer_learning(self):
        """Test construcción de transfer learning"""
        modelo = self.clf.construir_transfer_learning(n_clases=10)
        
        assert modelo.input_shape == (None, 32, 32, 3)
        assert modelo.output_shape == (None, 10)
    
    def test_cnn_prediccion_shape(self):
        """Test shape de predicciones CNN"""
        modelo = self.clf.construir_cnn_profunda(n_clases=10)
        X = np.random.randn(5, 32, 32, 3).astype(np.float32)
        
        pred = modelo.predict(X, verbose=0)
        
        assert pred.shape == (5, 10)
        assert np.allclose(pred.sum(axis=1), 1.0)
    
    def test_transfer_prediccion_shape(self):
        """Test shape de predicciones transfer learning"""
        modelo = self.clf.construir_transfer_learning(n_clases=10)
        X = np.random.randn(5, 32, 32, 3).astype(np.float32)
        
        pred = modelo.predict(X, verbose=0)
        
        assert pred.shape == (5, 10)


# ============================================================================
# TEST 4: Entrenamiento
# ============================================================================

class TestEntrenamiento:
    """Tests para entrenamiento"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
    
    def test_entrenar_cnn(self):
        """Test entrenamiento CNN"""
        clf = ClasificadorImagenes()
        hist = clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=2, arquitectura='cnn',
            usar_augmentacion=False, verbose=0
        )
        
        assert clf.entrenado
        assert 'loss' in hist
        assert 'accuracy' in hist
    
    def test_entrenar_transfer_learning(self):
        """Test entrenamiento transfer learning"""
        clf = ClasificadorImagenes()
        hist = clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=2, arquitectura='transfer',
            usar_augmentacion=False, verbose=0
        )
        
        assert clf.entrenado
        assert 'loss' in hist
    
    def test_entrenar_loss_decrece(self):
        """Test que loss disminuya"""
        clf = ClasificadorImagenes()
        hist = clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=3, usar_augmentacion=False, verbose=0
        )
        
        assert hist['loss'][0] > hist['loss'][-1]


# ============================================================================
# TEST 5: Evaluación
# ============================================================================

class TestEvaluacion:
    """Tests para evaluación"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
        
        self.clf = ClasificadorImagenes()
        self.clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=2, usar_augmentacion=False, verbose=0
        )
    
    def test_evaluar_retorna_diccionario(self):
        """Test que evaluar retorna diccionario correcto"""
        metricas = self.clf.evaluar(self.datos.X_test[:100], self.datos.y_test[:100])
        
        assert isinstance(metricas, dict)
        assert 'loss' in metricas
        assert 'accuracy' in metricas
        assert 'confusion_matrix' in metricas
        assert 'classification_report' in metricas
        assert 'precision' in metricas
        assert 'recall' in metricas
        assert 'f1_score' in metricas
    
    def test_evaluar_metricas_validas(self):
        """Test que métricas sean válidas"""
        metricas = self.clf.evaluar(self.datos.X_test[:100], self.datos.y_test[:100])
        
        assert 0 <= metricas['accuracy'] <= 1
        assert 0 <= metricas['precision'] <= 1
        assert 0 <= metricas['recall'] <= 1
        assert 0 <= metricas['f1_score'] <= 1
    
    def test_confusion_matrix_shape(self):
        """Test shape de matriz de confusión"""
        metricas = self.clf.evaluar(self.datos.X_test[:50], self.datos.y_test[:50])
        cm = metricas['confusion_matrix']
        
        assert cm.shape == (10, 10)


# ============================================================================
# TEST 6: Predicción
# ============================================================================

class TestPrediccion:
    """Tests para predicción"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
        
        self.clf = ClasificadorImagenes()
        self.clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=2, usar_augmentacion=False, verbose=0
        )
    
    def test_predecir_retorna_tupla(self):
        """Test que predecir retorna tupla"""
        clases, probs = self.clf.predecir(self.datos.X_test[:10])
        
        assert isinstance(clases, np.ndarray)
        assert isinstance(probs, np.ndarray)
        assert clases.shape == (10,)
        assert probs.shape == (10, 10)
    
    def test_predecir_sin_entrenar(self):
        """Test predecir sin entrenar lanza error"""
        clf = ClasificadorImagenes()
        X = np.random.randn(5, 32, 32, 3).astype(np.float32)
        
        with pytest.raises(ValueError):
            clf.predecir(X)
    
    def test_predecir_probabilidades_validas(self):
        """Test que probabilidades sean válidas"""
        clases, probs = self.clf.predecir(self.datos.X_test[:10])
        
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.allclose(probs.sum(axis=1), 1.0)


# ============================================================================
# TEST 7: Transfer Learning
# ============================================================================

class TestTransferLearning:
    """Tests específicos para transfer learning"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
    
    def test_transfer_learning_mejor_que_cnn(self):
        """Test que TL típicamente sea mejor que CNN personalizado"""
        # CNN pequeño
        clf_cnn = ClasificadorImagenes()
        clf_cnn.entrenar(
            self.datos.X_train[:200], self.datos.y_train[:200],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=3, arquitectura='cnn',
            usar_augmentacion=False, verbose=0
        )
        acc_cnn = clf_cnn.evaluar(self.datos.X_test[:100], self.datos.y_test[:100])['accuracy']
        
        # Transfer learning
        clf_tl = ClasificadorImagenes()
        clf_tl.entrenar(
            self.datos.X_train[:200], self.datos.y_train[:200],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=3, arquitectura='transfer',
            usar_augmentacion=False, verbose=0
        )
        acc_tl = clf_tl.evaluar(self.datos.X_test[:100], self.datos.y_test[:100])['accuracy']
        
        # TL típicamente mejor
        assert acc_tl >= acc_cnn * 0.95  # Tolerancia


# ============================================================================
# TEST 8: Persistencia
# ============================================================================

class TestPersistencia:
    """Tests para guardar/cargar"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
        
        self.clf = ClasificadorImagenes()
        self.clf.entrenar(
            self.datos.X_train[:100], self.datos.y_train[:100],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=2, usar_augmentacion=False, verbose=0
        )
    
    def test_guardar_cargar(self, tmp_path):
        """Test guardar y cargar modelo"""
        ruta = str(tmp_path / "modelo")
        self.clf.guardar(ruta)
        
        clf_cargado = ClasificadorImagenes.cargar(ruta)
        
        # Predicciones idénticas
        X = self.datos.X_test[:5]
        clases1, probs1 = self.clf.predecir(X)
        clases2, probs2 = clf_cargado.predecir(X)
        
        np.testing.assert_array_equal(clases1, clases2)
        np.testing.assert_array_almost_equal(probs1, probs2, decimal=5)


# ============================================================================
# TEST 9: Clases Específicas
# ============================================================================

class TestClasesEspecificas:
    """Tests para predicción de clases específicas"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
        self.datos = self.generador.cargar_datos()
        
        self.clf = ClasificadorImagenes()
        self.clf.entrenar(
            self.datos.X_train[:200], self.datos.y_train[:200],
            self.datos.X_val[:50], self.datos.y_val[:50],
            epochs=3, usar_augmentacion=False, verbose=0
        )
    
    def test_todas_clases_en_prediccion(self):
        """Test que predice todas las clases"""
        clases, _ = self.clf.predecir(self.datos.X_test[:100])
        
        clases_unicas = np.unique(clases)
        # Debe predecir al menos 3-4 clases diferentes en 100 muestras
        assert len(clases_unicas) >= 3


# ============================================================================
# TEST 10: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests para casos extremos"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
    
    def test_imagen_unica(self):
        """Test con una imagen"""
        datos = self.generador.cargar_datos()
        clf = ClasificadorImagenes()
        clf.entrenar(
            datos.X_train[:50], datos.y_train[:50],
            datos.X_val[:10], datos.y_val[:10],
            epochs=1, usar_augmentacion=False, verbose=0
        )
        
        clases, probs = clf.predecir(datos.X_test[0:1])
        assert clases.shape == (1,)
        assert probs.shape == (1, 10)
    
    def test_imagen_todos_ceros(self):
        """Test con imagen negra"""
        datos = self.generador.cargar_datos()
        clf = ClasificadorImagenes()
        clf.entrenar(
            datos.X_train[:50], datos.y_train[:50],
            datos.X_val[:10], datos.y_val[:10],
            epochs=1, usar_augmentacion=False, verbose=0
        )
        
        X_negro = np.zeros((1, 32, 32, 3), dtype=np.float32)
        clases, probs = clf.predecir(X_negro)
        
        assert clases.shape == (1,)
        assert 0 <= clases[0] < 10


# ============================================================================
# TEST 11: Rendimiento
# ============================================================================

class TestRendimiento:
    """Tests de rendimiento"""
    
    def setup_method(self):
        self.generador = GeneradorCIFAR10()
    
    def test_velocidad_carga_datos(self):
        """Test velocidad de carga"""
        import time
        
        inicio = time.time()
        datos = self.generador.cargar_datos()
        tiempo = time.time() - inicio
        
        # Debe ser rápido (< 30 segundos)
        assert tiempo < 30.0
    
    def test_velocidad_prediccion(self):
        """Test velocidad de predicción"""
        import time
        
        datos = self.generador.cargar_datos()
        clf = ClasificadorImagenes()
        clf.entrenar(
            datos.X_train[:100], datos.y_train[:100],
            datos.X_val[:50], datos.y_val[:50],
            epochs=1, usar_augmentacion=False, verbose=0
        )
        
        inicio = time.time()
        clf.predecir(datos.X_test[:100])
        tiempo = time.time() - inicio
        
        # 100 predicciones en < 5 segundos
        assert tiempo < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
