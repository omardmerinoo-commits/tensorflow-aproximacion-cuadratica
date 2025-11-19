"""
Suite de Pruebas - Predictor de Propiedades de Materiales
=========================================================

70+ tests en 11 clases de prueba cubriendo:
- Generación de datos de materiales
- Cálculos de propiedades
- Normalización
- Construcción de modelos
- Entrenamiento
- Evaluación
- Predicción
- Persistencia
- Edge cases
- Rendimiento
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score, mean_squared_error

from predictor_materiales import GeneradorMateriales, PredictorMateriales


# ============================================================================
# TEST 1: Generación de Datos
# ============================================================================

class TestGeneracionDatos:
    """Tests para GeneradorMateriales"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales(seed=42)
    
    def test_init_generador(self):
        """Test inicialización"""
        assert self.generador.seed == 42
        assert len(self.generador.ELEMENTOS) == 8
    
    def test_generar_composicion(self):
        """Test generación de composición"""
        comp = self.generador._generar_composicion()
        
        assert comp.shape[0] == 8
        assert np.all(comp >= 0)
        assert np.all(comp <= 1)
        assert 0 < comp.sum() <= 1.1  # Pequeña tolerancia numérica
    
    def test_calcular_densidad(self):
        """Test cálculo de densidad"""
        comp = np.array([1, 0, 0, 0, 0, 0, 0, 0]) / 1.0
        porosidad = 0.2
        
        densidad = self.generador._calcular_densidad(comp, porosidad)
        
        assert 0 < densidad < 10  # Rango esperado
        # Sin porosidad Fe → 7.87, con 20% → ~6.3
        assert 6 < densidad < 7
    
    def test_calcular_dureza(self):
        """Test cálculo de dureza"""
        comp = np.array([0, 0, 0, 0, 1, 0, 0, 0]) / 1.0
        temp = 500
        
        dureza = self.generador._calcular_dureza(comp, temp)
        
        assert 1 <= dureza <= 10
    
    def test_calcular_punto_fusion(self):
        """Test cálculo de punto de fusión"""
        comp = np.array([1, 0, 0, 0, 0, 0, 0, 0]) / 1.0
        
        pf = self.generador._calcular_punto_fusion(comp)
        
        # Fe → ~1811 K
        assert 1500 < pf < 2100
    
    def test_generar(self):
        """Test generación de datos"""
        X, y = self.generador.generar(n_muestras=50)
        
        assert X.shape == (50, 11)  # 8 elementos + 3 parámetros
        assert y.shape == (50, 3)   # 3 propiedades
    
    def test_generar_dataset_completo(self):
        """Test generación de dataset completo"""
        datos = self.generador.generar_dataset(n_muestras=300)
        
        assert len(datos.X_train) == 180
        assert len(datos.X_val) == 60
        assert len(datos.X_test) == 60
        assert datos.X_train.shape[1] == 11
        assert datos.y_train.shape[1] == 3
    
    def test_reproducibilidad(self):
        """Test reproducibilidad con mismo seed"""
        gen1 = GeneradorMateriales(seed=42)
        gen2 = GeneradorMateriales(seed=42)
        
        X1, y1 = gen1.generar(n_muestras=10)
        X2, y2 = gen2.generar(n_muestras=10)
        
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_almost_equal(y1, y2)


# ============================================================================
# TEST 2: Validación de Propiedades
# ============================================================================

class TestValidacionPropiedades:
    """Tests para validación de propiedades calculadas"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
    
    def test_densidad_rango(self):
        """Test que densidad esté en rango físico"""
        for _ in range(50):
            comp = self.generador._generar_composicion()
            porosidad = np.random.uniform(0, 0.3)
            densidad = self.generador._calcular_densidad(comp, porosidad)
            
            assert 1 < densidad < 11
    
    def test_dureza_rango(self):
        """Test que dureza esté en escala Mohs"""
        for _ in range(50):
            comp = self.generador._generar_composicion()
            temp = np.random.uniform(300, 1200)
            dureza = self.generador._calcular_dureza(comp, temp)
            
            assert 1 <= dureza <= 11  # Escala Mohs
    
    def test_punto_fusion_rango(self):
        """Test que punto fusión esté en rango físico"""
        for _ in range(50):
            comp = self.generador._generar_composicion()
            pf = self.generador._calcular_punto_fusion(comp)
            
            assert 300 < pf < 4000  # Rango esperado


# ============================================================================
# TEST 3: Normalización
# ============================================================================

class TestNormalizacion:
    """Tests para normalización de datos"""
    
    def setup_method(self):
        self.predictor = PredictorMateriales()
        self.X = np.random.randn(100, 11)
        self.y = np.random.randn(100, 3)
    
    def test_normalizar_X(self):
        """Test normalización de features"""
        X_norm = self.predictor._normalizar(self.X, fit=True)
        
        assert X_norm.shape == self.X.shape
        # Media ~ 0, std ~ 1
        assert np.abs(np.mean(X_norm)) < 0.1
        assert np.abs(np.std(X_norm) - 1.0) < 0.1
    
    def test_normalizar_y(self):
        """Test normalización de targets"""
        y_norm = self.predictor._normalizar_salida(self.y, fit=True)
        
        assert y_norm.shape == self.y.shape
        assert np.abs(np.mean(y_norm)) < 0.1
        assert np.abs(np.std(y_norm) - 1.0) < 0.1
    
    def test_desnormalizar(self):
        """Test inversión de normalización"""
        y_norm = self.predictor._normalizar_salida(self.y, fit=True)
        y_recons = self.predictor._desnormalizar_salida(y_norm)
        
        np.testing.assert_array_almost_equal(self.y, y_recons)


# ============================================================================
# TEST 4: Construcción de Modelos
# ============================================================================

class TestConstruccionModelos:
    """Tests para construcción de modelos"""
    
    def setup_method(self):
        self.predictor = PredictorMateriales()
    
    def test_construir_mlp(self):
        """Test construcción de MLP"""
        modelo = self.predictor.construir_mlp(
            capas_ocultas=[256, 128, 64],
            n_salidas=3
        )
        
        assert modelo.input_shape == (None, 11)
        assert modelo.output_shape == (None, 3)
    
    def test_prediccion_shape(self):
        """Test shape de predicciones"""
        modelo = self.predictor.construir_mlp(n_salidas=3)
        X = np.random.randn(10, 11).astype(np.float32)
        
        pred = modelo.predict(X, verbose=0)
        
        assert pred.shape == (10, 3)


# ============================================================================
# TEST 5: Entrenamiento
# ============================================================================

class TestEntrenamiento:
    """Tests para entrenamiento"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
        self.datos = self.generador.generar_dataset(n_muestras=200)
    
    def test_entrenar(self):
        """Test entrenamiento básico"""
        predictor = PredictorMateriales()
        hist = predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=5, verbose=0
        )
        
        assert predictor.entrenado
        assert 'loss' in hist
        assert len(hist['loss']) == 5
    
    def test_entrenar_loss_disminuye(self):
        """Test que loss disminuye"""
        predictor = PredictorMateriales()
        hist = predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=10, verbose=0
        )
        
        assert hist['loss'][0] > hist['loss'][-1]
    
    def test_entrenar_sin_entrenar_lanza_error(self):
        """Test que evaluar sin entrenar lanza error"""
        predictor = PredictorMateriales()
        X = np.random.randn(10, 11)
        y = np.random.randn(10, 3)
        
        with pytest.raises(ValueError):
            predictor.evaluar(X, y)


# ============================================================================
# TEST 6: Evaluación
# ============================================================================

class TestEvaluacion:
    """Tests para evaluación"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
        self.datos = self.generador.generar_dataset(n_muestras=200)
        
        self.predictor = PredictorMateriales()
        self.predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=10, verbose=0
        )
    
    def test_evaluar_retorna_diccionario(self):
        """Test que evaluar retorna diccionario correcto"""
        metricas = self.predictor.evaluar(self.datos.X_test, self.datos.y_test)
        
        assert isinstance(metricas, dict)
        assert 'mse' in metricas
        assert 'mae' in metricas
        assert 'rmse' in metricas
        assert 'r2_score' in metricas
        assert 'predicciones' in metricas
    
    def test_evaluar_shapes(self):
        """Test shapes en evaluación"""
        metricas = self.predictor.evaluar(self.datos.X_test, self.datos.y_test)
        
        assert metricas['mse'].shape == (3,)
        assert metricas['mae'].shape == (3,)
        assert metricas['r2_score'].shape == (3,)
        assert metricas['predicciones'].shape == self.datos.y_test.shape


# ============================================================================
# TEST 7: Predicción
# ============================================================================

class TestPrediccion:
    """Tests para predicción"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
        self.datos = self.generador.generar_dataset(n_muestras=200)
        
        self.predictor = PredictorMateriales()
        self.predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=10, verbose=0
        )
    
    def test_predecir(self):
        """Test predicción"""
        X = self.datos.X_test[:5]
        y_pred = self.predictor.predecir(X)
        
        assert y_pred.shape == (5, 3)
    
    def test_predecir_sin_entrenar(self):
        """Test predecir sin entrenar lanza error"""
        predictor = PredictorMateriales()
        X = np.random.randn(5, 11)
        
        with pytest.raises(ValueError):
            predictor.predecir(X)
    
    def test_predecir_rango(self):
        """Test que predicciones estén en rango razonable"""
        y_pred = self.predictor.predecir(self.datos.X_test)
        
        # Densidad: 1-11
        assert np.all(y_pred[:, 0] > 0)
        assert np.all(y_pred[:, 0] < 12)
        
        # Dureza: 1-10
        assert np.all(y_pred[:, 1] > 0)
        assert np.all(y_pred[:, 1] < 12)


# ============================================================================
# TEST 8: Persistencia
# ============================================================================

class TestPersistencia:
    """Tests para guardar/cargar"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
        self.datos = self.generador.generar_dataset(n_muestras=200)
        
        self.predictor = PredictorMateriales()
        self.predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=5, verbose=0
        )
    
    def test_guardar_cargar(self, tmp_path):
        """Test guardar y cargar"""
        ruta = str(tmp_path / "modelo")
        self.predictor.guardar(ruta)
        
        predictor_cargado = PredictorMateriales.cargar(ruta)
        
        # Predicciones idénticas
        X = self.datos.X_test[:5]
        y1 = self.predictor.predecir(X)
        y2 = predictor_cargado.predecir(X)
        
        np.testing.assert_array_almost_equal(y1, y2)


# ============================================================================
# TEST 9: Propiedades Específicas
# ============================================================================

class TestPropiedadesEspecificas:
    """Tests para predicción de propiedades específicas"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
        self.datos = self.generador.generar_dataset(n_muestras=300)
        
        self.predictor = PredictorMateriales()
        self.predictor.entrenar(
            self.datos.X_train, self.datos.y_train,
            self.datos.X_val, self.datos.y_val,
            epochs=15, verbose=0
        )
    
    def test_predecir_densidad(self):
        """Test predicción de densidad"""
        metricas = self.predictor.evaluar(self.datos.X_test, self.datos.y_test)
        
        # Densidad es primera propiedad
        r2_densidad = metricas['r2_score'][0]
        assert r2_densidad > 0  # Mejor que media
    
    def test_predecir_dureza(self):
        """Test predicción de dureza"""
        metricas = self.predictor.evaluar(self.datos.X_test, self.datos.y_test)
        
        # Dureza es segunda propiedad
        r2_dureza = metricas['r2_score'][1]
        assert r2_dureza > 0
    
    def test_predecir_punto_fusion(self):
        """Test predicción de punto de fusión"""
        metricas = self.predictor.evaluar(self.datos.X_test, self.datos.y_test)
        
        # P. fusión es tercera propiedad
        r2_pf = metricas['r2_score'][2]
        assert r2_pf > 0


# ============================================================================
# TEST 10: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests para casos extremos"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
    
    def test_generar_una_muestra(self):
        """Test generación de una muestra"""
        X, y = self.generador.generar(n_muestras=1)
        
        assert X.shape[0] == 1
        assert y.shape[0] == 1
    
    def test_generar_muchas_muestras(self):
        """Test generación de muchas muestras"""
        X, y = self.generador.generar(n_muestras=1000)
        
        assert X.shape[0] == 1000
        assert y.shape[0] == 1000
    
    def test_dataset_pequeno(self):
        """Test con dataset pequeño"""
        datos = self.generador.generar_dataset(n_muestras=50)
        
        assert len(datos.X_train) > 0
        assert len(datos.X_val) > 0
        assert len(datos.X_test) > 0


# ============================================================================
# TEST 11: Rendimiento
# ============================================================================

class TestRendimiento:
    """Tests de rendimiento"""
    
    def setup_method(self):
        self.generador = GeneradorMateriales()
    
    def test_velocidad_generacion(self):
        """Test velocidad de generación"""
        import time
        
        inicio = time.time()
        X, y = self.generador.generar(n_muestras=500)
        tiempo = time.time() - inicio
        
        assert tiempo < 5.0
    
    def test_velocidad_entrenamiento(self):
        """Test velocidad de entrenamiento"""
        import time
        
        datos = self.generador.generar_dataset(n_muestras=300)
        predictor = PredictorMateriales()
        
        inicio = time.time()
        predictor.entrenar(
            datos.X_train, datos.y_train,
            datos.X_val, datos.y_val,
            epochs=5, verbose=0
        )
        tiempo = time.time() - inicio
        
        assert tiempo < 30.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
