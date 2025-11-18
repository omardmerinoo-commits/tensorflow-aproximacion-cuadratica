"""Tests Proyecto 7."""

import pytest
import numpy as np
from predictor_materiales import GeneradorDatosMateriales, PredictorMateriales


class TestGenerador:
    def test_generar_composiciones(self):
        gen = GeneradorDatosMateriales()
        X, y = gen.generar_composiciones(100)
        assert X.shape == (100, 5)
        assert y.shape == (100, 3)
        assert np.allclose(X.sum(axis=1), 1.0)


class TestPredictor:
    @pytest.fixture
    def datos(self):
        gen = GeneradorDatosMateriales(seed=42)
        X, y = gen.generar_composiciones(100)
        return X, y
    
    def test_construccion(self):
        model = PredictorMateriales(5, 3)
        model.construir_modelo()
        assert model.modelo is not None
    
    def test_prediccion(self, datos):
        X, y = datos
        model = PredictorMateriales()
        model.construir_modelo()
        preds = model.predecir(X[:10])
        assert preds.shape == (10, 3)
    
    def test_entrenamiento(self, datos):
        X, y = datos
        model = PredictorMateriales()
        model.construir_modelo()
        hist = model.entrenar(X, y, epochs=5, verbose=0)
        assert hist['loss_final'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
