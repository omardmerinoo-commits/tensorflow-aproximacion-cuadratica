"""Tests Proyecto 8."""

import pytest
import numpy as np
from clasificador_musica import ExtractorCaracteristicasAudio, ClasificadorMusica


class TestExtractor:
    def test_generar_audio(self):
        y = ExtractorCaracteristicasAudio.generar_audio_sintetico(1.0, 22050, 'rock')
        assert y.shape[0] == 22050
    
    def test_extraer_caracteristicas(self):
        y = ExtractorCaracteristicasAudio.generar_audio_sintetico()
        caract = ExtractorCaracteristicasAudio.extraer_caracteristicas(y)
        assert caract.shape[0] == 20


class TestClasificador:
    @pytest.fixture
    def datos(self):
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randint(0, 3, 100).astype(np.int32)
        return X, y
    
    def test_construccion(self):
        model = ClasificadorMusica()
        model.construir_modelo()
        assert model.modelo is not None
    
    def test_prediccion(self, datos):
        X, _ = datos
        model = ClasificadorMusica()
        model.construir_modelo()
        preds, probs = model.predecir(X[:10])
        assert preds.shape == (10,)
        assert probs.shape == (10, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
