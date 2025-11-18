"""Tests Proyecto 9."""

import pytest
import numpy as np
from contador_objetos import GeneradorImagenesSinteticas, ContadorObjetos


class TestGenerador:
    def test_generar_imagen(self):
        img, count = GeneradorImagenesSinteticas.generar_imagen_con_objetos(5, 64, 8)
        assert img.shape == (64, 64, 3)
        assert 0 <= count <= 5
    
    def test_dataset(self):
        imgs, conteos = GeneradorImagenesSinteticas.generar_dataset(100, 64)
        assert imgs.shape == (100, 64, 64, 3)
        assert conteos.shape == (100,)


class TestContador:
    @pytest.fixture
    def datos(self):
        imgs = np.random.rand(50, 64, 64, 3).astype(np.float32)
        conteos = np.random.randint(0, 15, 50).astype(np.float32)
        return imgs, conteos
    
    def test_construccion(self):
        model = ContadorObjetos()
        model.construir_modelo()
        assert model.modelo is not None
    
    def test_prediccion(self, datos):
        imgs, _ = datos
        model = ContadorObjetos()
        model.construir_modelo()
        preds = model.predecir(imgs[:10])
        assert preds.shape == (10,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
