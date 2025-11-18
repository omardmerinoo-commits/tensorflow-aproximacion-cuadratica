"""Tests Proyecto 11."""

import pytest
import numpy as np
import qutip as qt
from simulador_decoherencia import SimuladorDecoherencia


class TestDecoherencia:
    @pytest.fixture
    def sim(self):
        return SimuladorDecoherencia(T1=2.0, T2=1.0)
    
    def test_operadores_lindblad(self, sim):
        ops = sim.crear_operadores_lindblad(2)
        assert len(ops) > 0
    
    def test_simulacion_decoherencia(self, sim):
        estado = qt.basis(2, 0)
        tiempos, x, z = sim.simular_decoherencia(estado, 10, 50)
        assert len(tiempos) == 50
        assert len(x) == 50
        assert len(z) == 50
    
    def test_eco_hahn(self, sim):
        estado = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        tiempos, coherencia = sim.simular_eco_hahn(estado, 10, 50)
        assert len(tiempos) == 50
        assert len(coherencia) == 50
        assert coherencia[0] > coherencia[-1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
