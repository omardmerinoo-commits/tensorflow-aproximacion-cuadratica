"""Tests Proyecto 12."""

import pytest
import numpy as np
import qutip as qt
from simulador_qubits_entrelazados import SimuladorDosQubits


class TestDosQubits:
    @pytest.fixture
    def sim(self):
        return SimuladorDosQubits()
    
    def test_estado_base(self, sim):
        estado = sim.crear_estado_base(0, 0)
        assert estado.shape == (4, 1)
    
    def test_estados_bell(self, sim):
        estados = sim.crear_estados_bell()
        assert len(estados) == 4
        for estado in estados.values():
            assert estado.shape == (4, 1)
    
    def test_puerta_cnot(self, sim):
        cnot = sim.crear_puerta_cnot()
        assert cnot.shape == (4, 4)
    
    def test_generar_entrelazado(self, sim):
        estado = sim.generar_par_entrelazado()
        assert estado.shape == (4, 1)
        corr = sim.calcular_correlacion(estado)
        assert abs(corr - (-1)) < 0.1
    
    def test_desigualdad_bell(self, sim):
        estado = sim.generar_par_entrelazado()
        S = sim.calcular_desigualdad_bell(estado)
        assert S > 2  # Violación de Bell


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
