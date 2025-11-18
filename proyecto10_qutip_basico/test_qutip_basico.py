"""Tests Proyecto 10."""

import pytest
import numpy as np
import qutip as qt
from simulador_qutip_basico import SimuladorCuanticoBasico


class TestSimulador:
    @pytest.fixture
    def sim(self):
        return SimuladorCuanticoBasico()
    
    def test_obtener_estado(self, sim):
        estado = sim.obtener_estado('|0>')
        assert estado is not None
        assert isinstance(estado, qt.Qobj)
    
    def test_estados_basicos(self, sim):
        for nombre in ['|0>', '|1>', '|+>']:
            estado = sim.obtener_estado(nombre)
            assert estado.shape == (2, 1)
    
    def test_bloch(self, sim):
        estado = sim.obtener_estado('|0>')
        x, y, z = sim.calcular_bloch(estado)
        assert isinstance(x, float)
        assert abs(z - (-1.0)) < 0.01
    
    def test_operadores(self, sim):
        estado = sim.obtener_estado('|0>')
        op_x = sim.obtener_operador('X')
        estado_nuevo = sim.aplicar_operador(estado, op_x)
        assert estado_nuevo.shape == (2, 1)
    
    def test_fidelidad(self, sim):
        e1 = sim.obtener_estado('|0>')
        e2 = sim.obtener_estado('|0>')
        fid = sim.calcular_fidelidad(e1, e2)
        assert abs(fid - 1.0) < 0.01
    
    def test_entropia(self, sim):
        estado = sim.obtener_estado('|0>')
        ent = sim.calcular_entropia(estado)
        assert ent >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
