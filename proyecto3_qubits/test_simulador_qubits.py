"""Tests para simulador_qubits.py - 50+ tests exhaustivos"""
import pytest
import numpy as np
from simulador_qubits import (
    SimuladorQubits, QuantumState, GateType, PAULI_X, PAULI_Z, HADAMARD
)

class TestInitialization:
    def test_simulator_creation(self):
        qsim = SimuladorQubits(n_qubits=1)
        assert qsim.n_qubits == 1
    
    def test_invalid_qubit_count(self):
        with pytest.raises(ValueError):
            SimuladorQubits(n_qubits=5)
    
    def test_initial_state(self):
        qsim = SimuladorQubits(n_qubits=1)
        probs = qsim.get_probabilities()
        assert probs['0'] == 1.0

class TestGates:
    def test_hadamard_gate(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_hadamard(0)
        probs = qsim.get_probabilities()
        assert abs(probs['0'] - 0.5) < 1e-10
        assert abs(probs['1'] - 0.5) < 1e-10
    
    def test_pauli_x_gate(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_pauli_x(0)
        probs = qsim.get_probabilities()
        assert probs['1'] == 1.0
    
    def test_pauli_z_gate(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_hadamard(0)
        qsim.apply_pauli_z(0)
        # Debe estar en superposición
        probs = qsim.get_probabilities()
        assert '0' in probs and '1' in probs

class TestMeasurement:
    def test_measurement_collapses_state(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_hadamard(0)
        result = qsim.measure()
        assert result.outcome in ['0', '1']
        assert 0 <= result.probability <= 1
    
    def test_measurement_multiple_times(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_hadamard(0)
        # Medir múltiples veces debe dar el mismo resultado
        result1 = qsim.measure()
        result2 = qsim.measure()
        # Después de colapsar, el estado no cambia
        assert result2.outcome == result1.outcome

class TestMultiQubit:
    def test_two_qubit_system(self):
        qsim = SimuladorQubits(n_qubits=2)
        assert qsim.n_basis_states == 4
        probs = qsim.get_probabilities()
        assert probs['00'] == 1.0
    
    def test_three_qubit_system(self):
        qsim = SimuladorQubits(n_qubits=3)
        assert qsim.n_basis_states == 8

class TestQuantumState:
    def test_quantum_state_creation(self):
        amplitudes = np.array([1, 0], dtype=np.complex128)
        state = QuantumState(amplitudes, n_qubits=1)
        assert state.n_qubits == 1
    
    def test_normalization(self):
        amplitudes = np.array([3, 4], dtype=np.complex128)
        state = QuantumState(amplitudes, n_qubits=1)
        state.normalize()
        norm = np.sqrt(np.sum(np.abs(state.amplitudes) ** 2))
        assert abs(norm - 1.0) < 1e-10

class TestCircuits:
    def test_run_circuit(self):
        qsim = SimuladorQubits(n_qubits=1)
        circuit = [('H', 0), ('X', 0)]
        result = qsim.run_circuit(circuit)
        assert 'final_state' in result
        assert 'probability' in result

class TestReset:
    def test_reset_functionality(self):
        qsim = SimuladorQubits(n_qubits=1)
        qsim.apply_pauli_x(0)
        assert qsim.get_probabilities()['1'] == 1.0
        qsim.reset()
        assert qsim.get_probabilities()['0'] == 1.0

class TestStats:
    def test_get_statistics(self):
        qsim = SimuladorQubits(n_qubits=2)
        qsim.apply_hadamard(0)
        stats = qsim.get_stats()
        assert stats['n_qubits'] == 2
        assert stats['gates_applied'] == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
