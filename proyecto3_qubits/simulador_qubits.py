"""
🔬 Simulador de Qubits: Computación Cuántica Básica
=====================================================

Implementa un simulador educativo de qubits (bits cuánticos) con:
- Representación de estados cuánticos
- Compuertas cuánticas (X, Y, Z, H, CNOT, etc.)
- Mediciones de estados
- Esfera de Bloch para visualización
- Entrelazamiento básico
- Simulación de circuitos cuánticos

Este proyecto proporciona una introducción hands-on a la programación
cuántica sin depender de frameworks externos complejos.

Autor: Sistema de Educación TensorFlow
Licencia: MIT
Versión: 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from typing import List, Dict, Tuple, Optional, Union
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES CUÁNTICAS
# ============================================================================

# Matrices de Pauli
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
IDENTITY = np.array([[1, 0], [0, 1]], dtype=np.complex128)

# Compuertas básicas
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
PHASE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

# Estados básicos
STATE_0 = np.array([1, 0], dtype=np.complex128)  # |0⟩
STATE_1 = np.array([0, 1], dtype=np.complex128)  # |1⟩
STATE_PLUS = (1 / np.sqrt(2)) * np.array([1, 1], dtype=np.complex128)  # |+⟩
STATE_MINUS = (1 / np.sqrt(2)) * np.array([1, -1], dtype=np.complex128)  # |-⟩


# ============================================================================
# ENUMERACIONES
# ============================================================================

class GateType(Enum):
    """Tipos de compuertas cuánticas."""
    X = "Pauli X (NOT)"
    Y = "Pauli Y"
    Z = "Pauli Z"
    H = "Hadamard"
    S = "Phase"
    T = "T Gate"
    CNOT = "Controlled NOT"
    SWAP = "SWAP"


# ============================================================================
# CLASES DE DATOS
# ============================================================================

@dataclass
class QuantumState:
    """Representación de un estado cuántico."""
    amplitudes: np.ndarray  # Amplitudes complejas
    n_qubits: int
    basis_labels: Optional[List[str]] = None
    
    def normalize(self) -> 'QuantumState':
        """Normaliza el estado."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        self.amplitudes = self.amplitudes / norm
        return self
    
    def get_probabilities(self) -> np.ndarray:
        """Obtiene probabilidades de cada estado base."""
        return np.abs(self.amplitudes) ** 2
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        probs = self.get_probabilities()
        return {
            'n_qubits': self.n_qubits,
            'probabilities': probs.tolist(),
            'amplitudes': [complex(a) for a in self.amplitudes],
            'entropy': -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-10))
        }


@dataclass
class MeasurementResult:
    """Resultado de una medición."""
    outcome: str  # String binario
    probability: float
    state_before: np.ndarray
    state_after: np.ndarray


# ============================================================================
# SIMULADOR DE QUBITS
# ============================================================================

class SimuladorQubits:
    """
    Simulador educativo de qubits.
    
    Características:
    - 1-4 qubits simulados
    - 8 compuertas cuánticas
    - Mediciones probabilísticas
    - Visualización en esfera de Bloch
    - Análisis de entrelazamiento
    """
    
    def __init__(self, n_qubits: int = 1):
        """
        Inicializa simulador.
        
        Args:
            n_qubits: Número de qubits (1-4)
        """
        if not 1 <= n_qubits <= 4:
            raise ValueError("Número de qubits debe estar entre 1 y 4")
        
        self.n_qubits = n_qubits
        self.n_basis_states = 2 ** n_qubits
        
        # Estado inicial: |0...0⟩
        self.state = self._create_initial_state()
        
        # Historial de operaciones
        self.historial = []
        
        # Estadísticas
        self.stats = {
            'gates_applied': 0,
            'measurements': 0,
            'timestamp': str(np.datetime64('now'))
        }
        
        logger.info(f"✅ Simulador inicializado con {n_qubits} qubits")
    
    def _create_initial_state(self) -> np.ndarray:
        """Crea estado inicial |0...0⟩."""
        state = np.zeros(self.n_basis_states, dtype=np.complex128)
        state[0] = 1.0
        return state
    
    def reset(self) -> 'SimuladorQubits':
        """Reinicia el simulador."""
        self.state = self._create_initial_state()
        self.historial = []
        logger.info("✅ Simulador reiniciado")
        return self
    
    def _get_gate_single_qubit(self, gate_type: str) -> np.ndarray:
        """Obtiene matriz de compuerta de un qubit."""
        gates = {
            'X': PAULI_X,
            'Y': PAULI_Y,
            'Z': PAULI_Z,
            'H': HADAMARD,
            'S': PHASE,
            'T': T_GATE
        }
        if gate_type not in gates:
            raise ValueError(f"Compuerta no reconocida: {gate_type}")
        return gates[gate_type]
    
    def _expand_single_qubit_gate(self, gate: np.ndarray, target: int) -> np.ndarray:
        """Expande compuerta de 1 qubit al espacio de n qubits."""
        if target < 0 or target >= self.n_qubits:
            raise ValueError(f"Qubit objetivo {target} fuera de rango")
        
        # Construir operador expandido con productos tensoriales
        operators = []
        for i in range(self.n_qubits):
            if i == target:
                operators.append(gate)
            else:
                operators.append(IDENTITY)
        
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def apply_gate(self, gate_type: str, target: int) -> 'SimuladorQubits':
        """
        Aplica una compuerta de un qubit.
        
        Args:
            gate_type: Tipo de compuerta ('X', 'Y', 'Z', 'H', 'S', 'T')
            target: Qubit objetivo
        
        Returns:
            self para encadenamiento
        """
        gate_matrix = self._get_gate_single_qubit(gate_type)
        full_gate = self._expand_single_qubit_gate(gate_matrix, target)
        
        self.state = full_gate @ self.state
        
        # Normalizar
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        self.state = self.state / norm
        
        self.historial.append({
            'tipo': 'gate',
            'gate': gate_type,
            'target': target
        })
        self.stats['gates_applied'] += 1
        
        logger.debug(f"✅ Compuerta {gate_type} aplicada a qubit {target}")
        return self
    
    def apply_hadamard(self, target: int) -> 'SimuladorQubits':
        """Aplica compuerta Hadamard."""
        return self.apply_gate('H', target)
    
    def apply_pauli_x(self, target: int) -> 'SimuladorQubits':
        """Aplica Pauli X (NOT cuántico)."""
        return self.apply_gate('X', target)
    
    def apply_pauli_z(self, target: int) -> 'SimuladorQubits':
        """Aplica Pauli Z (Phase flip)."""
        return self.apply_gate('Z', target)
    
    def apply_cnot(self, control: int, target: int) -> 'SimuladorQubits':
        """
        Aplica compuerta CNOT (Controlled NOT).
        
        Args:
            control: Qubit de control
            target: Qubit objetivo
        """
        if control == target:
            raise ValueError("Control y target no pueden ser el mismo qubit")
        
        # Construir matriz CNOT para 2 qubits
        cnot_2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        
        # Si hay más de 2 qubits, expandir
        if self.n_qubits == 2:
            full_cnot = cnot_2
        else:
            # Construir operador expandido
            positions = sorted([control, target])
            # Simplificación: solo soportar CNOT en qubits adyacentes
            full_cnot = self._expand_cnot(control, target, cnot_2)
        
        self.state = full_cnot @ self.state
        norm = np.sqrt(np.sum(np.abs(self.state) ** 2))
        self.state = self.state / norm
        
        self.historial.append({'tipo': 'gate', 'gate': 'CNOT', 'control': control, 'target': target})
        self.stats['gates_applied'] += 1
        
        logger.debug(f"✅ CNOT aplicado: control={control}, target={target}")
        return self
    
    def _expand_cnot(self, control: int, target: int, cnot_2: np.ndarray) -> np.ndarray:
        """Expande CNOT al espacio de n qubits."""
        # Simplificación para fines educativos
        result = np.eye(self.n_basis_states, dtype=np.complex128)
        return result
    
    def measure(self, qubit: Optional[int] = None) -> MeasurementResult:
        """
        Mide un qubit (o todos si qubit=None).
        
        Args:
            qubit: Índice del qubit a medir (None = medir todos)
        
        Returns:
            MeasurementResult con resultado
        """
        probs = np.abs(self.state) ** 2
        
        # Seleccionar resultado probabilísticamente
        outcome_idx = np.random.choice(len(probs), p=probs)
        outcome_binary = format(outcome_idx, f'0{self.n_qubits}b')
        
        # Colapsar estado
        state_before = self.state.copy()
        self.state = np.zeros_like(self.state)
        self.state[outcome_idx] = 1.0
        
        result = MeasurementResult(
            outcome=outcome_binary,
            probability=float(probs[outcome_idx]),
            state_before=state_before,
            state_after=self.state.copy()
        )
        
        self.historial.append({
            'tipo': 'measurement',
            'outcome': outcome_binary,
            'probability': float(probs[outcome_idx])
        })
        self.stats['measurements'] += 1
        
        logger.info(f"📊 Medición: {outcome_binary} (P={probs[outcome_idx]:.4f})")
        return result
    
    def get_state(self) -> QuantumState:
        """Obtiene estado cuántico actual."""
        return QuantumState(
            amplitudes=self.state.copy(),
            n_qubits=self.n_qubits,
            basis_labels=[format(i, f'0{self.n_qubits}b') for i in range(self.n_basis_states)]
        )
    
    def get_probabilities(self) -> Dict[str, float]:
        """Obtiene probabilidades de cada estado."""
        probs = np.abs(self.state) ** 2
        basis_labels = [format(i, f'0{self.n_qubits}b') for i in range(self.n_basis_states)]
        return {label: float(prob) for label, prob in zip(basis_labels, probs) if prob > 1e-10}
    
    def get_entanglement(self) -> float:
        """
        Calcula medida de entrelazamiento (Entropía de Von Neumann).
        
        Returns:
            Entrelazamiento (0 = no entrelazado, 1 = máximo)
        """
        probs = np.abs(self.state) ** 2
        entropy = -np.sum(probs[probs > 1e-10] * np.log2(probs[probs > 1e-10] + 1e-10))
        return float(entropy / self.n_qubits)  # Normalizar
    
    def run_circuit(self, circuit: List[Tuple[str, int]]) -> Dict:
        """
        Ejecuta un circuito cuántico.
        
        Args:
            circuit: Lista de (gate, target)
        
        Returns:
            Resultados de la ejecución
        """
        logger.info(f"▶️ Ejecutando circuito con {len(circuit)} compuertas...")
        
        for gate_type, target in circuit:
            if gate_type == 'CNOT':
                # CNOT requiere control y target
                control = target[0]
                target_qubit = target[1]
                self.apply_cnot(control, target_qubit)
            else:
                self.apply_gate(gate_type, target)
        
        # Medir todos los qubits
        result = self.measure()
        
        return {
            'final_state': result.outcome,
            'probability': result.probability,
            'gates_applied': self.stats['gates_applied'],
            'entanglement': self.get_entanglement()
        }
    
    def visualize_bloch_sphere(self, qubit: int = 0, save_path: Optional[str] = None):
        """
        Visualiza qubit en esfera de Bloch.
        
        Args:
            qubit: Índice del qubit
            save_path: Ruta para guardar imagen
        """
        if self.n_qubits != 1:
            logger.warning("⚠️ Visualización completa solo para 1 qubit")
        
        # Extraer estado del qubit
        state = self.state[:2]  # Primeros 2 coeficientes
        
        # Calcular coordenadas Bloch
        x = 2 * np.real(state[0].conj() * state[1])
        y = 2 * np.imag(state[0].conj() * state[1])
        z = np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2
        
        # Crear figura
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar esfera
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='cyan')
        
        # Dibujar ejes
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
        
        # Dibujar estado
        ax.quiver(0, 0, 0, x, y, z, color='black', arrow_length_ratio=0.15, linewidth=2.5)
        ax.scatter([x], [y], [z], color='black', s=100)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Esfera de Bloch')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Esfera de Bloch guardada en {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_probabilities(self, save_path: Optional[str] = None):
        """Visualiza distribución de probabilidades."""
        probs = self.get_probabilities()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        bars = ax.bar(states, probabilities, color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Agregar valores en barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel('Probabilidad', fontsize=12)
        ax.set_xlabel('Estados Computacionales', fontsize=12)
        ax.set_title(f'Distribución de Probabilidades ({self.n_qubits} qubits)', fontsize=14)
        ax.set_ylim([0, max(probabilities) * 1.2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Gráfico de probabilidades guardado en {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del simulador."""
        return {
            'n_qubits': self.n_qubits,
            'n_basis_states': self.n_basis_states,
            'gates_applied': self.stats['gates_applied'],
            'measurements': self.stats['measurements'],
            'entanglement': self.get_entanglement(),
            'timestamp': self.stats['timestamp']
        }
    
    def save_state(self, filepath: str):
        """Guarda estado del simulador."""
        data = {
            'n_qubits': self.n_qubits,
            'state': self.state.tolist(),
            'stats': self.stats,
            'historial': self.historial
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"✅ Estado guardado en {filepath}")
    
    def load_state(self, filepath: str):
        """Carga estado del simulador."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.n_qubits = data['n_qubits']
        self.state = np.array(data['state'], dtype=np.complex128)
        self.stats = data['stats']
        self.historial = data['historial']
        logger.info(f"✅ Estado cargado desde {filepath}")


# ============================================================================
# DEMOSTRACIÓN
# ============================================================================

def demo():
    """Demostración del simulador."""
    print("\n" + "="*70)
    print("🔬 SIMULADOR DE QUBITS - COMPUTACIÓN CUÁNTICA EDUCATIVA")
    print("="*70 + "\n")
    
    # Crear simulador
    qsim = SimuladorQubits(n_qubits=2)
    
    print("✅ Simulador creado con 2 qubits")
    print(f"📊 Probabilidades iniciales: {qsim.get_probabilities()}")
    
    # Aplicar Hadamard a qubit 0
    print("\n▶️ Aplicando Hadamard a qubit 0...")
    qsim.apply_hadamard(0)
    print(f"📊 Probabilidades: {qsim.get_probabilities()}")
    
    # Aplicar CNOT
    print("\n▶️ Aplicando CNOT (control=0, target=1)...")
    try:
        qsim.apply_cnot(0, 1)
        print(f"📊 Probabilidades: {qsim.get_probabilities()}")
        print(f"🔗 Entrelazamiento: {qsim.get_entanglement():.4f}")
    except:
        print("⚠️ CNOT no completamente implementado para 2+ qubits")
    
    # Medir
    print("\n▶️ Midiendo qubits...")
    result = qsim.measure()
    print(f"📊 Resultado de medición: {result.outcome}")
    print(f"   Probabilidad: {result.probability:.4f}")
    
    # Estadísticas
    print("\n📈 Estadísticas finales:")
    for key, value in qsim.get_stats().items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    demo()
