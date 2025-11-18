"""
Modelo de dos qubits entrelazados.

Simula estados Bell, operaciones CNOT, y medidas correlacionadas
para demostrar entrelazamiento cuántico.
"""

import numpy as np
import qutip as qt
from typing import Tuple, Dict, Any
import json


class SimuladorDosQubits:
    """Simulador de dos qubits entrelazados."""
    
    def __init__(self):
        """Inicializar simulador."""
        self.I = qt.qeye(2)
        self.X = qt.sigmax()
        self.Y = qt.sigmay()
        self.Z = qt.sigmaz()
        self.H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    
    def crear_estado_base(self, q1: int, q2: int) -> qt.Qobj:
        """
        Crear estado de base computacional.
        
        Parameters
        ----------
        q1, q2 : int
            Estados del qubit 1 y 2 (0 o 1)
            
        Returns
        -------
        estado : qt.Qobj
            Estado producto |q1q2>
        """
        return qt.tensor(qt.basis(2, q1), qt.basis(2, q2))
    
    def crear_estados_bell(self) -> Dict[str, qt.Qobj]:
        """
        Crear estados de Bell.
        
        Returns
        -------
        estados : Dict[str, qt.Qobj]
            Cuatro estados de Bell
        """
        estados = {}
        
        # |Φ+> = (|00> + |11>) / √2
        estados['Φ+'] = (self.crear_estado_base(0, 0) + 
                        self.crear_estado_base(1, 1)).unit()
        
        # |Φ-> = (|00> - |11>) / √2
        estados['Φ-'] = (self.crear_estado_base(0, 0) - 
                        self.crear_estado_base(1, 1)).unit()
        
        # |Ψ+> = (|01> + |10>) / √2
        estados['Ψ+'] = (self.crear_estado_base(0, 1) + 
                        self.crear_estado_base(1, 0)).unit()
        
        # |Ψ-> = (|01> - |10>) / √2
        estados['Ψ-'] = (self.crear_estado_base(0, 1) - 
                        self.crear_estado_base(1, 0)).unit()
        
        return estados
    
    def crear_puerta_cnot(self) -> qt.Qobj:
        """
        Crear puerta CNOT.
        
        Returns
        -------
        CNOT : qt.Qobj
            Operador CNOT (control en qubit 1)
        """
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        return qt.Qobj(CNOT)
    
    def crear_hadamard_dos_qubits(self) -> qt.Qobj:
        """
        Crear puerta Hadamard en dos qubits.
        
        Returns
        -------
        H_2q : qt.Qobj
            H ⊗ I aplicado a dos qubits
        """
        return qt.tensor(qt.qeye(2), self.H)
    
    def generar_par_entrelazado(self) -> qt.Qobj:
        """
        Generar estado de Bell |Φ+> desde |00>.
        
        Aplica: H ⊗ I seguido de CNOT
        
        Returns
        -------
        estado_bell : qt.Qobj
            Estado |Φ+>
        """
        estado = self.crear_estado_base(0, 0)
        
        # H en primer qubit
        H_I = qt.tensor(self.H, self.I)
        estado = H_I * estado
        
        # CNOT
        CNOT = self.crear_puerta_cnot()
        estado = CNOT * estado
        
        return estado
    
    def medir_qubit(self, estado: qt.Qobj, qubit: int) -> Tuple[int, qt.Qobj]:
        """
        Medir un qubit (simulado).
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado a medir
        qubit : int
            Qubit a medir (0 o 1)
            
        Returns
        -------
        resultado : int
            Resultado de medida (0 o 1)
        estado_colapso : qt.Qobj
            Estado colapsado
        """
        # Operadores de proyección
        if qubit == 0:
            P0 = qt.tensor(qt.basis(2, 0) * qt.basis(2, 0).dag(), self.I)
            P1 = qt.tensor(qt.basis(2, 1) * qt.basis(2, 1).dag(), self.I)
        else:
            P0 = qt.tensor(self.I, qt.basis(2, 0) * qt.basis(2, 0).dag())
            P1 = qt.tensor(self.I, qt.basis(2, 1) * qt.basis(2, 1).dag())
        
        # Probabilidades
        rho = estado * estado.dag()
        p0 = (P0 * rho).tr()
        p1 = (P1 * rho).tr()
        
        # Resultado aleatorio
        resultado = np.random.binomial(1, p1)
        
        # Estado colapsado
        if resultado == 0:
            estado_nuevo = (P0 * estado).unit()
        else:
            estado_nuevo = (P1 * estado).unit()
        
        return resultado, estado_nuevo
    
    def calcular_correlacion(self, estado: qt.Qobj) -> float:
        """
        Calcular correlación ZZ entre qubits.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado entrelazado
            
        Returns
        -------
        correlacion : float
            <Z1 Z2>
        """
        Z1Z2 = qt.tensor(self.Z, self.Z)
        correlacion = qt.expect(Z1Z2, estado)
        return float(correlacion)
    
    def calcular_desigualdad_bell(self, estado: qt.Qobj) -> float:
        """
        Calcular S para la desigualdad CHSH.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado entrelazado
            
        Returns
        -------
        S : float
            Valor de S (máximo clásico: 2, máximo cuántico: 2√2)
        """
        # Operadores
        A0 = qt.tensor(self.Z, self.I)
        A1 = qt.tensor(self.X, self.I)
        B0 = qt.tensor(self.I, (self.Z + self.X) / np.sqrt(2))
        B1 = qt.tensor(self.I, (self.Z - self.X) / np.sqrt(2))
        
        # Correlaciones
        E00 = qt.expect(qt.tensor(A0, B0), estado)
        E01 = qt.expect(qt.tensor(A0, B1), estado)
        E10 = qt.expect(qt.tensor(A1, B0), estado)
        E11 = qt.expect(qt.tensor(A1, B1), estado)
        
        S = abs(E00 + E01 + E10 - E11)
        return float(S)
