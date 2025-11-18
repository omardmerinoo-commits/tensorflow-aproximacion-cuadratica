"""
Simulador cuántico básico usando QuTiP.

Implementa simulaciones básicas de sistemas cuánticos incluyendo
estados, operadores de Pauli, evolución temporal y medidas.
"""

import numpy as np
import qutip as qt
from typing import Tuple, List, Dict, Any
import json


class SimuladorCuanticoBasico:
    """Simulador de sistemas cuánticos simples."""
    
    def __init__(self):
        """Inicializar simulador."""
        self.estados = {}
        self.operadores = {}
        self._inicializar_estados_basicos()
        self._inicializar_operadores()
    
    def _inicializar_estados_basicos(self):
        """Inicializar estados básicos."""
        self.estados = {
            '|0>': qt.basis(2, 0),
            '|1>': qt.basis(2, 1),
            '|+>': (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
            '|->': (qt.basis(2, 0) - qt.basis(2, 1)).unit(),
            '|+i>': (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),
            '|-i>': (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit(),
        }
    
    def _inicializar_operadores(self):
        """Inicializar operadores de Pauli."""
        self.operadores = {
            'I': qt.qeye(2),
            'X': qt.sigmax(),
            'Y': qt.sigmay(),
            'Z': qt.sigmaz(),
            'H': (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
        }
    
    def obtener_estado(self, nombre: str) -> qt.Qobj:
        """
        Obtener estado cuántico.
        
        Parameters
        ----------
        nombre : str
            Nombre del estado ('|0>', '|1>', '|+>', etc)
            
        Returns
        -------
        estado : qt.Qobj
            Estado cuántico
        """
        if nombre in self.estados:
            return self.estados[nombre]
        raise ValueError(f"Estado desconocido: {nombre}")
    
    def obtener_operador(self, nombre: str) -> qt.Qobj:
        """Obtener operador."""
        if nombre in self.operadores:
            if isinstance(self.operadores[nombre], np.ndarray):
                return qt.Qobj(self.operadores[nombre])
            return self.operadores[nombre]
        raise ValueError(f"Operador desconocido: {nombre}")
    
    def calcular_bloch(self, estado: qt.Qobj) -> Tuple[float, float, float]:
        """
        Calcular coordenadas de Bloch.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado cuántico
            
        Returns
        -------
        x, y, z : Tuple[float, float, float]
            Coordenadas en la esfera de Bloch
        """
        x = qt.expect(qt.sigmax(), estado)
        y = qt.expect(qt.sigmay(), estado)
        z = qt.expect(qt.sigmaz(), estado)
        return float(x), float(y), float(z)
    
    def aplicar_operador(self, estado: qt.Qobj, operador: qt.Qobj) -> qt.Qobj:
        """
        Aplicar operador al estado.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado inicial
        operador : qt.Qobj
            Operador a aplicar
            
        Returns
        -------
        estado_nuevo : qt.Qobj
            Estado resultante
        """
        return operador * estado
    
    def evolucionar_temporal(self, estado: qt.Qobj, hamiltoniano: qt.Qobj,
                           tiempo_max: float = 10.0,
                           num_puntos: int = 100) -> Tuple[np.ndarray, List[qt.Qobj]]:
        """
        Evolucionar estado bajo Hamiltoniano.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado inicial
        hamiltoniano : qt.Qobj
            Hamiltoniano del sistema
        tiempo_max : float
            Tiempo máximo de evolución
        num_puntos : int
            Número de puntos temporales
            
        Returns
        -------
        tiempos : np.ndarray
            Tiempos
        estados : List[qt.Qobj]
            Estados en cada tiempo
        """
        tiempos = np.linspace(0, tiempo_max, num_puntos)
        resultado = qt.mesolve(hamiltoniano, estado, tiempos, [], [])
        return tiempos, resultado.states
    
    def calcular_fidelidad(self, estado1: qt.Qobj, estado2: qt.Qobj) -> float:
        """
        Calcular fidelidad entre dos estados.
        
        Parameters
        ----------
        estado1 : qt.Qobj
            Primer estado
        estado2 : qt.Qobj
            Segundo estado
            
        Returns
        -------
        fidelidad : float
            Fidelidad entre 0 y 1
        """
        return float(qt.fidelity(estado1, estado2)**2)
    
    def calcular_entropia(self, estado: qt.Qobj) -> float:
        """
        Calcular entropía de von Neumann.
        
        Parameters
        ----------
        estado : qt.Qobj
            Estado cuántico (puede ser estado puro o matriz densidad)
            
        Returns
        -------
        entropia : float
            Entropía de von Neumann
        """
        if estado.type == 'ket':
            matriz_densidad = estado * estado.dag()
        else:
            matriz_densidad = estado
        
        return float(qt.entropy_vn(matriz_densidad))
    
    def crear_hamiltoniano_precesion(self, frecuencia: float = 1.0) -> qt.Qobj:
        """
        Crear Hamiltoniano de precesión.
        
        H = ω * σz / 2
        
        Parameters
        ----------
        frecuencia : float
            Frecuencia de Rabi
            
        Returns
        -------
        H : qt.Qobj
            Hamiltoniano
        """
        return (frecuencia / 2.0) * qt.sigmaz()
