"""
Simulación de decoherencia en sistemas cuánticos abiertos.

Simula T1 (relajación de energía) y T2 (dephasing) usando
operadores de Lindblad en QuTiP.
"""

import numpy as np
import qutip as qt
from typing import Tuple, List, Dict, Any
import json


class SimuladorDecoherencia:
    """Simulador de decoherencia usando ecuación maestra de Lindblad."""
    
    def __init__(self, T1: float = 1.0, T2: float = 1.0):
        """
        Inicializar simulador.
        
        Parameters
        ----------
        T1 : float
            Tiempo de relajación de energía (segundos)
        T2 : float
            Tiempo de dephasing (segundos)
        """
        self.T1 = T1
        self.T2 = T2
    
    def crear_operadores_lindblad(self, N: int = 2) -> List[qt.Qobj]:
        """
        Crear operadores de Lindblad.
        
        Parameters
        ----------
        N : int
            Dimensión del espacio de Hilbert
            
        Returns
        -------
        operadores : List[qt.Qobj]
            Operadores de decaimiento
        """
        a = qt.destroy(N)
        sm = qt.destroy(N)
        
        operadores = []
        
        # T1: relajación de energía
        if self.T1 > 0:
            gamma_t1 = 1.0 / self.T1
            operadores.append(np.sqrt(gamma_t1) * a)
        
        # T2: dephasing
        if self.T2 > 0:
            gamma_t2 = 1.0 / self.T2 - 1.0 / (2.0 * self.T1)
            if gamma_t2 > 0:
                operadores.append(np.sqrt(gamma_t2) * qt.sigmaz())
        
        return operadores
    
    def simular_decoherencia(self, estado_inicial: qt.Qobj,
                            tiempo_max: float = 10.0,
                            num_puntos: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simular decoherencia de estado inicial.
        
        Parameters
        ----------
        estado_inicial : qt.Qobj
            Estado inicial (usualmente |0> o superposición)
        tiempo_max : float
            Tiempo máximo de simulación
        num_puntos : int
            Número de puntos temporales
            
        Returns
        -------
        tiempos : np.ndarray
            Tiempos
        x_valores : np.ndarray
            Valores esperados <σx>
        z_valores : np.ndarray
            Valores esperados <σz>
        """
        tiempos = np.linspace(0, tiempo_max, num_puntos)
        
        # Hamiltoniano nulo (solo decoherencia)
        H = 0 * qt.sigmaz()
        
        # Operadores de Lindblad
        c_ops = self.crear_operadores_lindblad(2)
        
        # Resolver ecuación maestra
        resultado = qt.mesolve(H, estado_inicial, tiempos, c_ops,
                             [qt.sigmax(), qt.sigmaz()])
        
        return tiempos, np.array(resultado.expect[0]), np.array(resultado.expect[1])
    
    def simular_eco_hahn(self, estado_inicial: qt.Qobj,
                        tiempo_max: float = 10.0,
                        num_puntos: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simular eco de Hahn (refocalización de dephasing).
        
        Parameters
        ----------
        estado_inicial : qt.Qobj
            Estado inicial
        tiempo_max : float
            Tiempo máximo
        num_puntos : int
            Número de puntos
            
        Returns
        -------
        tiempos : np.ndarray
            Tiempos
        cohencia : np.ndarray
            Coeficiente de coherencia
        """
        tiempos = np.linspace(0, tiempo_max, num_puntos)
        
        H = 0 * qt.sigmaz()
        c_ops = self.crear_operadores_lindblad(2)
        
        coherencia = np.zeros(len(tiempos))
        
        for i, t in enumerate(tiempos):
            # Evolución sin pulso (decoherencia)
            resultado1 = qt.mesolve(H, estado_inicial, [0, t/2], c_ops, [])
            estado_medio = resultado1.states[-1]
            
            # Aplicar pulso π
            estado_pulsado = qt.sigmax() * estado_medio
            
            # Evolución con pulso (refocalización)
            resultado2 = qt.mesolve(H, estado_pulsado, [t/2, t], c_ops, [])
            estado_final = resultado2.states[-1]
            
            # Calcular coherencia como <σx>
            coherencia[i] = abs(qt.expect(qt.sigmax(), estado_final))
        
        return tiempos, coherencia
    
    def tasa_relajacion_t1(self) -> float:
        """Obtener tasa de relajación T1."""
        return 1.0 / self.T1 if self.T1 > 0 else 0.0
    
    def tasa_dephasing_t2(self) -> float:
        """Obtener tasa de dephasing T2."""
        return 1.0 / self.T2 if self.T2 > 0 else 0.0
