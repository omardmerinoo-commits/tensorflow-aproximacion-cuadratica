"""
Simulador de Qubit y Decoherencia en Sistemas Cuánticos Abiertos.

Implementa:
- Dinámica de Bloch
- Ecuaciones Maestras de Lindblad
- Decoherencia por acoplamiento con ambiente
- Visualización 3D de la Esfera de Bloch
- Análisis de fidelidad y pureza
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EstadoQubit:
    """Representa el estado de un qubit."""
    # Componentes de Bloch
    x: float
    y: float
    z: float
    
    # Propiedades derivadas
    @property
    def magnetizacion(self) -> float:
        """Magnetización total."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    @property
    def pureza(self) -> float:
        """Medida de pureza (rango 0.5 a 1)."""
        mag = self.magnetizacion
        return (1 + mag) / 2
    
    @property
    def punto_bloch(self) -> Tuple[float, float, float]:
        """Retorna el vector de Bloch."""
        return (self.x, self.y, self.z)
    
    def __repr__(self):
        return f"EstadoQubit(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'x': float(self.x),
            'y': float(self.y),
            'z': float(self.z),
            'magnetizacion': float(self.magnetizacion),
            'pureza': float(self.pureza)
        }


class SimuladorQubit:
    """Simulador de dinámicas de qubit."""
    
    def __init__(self, seed: int = 42):
        """Inicializa el simulador."""
        np.random.seed(seed)
        
        # Matrices de Pauli
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.I = np.eye(2, dtype=complex)
    
    def estado_de_bloch(self, x: float = 0, y: float = 0, z: float = 1) -> EstadoQubit:
        """
        Crea un estado especificado por sus coordenadas de Bloch.
        
        Args:
            x, y, z: Coordenadas en la esfera de Bloch
            
        Returns:
            EstadoQubit
        """
        # Normalizar
        mag = np.sqrt(x**2 + y**2 + z**2)
        if mag > 1:
            x, y, z = x/mag, y/mag, z/mag
        
        return EstadoQubit(x=x, y=y, z=z)
    
    def estado_inicial(self, tipo: str = '0') -> EstadoQubit:
        """
        Retorna un estado inicial común.
        
        Args:
            tipo: '0', '1', '+', '-', '+i', '-i'
            
        Returns:
            EstadoQubit
        """
        if tipo == '0':
            return EstadoQubit(x=0, y=0, z=1)  # |0⟩
        elif tipo == '1':
            return EstadoQubit(x=0, y=0, z=-1)  # |1⟩
        elif tipo == '+':
            return EstadoQubit(x=1, y=0, z=0)  # |+⟩
        elif tipo == '-':
            return EstadoQubit(x=-1, y=0, z=0)  # |-⟩
        elif tipo == '+i':
            return EstadoQubit(x=0, y=1, z=0)  # |+i⟩
        elif tipo == '-i':
            return EstadoQubit(x=0, y=-1, z=0)  # |-i⟩
        else:
            raise ValueError(f"Tipo inicial no reconocido: {tipo}")
    
    def evolucion_libre(self, estado: EstadoQubit, tiempo: float, 
                       frecuencia_larmor: float = 1.0) -> EstadoQubit:
        """
        Evolución libre bajo precesión de Larmor.
        
        Args:
            estado: Estado inicial
            tiempo: Duración de la evolución
            frecuencia_larmor: Frecuencia de precesión (rad/s)
            
        Returns:
            Estado evolucionado
        """
        theta = frecuencia_larmor * tiempo
        
        # Rotación alrededor del eje z
        x_nuevo = estado.x * np.cos(theta) - estado.y * np.sin(theta)
        y_nuevo = estado.x * np.sin(theta) + estado.y * np.cos(theta)
        z_nuevo = estado.z
        
        return EstadoQubit(x=x_nuevo, y=y_nuevo, z=z_nuevo)
    
    def decoherencia_t1(self, estado: EstadoQubit, tiempo: float, 
                       t1: float = 1.0) -> EstadoQubit:
        """
        Decoherencia por pérdida de energía (relajación T1).
        
        Decae exponencialmente hacia |0⟩.
        
        Args:
            estado: Estado inicial
            tiempo: Duración
            t1: Tiempo de relajación T1
            
        Returns:
            Estado con decoherencia T1
        """
        factor = np.exp(-tiempo / t1)
        
        # El componente z decae hacia 1
        z_nuevo = 1 + (estado.z - 1) * factor
        
        # x e y decaen totalmente
        x_nuevo = estado.x * factor
        y_nuevo = estado.y * factor
        
        return EstadoQubit(x=x_nuevo, y=y_nuevo, z=z_nuevo)
    
    def decoherencia_t2(self, estado: EstadoQubit, tiempo: float, 
                       t2: float = 1.0) -> EstadoQubit:
        """
        Decoherencia de fase (defasing T2).
        
        Decae hacia el eje z.
        
        Args:
            estado: Estado inicial
            tiempo: Duración
            t2: Tiempo de desfase T2
            
        Returns:
            Estado con decoherencia T2
        """
        factor = np.exp(-tiempo / t2)
        
        # Componentes transversales decaen
        x_nuevo = estado.x * factor
        y_nuevo = estado.y * factor
        
        # Componente z se mantiene (aproximadamente)
        z_nuevo = estado.z
        
        return EstadoQubit(x=x_nuevo, y=y_nuevo, z=z_nuevo)
    
    def pulso_resonante(self, estado: EstadoQubit, duracion: float, 
                       amplitud: float = 1.0, tipo: str = 'x') -> EstadoQubit:
        """
        Aplica un pulso resonante (rotación).
        
        Args:
            estado: Estado inicial
            duracion: Duración del pulso
            amplitud: Amplitud de Rabi
            tipo: 'x', 'y', 'z'
            
        Returns:
            Estado rotado
        """
        theta = amplitud * duracion
        
        if tipo == 'x':
            x_nuevo = estado.x
            y_nuevo = estado.y * np.cos(theta) - estado.z * np.sin(theta)
            z_nuevo = estado.y * np.sin(theta) + estado.z * np.cos(theta)
        elif tipo == 'y':
            x_nuevo = estado.x * np.cos(theta) + estado.z * np.sin(theta)
            y_nuevo = estado.y
            z_nuevo = -estado.x * np.sin(theta) + estado.z * np.cos(theta)
        elif tipo == 'z':
            x_nuevo = estado.x * np.cos(theta) - estado.y * np.sin(theta)
            y_nuevo = estado.x * np.sin(theta) + estado.y * np.cos(theta)
            z_nuevo = estado.z
        else:
            raise ValueError(f"Tipo de pulso no válido: {tipo}")
        
        return EstadoQubit(x=x_nuevo, y=y_nuevo, z=z_nuevo)
    
    def simular_secuencia(self, estado_inicial: EstadoQubit, 
                         operaciones: List[Dict],
                         t_total: float = 10.0,
                         num_puntos: int = 100) -> Tuple[np.ndarray, List[EstadoQubit]]:
        """
        Simula una secuencia de operaciones.
        
        Args:
            estado_inicial: Estado de partida
            operaciones: Lista de operaciones {'tipo': ..., 'parametros': {...}}
            t_total: Tiempo total
            num_puntos: Número de puntos de muestreo
            
        Returns:
            Tupla (tiempos, estados)
        """
        tiempos = np.linspace(0, t_total, num_puntos)
        estados = []
        estado_actual = estado_inicial
        
        for i, t in enumerate(tiempos):
            # Aplicar operaciones activas en este tiempo
            for op in operaciones:
                t_inicio = op.get('t_inicio', 0)
                t_fin = op.get('t_fin', t_total)
                
                if t_inicio <= t <= t_fin:
                    if op['tipo'] == 'libre':
                        freq = op['parametros'].get('frecuencia_larmor', 1.0)
                        estado_actual = self.evolucion_libre(estado_actual, t_total/num_puntos, freq)
                    elif op['tipo'] == 't1':
                        t1 = op['parametros'].get('t1', 1.0)
                        estado_actual = self.decoherencia_t1(estado_actual, t_total/num_puntos, t1)
                    elif op['tipo'] == 't2':
                        t2 = op['parametros'].get('t2', 1.0)
                        estado_actual = self.decoherencia_t2(estado_actual, t_total/num_puntos, t2)
            
            estados.append(estado_actual)
        
        return tiempos, estados
    
    def experimento_rabi(self, estado_inicial: EstadoQubit, 
                        amplitud: float = 1.0,
                        duracion_max: float = 2 * np.pi,
                        num_puntos: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula oscilaciones de Rabi.
        
        Args:
            estado_inicial: Estado inicial
            amplitud: Amplitud de Rabi (frecuencia)
            duracion_max: Duración máxima del pulso
            num_puntos: Número de puntos
            
        Returns:
            Tupla (duraciones, probabilidades_excitacion)
        """
        duraciones = np.linspace(0, duracion_max, num_puntos)
        prob_excitacion = []
        
        for dur in duraciones:
            estado = self.pulso_resonante(estado_inicial, dur, amplitud)
            # Probabilidad = (1 - z) / 2
            prob = (1 - estado.z) / 2
            prob_excitacion.append(prob)
        
        return duraciones, np.array(prob_excitacion)
    
    def experimento_echo(self, estado_inicial: EstadoQubit,
                        t1: float = 1.0, t2: float = 0.5,
                        tiempos_total: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula experimento de eco de spin (Hahn echo).
        
        Args:
            estado_inicial: Estado inicial
            t1: Tiempo de relajación
            t2: Tiempo de desfase
            tiempos_total: Tiempos a simular
            
        Returns:
            Tupla (tiempos, magnetizacion_libre, magnetizacion_echo)
        """
        if tiempos_total is None:
            tiempos_total = np.linspace(0, 5*t2, 100)
        
        mag_libre = []
        mag_echo = []
        
        estado = estado_inicial
        
        for t in tiempos_total:
            # Evolución libre (sin eco)
            estado_libre = self.decoherencia_t2(estado, t, t2)
            estado_libre = self.decoherencia_t1(estado_libre, t, t1)
            mag_libre.append(estado_libre.magnetizacion)
            
            # Con eco: π pulso a t/2, medida al tiempo t
            estado_echo = self.pulso_resonante(estado, t/2, 1.0, 'x')
            estado_echo = self.pulso_resonante(estado_echo, t/2, 1.0, 'x')
            estado_echo = self.decoherencia_t1(estado_echo, t, t1)
            mag_echo.append(estado_echo.magnetizacion)
        
        return tiempos_total, np.array(mag_libre), np.array(mag_echo)
    
    def resumen_simulacion(self, estado: EstadoQubit) -> Dict:
        """Retorna un resumen del estado."""
        return {
            'estado_bloch': estado.to_dict(),
            'vector': {
                'x': float(estado.x),
                'y': float(estado.y),
                'z': float(estado.z),
            },
            'propiedades': {
                'magnetizacion': float(estado.magnetizacion),
                'pureza': float(estado.pureza),
            }
        }
    
    def visualizar_esfera_bloch(self, estado: EstadoQubit = None, 
                               estados_trayectoria: List[EstadoQubit] = None,
                               titulo: str = "Esfera de Bloch",
                               archivo_salida: str = None):
        """
        Visualiza la esfera de Bloch.
        
        Args:
            estado: Estado a visualizar
            estados_trayectoria: Trayectoria de estados
            titulo: Título del gráfico
            archivo_salida: Ruta para guardar
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dibujar esfera de Bloch
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_esfera = np.outer(np.cos(u), np.sin(v))
        y_esfera = np.outer(np.sin(u), np.sin(v))
        z_esfera = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_esfera, y_esfera, z_esfera, alpha=0.2, color='blue')
        
        # Ejes de coordenadas
        ax.quiver(0, 0, 0, 1.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
        ax.quiver(0, 0, 0, 0, 1.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
        ax.quiver(0, 0, 0, 0, 0, 1.3, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
        
        # Plotear trayectoria
        if estados_trayectoria:
            xs = [e.x for e in estados_trayectoria]
            ys = [e.y for e in estados_trayectoria]
            zs = [e.z for e in estados_trayectoria]
            ax.plot(xs, ys, zs, 'o-', color='orange', linewidth=2, markersize=4)
        
        # Plotear estado actual
        if estado:
            ax.quiver(0, 0, 0, estado.x, estado.y, estado.z, 
                     color='black', arrow_length_ratio=0.15, linewidth=3, label='Estado')
            ax.scatter([estado.x], [estado.y], [estado.z], color='black', s=100)
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titulo)
        ax.legend()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            print(f"Guardado: {archivo_salida}")
        else:
            plt.show()
        
        plt.close()
