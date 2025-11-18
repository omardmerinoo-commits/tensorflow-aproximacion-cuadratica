"""
Generador de datos sintéticos para clasificación de fases.

Este módulo genera datos simulados que representan diferentes fases
de un material (sólido, líquido, gas) basado en características
como temperatura, presión y densidad.
"""

import numpy as np
from typing import Tuple, List


class GeneradorDatosFases:
    """
    Generador de datos sintéticos para clasificación de fases.
    
    Genera características físicas que simulan diferentes estados
    de la materia y sus transiciones de fase.
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializar el generador.
        
        Parameters
        ----------
        seed : int
            Semilla para reproducibilidad
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generar_fase_solida(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar características para fase sólida.
        
        Parameters
        ----------
        n_samples : int
            Número de muestras a generar
            
        Returns
        -------
        X : np.ndarray
            Características [temperatura, presión, densidad, dureza, modulo_elastico]
        y : np.ndarray
            Etiquetas (0 para sólido)
        """
        # Fase sólida: baja temperatura, densidad alta, rigidez alta
        temp = np.random.normal(300, 100, n_samples)  # 300K promedio
        pres = np.random.normal(1.5, 0.5, n_samples)  # 1.5 atm promedio
        dens = np.random.normal(8.0, 1.5, n_samples)  # 8 g/cm³ promedio
        dureza = np.random.normal(7.0, 1.0, n_samples)  # 7 dureza Mohs
        modulo = np.random.normal(200, 30, n_samples)  # 200 GPa promedio
        
        X = np.column_stack([temp, pres, dens, dureza, modulo])
        y = np.zeros(n_samples, dtype=np.int32)
        
        return X, y
    
    def generar_fase_liquida(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar características para fase líquida.
        
        Parameters
        ----------
        n_samples : int
            Número de muestras a generar
            
        Returns
        -------
        X : np.ndarray
            Características [temperatura, presión, densidad, viscosidad, tension_superficial]
        y : np.ndarray
            Etiquetas (1 para líquido)
        """
        # Fase líquida: temperatura intermedia, densidad intermedia
        temp = np.random.normal(500, 150, n_samples)  # 500K promedio
        pres = np.random.normal(5.0, 2.0, n_samples)  # 5 atm promedio
        dens = np.random.normal(4.5, 1.2, n_samples)  # 4.5 g/cm³ promedio
        viscosidad = np.random.normal(0.8, 0.3, n_samples)  # 0.8 mPa·s
        tension = np.random.normal(0.07, 0.02, n_samples)  # 0.07 N/m
        
        X = np.column_stack([temp, pres, dens, viscosidad, tension])
        y = np.ones(n_samples, dtype=np.int32)
        
        return X, y
    
    def generar_fase_gaseosa(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar características para fase gaseosa.
        
        Parameters
        ----------
        n_samples : int
            Número de muestras a generar
            
        Returns
        -------
        X : np.ndarray
            Características [temperatura, presión, densidad, compresibilidad, velocidad_sonido]
        y : np.ndarray
            Etiquetas (2 para gas)
        """
        # Fase gaseosa: alta temperatura, baja densidad, muy compresible
        temp = np.random.normal(1000, 300, n_samples)  # 1000K promedio
        pres = np.random.normal(20.0, 5.0, n_samples)  # 20 atm promedio
        dens = np.random.normal(0.5, 0.2, n_samples)  # 0.5 g/cm³ promedio
        compresibilidad = np.random.normal(0.95, 0.05, n_samples)  # ~1 (muy compresible)
        velocidad_sonido = np.random.normal(500, 100, n_samples)  # 500 m/s
        
        X = np.column_stack([temp, pres, dens, compresibilidad, velocidad_sonido])
        y = 2 * np.ones(n_samples, dtype=np.int32)
        
        return X, y
    
    def generar_datos_completos(self, n_samples_por_clase: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar dataset completo balanceado.
        
        Parameters
        ----------
        n_samples_por_clase : int
            Número de muestras por cada clase
            
        Returns
        -------
        X : np.ndarray
            Todas las características
        y : np.ndarray
            Todas las etiquetas
        """
        X_solida, y_solida = self.generar_fase_solida(n_samples_por_clase)
        X_liquida, y_liquida = self.generar_fase_liquida(n_samples_por_clase)
        X_gaseosa, y_gaseosa = self.generar_fase_gaseosa(n_samples_por_clase)
        
        X = np.vstack([X_solida, X_liquida, X_gaseosa])
        y = np.concatenate([y_solida, y_liquida, y_gaseosa])
        
        # Mezclar datos
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def normalizar_datos(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalizar características a rango [0, 1].
        
        Parameters
        ----------
        X : np.ndarray
            Datos a normalizar
            
        Returns
        -------
        X_norm : np.ndarray
            Datos normalizados
        min_vals : np.ndarray
            Valores mínimos por característica
        max_vals : np.ndarray
            Valores máximos por característica
        """
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
        
        return X_norm, min_vals, max_vals
