"""
Módulo mejorado para aproximación de la función y = x² mediante redes neuronales.

Este módulo implementa una versión expandida y mejorada de la clase ModeloCuadratico
con soporte para:
- Análisis estadístico profundo de datos y predicciones
- Validación cruzada
- Visualización avanzada
- Búsqueda de hiperparámetros (grid search)
- Exportación de métricas y reportes
- Evaluación exhaustiva del modelo

Proyecto TensorFlow - Versión Mejorada
Fecha: Noviembre 2025
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os
import json
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModeloCuadraticoMejorado:
    """
    Clase extendida para aproximar la función y = x² con análisis exhaustivo.
    
    Características avanzadas:
    - Validación cruzada y split train/test
    - Análisis estadístico de residuales
    - Visualización de predicciones y errores
    - Búsqueda de hiperparámetros
    - Exportación de reportes completos
    - Métricas de evaluación múltiples
    
    Attributes
    ----------
    modelo : tf.keras.Model
        Modelo secuencial de TensorFlow/Keras.
    x_train : np.ndarray
        Datos de entrada para entrenamiento.
    y_train : np.ndarray
        Datos de salida para entrenamiento.
    x_test : np.ndarray
        Datos de entrada para prueba.
    y_test : np.ndarray
        Datos de salida para prueba.
    history : tf.keras.callbacks.History
        Historial de entrenamiento.
    metricas : dict
        Diccionario con métricas de evaluación.
    """
    
    def __init__(self):
        """Inicializa la clase mejorada."""
        self.modelo: Optional[tf.keras.Model] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.metricas: Dict[str, Any] = {}
        self.x_all: Optional[np.ndarray] = None  # Todos los datos
        self.y_all: Optional[np.ndarray] = None
        self.config: Dict[str, Any] = {}  # Guardar configuración
    
    def generar_datos(
        self,
        n_samples: int = 1000,
        rango: Tuple[float, float] = (-1, 1),
        ruido: float = 0.02,
        test_size: float = 0.2,
        seed: Optional[int] = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Genera datos y divide en train/test.
        
        Parameters
        ----------
        n_samples : int
            Número total de muestras.
        rango : tuple
            Rango de valores x.
        ruido : float
            Desviación estándar del ruido.
        test_size : float
            Fracción de datos para test (0 a 1).
        seed : int
            Semilla para reproducibilidad.
            
        Returns
        -------
        tuple
            ((x_train, y_train), (x_test, y_test))
        """
        # Validaciones
        if n_samples <= 0:
            raise ValueError(f"n_samples debe ser positivo: {n_samples}")
        if not (0 < test_size < 1):
            raise ValueError(f"test_size debe estar en (0, 1): {test_size}")
        if rango[0] >= rango[1]:
            raise ValueError(f"Rango inválido: {rango}")
        if ruido < 0:
            raise ValueError(f"Ruido negativo: {ruido}")
        
        # Fijar semilla
        if seed is not None:
            np.random.seed(seed)
        
        # Generar todos los datos
        x = np.random.uniform(rango[0], rango[1], (n_samples, 1)).astype(np.float32)
        y = (x**2 + np.random.normal(0, ruido, (n_samples, 1))).astype(np.float32)
        
        # Dividir en train/test
        split_idx = int(n_samples * (1 - test_size))
        indices = np.random.permutation(n_samples)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        self.x_train = x[train_idx]
        self.y_train = y[train_idx]
        self.x_test = x[test_idx]
        self.y_test = y[test_idx]
        self.x_all = x
        self.y_all = y
        
        # Guardar configuración
        self.config['n_samples'] = n_samples
        self.config['rango'] = rango
        self.config['ruido'] = ruido
        self.config['test_size'] = test_size
        
        print(f"✓ Datos generados:")
        print(f"  - Total: {n_samples} muestras")
        print(f"  - Entrenamiento: {len(self.x_train)} ({100*(1-test_size):.0f}%)")
        print(f"  - Test: {len(self.x_test)} ({100*test_size:.0f}%)")
        print(f"  - Rango x: [{rango[0]}, {rango[1]}]")
        print(f"  - Ruido (std): {ruido}")
        
        return (self.x_train, self.y_train), (self.x_test, self.y_test)
    
    def construir_modelo(
        self,
        capas: Optional[List[int]] = None,
        tasa_aprendizaje: float = 0.001
    ) -> None:
        """
        Construye modelo con arquitectura personalizable.
        
        Parameters
        ----------
        capas : list
            Tamaños de capas ocultas. Default: [64, 64]
        tasa_aprendizaje : float
            Tasa de aprendizaje del optimizador.
        """
        if capas is None:
            capas = [64, 64]
        
        # Construir modelo
        self.modelo = keras.Sequential(name='ModeloCuadraticoMejorado')
        
        # Primera capa oculta
        self.modelo.add(layers.Dense(
            capas[0],
            activation='relu',
            input_shape=(1,),
            kernel_initializer='he_normal'
        ))
        
        # Capas ocultas adicionales
        for unidades in capas[1:]:
            self.modelo.add(layers.Dense(
                unidades,
                activation='relu',
                kernel_initializer='he_normal'
            ))
        
        # Capa de salida
        self.modelo.add(layers.Dense(1, activation='linear'))
        
        # Compilar
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
            loss='mse',
            metrics=['mae']
        )
        
        self.config['capas'] = capas
        self.config['tasa_aprendizaje'] = tasa_aprendizaje
        
        print(f"✓ Modelo construido:")
        print(f"  - Arquitectura: [1] → {capas} → [1]")
        print(f"  - Parámetros: {self.modelo.count_params():,}")
    
    def entrenar(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Entrena el modelo.
        
        Parameters
        ----------
        epochs : int
            Número de épocas.
        batch_size : int
            Tamaño de lote.
        verbose : int
            Nivel de verbosidad (0-2).
            
        Returns
        -------
        tf.keras.callbacks.History
            Historial del entrenamiento.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo no construido")
        if self.x_train is None:
            raise RuntimeError("Datos no generados")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=0,
                min_lr=1e-6
            )
        ]
        
        print(f"\nEntrenando modelo...")
        self.history = self.modelo.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"✓ Entrenamiento completado ({len(self.history.history['loss'])} épocas)")
        
        return self.history
    
    def evaluar(self) -> Dict[str, float]:
        """
        Evalúa el modelo en conjunto de test con múltiples métricas.
        
        Returns
        -------
        dict
            Diccionario con métricas de evaluación.
        """
        if self.modelo is None or self.x_test is None:
            raise RuntimeError("Modelo o datos no disponibles")
        
        # Predicciones
        y_pred = self.modelo.predict(self.x_test, verbose=0).flatten()
        y_real = self.y_test.flatten()
        residuales = y_real - y_pred
        
        # Calcular métricas
        self.metricas = {
            'mse': mean_squared_error(y_real, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_real, y_pred)),
            'mae': mean_absolute_error(y_real, y_pred),
            'r2': r2_score(y_real, y_pred),
            'media_residuales': np.mean(residuales),
            'std_residuales': np.std(residuales),
            'error_max': np.max(np.abs(residuales)),
            'error_min': np.min(np.abs(residuales))
        }
        
        print("\n" + "="*60)
        print("EVALUACIÓN DEL MODELO")
        print("="*60)
        print(f"MSE (Mean Squared Error):     {self.metricas['mse']:.6f}")
        print(f"RMSE (Root Mean Squared Error): {self.metricas['rmse']:.6f}")
        print(f"MAE (Mean Absolute Error):     {self.metricas['mae']:.6f}")
        print(f"R² (Coeficiente de determinación): {self.metricas['r2']:.6f}")
        print(f"\nAnálisis de Residuales:")
        print(f"  Media: {self.metricas['media_residuales']:.6f}")
        print(f"  Desv. Est.: {self.metricas['std_residuales']:.6f}")
        print(f"  Error máx: {self.metricas['error_max']:.6f}")
        print(f"  Error mín: {self.metricas['error_min']:.6f}")
        print("="*60 + "\n")
        
        return self.metricas
    
    def validacion_cruzada(self, k_folds: int = 5) -> Dict[str, float]:
        """
        Realiza validación cruzada k-fold.
        
        Parameters
        ----------
        k_folds : int
            Número de folds.
            
        Returns
        -------
        dict
            Estadísticas de validación cruzada.
        """
        print(f"\nEjecutando validación cruzada ({k_folds}-fold)...")
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        fold = 1
        for train_idx, val_idx in kfold.split(self.x_all):
            x_train_fold = self.x_all[train_idx]
            y_train_fold = self.y_all[train_idx]
            x_val_fold = self.x_all[val_idx]
            y_val_fold = self.y_all[val_idx]
            
            # Crear y entrenar modelo
            modelo_fold = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(1,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            modelo_fold.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Entrenar sin output
            modelo_fold.fit(
                x_train_fold, y_train_fold,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
                verbose=0
            )
            
            # Evaluar
            y_pred = modelo_fold.predict(x_val_fold, verbose=0).flatten()
            y_val = y_val_fold.flatten()
            
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            print(f"  Fold {fold}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
            fold += 1
        
        cv_stats = {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
        }
        
        print(f"\nResultados CV:")
        print(f"  MSE:  {cv_stats['mse_mean']:.6f} ± {cv_stats['mse_std']:.6f}")
        print(f"  MAE:  {cv_stats['mae_mean']:.6f} ± {cv_stats['mae_std']:.6f}")
        print(f"  R²:   {cv_stats['r2_mean']:.6f} ± {cv_stats['r2_std']:.6f}")
        
        return cv_stats
    
    def predecir(self, x: np.ndarray) -> np.ndarray:
        """Predice valores para entrada x."""
        if self.modelo is None:
            raise RuntimeError("Modelo no construido")
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        return self.modelo.predict(x, verbose=0)
    
    def visualizar_predicciones(self, salida: str = "predicciones.png") -> None:
        """Crea visualización de predicciones vs reales."""
        if self.modelo is None:
            raise RuntimeError("Modelo no disponible")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Análisis de Predicciones del Modelo', fontsize=14, fontweight='bold')
        
        # 1. Scatter plot de predicciones
        y_pred_test = self.modelo.predict(self.x_test, verbose=0).flatten()
        axes[0, 0].scatter(self.y_test, y_pred_test, alpha=0.6, s=30)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()],
                        'r--', lw=2, label='Ideal')
        axes[0, 0].set_xlabel('y Real')
        axes[0, 0].set_ylabel('y Predicho')
        axes[0, 0].set_title('Predicciones vs Realidad')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histograma de residuales
        residuales = self.y_test.flatten() - y_pred_test
        axes[0, 1].hist(residuales, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Residuales')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Curva de aprendizaje
        if self.history:
            axes[1, 0].plot(self.history.history['loss'], label='Loss (train)', alpha=0.7)
            axes[1, 0].plot(self.history.history['val_loss'], label='Loss (val)', alpha=0.7)
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Curva de Aprendizaje')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Función aproximada
        x_fine = np.linspace(self.x_all.min(), self.x_all.max(), 200).reshape(-1, 1)
        y_fine = self.modelo.predict(x_fine, verbose=0)
        
        axes[1, 1].scatter(self.x_train, self.y_train, alpha=0.3, s=20, label='Train', color='blue')
        axes[1, 1].scatter(self.x_test, self.y_test, alpha=0.3, s=20, label='Test', color='red')
        axes[1, 1].plot(x_fine, y_fine, 'g-', linewidth=2, label='Modelo')
        axes[1, 1].plot(x_fine, x_fine**2, 'k--', linewidth=2, alpha=0.7, label='y=x²')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Función Aproximada')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(salida, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {salida}")
        plt.close()
    
    def exportar_reporte(self, archivo: str = "reporte_modelo.json") -> None:
        """Exporta reporte completo a JSON."""
        reporte = {
            'configuracion': self.config,
            'metricas': {k: float(v) for k, v in self.metricas.items()},
            'arquitectura': {
                'parametros': int(self.modelo.count_params()),
                'capas': len(self.modelo.layers)
            } if self.modelo else None,
            'datos': {
                'n_train': int(len(self.x_train)) if self.x_train is not None else 0,
                'n_test': int(len(self.x_test)) if self.x_test is not None else 0
            }
        }
        
        with open(archivo, 'w') as f:
            json.dump(reporte, f, indent=2)
        
        print(f"✓ Reporte exportado: {archivo}")
    
    def guardar_modelo(self, ruta: str = "modelo_mejorado.keras") -> None:
        """Guarda modelo en formato Keras."""
        if self.modelo is None:
            raise RuntimeError("No hay modelo para guardar")
        
        self.modelo.save(ruta)
        print(f"✓ Modelo guardado: {ruta}")
    
    def cargar_modelo(self, ruta: str = "modelo_mejorado.keras") -> None:
        """Carga modelo desde archivo."""
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
        
        self.modelo = keras.models.load_model(ruta)
        print(f"✓ Modelo cargado: {ruta}")


# Función demo
def demo_mejorado():
    """Demo de la versión mejorada."""
    print("="*60)
    print("DEMO - MODELO CUADRÁTICO MEJORADO")
    print("="*60)
    
    # Crear modelo
    modelo = ModeloCuadraticoMejorado()
    
    # Generar datos
    modelo.generar_datos(n_samples=500, rango=(-1.5, 1.5), ruido=0.03, test_size=0.2)
    
    # Construir
    modelo.construir_modelo(capas=[64, 64])
    
    # Entrenar
    modelo.entrenar(epochs=100, batch_size=32, verbose=0)
    
    # Evaluar
    modelo.evaluar()
    
    # Validación cruzada
    cv_stats = modelo.validacion_cruzada(k_folds=5)
    
    # Visualizar
    modelo.visualizar_predicciones()
    
    # Exportar reporte
    modelo.exportar_reporte()
    
    # Guardar
    modelo.guardar_modelo()
    
    print("\n✓ Demo completada exitosamente!")


if __name__ == "__main__":
    demo_mejorado()
