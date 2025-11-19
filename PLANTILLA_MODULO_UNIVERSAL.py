"""
PLANTILLA UNIVERSAL PARA MÓDULOS TENSORFLOW

Esta plantilla proporciona la estructura estándar que deben seguir
todos los módulos de los 12 proyectos para garantizar consistencia,
calidad y humanización.

Usar como referencia para crear nuevos módulos.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime


class PlantillaModelo:
    """
    [TÍTULO DEL MÓDULO]
    
    [Descripción detallada del problema que resuelve]
    [Ecuaciones matemáticas si aplica]
    [Explicación de parámetros]
    
    Atributos:
        model: Modelo Keras compilado
        history: Historial de entrenamiento
        config: Configuración del modelo y datos
        scaler_X: Escalador para normalización de entrada
        scaler_y: Escalador para normalización de salida
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa la clase del modelo.
        
        Args:
            seed: Semilla para reproducibilidad
        """
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Atributos de modelo
        self.model = None
        self.history = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Configuración
        self.config = {
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'framework': 'TensorFlow 2.16+'
        }
    
    def generar_datos(
        self,
        num_muestras: int = 1000,
        test_size: float = 0.2,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera datos sintéticos para entrenamiento.
        
        [DOCUMENTO LOS PARÁMETROS Y EL PROCESO ESPECÍFICO]
        
        Args:
            num_muestras: Número de muestras a generar
            test_size: Fracción para conjunto de prueba
            **kwargs: Parámetros adicionales específicos del modelo
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        raise NotImplementedError("Implementar generar_datos en subclase")
    
    def construir_modelo(
        self,
        input_shape: int,
        capas_ocultas: Optional[List[int]] = None,
        tasa_aprendizaje: float = 0.001,
        dropout_rate: float = 0.2
    ) -> keras.Model:
        """
        Construye la arquitectura de la red neuronal.
        
        Args:
            input_shape: Número de características de entrada
            capas_ocultas: Lista con número de neuronas por capa oculta
            tasa_aprendizaje: Tasa de aprendizaje del optimizador
            dropout_rate: Probabilidad de dropout
        
        Returns:
            Modelo Keras compilado
        """
        if capas_ocultas is None:
            capas_ocultas = [256, 128, 64, 32]
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape,)))
        
        # Capas ocultas con normalización y dropout
        for unidades in capas_ocultas:
            model.add(layers.Dense(unidades, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(layers.Dense(1, activation='linear'))
        
        # Compilar
        optimizer = keras.optimizers.Adam(learning_rate=tasa_aprendizaje)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        # Guardar configuración
        self.config.update({
            'capas_ocultas': capas_ocultas,
            'tasa_aprendizaje': tasa_aprendizaje,
            'dropout_rate': dropout_rate,
            'parametros_totales': model.count_params()
        })
        
        return model
    
    def entrenar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Entrena el modelo.
        
        Args:
            X: Datos de entrada
            y: Datos de salida
            epochs: Número de épocas
            batch_size: Tamaño del batch
            validation_split: Fracción para validación
            early_stopping_patience: Paciencia de early stopping
            verbose: Nivel de verbosidad
        
        Returns:
            Dict con información del entrenamiento
        """
        if self.model is None:
            self.construir_modelo(input_shape=X.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        info = {
            'epochs_entrenadas': len(self.history.history['loss']),
            'loss_final': float(self.history.history['loss'][-1]),
            'val_loss_final': float(self.history.history['val_loss'][-1]),
            'mae_final': float(self.history.history['mae'][-1]),
            'timestamp': datetime.now().isoformat()
        }
        
        return info
    
    def evaluar(self) -> Dict[str, float]:
        """
        Evalúa el modelo en los datos de prueba.
        
        Returns:
            Dict con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Predicciones
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_test_original = self.scaler_y.inverse_transform(self.y_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        
        # Métricas
        mse = np.mean((y_test_original - y_pred_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_original - y_pred_original))
        r2 = 1 - (np.sum((y_test_original - y_pred_original) ** 2) / 
                  np.sum((y_test_original - np.mean(y_test_original)) ** 2))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_test': len(self.X_test)
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Args:
            X: Datos de entrada (sin escalar)
        
        Returns:
            Predicciones escaladas originalmente
        """
        if self.model is None:
            raise ValueError("Modelo no construido")
        
        X_scaled = self.scaler_X.transform(X).astype(np.float32)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def validacion_cruzada(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Realiza validación cruzada k-fold.
        
        Args:
            X: Datos de entrada
            y: Datos de salida
            k_folds: Número de folds
            epochs: Épocas por modelo
            batch_size: Tamaño del batch
        
        Returns:
            Dict con resultados de CV
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Crear y entrenar modelo
            modelo_fold = self._crear_modelo(X_train_fold.shape[1])
            
            modelo_fold.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            # Evaluar
            y_pred = modelo_fold.predict(X_val_fold, verbose=0)
            mse = np.mean((y_val_fold - y_pred) ** 2)
            scores.append(mse)
        
        return {
            'mse_mean': float(np.mean(scores)),
            'mse_std': float(np.std(scores)),
            'scores': [float(s) for s in scores]
        }
    
    def _crear_modelo(self, input_shape: int) -> keras.Model:
        """Método auxiliar para crear modelo en CV."""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def visualizar_predicciones(
        self,
        X_visual: Optional[np.ndarray] = None,
        y_visual: Optional[np.ndarray] = None,
        salida: str = 'predicciones.png'
    ) -> None:
        """
        Crea visualizaciones del rendimiento del modelo.
        
        Args:
            X_visual: Datos para visualizar
            y_visual: Valores reales
            salida: Ruta del archivo de salida
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        X_vis = X_visual if X_visual is not None else self.X_test
        y_vis = y_visual if y_visual is not None else self.y_test
        
        y_pred = self.model.predict(X_vis, verbose=0)
        
        y_vis_original = self.scaler_y.inverse_transform(y_vis)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predicciones vs Reales
        axes[0, 0].scatter(y_vis_original, y_pred_original, alpha=0.5)
        axes[0, 0].plot([y_vis_original.min(), y_vis_original.max()],
                       [y_vis_original.min(), y_vis_original.max()], 'r--')
        axes[0, 0].set_title('Predicciones vs Valores Reales')
        axes[0, 0].set_xlabel('Reales')
        axes[0, 0].set_ylabel('Predicciones')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuos
        residuos = y_vis_original - y_pred_original
        axes[0, 1].scatter(y_pred_original, residuos, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Análisis de Residuos')
        axes[0, 1].set_xlabel('Predicciones')
        axes[0, 1].set_ylabel('Residuos')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribución residuos
        axes[1, 0].hist(residuos, bins=50, edgecolor='black')
        axes[1, 0].set_title('Distribución de Residuos')
        axes[1, 0].set_xlabel('Residuos')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Curva de aprendizaje
        if self.history:
            axes[1, 1].plot(self.history.history['loss'], label='Training')
            axes[1, 1].plot(self.history.history['val_loss'], label='Validation')
            axes[1, 1].set_title('Curva de Aprendizaje')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(salida, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada: {salida}")
        plt.close()
    
    def guardar_modelo(self, ruta: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            ruta: Ruta del archivo (sin extensión)
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        self.model.save(f"{ruta}.keras")
        
        config_path = f"{ruta}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        scaler_path = f"{ruta}_scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, f)
        
        print(f"✅ Modelo guardado: {ruta}.keras")
    
    def cargar_modelo(self, ruta: str) -> None:
        """
        Carga un modelo guardado.
        
        Args:
            ruta: Ruta del archivo (sin extensión)
        """
        self.model = keras.models.load_model(f"{ruta}.keras")
        
        config_path = f"{ruta}_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        scaler_path = f"{ruta}_scalers.pkl"
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
        
        print(f"✅ Modelo cargado: {ruta}.keras")
    
    def resumen_modelo(self) -> Dict[str, Any]:
        """
        Retorna un resumen del modelo y su configuración.
        
        Returns:
            Dict con información del modelo
        """
        return {
            'estado': 'Entrenado' if self.model is not None else 'No entrenado',
            'configuración': self.config
        }


if __name__ == '__main__':
    print("Esta es una plantilla universal. Implementar generar_datos() en subclase.")
