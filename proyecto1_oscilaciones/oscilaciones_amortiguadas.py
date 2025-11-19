"""
Módulo de Oscilaciones Amortiguadas
=====================================

Este módulo implementa una red neuronal para predecir el comportamiento de sistemas
de osciladores amortiguados. Proporciona herramientas para:

1. Generar datos de oscilaciones amortiguadas con parámetros variables
2. Resolver la ecuación diferencial analíticamente
3. Entrenar redes neuronales para aproximar el comportamiento
4. Evaluar y validar el rendimiento del modelo
5. Exportar reportes y visualizaciones

Autor: Sistema de Educación TensorFlow
Licencia: MIT
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


class OscilacionesAmortiguadas:
    """
    Clase para modelar y predecir oscilaciones amortiguadas usando redes neuronales.
    
    Resuelve la ecuación diferencial: m*d²x/dt² + c*dx/dt + k*x = 0
    
    Donde:
    - m: masa del sistema
    - c: coeficiente de amortiguamiento
    - k: constante de rigidez
    - x: posición
    - t: tiempo
    
    Atributos:
        model: Modelo Keras compilado
        history: Historial de entrenamiento
        config: Configuración del modelo y datos
        scaler: Escalador para normalización de datos
        X_train, X_test: Datos de entrada
        y_train, y_test: Datos de salida
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa la clase de oscilaciones amortiguadas.
        
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
    
    @staticmethod
    def solucion_analitica(
        t: np.ndarray,
        m: float = 1.0,
        c: float = 0.5,
        k: float = 1.0,
        x0: float = 1.0,
        v0: float = 0.0
    ) -> np.ndarray:
        """
        Calcula la solución analítica de la ecuación de oscilador amortiguado.
        
        Resuelve: m*d²x/dt² + c*dx/dt + k*x = 0
        
        Casos:
        1. Subamortiguado (ζ < 1): Oscilación con decaimiento exponencial
        2. Críticamente amortiguado (ζ = 1): Sin oscilación, decaimiento más rápido
        3. Sobreamortiguado (ζ > 1): Sin oscilación, decaimiento lento
        
        Args:
            t: Array de tiempos
            m: Masa
            c: Coeficiente de amortiguamiento
            k: Constante de rigidez
            x0: Posición inicial
            v0: Velocidad inicial
        
        Returns:
            Array con posiciones calculadas analíticamente
        """
        # Frecuencia natural y ratio de amortiguamiento
        w0 = np.sqrt(k / m)  # Frecuencia natural
        zeta = c / (2 * np.sqrt(k * m))  # Ratio de amortiguamiento
        
        if zeta < 1:  # Subamortiguado
            w_d = w0 * np.sqrt(1 - zeta**2)  # Frecuencia amortiguada
            A = x0
            B = (v0 + zeta * w0 * x0) / w_d
            x = np.exp(-zeta * w0 * t) * (A * np.cos(w_d * t) + B * np.sin(w_d * t))
        
        elif np.isclose(zeta, 1):  # Críticamente amortiguado
            x = (x0 + (v0 + w0 * x0) * t) * np.exp(-w0 * t)
        
        else:  # Sobreamortiguado
            r1 = -zeta * w0 - w0 * np.sqrt(zeta**2 - 1)
            r2 = -zeta * w0 + w0 * np.sqrt(zeta**2 - 1)
            C1 = (v0 - r2 * x0) / (r1 - r2)
            C2 = x0 - C1
            x = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
        
        return x.astype(np.float32)
    
    def generar_datos(
        self,
        num_muestras: int = 1000,
        tiempo_max: float = 10.0,
        puntos_tiempo: int = 100,
        params_sistema: Optional[Dict[str, float]] = None,
        ruido: float = 0.01,
        test_size: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera datos sintéticos de oscilaciones amortiguadas.
        
        Args:
            num_muestras: Número de conjuntos de parámetros a generar
            tiempo_max: Tiempo máximo de simulación
            puntos_tiempo: Número de puntos de tiempo por muestra
            params_sistema: Dict con rangos de parámetros (m, c, k, x0, v0)
            ruido: Nivel de ruido gaussiano a agregar
            test_size: Fracción para conjunto de prueba
            seed: Semilla para reproducibilidad
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Parámetros por defecto
        if params_sistema is None:
            params_sistema = {
                'm': (0.5, 2.0),      # masa
                'c': (0.1, 2.0),      # amortiguamiento
                'k': (0.5, 5.0),      # rigidez
                'x0': (-2.0, 2.0),    # posición inicial
                'v0': (-1.0, 1.0)     # velocidad inicial
            }
        
        X_data = []
        y_data = []
        
        tiempo = np.linspace(0, tiempo_max, puntos_tiempo)
        
        for _ in range(num_muestras):
            # Generar parámetros aleatorios
            m = np.random.uniform(*params_sistema['m'])
            c = np.random.uniform(*params_sistema['c'])
            k = np.random.uniform(*params_sistema['k'])
            x0 = np.random.uniform(*params_sistema['x0'])
            v0 = np.random.uniform(*params_sistema['v0'])
            
            # Calcular solución analítica
            x_solucion = self.solucion_analitica(tiempo, m, c, k, x0, v0)
            
            # Agregar ruido
            x_ruidosa = x_solucion + np.random.normal(0, ruido, len(x_solucion)).astype(np.float32)
            
            # Calcular ratio de amortiguamiento (zeta)
            zeta = c / (2 * np.sqrt(k * m))
            
            # Para cada punto de tiempo, crear entrada
            for i, t in enumerate(tiempo):
                X_data.append([t, m, c, k, x0, v0, zeta])
                y_data.append([x_ruidosa[i]])
        
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.float32)
        
        # Dividir en entrenamiento y prueba
        n_split = int(len(X) * (1 - test_size))
        indices = np.random.permutation(len(X))
        
        X_train = X[indices[:n_split]]
        X_test = X[indices[n_split:]]
        y_train = y[indices[:n_split]]
        y_test = y[indices[n_split:]]
        
        # Escalar datos
        X_train = self.scaler_X.fit_transform(X_train).astype(np.float32)
        X_test = self.scaler_X.transform(X_test).astype(np.float32)
        y_train = self.scaler_y.fit_transform(y_train).astype(np.float32)
        y_test = self.scaler_y.transform(y_test).astype(np.float32)
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Guardar configuración
        self.config.update({
            'num_muestras': num_muestras,
            'tiempo_max': tiempo_max,
            'puntos_tiempo': puntos_tiempo,
            'ruido': ruido,
            'test_size': test_size,
            'tamaño_entrada': X_train.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
        
        return X_train, X_test, y_train, y_test
    
    def construir_modelo(
        self,
        input_shape: int = 7,
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
            self.construir_modelo()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Entrenar
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
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
        if self.model is None or self.X_test is None:
            raise ValueError("Modelo no entrenado o datos no disponibles")
        
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
        
        # Análisis de residuos
        residuos = y_test_original - y_pred_original
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'residuos_media': float(np.mean(residuos)),
            'residuos_std': float(np.std(residuos)),
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
            raise ValueError("Modelo no construido. Llama primero a construir_modelo()")
        
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
        
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Crear y entrenar modelo
            modelo_fold = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(X_train_fold.shape[1],)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            
            modelo_fold.compile(
                optimizer=keras.optimizers.Adam(0.001),
                loss='mse',
                metrics=['mae']
            )
            
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
            mae = np.mean(np.abs(y_val_fold - y_pred))
            r2 = 1 - (np.sum((y_val_fold - y_pred) ** 2) / 
                     np.sum((y_val_fold - np.mean(y_val_fold)) ** 2))
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        return {
            'mse_mean': float(np.mean(mse_scores)),
            'mse_std': float(np.std(mse_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'mae_std': float(np.std(mae_scores)),
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'scores_por_fold': {
                'mse': [float(m) for m in mse_scores],
                'mae': [float(m) for m in mae_scores],
                'r2': [float(r) for r in r2_scores]
            }
        }
    
    def visualizar_predicciones(
        self,
        X_visual: Optional[np.ndarray] = None,
        y_visual: Optional[np.ndarray] = None,
        salida: str = 'predicciones_oscilaciones.png'
    ) -> None:
        """
        Crea visualizaciones del rendimiento del modelo.
        
        Args:
            X_visual: Datos para visualizar (None = usa X_test)
            y_visual: Valores reales (None = usa y_test)
            salida: Ruta del archivo de salida
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        X_vis = X_visual if X_visual is not None else self.X_test
        y_vis = y_visual if y_visual is not None else self.y_test
        
        y_pred = self.model.predict(X_vis, verbose=0)
        
        # Desescalar
        y_vis_original = self.scaler_y.inverse_transform(y_vis)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predicciones vs Reales
        ax = axes[0, 0]
        ax.scatter(y_vis_original, y_pred_original, alpha=0.5, s=10)
        ax.plot([y_vis_original.min(), y_vis_original.max()],
                [y_vis_original.min(), y_vis_original.max()], 'r--', lw=2)
        ax.set_xlabel('Valores Reales', fontsize=11)
        ax.set_ylabel('Predicciones', fontsize=11)
        ax.set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuos
        ax = axes[0, 1]
        residuos = y_vis_original - y_pred_original
        ax.scatter(y_pred_original, residuos, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicciones', fontsize=11)
        ax.set_ylabel('Residuos', fontsize=11)
        ax.set_title('Análisis de Residuos', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribución de residuos
        ax = axes[1, 0]
        ax.hist(residuos, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residuos', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Curva de aprendizaje
        ax = axes[1, 1]
        if self.history is not None:
            ax.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
            ax.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax.set_xlabel('Época', fontsize=11)
            ax.set_ylabel('Loss (MSE)', fontsize=11)
            ax.set_title('Curva de Aprendizaje', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
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
        
        # Guardar modelo Keras
        self.model.save(f"{ruta}.keras")
        
        # Guardar configuración
        config_path = f"{ruta}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Guardar escaladores
        scaler_path = f"{ruta}_scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, f)
        
        print(f"✅ Modelo guardado: {ruta}.keras")
        print(f"✅ Configuración guardada: {config_path}")
        print(f"✅ Escaladores guardados: {scaler_path}")
    
    def cargar_modelo(self, ruta: str) -> None:
        """
        Carga un modelo guardado.
        
        Args:
            ruta: Ruta del archivo (sin extensión)
        """
        # Cargar modelo
        self.model = keras.models.load_model(f"{ruta}.keras")
        
        # Cargar configuración
        config_path = f"{ruta}_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Cargar escaladores
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
            'tipo_modelo': 'OscilacionesAmortiguadas',
            'estado': 'Entrenado' if self.model is not None else 'No entrenado',
            'capas': self.config.get('capas_ocultas', 'N/A'),
            'parametros_totales': self.config.get('parametros_totales', 0),
            'tasa_aprendizaje': self.config.get('tasa_aprendizaje', 'N/A'),
            'tamaño_entrada': self.config.get('tamaño_entrada', 'N/A'),
            'configuración': self.config
        }


def demo():
    """Demostración completa del módulo."""
    print("=" * 70)
    print("🌊 DEMO: OSCILACIONES AMORTIGUADAS CON REDES NEURONALES")
    print("=" * 70)
    
    # Crear instancia
    modelo = OscilacionesAmortiguadas()
    print("\n✅ Instancia creada\n")
    
    # Generar datos
    print("📊 Generando datos...")
    X_train, X_test, y_train, y_test = modelo.generar_datos(
        num_muestras=500,
        tiempo_max=10.0,
        puntos_tiempo=50,
        ruido=0.02
    )
    print(f"✅ Datos generados: {X_train.shape[0]} muestras de entrenamiento")
    print(f"✅ {X_test.shape[0]} muestras de prueba\n")
    
    # Construir modelo
    print("🏗️  Construyendo modelo...")
    modelo.construir_modelo(
        input_shape=7,
        capas_ocultas=[256, 128, 64, 32]
    )
    print(f"✅ Modelo creado con {modelo.config['parametros_totales']} parámetros\n")
    
    # Entrenar
    print("🎯 Entrenando modelo...")
    info = modelo.entrenar(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    print(f"✅ Entrenamiento completado en {info['epochs_entrenadas']} épocas")
    print(f"✅ Loss final: {info['loss_final']:.6f}\n")
    
    # Evaluar
    print("📈 Evaluando modelo...")
    metricas = modelo.evaluar()
    print(f"✅ MSE:  {metricas['mse']:.6f}")
    print(f"✅ RMSE: {metricas['rmse']:.6f}")
    print(f"✅ MAE:  {metricas['mae']:.6f}")
    print(f"✅ R²:   {metricas['r2']:.4f}\n")
    
    # Validación cruzada
    print("🔄 Realizando validación cruzada (5-fold)...")
    cv_results = modelo.validacion_cruzada(
        X_train, y_train,
        k_folds=5,
        epochs=30
    )
    print(f"✅ MSE promedio: {cv_results['mse_mean']:.6f} ± {cv_results['mse_std']:.6f}")
    print(f"✅ R² promedio:  {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}\n")
    
    # Visualizar
    print("🎨 Creando visualizaciones...")
    modelo.visualizar_predicciones(salida='oscilaciones_predicciones.png')
    
    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN DEL MODELO")
    print("=" * 70)
    resumen = modelo.resumen_modelo()
    for key, value in resumen.items():
        if key != 'configuración':
            print(f"{key}: {value}")
    
    print("\n✅ Demo completada exitosamente!\n")


if __name__ == '__main__':
    demo()
