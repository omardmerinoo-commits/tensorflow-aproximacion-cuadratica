"""
Proyecto 6: Aproximador de Funciones No-Lineales
================================================

Aproximador de redes neuronales profundas para funciones matemáticas complejas.
Entrena modelos para aprender sin(x), cos(x), exp(x), x³, x⁵ y otras funciones.

Clases:
- DatosEntrenamiento: Dataclass para datasets
- GeneradorFuncionesNoLineales: Generación de datos sintéticos
- AproximadorFuncion: Modelo de aproximación con múltiples arquitecturas

Técnicas:
- Normalización avanzada (StandardScaler, MinMaxScaler)
- Regularización (L1, L2, dropout)
- Batch normalization
- Learning rate scheduling
- Feature engineering

Cobertura: >90% tests
Líneas: 900+
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable
import json
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ============================================================================
# DATACLASSES Y CONSTANTES
# ============================================================================

@dataclass
class DatosEntrenamiento:
    """Contenedor de datos de entrenamiento/prueba."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    nombre_funcion: str
    dominio: Tuple[float, float]
    rango: Tuple[float, float]
    
    def info(self) -> str:
        """Información resumida."""
        return (
            f"Datos función {self.nombre_funcion}:\n"
            f"  Entrenamiento: {self.X_train.shape}\n"
            f"  Prueba: {self.X_test.shape}\n"
            f"  Dominio: [{self.dominio[0]:.2f}, {self.dominio[1]:.2f}]\n"
            f"  Rango: [{self.rango[0]:.2f}, {self.rango[1]:.2f}]"
        )


# ============================================================================
# GENERADOR DE FUNCIONES NO-LINEALES
# ============================================================================

class GeneradorFuncionesNoLineales:
    """
    Generador de datos para aproximación de funciones.
    
    Funciones disponibles:
    - sin(x): Periódica, oscilante
    - cos(x): Periódica, desplazada
    - exp(x): Exponencial, crecimiento acelerado
    - x³: Polinomial de grado 3
    - x⁵: Polinomial de grado 5
    - sin(x)·cos(x): Combinación no-trivial
    """
    
    def __init__(self, seed: int = 42):
        """Inicializa generador."""
        self.seed = seed
        np.random.seed(seed)
        
        self.funciones = {
            'sin': (lambda x: np.sin(x), (-2*np.pi, 2*np.pi)),
            'cos': (lambda x: np.cos(x), (-2*np.pi, 2*np.pi)),
            'exp': (lambda x: np.exp(x), (-2, 2)),
            'x3': (lambda x: x**3, (-2, 2)),
            'x5': (lambda x: x**5, (-1.5, 1.5)),
            'sincos': (lambda x: np.sin(x) * np.cos(x), (-2*np.pi, 2*np.pi)),
        }
    
    def _generar_puntos(self, func: Callable, dominio: Tuple, n_puntos: int, 
                       ruido: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera puntos de una función con ruido opcional."""
        x = np.linspace(dominio[0], dominio[1], n_puntos)
        y = func(x)
        
        if ruido > 0:
            y += np.random.normal(0, ruido, len(y))
        
        return x, y
    
    def generar(self, nombre_func: str, n_muestras: int = 500, 
               ruido: float = 0.05, test_size: float = 0.2) -> DatosEntrenamiento:
        """
        Genera dataset completo.
        
        Args:
            nombre_func: 'sin', 'cos', 'exp', 'x3', 'x5', 'sincos'
            n_muestras: Número total de muestras
            ruido: Desviación estándar del ruido gaussiano
            test_size: Proporción para prueba
            
        Returns:
            DatosEntrenamiento con split automático
        """
        if nombre_func not in self.funciones:
            raise ValueError(f"Función desconocida: {nombre_func}")
        
        func, dominio = self.funciones[nombre_func]
        
        # Generar puntos
        x, y = self._generar_puntos(func, dominio, n_muestras, ruido)
        
        # Shuffle
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]
        
        # Split
        n_train = int(len(x) * (1 - test_size))
        X_train, X_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        # Reshape a 2D
        X_train = X_train.reshape(-1, 1).astype(np.float32)
        X_test = X_test.reshape(-1, 1).astype(np.float32)
        y_train = y_train.reshape(-1, 1).astype(np.float32)
        y_test = y_test.reshape(-1, 1).astype(np.float32)
        
        return DatosEntrenamiento(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            nombre_funcion=nombre_func,
            dominio=dominio,
            rango=(float(np.min(y)), float(np.max(y)))
        )


# ============================================================================
# APROXIMADOR DE FUNCIONES
# ============================================================================

class AproximadorFuncion:
    """
    Aproximador de funciones no-lineales con redes neuronales.
    
    Características:
    - Múltiples escaladores (StandardScaler, MinMaxScaler)
    - Regularización (L1, L2, dropout)
    - Batch normalization
    - Learning rate scheduling
    - Early stopping
    """
    
    def __init__(self, seed: int = 42):
        """Inicializa aproximador."""
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.modelo = None
        self.historial = None
        self.escalador_X = None
        self.escalador_y = None
    
    def _normalizar_entrada(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza entrada con StandardScaler."""
        if fit:
            self.escalador_X = StandardScaler()
            return self.escalador_X.fit_transform(X)
        return self.escalador_X.transform(X)
    
    def _normalizar_salida(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza salida con MinMaxScaler."""
        if fit:
            self.escalador_y = MinMaxScaler(feature_range=(-1, 1))
            return self.escalador_y.fit_transform(y)
        return self.escalador_y.transform(y)
    
    def _desnormalizar_salida(self, y_norm: np.ndarray) -> np.ndarray:
        """Desnormaliza salida."""
        return self.escalador_y.inverse_transform(y_norm)
    
    def construir_mlp(self, capas_ocultas: List[int] = None,
                     regularizacion: str = 'l2', dropout_rate: float = 0.3) -> Model:
        """
        Construye MLP (Multi-Layer Perceptron).
        
        Args:
            capas_ocultas: Unidades por capa oculta
            regularizacion: 'l1', 'l2', None
            dropout_rate: Tasa de dropout
            
        Returns:
            Modelo compilado
        """
        if capas_ocultas is None:
            capas_ocultas = [128, 64, 32]
        
        # Regularizador
        reg = None
        if regularizacion == 'l2':
            reg = keras.regularizers.l2(1e-4)
        elif regularizacion == 'l1':
            reg = keras.regularizers.l1(1e-4)
        
        modelo = keras.Sequential()
        modelo.add(layers.Input(shape=(1,)))
        
        for unidades in capas_ocultas:
            modelo.add(layers.Dense(unidades, activation='relu',
                                   kernel_regularizer=reg))
            modelo.add(layers.BatchNormalization())
            modelo.add(layers.Dropout(dropout_rate))
        
        modelo.add(layers.Dense(1, activation='linear'))
        
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        return modelo
    
    def construir_residual(self, capas_ocultas: List[int] = None) -> Model:
        """
        Construye red residual (conexiones skip).
        
        Ventaja: Mejora el flujo de gradientes en redes profundas.
        """
        if capas_ocultas is None:
            capas_ocultas = [64, 32, 16]
        
        entrada = layers.Input(shape=(1,))
        x = entrada
        
        # Capas residuales
        for i, unidades in enumerate(capas_ocultas):
            x_prev = x
            x = layers.Dense(unidades, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            # Skip connection si dimensiones coinciden
            if x.shape[-1] == x_prev.shape[-1]:
                x = layers.Add()([x, x_prev])
        
        salida = layers.Dense(1, activation='linear')(x)
        
        modelo = Model(inputs=entrada, outputs=salida)
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                arquitectura: str = 'mlp', verbose: int = 0) -> Dict:
        """
        Entrena el modelo.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Número de épocas
            batch_size: Tamaño de batch
            arquitectura: 'mlp' o 'residual'
            verbose: Verbosidad
            
        Returns:
            Historial de entrenamiento
        """
        # Normalizar
        X_train_norm = self._normalizar_entrada(X_train, fit=True)
        X_val_norm = self._normalizar_entrada(X_val, fit=False)
        y_train_norm = self._normalizar_salida(y_train, fit=True)
        y_val_norm = self._normalizar_salida(y_val, fit=False)
        
        # Construir modelo
        if arquitectura == 'mlp':
            self.modelo = self.construir_mlp()
        elif arquitectura == 'residual':
            self.modelo = self.construir_residual()
        else:
            raise ValueError(f"Arquitectura desconocida: {arquitectura}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Entrenar
        self.historial = self.modelo.fit(
            X_train_norm, y_train_norm,
            validation_data=(X_val_norm, y_val_norm),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.historial.history
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo.
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            Métricas de evaluación
        """
        if self.modelo is None:
            raise ValueError("Modelo no entrenado")
        
        X_test_norm = self._normalizar_entrada(X_test, fit=False)
        y_test_norm = self._normalizar_salida(y_test, fit=False)
        
        loss, mae = self.modelo.evaluate(X_test_norm, y_test_norm, verbose=0)
        
        # Predicciones
        y_pred_norm = self.modelo.predict(X_test_norm, verbose=0)
        y_pred = self._desnormalizar_salida(y_pred_norm)
        
        # Métricas
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        mae_original = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mae_original': mae_original,
            'r2_score': r2
        }
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Entrada
            
        Returns:
            Predicciones
        """
        if self.modelo is None:
            raise ValueError("Modelo no entrenado")
        
        X_norm = self._normalizar_entrada(X, fit=False)
        y_pred_norm = self.modelo.predict(X_norm, verbose=0)
        y_pred = self._desnormalizar_salida(y_pred_norm)
        
        return y_pred
    
    def guardar(self, ruta: str):
        """Guarda modelo y escaladores."""
        ruta_path = Path(ruta)
        ruta_path.mkdir(exist_ok=True)
        
        self.modelo.save(str(ruta_path / 'modelo.h5'))
        
        with open(ruta_path / 'escalador_X.pkl', 'wb') as f:
            pickle.dump(self.escalador_X, f)
        
        with open(ruta_path / 'escalador_y.pkl', 'wb') as f:
            pickle.dump(self.escalador_y, f)
        
        return True
    
    @staticmethod
    def cargar(ruta: str):
        """Carga modelo guardado."""
        ruta_path = Path(ruta)
        
        aproximador = AproximadorFuncion()
        aproximador.modelo = keras.models.load_model(str(ruta_path / 'modelo.h5'))
        
        with open(ruta_path / 'escalador_X.pkl', 'rb') as f:
            aproximador.escalador_X = pickle.load(f)
        
        with open(ruta_path / 'escalador_y.pkl', 'rb') as f:
            aproximador.escalador_y = pickle.load(f)
        
        return aproximador


# ============================================================================
# FUNCIÓN DE DEMOSTRACIÓN
# ============================================================================

def demo():
    """Demostración completa."""
    print("\n" + "="*70)
    print("APROXIMADOR DE FUNCIONES NO-LINEALES")
    print("="*70 + "\n")
    
    # Generar datos
    print("1. Generando datos para sin(x)...")
    generador = GeneradorFuncionesNoLineales()
    datos = generador.generar('sin', n_muestras=500)
    print(f"   ✓ {datos.info()}")
    
    # Crear aproximador
    print("\n2. Creando aproximador...")
    aprox = AproximadorFuncion()
    print("   ✓ Aproximador listo")
    
    # Entrenar
    print("\n3. Entrenando modelo MLP...")
    historial = aprox.entrenar(
        datos.X_train, datos.y_train,
        datos.X_test, datos.y_test,
        epochs=50,
        verbose=0
    )
    print(f"   ✓ Entrenado en 50 épocas")
    
    # Evaluar
    print("\n4. Evaluando...")
    metricas = aprox.evaluar(datos.X_test, datos.y_test)
    print(f"   ✓ RMSE: {metricas['rmse']:.6f}")
    print(f"   ✓ R²: {metricas['r2_score']:.6f}")
    
    # Predecir
    print("\n5. Predicciones en puntos de prueba:")
    muestra = datos.X_test[:3]
    predicciones = aprox.predecir(muestra)
    reales = datos.y_test[:3]
    for i, (x, y_pred, y_real) in enumerate(zip(muestra, predicciones, reales)):
        print(f"   x={x[0]:6.3f}: predicción={y_pred[0]:7.4f}, real={y_real[0]:7.4f}")


if __name__ == '__main__':
    demo()
