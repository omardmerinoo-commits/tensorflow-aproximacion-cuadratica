"""
Módulo para aproximación de la función y = x² mediante redes neuronales.

Este módulo implementa la clase ModeloCuadratico que utiliza TensorFlow
para construir, entrenar y evaluar una red neuronal simple que aprende
la relación cuadrática y = x².

Proyecto TensorFlow
Fecha: Noviembre 2025
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
from typing import Tuple, Optional, List


class ModeloCuadratico:
    """
    Clase para aproximar la función y = x² usando una red neuronal simple.
    
    Esta clase encapsula todo el flujo de trabajo de aprendizaje automático:
    generación de datos, construcción del modelo, entrenamiento, predicción
    y persistencia del modelo entrenado.
    
    Attributes
    ----------
    modelo : tf.keras.Model
        Modelo secuencial de TensorFlow/Keras.
    x_train : np.ndarray
        Datos de entrada para entrenamiento.
    y_train : np.ndarray
        Datos de salida para entrenamiento.
    history : tf.keras.callbacks.History
        Historial de entrenamiento con métricas por época.
    
    Examples
    --------
    >>> modelo_cuad = ModeloCuadratico()
    >>> modelo_cuad.generar_datos(n_samples=1000, rango=(-1, 1))
    >>> modelo_cuad.construir_modelo()
    >>> modelo_cuad.entrenar(epochs=100, batch_size=32)
    >>> predicciones = modelo_cuad.predecir(np.array([[0.5], [1.0]]))
    """
    
    def __init__(self):
        """
        Inicializa una nueva instancia de ModeloCuadratico.
        
        El modelo se inicializa como None y debe ser construido
        explícitamente usando el método construir_modelo().
        """
        self.modelo: Optional[tf.keras.Model] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        
    def generar_datos(
        self, 
        n_samples: int = 1000, 
        rango: Tuple[float, float] = (-1, 1),
        ruido: float = 0.02,
        seed: Optional[int] = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera datos de entrenamiento para la función y = x².
        
        Crea un conjunto de datos sintético generando valores x uniformemente
        distribuidos en el rango especificado y calculando y = x² con la
        adición de ruido gaussiano para simular datos del mundo real.
        
        Parameters
        ----------
        n_samples : int, optional
            Número de muestras a generar (default: 1000).
        rango : tuple of float, optional
            Tupla (min, max) que define el rango de valores x (default: (-1, 1)).
        ruido : float, optional
            Desviación estándar del ruido gaussiano a añadir (default: 0.02).
        seed : int or None, optional
            Semilla para reproducibilidad. Si es None, no se fija semilla (default: 42).
            
        Returns
        -------
        tuple of np.ndarray
            Tupla (x, y) donde:
            - x: array de forma (n_samples, 1) con valores de entrada
            - y: array de forma (n_samples, 1) con valores y = x² + ruido
            
        Raises
        ------
        ValueError
            Si n_samples <= 0 o si rango[0] >= rango[1] o si ruido < 0.
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> x, y = modelo.generar_datos(n_samples=500, rango=(-2, 2), ruido=0.01)
        >>> print(x.shape, y.shape)
        (500, 1) (500, 1)
        """
        # Validación de parámetros
        if n_samples <= 0:
            raise ValueError(f"n_samples debe ser positivo, se recibió: {n_samples}")
        
        if rango[0] >= rango[1]:
            raise ValueError(f"rango inválido: {rango}. El mínimo debe ser menor que el máximo.")
        
        if ruido < 0:
            raise ValueError(f"ruido debe ser no negativo, se recibió: {ruido}")
        
        # Fijar semilla para reproducibilidad
        if seed is not None:
            np.random.seed(seed)
        
        # Generar valores x uniformemente distribuidos
        x = np.random.uniform(low=rango[0], high=rango[1], size=(n_samples, 1))
        
        # Calcular y = x² con ruido gaussiano
        y = x ** 2 + np.random.normal(loc=0.0, scale=ruido, size=(n_samples, 1))
        
        # Almacenar los datos generados
        self.x_train = x.astype(np.float32)
        self.y_train = y.astype(np.float32)
        
        print(f"✓ Datos generados exitosamente:")
        print(f"  - Muestras: {n_samples}")
        print(f"  - Rango de x: [{rango[0]}, {rango[1]}]")
        print(f"  - Ruido (std): {ruido}")
        print(f"  - Forma de x: {self.x_train.shape}")
        print(f"  - Forma de y: {self.y_train.shape}")
        
        return self.x_train, self.y_train
    
    def construir_modelo(self) -> None:
        """
        Construye y compila el modelo de red neuronal secuencial.
        
        Crea una arquitectura de red neuronal feedforward con:
        - Capa de entrada: 1 neurona (valor x)
        - Primera capa oculta: 64 neuronas, activación ReLU
        - Segunda capa oculta: 64 neuronas, activación ReLU
        - Capa de salida: 1 neurona, activación lineal (valor y)
        
        El modelo se compila con:
        - Optimizador: Adam (tasa de aprendizaje adaptativa)
        - Función de pérdida: MSE (Mean Squared Error)
        - Métricas: MAE (Mean Absolute Error)
        
        Returns
        -------
        None
            El modelo se almacena en self.modelo
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> modelo.construir_modelo()
        >>> modelo.modelo.summary()
        """
        # Crear modelo secuencial
        self.modelo = keras.Sequential([
            # Capa de entrada implícita (shape=(1,))
            
            # Primera capa oculta: 64 neuronas con activación ReLU
            # ReLU (Rectified Linear Unit) introduce no-linealidad: f(x) = max(0, x)
            layers.Dense(
                units=64,
                activation='relu',
                input_shape=(1,),
                kernel_initializer='he_normal',  # Inicialización óptima para ReLU
                name='capa_oculta_1'
            ),
            
            # Segunda capa oculta: 64 neuronas con activación ReLU
            # Permite al modelo aprender representaciones más complejas
            layers.Dense(
                units=64,
                activation='relu',
                kernel_initializer='he_normal',
                name='capa_oculta_2'
            ),
            
            # Capa de salida: 1 neurona con activación lineal
            # Activación lineal para regresión (sin restricciones en el rango de salida)
            layers.Dense(
                units=1,
                activation='linear',
                kernel_initializer='glorot_uniform',
                name='capa_salida'
            )
        ], name='ModeloCuadratico')
        
        # Compilar el modelo
        # Esto configura el proceso de aprendizaje
        self.modelo.compile(
            # Adam: optimizador adaptativo que ajusta la tasa de aprendizaje
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            
            # MSE: función de pérdida para regresión
            # Penaliza errores grandes más que errores pequeños
            loss='mse',
            
            # MAE: métrica adicional para monitorear el entrenamiento
            # Más interpretable que MSE (mismas unidades que y)
            metrics=['mae']
        )
        
        print("✓ Modelo construido y compilado exitosamente:")
        print(f"  - Arquitectura: [1] → [64, ReLU] → [64, ReLU] → [1, Linear]")
        print(f"  - Optimizador: Adam (lr=0.001)")
        print(f"  - Función de pérdida: MSE")
        print(f"  - Métricas: MAE")
        print(f"  - Parámetros entrenables: {self.modelo.count_params():,}")
        
    def entrenar(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[List] = None
    ) -> tf.keras.callbacks.History:
        """
        Entrena el modelo con los datos generados.
        
        Ejecuta el proceso de entrenamiento utilizando backpropagation y
        descenso de gradiente. Incluye validación automática y callbacks
        para early stopping y guardado del mejor modelo.
        
        Parameters
        ----------
        epochs : int, optional
            Número máximo de épocas de entrenamiento (default: 100).
        batch_size : int, optional
            Tamaño del lote para entrenamiento por mini-batches (default: 32).
        validation_split : float, optional
            Fracción de datos a usar para validación (default: 0.2).
        callbacks : list or None, optional
            Lista de callbacks personalizados. Si es None, se usan
            EarlyStopping y ModelCheckpoint por defecto.
            
        Returns
        -------
        tf.keras.callbacks.History
            Objeto History con métricas de entrenamiento y validación
            por época (loss, mae, val_loss, val_mae).
            
        Raises
        ------
        RuntimeError
            Si el modelo no ha sido construido o si no hay datos generados.
        ValueError
            Si epochs <= 0 o batch_size <= 0 o validation_split no está en (0, 1).
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> modelo.generar_datos()
        >>> modelo.construir_modelo()
        >>> history = modelo.entrenar(epochs=150, batch_size=64)
        """
        # Validaciones
        if self.modelo is None:
            raise RuntimeError("Debe construir el modelo antes de entrenar. Use construir_modelo().")
        
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Debe generar datos antes de entrenar. Use generar_datos().")
        
        if epochs <= 0:
            raise ValueError(f"epochs debe ser positivo, se recibió: {epochs}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size debe ser positivo, se recibió: {batch_size}")
        
        if not (0 < validation_split < 1):
            raise ValueError(f"validation_split debe estar en (0, 1), se recibió: {validation_split}")
        
        # Configurar callbacks por defecto si no se proporcionan
        if callbacks is None:
            callbacks = [
                # EarlyStopping: detiene el entrenamiento si no hay mejora
                EarlyStopping(
                    monitor='val_loss',      # Métrica a monitorear
                    patience=15,              # Épocas sin mejora antes de detener
                    restore_best_weights=True, # Restaurar pesos del mejor modelo
                    verbose=1,
                    mode='min'                # Minimizar la pérdida
                ),
                
                # ModelCheckpoint: guarda el mejor modelo durante entrenamiento
                ModelCheckpoint(
                    filepath='mejor_modelo_temp.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0,
                    mode='min'
                )
            ]
        
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento...")
        print(f"{'='*60}")
        print(f"Configuración:")
        print(f"  - Épocas máximas: {epochs}")
        print(f"  - Tamaño de lote: {batch_size}")
        print(f"  - División de validación: {validation_split*100:.0f}%")
        print(f"  - Muestras de entrenamiento: {int(len(self.x_train)*(1-validation_split))}")
        print(f"  - Muestras de validación: {int(len(self.x_train)*validation_split)}")
        print(f"{'='*60}\n")
        
        # Entrenar el modelo
        self.history = self.modelo.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1  # Mostrar barra de progreso
        )
        
        # Mostrar resumen del entrenamiento
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_mae = self.history.history['mae'][-1]
        final_val_mae = self.history.history['val_mae'][-1]
        
        print(f"\n{'='*60}")
        print(f"Entrenamiento completado")
        print(f"{'='*60}")
        print(f"Resultados finales:")
        print(f"  - Loss (entrenamiento): {final_loss:.6f}")
        print(f"  - Loss (validación): {final_val_loss:.6f}")
        print(f"  - MAE (entrenamiento): {final_mae:.6f}")
        print(f"  - MAE (validación): {final_val_mae:.6f}")
        print(f"  - Épocas ejecutadas: {len(self.history.history['loss'])}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def predecir(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el modelo entrenado.
        
        Parameters
        ----------
        x : np.ndarray
            Array de valores de entrada. Puede ser de forma (n,) o (n, 1).
            
        Returns
        -------
        np.ndarray
            Array de predicciones de forma (n, 1).
            
        Raises
        ------
        RuntimeError
            Si el modelo no ha sido construido o entrenado.
        ValueError
            Si x no es un array de numpy o tiene dimensiones incorrectas.
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> # ... generar datos, construir y entrenar modelo ...
        >>> x_test = np.array([[0.5], [1.0], [1.5]])
        >>> predicciones = modelo.predecir(x_test)
        >>> print(predicciones)
        """
        if self.modelo is None:
            raise RuntimeError("Debe construir el modelo antes de predecir. Use construir_modelo().")
        
        if not isinstance(x, np.ndarray):
            raise ValueError("x debe ser un array de numpy.")
        
        # Asegurar que x tenga la forma correcta (n, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 2 and x.shape[1] != 1:
            raise ValueError(f"x debe tener forma (n, 1), se recibió: {x.shape}")
        
        # Realizar predicción
        predicciones = self.modelo.predict(x, verbose=0)
        
        return predicciones
    
    def guardar_modelo(
        self,
        path_tf: str = "modelo_entrenado.h5",
        path_pkl: str = "modelo_entrenado.pkl"
    ) -> None:
        """
        Guarda el modelo entrenado en formatos TensorFlow (.h5) y pickle (.pkl).
        
        El formato .h5 es nativo de Keras y preserva la arquitectura completa,
        pesos y configuración del optimizador. El formato .pkl permite
        serialización completa del objeto para compatibilidad con otras herramientas.
        
        Parameters
        ----------
        path_tf : str, optional
            Ruta para guardar el modelo en formato TensorFlow (default: "modelo_entrenado.h5").
        path_pkl : str, optional
            Ruta para guardar el modelo en formato pickle (default: "modelo_entrenado.pkl").
            
        Returns
        -------
        None
        
        Raises
        ------
        RuntimeError
            Si el modelo no ha sido construido.
        IOError
            Si hay problemas al escribir los archivos.
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> # ... entrenar modelo ...
        >>> modelo.guardar_modelo("mi_modelo.h5", "mi_modelo.pkl")
        """
        if self.modelo is None:
            raise RuntimeError("No hay modelo para guardar. Debe construir y entrenar el modelo primero.")
        
        try:
            # Guardar en formato TensorFlow (.h5)
            self.modelo.save(path_tf)
            size_h5 = os.path.getsize(path_tf) / 1024  # KB
            print(f"✓ Modelo guardado en formato TensorFlow: {path_tf} ({size_h5:.2f} KB)")
            
            # Guardar en formato pickle (.pkl)
            # Serializamos un diccionario con el modelo y metadatos
            modelo_data = {
                'modelo': self.modelo,
                'history': self.history.history if self.history else None,
                'arquitectura': {
                    'capas': len(self.modelo.layers),
                    'parametros': self.modelo.count_params()
                }
            }
            
            with open(path_pkl, 'wb') as f:
                pickle.dump(modelo_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            size_pkl = os.path.getsize(path_pkl) / 1024  # KB
            print(f"✓ Modelo guardado en formato pickle: {path_pkl} ({size_pkl:.2f} KB)")
            
        except Exception as e:
            raise IOError(f"Error al guardar el modelo: {str(e)}")
    
    def cargar_modelo(
        self,
        path_tf: Optional[str] = None,
        path_pkl: Optional[str] = None
    ) -> None:
        """
        Carga un modelo previamente guardado desde archivo .h5 o .pkl.
        
        Permite cargar el modelo desde cualquiera de los dos formatos.
        Si se proporcionan ambos, se prioriza el formato TensorFlow (.h5).
        
        Parameters
        ----------
        path_tf : str or None, optional
            Ruta del archivo .h5 a cargar (default: None).
        path_pkl : str or None, optional
            Ruta del archivo .pkl a cargar (default: None).
            
        Returns
        -------
        None
            El modelo se carga en self.modelo
            
        Raises
        ------
        ValueError
            Si no se proporciona ninguna ruta o si los archivos no existen.
        IOError
            Si hay problemas al leer los archivos.
            
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> modelo.cargar_modelo(path_tf="modelo_entrenado.h5")
        >>> # O alternativamente:
        >>> modelo.cargar_modelo(path_pkl="modelo_entrenado.pkl")
        """
        if path_tf is None and path_pkl is None:
            raise ValueError("Debe proporcionar al menos una ruta (path_tf o path_pkl).")
        
        try:
            # Priorizar carga desde TensorFlow si está disponible
            if path_tf is not None:
                if not os.path.exists(path_tf):
                    raise ValueError(f"El archivo no existe: {path_tf}")
                
                self.modelo = keras.models.load_model(path_tf)
                print(f"✓ Modelo cargado desde formato TensorFlow: {path_tf}")
                print(f"  - Parámetros: {self.modelo.count_params():,}")
                
            # Cargar desde pickle si no hay .h5 o como alternativa
            elif path_pkl is not None:
                if not os.path.exists(path_pkl):
                    raise ValueError(f"El archivo no existe: {path_pkl}")
                
                with open(path_pkl, 'rb') as f:
                    modelo_data = pickle.load(f)
                
                self.modelo = modelo_data['modelo']
                
                # Restaurar history si está disponible
                if modelo_data.get('history'):
                    # Crear un objeto History mock
                    self.history = type('History', (), {'history': modelo_data['history']})()
                
                print(f"✓ Modelo cargado desde formato pickle: {path_pkl}")
                print(f"  - Parámetros: {self.modelo.count_params():,}")
                
        except Exception as e:
            raise IOError(f"Error al cargar el modelo: {str(e)}")
    
    def resumen(self) -> None:
        """
        Muestra un resumen completo del modelo y su estado.
        
        Imprime información sobre la arquitectura, parámetros,
        estado de entrenamiento y datos disponibles.
        
        Examples
        --------
        >>> modelo = ModeloCuadratico()
        >>> modelo.construir_modelo()
        >>> modelo.resumen()
        """
        print(f"\n{'='*60}")
        print(f"RESUMEN DEL MODELO CUADRÁTICO")
        print(f"{'='*60}")
        
        if self.modelo is not None:
            print("\n📊 Arquitectura del modelo:")
            self.modelo.summary()
        else:
            print("\n⚠ Modelo no construido")
        
        print(f"\n📈 Estado de los datos:")
        if self.x_train is not None:
            print(f"  - Datos de entrenamiento: {self.x_train.shape[0]} muestras")
            print(f"  - Rango de x: [{self.x_train.min():.3f}, {self.x_train.max():.3f}]")
            print(f"  - Rango de y: [{self.y_train.min():.3f}, {self.y_train.max():.3f}]")
        else:
            print("  - No hay datos generados")
        
        print(f"\n🎯 Estado del entrenamiento:")
        if self.history is not None:
            print(f"  - Épocas completadas: {len(self.history.history['loss'])}")
            print(f"  - Mejor val_loss: {min(self.history.history['val_loss']):.6f}")
        else:
            print("  - Modelo no entrenado")
        
        print(f"\n{'='*60}\n")


# Función auxiliar para pruebas rápidas
def ejemplo_rapido():
    """
    Función de demostración rápida del uso de ModeloCuadratico.
    
    Ejecuta un ejemplo completo de generación de datos, construcción,
    entrenamiento y predicción del modelo.
    """
    print("Ejemplo de uso de ModeloCuadratico\n")
    
    # Crear instancia
    modelo = ModeloCuadratico()
    
    # Generar datos
    modelo.generar_datos(n_samples=1000, rango=(-1, 1), ruido=0.02)
    
    # Construir modelo
    modelo.construir_modelo()
    
    # Entrenar
    modelo.entrenar(epochs=50, batch_size=32)
    
    # Hacer predicciones
    x_test = np.array([[0.0], [0.5], [1.0]])
    predicciones = modelo.predecir(x_test)
    
    print("\nPredicciones de prueba:")
    for x_val, y_pred in zip(x_test, predicciones):
        y_real = x_val[0] ** 2
        print(f"  x={x_val[0]:.2f} → y_pred={y_pred[0]:.4f}, y_real={y_real:.4f}")
    
    # Guardar modelo
    modelo.guardar_modelo()
    
    # Mostrar resumen
    modelo.resumen()


if __name__ == "__main__":
    # Ejecutar ejemplo si se ejecuta el script directamente
    ejemplo_rapido()
