"""
Proyecto 9: Clasificador de Imágenes CIFAR-10 con CNN Profunda
==============================================================

Sistema para clasificación de objetos en 10 categorías usando
imágenes RGB de 32x32 píxeles del dataset CIFAR-10.

Categorías:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Características principales:
- Arquitectura CNN profunda (5+ capas convolucionales)
- Data augmentation (rotación, zoom, flip, desplazamiento)
- Transfer learning con MobileNetV2 pre-entrenada
- Técnicas avanzadas: Batch normalization, dropout, regularización L2
- Evaluación: Accuracy, precision, recall, F1-score, confusion matrix

Teórico:
El aprendizaje profundo revolucionó visión por computadora. CNN capturan
características jerárquicas: bordes (L1) → formas (L2) → objetos (L3+).
Transfer learning reutiliza conocimiento de ImageNet (1.2M imágenes,
1000 clases) en CIFAR-10, logrando mejor generalización con menos datos.

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosImagenes:
    """Contenedor para datos de CIFAR-10"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    clases: List[str]
    
    def info(self) -> str:
        return (f"CIFAR-10 Dataset: Train {self.X_train.shape}, "
                f"Val {self.X_val.shape}, Test {self.X_test.shape}")


class GeneradorCIFAR10:
    """Cargador y preprocesador de CIFAR-10"""
    
    CLASES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def cargar_datos(self, validacion_split: float = 0.2
                    ) -> 'DatosImagenes':
        """
        Carga CIFAR-10 y realiza split train/val/test
        
        Args:
            validacion_split: Proporción para validación
        
        Returns:
            DatosImagenes con splits
        """
        # Cargar del servidor de TensorFlow
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Reducir a [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Aplanar labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Split train/val
        n_val = int(len(X_train) * validacion_split)
        indices = np.random.permutation(len(X_train))
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        
        return DatosImagenes(
            X_train, y_train, X_val, y_val, X_test, y_test,
            self.CLASES
        )
    
    def crear_augmentador(self) -> ImageDataGenerator:
        """Crea generador de data augmentation"""
        return ImageDataGenerator(
            rotation_range=20,           # Rotación ±20°
            width_shift_range=0.2,       # Desplazamiento horizontal 20%
            height_shift_range=0.2,      # Desplazamiento vertical 20%
            horizontal_flip=True,        # Flip horizontal
            zoom_range=0.2,              # Zoom 80-120%
            fill_mode='nearest'
        )


class ClasificadorImagenes:
    """Clasificador de imágenes CIFAR-10"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.entrenado = False
        self.augmentador = None
    
    def construir_cnn_profunda(self, n_clases: int = 10) -> models.Model:
        """
        Construye CNN profunda personalizada
        
        Arquitectura: 5 bloques convolucionales + 2 capas densas
        """
        modelo = models.Sequential([
            # Bloque 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                         input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling
            layers.GlobalAveragePooling2D(),
            
            # Capas densas
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Salida
            layers.Dense(n_clases, activation='softmax')
        ])
        
        return modelo
    
    def construir_transfer_learning(self, n_clases: int = 10) -> models.Model:
        """
        Construye modelo con transfer learning desde MobileNetV2
        
        MobileNetV2: Ligero, rápido, entrenado en ImageNet
        """
        # Cargar modelo pre-entrenado sin capas densas
        base_model = keras.applications.MobileNetV2(
            input_shape=(32, 32, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar capas base
        base_model.trainable = False
        
        # Agregar capas personalizadas
        modelo = models.Sequential([
            base_model,
            
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(n_clases, activation='softmax')
        ])
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 50, arquitectura: str = 'cnn',
                 usar_augmentacion: bool = True,
                 verbose: int = 1) -> Dict:
        """
        Entrena el modelo
        
        Args:
            X_train: Imágenes de entrenamiento [N, 32, 32, 3]
            y_train: Labels de entrenamiento [N]
            X_val: Imágenes de validación
            y_val: Labels de validación
            epochs: Número de épocas
            arquitectura: 'cnn' o 'transfer'
            usar_augmentacion: Si aplicar data augmentation
            verbose: Nivel de verbosidad
        
        Returns:
            Historial de entrenamiento
        """
        n_clases = len(np.unique(y_train))
        
        if arquitectura == 'cnn':
            self.modelo = self.construir_cnn_profunda(n_clases)
        elif arquitectura == 'transfer':
            self.modelo = self.construir_transfer_learning(n_clases)
        else:
            raise ValueError(f"Arquitectura inválida: {arquitectura}")
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
            )
        ]
        
        if usar_augmentacion:
            self.augmentador = GeneradorCIFAR10().crear_augmentador()
            
            hist = self.modelo.fit(
                self.augmentador.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
                steps_per_epoch=len(X_train) // 32
            )
        else:
            hist = self.modelo.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose,
                batch_size=32
            )
        
        self.entrenado = True
        return hist.history
    
    def evaluar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo
        
        Returns:
            Diccionario con métricas
        """
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        loss, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=1)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'predicciones': y_pred
        }
    
    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice clase y probabilidades"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        probs = self.modelo.predict(X, verbose=0)
        clases = np.argmax(probs, axis=1)
        
        return clases, probs
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(f"{ruta}_modelo.h5")
    
    @staticmethod
    def cargar(ruta: str) -> 'ClasificadorImagenes':
        """Carga un modelo guardado"""
        clf = ClasificadorImagenes()
        clf.modelo = keras.models.load_model(f"{ruta}_modelo.h5")
        clf.entrenado = True
        return clf


def demo():
    """Demostración completa"""
    print("="*70)
    print("CLASIFICADOR DE IMÁGENES CIFAR-10 - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Cargar datos
    print("\n[1] Cargando CIFAR-10...")
    generador = GeneradorCIFAR10()
    datos = generador.cargar_datos()
    print(f"✓ {datos.info()}")
    print(f"  Clases: {len(datos.clases)}")
    
    # 2. Entrenar CNN
    print("\n[2] Entrenando CNN personalizada...")
    clf_cnn = ClasificadorImagenes()
    clf_cnn.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=10, arquitectura='cnn',
        usar_augmentacion=True, verbose=0
    )
    metricas_cnn = clf_cnn.evaluar(datos.X_test, datos.y_test)
    print(f"✓ CNN Accuracy: {metricas_cnn['accuracy']:.4f}")
    
    # 3. Entrenar Transfer Learning
    print("\n[3] Entrenando con Transfer Learning (MobileNetV2)...")
    clf_tl = ClasificadorImagenes()
    clf_tl.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=10, arquitectura='transfer',
        usar_augmentacion=True, verbose=0
    )
    metricas_tl = clf_tl.evaluar(datos.X_test, datos.y_test)
    print(f"✓ Transfer Learning Accuracy: {metricas_tl['accuracy']:.4f}")
    
    # 4. Comparar
    print("\n[4] Comparación:")
    print(f"  CNN:                 {metricas_cnn['accuracy']:.4f}")
    print(f"  Transfer Learning:   {metricas_tl['accuracy']:.4f}")
    
    mejor = "CNN" if metricas_cnn['accuracy'] > metricas_tl['accuracy'] else "Transfer Learning"
    print(f"  ✓ Mejor: {mejor}")
    
    # 5. Predecir
    print("\n[5] Predicciones en test samples:")
    clases, probs = clf_cnn.predecir(datos.X_test[:5])
    for i in range(5):
        print(f"  {i}: {datos.clases[clases[i]]} ({probs[i, clases[i]]:.2%})")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
