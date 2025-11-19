"""
Aplicación: Reconocedor de Dígitos Manuscritos
=============================================

Caso de uso real: Clasificación de dígitos manuscritos usando CNN.

Características:
- Entrenamiento en MNIST
- Predicción con confianza
- Visualización de predicciones
- Análisis de errores

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json


class GeneradorDigitosMNIST:
    """Generador de datos MNIST."""
    
    @staticmethod
    def cargar_datos():
        """Carga dataset MNIST."""
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalizar
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Reformatear
        X_train = X_train.reshape(-1, 28, 28, 1).astype(np.float32)
        X_test = X_test.reshape(-1, 28, 28, 1).astype(np.float32)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }


class ReconocedorDigitos:
    """Reconocedor de dígitos manuscritos."""
    
    def __init__(self, seed=42):
        """Inicializa el reconocedor."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.historial = None
        self.metricas = {}
    
    def construir_modelo(self):
        """Construye CNN."""
        self.modelo = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo CNN construido")
        print(f"   Parámetros: {self.modelo.count_params():,}")
    
    def entrenar(self, X_train, y_train, epochs=5, batch_size=128):
        """Entrena el modelo."""
        self.historial = self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo."""
        pérdida, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.predecir_clases(X_test)
        
        self.metricas = {
            'loss': float(pérdida),
            'accuracy': float(accuracy),
            'accuracy_percent': float(accuracy * 100)
        }
        
        print(f"\n📊 Métricas de evaluación:")
        print(f"   Pérdida: {pérdida:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Reporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        return self.metricas
    
    def predecir(self, X):
        """Predice probabilidades."""
        return self.modelo.predict(X, verbose=0)
    
    def predecir_clases(self, X):
        """Predice clases."""
        probabilidades = self.predecir(X)
        return np.argmax(probabilidades, axis=1)
    
    def analizar_imagen(self, imagen):
        """
        Analiza una imagen individual.
        
        Args:
            imagen: Array (28, 28) o (28, 28, 1)
        
        Returns:
            Predicción con confianza
        """
        if len(imagen.shape) == 2:
            imagen = imagen.reshape(28, 28, 1)
        
        imagen = imagen.astype(np.float32) / 255.0
        imagen_batch = imagen.reshape(1, 28, 28, 1)
        
        probabilidades = self.predecir(imagen_batch)[0]
        digito_predicho = np.argmax(probabilidades)
        confianza = probabilidades[digito_predicho]
        
        return {
            'digito': int(digito_predicho),
            'confianza': float(confianza),
            'probabilidades': {
                f"Dígito {i}": float(p)
                for i, p in enumerate(probabilidades)
            }
        }
    
    def analizar_errores(self, X_test, y_test, top_n=5):
        """Analiza los errores más confiantes."""
        probabilidades = self.predecir(X_test)
        y_pred = np.argmax(probabilidades, axis=1)
        
        # Encontrar errores
        errores = np.where(y_pred != y_test)[0]
        
        if len(errores) == 0:
            print(f"\n✅ ¡Sin errores en el conjunto de test!")
            return []
        
        # Obtener confianza de errores
        confianza_errores = np.max(probabilidades[errores], axis=1)
        
        # Top errores por confianza
        top_error_idx = errores[np.argsort(-confianza_errores)[:top_n]]
        
        print(f"\n🔍 Top {top_n} errores más confiantes:")
        resultados = []
        
        for i, idx in enumerate(top_error_idx):
            y_verdadero = y_test[idx]
            y_predicho = y_pred[idx]
            conf = probabilidades[idx, y_predicho]
            
            print(f"\n   Error {i+1}:")
            print(f"     Verdadero: {y_verdadero}")
            print(f"     Predicho: {y_predicho} (confianza: {conf:.2%})")
            
            resultados.append({
                'index': int(idx),
                'verdadero': int(y_verdadero),
                'predicho': int(y_predicho),
                'confianza': float(conf)
            })
        
        return resultados


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🔢 RECONOCEDOR DE DÍGITOS MANUSCRITOS - CNN")
    print("="*80)
    
    # Paso 1: Cargar datos
    print("\n[1] Cargando dataset MNIST...")
    datos = GeneradorDigitosMNIST.cargar_datos()
    X_train = datos['X_train'][:10000]  # Usar subset para demo
    y_train = datos['y_train'][:10000]
    X_test = datos['X_test'][:2000]
    y_test = datos['y_test'][:2000]
    
    print(f"✅ Datos cargados:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Paso 2: Construir modelo
    print("\n[2] Construyendo modelo CNN...")
    reconocedor = ReconocedorDigitos()
    reconocedor.construir_modelo()
    
    # Paso 3: Entrenar
    print("\n[3] Entrenando modelo...")
    reconocedor.entrenar(X_train, y_train, epochs=5, batch_size=128)
    
    # Paso 4: Evaluar
    print("\n[4] Evaluando...")
    reconocedor.evaluar(X_test, y_test)
    
    # Paso 5: Predicciones individuales
    print("\n[5] Predicciones individuales:")
    for i in range(3):
        resultado = reconocedor.analizar_imagen(X_test[i])
        print(f"\n   Imagen {i+1}:")
        print(f"     → Dígito: {resultado['digito']} (confianza: {resultado['confianza']:.2%})")
    
    # Paso 6: Analizar errores
    print("\n[6] Analizando errores...")
    reconocedor.analizar_errores(X_test, y_test, top_n=5)
    
    # Paso 7: Visualizar
    print("\n[7] Generando visualizaciones...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Predicciones de dígitos MNIST')
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[i].squeeze(), cmap='gray')
        
        resultado = reconocedor.analizar_imagen(X_test[i])
        titulo = f"Predicho: {resultado['digito']}\nVerdadero: {y_test[i]}"
        ax.set_title(titulo, color='green' if resultado['digito'] == y_test[i] else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=100)
    plt.close()
    
    print(f"✅ Visualización guardada")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'CNN Reconocedor de Dígitos',
        'dataset': 'MNIST',
        'muestras': len(X_test),
        'metricas': reconocedor.metricas,
        'configuracion': {
            'épocas': 5,
            'batch_size': 128,
            'arquitectura': 'CNN 3 capas convolucionales'
        }
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
