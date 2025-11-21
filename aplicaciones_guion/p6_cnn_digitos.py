#!/usr/bin/env python3
"""
P6: Reconocedor de Digitos - CNN para MNIST
Clasificar imagenes manuscritas de digitos 0-9
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score


class GeneradorDigitos:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=1000):
        """Generar digitos manuscritos sinteticos"""
        imagenes = []
        etiquetas = []
        
        for digito in range(10):
            for _ in range(n_samples // 10):
                img = np.random.rand(28, 28) * 0.3
                # Agregar circulo/lineas segun digito
                y, x = np.ogrid[:28, :28]
                mask = (x - 14)**2 + (y - 14)**2 <= (6 + digito)**2
                img[mask] = 0.8
                
                # Ruido
                img += np.random.randn(28, 28) * 0.1
                img = np.clip(img, 0, 1)
                
                imagenes.append(img)
                etiquetas.append(digito)
        
        return {
            'X': np.array(imagenes).reshape(-1, 28, 28, 1).astype('float32'),
            'y': np.array(etiquetas)
        }


class ReconocedorDigitos:
    def __init__(self):
        self.modelo = None
    
    def construir_modelo(self):
        """CNN para clasificacion de digitos"""
        self.modelo = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[+] Modelo CNN P6 construido")
    
    def entrenar(self, X_train, y_train, epochs=10):
        self.modelo.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=32)
    
    def evaluar(self, X_test, y_test):
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {'accuracy': float(acc), 'f1': float(f1)}


def main():
    print("\n" + "="*60)
    print("P6: RECONOCEDOR DE DIGITOS (CNN MNIST)")
    print("="*60)
    
    generador = GeneradorDigitos()
    datos = generador.generar_dataset(1000)
    
    n_train = int(0.8 * len(datos['X']))
    X_train, X_test = datos['X'][:n_train], datos['X'][n_train:]
    y_train, y_test = datos['y'][:n_train], datos['y'][n_train:]
    
    modelo = ReconocedorDigitos()
    modelo.construir_modelo()
    modelo.entrenar(X_train, y_train, epochs=10)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
