#!/usr/bin/env python3
"""
P3: Clasificador de Diagnostico - Multiclase
Clasificar 3 enfermedades basado en sintomas
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class GeneradorDatosdiagnostico:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=500):
        """Generar sintomas y diagnosticos"""
        # 3 enfermedades
        X = np.random.randn(n_samples, 20) * 10 + 50
        
        # Crear separabilidad
        y = np.random.randint(0, 3, n_samples)
        for i in range(n_samples):
            if y[i] == 0:
                X[i, :5] += 20
            elif y[i] == 1:
                X[i, 5:10] += 20
            else:
                X[i, 10:15] += 20
        
        return {'X': X, 'y': y, 'features': [f'Sintoma_{i}' for i in range(20)]}


class ClasificadorDiagnostico:
    def __init__(self):
        self.scaler = StandardScaler()
        self.modelo = None
    
    def construir_modelo(self, input_dim=20):
        """Red para 3 clases"""
        self.modelo = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[+] Modelo P3 construido")
    
    def entrenar(self, X_train, y_train, epochs=20):
        X_scaled = self.scaler.fit_transform(X_train)
        self.modelo.fit(X_scaled, y_train, epochs=epochs, verbose=0)
    
    def evaluar(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        y_pred = np.argmax(self.modelo.predict(X_scaled, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {'accuracy': float(acc), 'f1': float(f1)}


def main():
    print("\n" + "="*60)
    print("P3: CLASIFICADOR DE DIAGNOSTICO (MULTICLASE)")
    print("="*60)
    
    generador = GeneradorDatosdiagnostico()
    datos = generador.generar_dataset(500)
    
    X_train, X_test, y_train, y_test = train_test_split(
        datos['X'], datos['y'], test_size=0.2, random_state=42)
    
    modelo = ClasificadorDiagnostico()
    modelo.construir_modelo(20)
    modelo.entrenar(X_train, y_train, epochs=20)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
