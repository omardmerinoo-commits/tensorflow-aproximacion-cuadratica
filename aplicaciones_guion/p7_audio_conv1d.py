#!/usr/bin/env python3
"""
P7: Clasificador de Ruido - Conv1D para Espectrogramas
Clasificar 3 tipos de ruido (blanco, trafico, lluvia)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score


class GeneradorEspectrogramas:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=300):
        """Generar espectrogramas de 3 tipos de ruido"""
        espectrogramas = []
        etiquetas = []
        
        tipos_ruido = ['blanco', 'trafico', 'lluvia']
        
        for tipo_idx, tipo in enumerate(tipos_ruido):
            for _ in range(n_samples // 3):
                if tipo == 'blanco':
                    # Ruido blanco: distribucion uniforme
                    spec = np.random.rand(128, 64) * 0.5
                elif tipo == 'trafico':
                    # Trafico: picos de baja frecuencia
                    spec = np.random.rand(128, 64) * 0.3
                    spec[:, :20] += 0.4
                else:  # lluvia
                    # Lluvia: distribucion aleatoria
                    spec = np.random.exponential(0.3, (128, 64))
                    spec = np.clip(spec, 0, 1)
                
                espectrogramas.append(spec)
                etiquetas.append(tipo_idx)
        
        return {
            'X': np.array(espectrogramas).reshape(-1, 128, 1).astype('float32'),
            'y': np.array(etiquetas)
        }


class ClasificadorRuido:
    def __init__(self):
        self.modelo = None
    
    def construir_modelo(self):
        """Conv1D para clasificacion de ruido"""
        self.modelo = keras.Sequential([
            keras.layers.Conv1D(32, 3, activation='relu', input_shape=(128, 1)),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[+] Modelo Conv1D P7 construido")
    
    def entrenar(self, X_train, y_train, epochs=15):
        self.modelo.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=16)
    
    def evaluar(self, X_test, y_test):
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {'accuracy': float(acc), 'f1': float(f1)}


def main():
    print("\n" + "="*60)
    print("P7: CLASIFICADOR DE RUIDO (Conv1D Audio)")
    print("="*60)
    
    generador = GeneradorEspectrogramas()
    datos = generador.generar_dataset(300)
    
    n_train = int(0.8 * len(datos['X']))
    X_train, X_test = datos['X'][:n_train], datos['X'][n_train:]
    y_train, y_test = datos['y'][:n_train], datos['y'][n_train:]
    
    modelo = ClasificadorRuido()
    modelo.construir_modelo()
    modelo.entrenar(X_train, y_train, epochs=15)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
