#!/usr/bin/env python3
"""
P8: Detector de Objetos - CNN con Region Proposals
Detectar objetos y localizar con bounding boxes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class GeneradorObjetos:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=200):
        """Generar imagenes con objetos y bounding boxes"""
        imagenes = []
        bboxes = []
        clases = []
        
        for _ in range(n_samples):
            img = np.random.rand(64, 64, 3) * 0.3
            
            # Agregar 1-3 objetos por imagen
            n_objetos = np.random.randint(1, 4)
            for _ in range(n_objetos):
                x = np.random.randint(5, 50)
                y = np.random.randint(5, 50)
                w = np.random.randint(5, 20)
                h = np.random.randint(5, 20)
                clase = np.random.randint(0, 3)
                
                # Dibujar rectangulo
                img[y:y+h, x:x+w, clase] = 0.9
                
                imagenes.append(img.copy())
                bboxes.append([x, y, w, h])
                clases.append(clase)
        
        return {
            'X': np.array(imagenes).astype('float32'),
            'bboxes': np.array(bboxes).astype('float32'),
            'clases': np.array(clases)
        }


class DetectorObjetos:
    def __init__(self):
        self.modelo = None
    
    def construir_modelo(self):
        """CNN con salidas para clasificacion y bounding box"""
        entrada = keras.Input(shape=(64, 64, 3))
        
        # Backbone CNN
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(entrada)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Flatten()(x)
        
        # Classification head
        clf = keras.layers.Dense(64, activation='relu')(x)
        clf = keras.layers.Dense(3, activation='softmax', name='clasificacion')(clf)
        
        # Regression head (bounding box)
        bbox = keras.layers.Dense(64, activation='relu')(x)
        bbox = keras.layers.Dense(4, activation='sigmoid', name='bounding_box')(bbox)
        
        self.modelo = keras.Model(entrada, [clf, bbox])
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss=['sparse_categorical_crossentropy', 'mse'],
            metrics=['accuracy']
        )
        print("[+] Modelo detector P8 construido")
    
    def entrenar(self, X, clases, bboxes, epochs=10):
        self.modelo.fit(X, [clases, bboxes], epochs=epochs, verbose=0, batch_size=16)


def main():
    print("\n" + "="*60)
    print("P8: DETECTOR DE OBJETOS (CNN + Region Proposals)")
    print("="*60)
    
    generador = GeneradorObjetos()
    datos = generador.generar_dataset(200)
    
    n_train = int(0.8 * len(datos['X']))
    X_train = datos['X'][:n_train]
    clases_train = datos['clases'][:n_train]
    bboxes_train = datos['bboxes'][:n_train]
    
    modelo = DetectorObjetos()
    modelo.construir_modelo()
    modelo.entrenar(X_train, clases_train, bboxes_train, epochs=10)
    
    print("Detector de objetos entrenado exitosamente")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
