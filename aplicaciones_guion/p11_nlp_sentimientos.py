#!/usr/bin/env python3
"""
P11: Clasificador de Sentimientos - RNN + Embedding
Clasificar textos en positivo, negativo, neutral
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score


class GeneradorTextos:
    def __init__(self, seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def generar_dataset(self, n_samples=300):
        """Generar textos con sentimientos"""
        textos_positivos = [
            "me encanta este producto es excelente",
            "muy bueno recomendado totalmente",
            "perfecto funciona muy bien",
            "esta increible me fascina",
        ]
        
        textos_negativos = [
            "horrible no funciona nada",
            "muy malo decepcionante totalmente",
            "pesimo calidad terrible",
            "no recomiendo nada bueno",
        ]
        
        textos_neutros = [
            "el producto llego a tiempo",
            "es un producto normal",
            "funciona como se describe",
            "nada especial solo lo basico",
        ]
        
        X, y = [], []
        for _ in range(n_samples // 3):
            X.append(np.random.choice(textos_positivos))
            y.append(1)
            X.append(np.random.choice(textos_negativos))
            y.append(0)
            X.append(np.random.choice(textos_neutros))
            y.append(2)
        
        return np.array(X), np.array(y)


class ClasificadorSentimientos:
    def __init__(self, vocab_size=500, max_len=20, embedding_dim=16):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.modelo = None
    
    def construir_modelo(self):
        """Red RNN + Embedding para sentimientos"""
        self.modelo = keras.Sequential([
            keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
            keras.layers.LSTM(64, activation='relu', return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[+] Modelo RNN P11 construido")
    
    def preparar_textos(self, textos):
        """Tokenizar y padding"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(textos)
        
        secuencias = self.tokenizer.texts_to_sequences(textos)
        return pad_sequences(secuencias, maxlen=self.max_len, padding='post').astype(np.float32)
    
    def entrenar(self, X_textos, y, epochs=20):
        X_prep = self.preparar_textos(X_textos)
        self.modelo.fit(X_prep, y, epochs=epochs, verbose=0, batch_size=16)
    
    def evaluar(self, X_textos, y):
        X_prep = self.preparar_textos(X_textos)
        y_pred = np.argmax(self.modelo.predict(X_prep, verbose=0), axis=1)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return {'accuracy': float(acc), 'f1': float(f1)}


def main():
    print("\n" + "="*60)
    print("P11: CLASIFICADOR DE SENTIMIENTOS (RNN + Embedding)")
    print("="*60)
    
    generador = GeneradorTextos()
    X_textos, y = generador.generar_dataset(300)
    
    n_train = int(0.8 * len(X_textos))
    X_train, X_test = X_textos[:n_train], X_textos[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    modelo = ClasificadorSentimientos(vocab_size=500, max_len=20, embedding_dim=16)
    modelo.construir_modelo()
    modelo.entrenar(X_train, y_train, epochs=20)
    metricas = modelo.evaluar(X_test, y_test)
    
    print(f"Accuracy: {metricas['accuracy']:.4f}")
    print(f"F1-Score: {metricas['f1']:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
