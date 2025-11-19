"""
P11: CLASIFICADOR DE SENTIMIENTOS CON RNN Y EMBEDDING
Aplicación para análisis de sentimientos usando redes neuronales recurrentes.

Técnica: RNN + Embedding + LSTM
Dataset: Textos sintéticos con sentimientos etiquetados
Métrica: Accuracy, Precision, Recall, F1-Score
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class GeneradorTextos:
    """Generador de textos sintéticos con sentimientos"""
    
    @staticmethod
    def generar_dataset(n_samples=1000, seed=42):
        """Genera textos sintéticos con sentimientos"""
        np.random.seed(seed)
        
        textos_positivos = [
            "me encanta este producto es excelente",
            "muy bueno recomendado totalmente",
            "perfecto funciona muy bien",
            "esta increible me fascina",
            "lo mejor que he comprado",
            "calidad premium excelente servicio",
            "muy satisfecho con la compra",
            "fantastico producto realmente",
            "adoro todo sobre esto",
            "genial superó mis expectativas"
        ]
        
        textos_negativos = [
            "horrible no funciona nada",
            "muy malo decepcionante totalmente",
            "pésimo calidad terrible",
            "no recomiendo nada bueno",
            "estoy muy decepcionado",
            "producto defectuoso no vale",
            "perdí mi dinero aquí",
            "lo peor que compré",
            "completamente insatisfecho",
            "no sirve para nada"
        ]
        
        textos_neutros = [
            "el producto llego a tiempo",
            "es un producto normal",
            "funciona como se describe",
            "nada especial solo lo básico",
            "precio promedio calidad media",
            "ni bueno ni malo",
            "cumple su función",
            "es aceptable",
            "funciona correctamente",
            "producto estándar"
        ]
        
        X, y = [], []
        for _ in range(n_samples // 3):
            X.append(np.random.choice(textos_positivos))
            y.append(1)  # Positivo
            X.append(np.random.choice(textos_negativos))
            y.append(0)  # Negativo
            X.append(np.random.choice(textos_neutros))
            y.append(2)  # Neutro
        
        return np.array(X), np.array(y)


class ClasificadorSentimientos:
    """Clasificador de sentimientos con RNN + Embedding"""
    
    def __init__(self, vocab_size=500, max_len=20, embedding_dim=16):
        self.modelo = None
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        
    def construir_modelo(self):
        """Construye modelo RNN con Embedding"""
        self.modelo = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, 
                           input_length=self.max_len, name='embedding'),
            layers.LSTM(64, activation='relu', return_sequences=True, name='lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', name='lstm_2'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dense(3, activation='softmax', name='salida')  # 3 clases
        ], name='ClasificadorSentimientos')
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preparar_textos(self, textos):
        """Prepara textos para el modelo"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(textos)
        
        secuencias = self.tokenizer.texts_to_sequences(textos)
        datos = pad_sequences(secuencias, maxlen=self.max_len, padding='post')
        return datos.astype(np.float32)
    
    def entrenar(self, X_train, y_train, epochs=20, batch_size=32):
        """Entrena el modelo"""
        self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
    
    def predecir(self, X):
        """Realiza predicciones"""
        probs = self.modelo.predict(X, verbose=0)
        return np.argmax(probs, axis=1), np.max(probs, axis=1)


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(" "*10 + "P11: CLASIFICADOR DE SENTIMIENTOS CON RNN")
    print("="*70 + "\n")
    
    # Generar datos
    print("[1/7] Generando textos sintéticos...")
    generador = GeneradorTextos()
    X_textos, y_etiquetas = generador.generar_dataset(n_samples=900, seed=42)
    print(f"     {len(X_textos)} textos generados")
    
    # Crear clasificador
    print("[2/7] Inicializando clasificador...")
    clasificador = ClasificadorSentimientos(vocab_size=500, max_len=20, embedding_dim=16)
    
    # Preparar datos
    print("[3/7] Preparando textos...")
    X_preparado = clasificador.preparar_textos(X_textos)
    split = int(0.8 * len(X_preparado))
    X_train, X_test = X_preparado[:split], X_preparado[split:]
    y_train, y_test = y_etiquetas[:split], y_etiquetas[split:]
    print(f"     Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Construir modelo
    print("[4/7] Construyendo modelo...")
    clasificador.construir_modelo()
    clasificador.modelo.summary()
    
    # Entrenar
    print("[5/7] Entrenando modelo...")
    clasificador.entrenar(X_train, y_train, epochs=20, batch_size=32)
    print("     Entrenamiento completado")
    
    # Evaluar
    print("[6/7] Evaluando modelo...")
    y_pred_train, conf_train = clasificador.predecir(X_train)
    y_pred_test, conf_test = clasificador.predecir(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    print(f"     Accuracy Train: {acc_train:.6f}")
    print(f"     Accuracy Test:  {acc_test:.6f}")
    print(f"     Precision:      {precision:.6f}")
    print(f"     Recall:         {recall:.6f}")
    print(f"     F1-Score:       {f1:.6f}")
    
    # Predicciones de ejemplo
    print("[7/7] Predicciones de ejemplo...")
    sentimientos = {0: "Negativo", 1: "Positivo", 2: "Neutro"}
    for i in range(min(5, len(X_test))):
        pred = y_pred_test[i]
        confianza = conf_test[i]
        print(f"     Sentimiento: {sentimientos[pred]}, Confianza: {confianza:.4f}")
    
    # Guardar resultados
    reporte = {
        "proyecto": "P11 - Clasificador de Sentimientos",
        "tecnica": "RNN + Embedding + LSTM",
        "fecha": datetime.now().isoformat(),
        "configuracion": {
            "vocab_size": 500,
            "max_len": 20,
            "embedding_dim": 16,
            "epochs": 20,
            "batch_size": 32
        },
        "metricas": {
            "accuracy_train": float(acc_train),
            "accuracy_test": float(acc_test),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "muestras_train": len(X_train),
            "muestras_test": len(X_test)
        },
        "modelo": {
            "parametros": int(clasificador.modelo.count_params()),
            "capas": "Embedding -> LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(3)"
        }
    }
    
    os.makedirs('reportes', exist_ok=True)
    with open('reportes/reporte_p11.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print("\n" + "="*70)
    print("[OK] Aplicación P11 completada correctamente")
    print(f"[REPORTE] Guardado en: reportes/reporte_p11.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
