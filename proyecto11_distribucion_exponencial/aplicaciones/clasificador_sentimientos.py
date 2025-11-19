"""
Aplicación: Clasificador de Sentimientos
======================================

Caso de uso real: Análisis de sentimientos en textos (reseñas, tweets)

Características:
- Generación de textos sintéticos
- Tokenización y embedding
- RNN para clasificación de texto
- Análisis de sentimientos

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import re
from datetime import datetime
from pathlib import Path
import json


class GeneradorTextos:
    """Generador de textos con sentimientos."""
    
    PALABRAS_POSITIVAS = [
        'excelente', 'fantástico', 'maravilloso', 'increíble', 'perfecto',
        'hermoso', 'genial', 'asombroso', 'espléndido', 'magnífico',
        'amor', 'adoro', 'me encanta', 'incrustado', 'bendición'
    ]
    
    PALABRAS_NEGATIVAS = [
        'horrible', 'terrible', 'malo', 'peor', 'decepcionante',
        'odio', 'asco', 'basura', 'patético', 'desagradable',
        'triste', 'deprimente', 'frustrado', 'enfadado', 'furioso'
    ]
    
    PALABRAS_NEUTRAS = [
        'normal', 'promedio', 'regular', 'estándar', 'típico',
        'común', 'usual', 'corriente', 'ordinario', 'mediocre'
    ]
    
    @staticmethod
    def generar_texto_positivo():
        """Genera texto con sentimiento positivo."""
        palabras_pos = np.random.choice(GeneradorTextos.PALABRAS_POSITIVAS, np.random.randint(2, 5))
        palabras_generales = ['es', 'fue', 'será', 'considerado', 'realmente']
        
        texto = ' '.join(palabras_pos) + ' y ' + np.random.choice(palabras_generales)
        return texto
    
    @staticmethod
    def generar_texto_negativo():
        """Genera texto con sentimiento negativo."""
        palabras_neg = np.random.choice(GeneradorTextos.PALABRAS_NEGATIVAS, np.random.randint(2, 5))
        palabras_generales = ['totalmente', 'absolutamente', 'completamente']
        
        texto = np.random.choice(palabras_generales) + ' ' + ' y '.join(palabras_neg)
        return texto
    
    @staticmethod
    def generar_texto_neutro():
        """Genera texto con sentimiento neutro."""
        palabras_neu = np.random.choice(GeneradorTextos.PALABRAS_NEUTRAS, np.random.randint(2, 4))
        palabras_generales = ['es', 'parece', 'se ve', 'resulta ser']
        
        texto = np.random.choice(palabras_generales) + ' ' + ' '.join(palabras_neu)
        return texto
    
    @staticmethod
    def generar_dataset(n_samples_por_clase=200):
        """Genera dataset de textos."""
        generadores = {
            0: GeneradorTextos.generar_texto_positivo,
            1: GeneradorTextos.generar_texto_negativo,
            2: GeneradorTextos.generar_texto_neutro
        }
        
        textos = []
        etiquetas = []
        
        for clase_id, generador in generadores.items():
            for _ in range(n_samples_por_clase):
                texto = generador()
                textos.append(texto)
                etiquetas.append(clase_id)
        
        return {
            'textos': textos,
            'etiquetas': np.array(etiquetas),
            'clases': ['positivo', 'negativo', 'neutro']
        }


class ClasificadorSentimientos:
    """Clasificador de sentimientos."""
    
    def __init__(self, vocab_size=1000, embedding_dim=16, seed=42):
        """Inicializa el clasificador."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.modelo = None
        self.tokenizer = None
        self.metricas = {}
    
    def construir_tokenizer(self, textos):
        """Construye tokenizador."""
        self.tokenizer = keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(textos)
        print(f"✅ Tokenizador construido con {len(self.tokenizer.word_index)} palabras únicas")
    
    def textos_a_secuencias(self, textos, maxlen=20):
        """Convierte textos a secuencias."""
        secuencias = self.tokenizer.texts_to_sequences(textos)
        secuencias = keras.preprocessing.sequence.pad_sequences(
            secuencias, maxlen=maxlen, padding='post'
        )
        return secuencias.astype(np.float32)
    
    def construir_modelo(self, maxlen=20):
        """Construye modelo RNN."""
        self.modelo = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=maxlen),
            
            layers.LSTM(32, return_sequences=True),
            layers.Dropout(0.2),
            
            layers.LSTM(16),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo RNN de clasificación de sentimientos construido")
    
    def entrenar(self, X_train, y_train, epochs=15):
        """Entrena el modelo."""
        self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test, y_test, clases=None):
        """Evalúa el modelo."""
        pérdida, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=1)
        
        self.metricas = {
            'loss': float(pérdida),
            'accuracy': float(accuracy)
        }
        
        print(f"\n📊 Métricas:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Pérdida: {pérdida:.4f}")
        
        if clases:
            print(f"\n   Precisión por clase:")
            for i, clase in enumerate(clases):
                mask = y_test == i
                if mask.sum() > 0:
                    acc_clase = (y_pred[mask] == i).mean()
                    print(f"     {clase}: {acc_clase:.4f}")
        
        return self.metricas
    
    def clasificar_sentimiento(self, texto, clases=None, maxlen=20):
        """Clasifica el sentimiento de un texto."""
        secuencia = self.textos_a_secuencias([texto], maxlen=maxlen)
        
        probabilidades = self.modelo.predict(secuencia, verbose=0)[0]
        clase_predicha = np.argmax(probabilidades)
        confianza = probabilidades[clase_predicha]
        
        resultado = {
            'clase': int(clase_predicha),
            'confianza': float(confianza),
            'probabilidades': {
                f"{clases[i] if clases else f'Clase {i}'}": float(p)
                for i, p in enumerate(probabilidades)
            }
        }
        
        if clases:
            resultado['clase_nombre'] = clases[clase_predicha]
        
        return resultado


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("💬 CLASIFICADOR DE SENTIMIENTOS - RNN")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando textos con sentimientos...")
    datos = GeneradorTextos.generar_dataset(n_samples_por_clase=200)
    
    textos = datos['textos']
    etiquetas = datos['etiquetas']
    clases = datos['clases']
    
    print(f"✅ Dataset generado: {len(textos)} textos")
    print(f"   Clases: {clases}")
    print(f"   Distribución: {[(etiquetas==i).sum() for i in range(len(clases))]}")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
    )
    
    print(f"✅ Train: {len(X_train_text)}, Test: {len(X_test_text)}")
    
    # Paso 3: Construir y tokenizar
    print("\n[3] Construyendo tokenizador...")
    clasificador = ClasificadorSentimientos(vocab_size=500, embedding_dim=16)
    clasificador.construir_tokenizer(X_train_text)
    
    X_train = clasificador.textos_a_secuencias(X_train_text, maxlen=20)
    X_test = clasificador.textos_a_secuencias(X_test_text, maxlen=20)
    
    print(f"✅ Textos convertidos a secuencias: {X_train.shape}")
    
    # Paso 4: Construir modelo
    print("\n[4] Construyendo modelo RNN...")
    clasificador.construir_modelo(maxlen=20)
    
    # Paso 5: Entrenar
    print("\n[5] Entrenando...")
    clasificador.entrenar(X_train, y_train, epochs=15)
    
    # Paso 6: Evaluar
    print("\n[6] Evaluando...")
    clasificador.evaluar(X_test, y_test, clases=clases)
    
    # Paso 7: Clasificar textos
    print("\n[7] Clasificando sentimientos:")
    textos_ejemplo = [
        "excelente maravilloso fantástico",
        "horrible terrible malo",
        "normal regular promedio"
    ]
    
    for texto in textos_ejemplo:
        resultado = clasificador.clasificar_sentimiento(texto, clases=clases)
        print(f"\n   Texto: '{texto}'")
        print(f"     Sentimiento: {resultado.get('clase_nombre', f'Clase {resultado['clase']}')} ({resultado['confianza']:.2%})")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'RNN Clasificador de Sentimientos',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'clases': clases,
        'vocab_size': 500,
        'metricas': clasificador.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
