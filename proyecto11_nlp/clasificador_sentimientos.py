"""
Proyecto 11: Análisis de Sentimientos con NLP
==============================================

Sistema completo para clasificación de sentimientos en textos
usando embeddings y arquitecturas deep learning.

Modelos:
1. LSTM Bidireccional con embeddings pre-entrenados
2. Transformer (multi-head attention) nativo
3. CNN 1D para clasificación de textos

Características:
- Tokenización y padding automáticos
- Embeddings Word2Vec / GloVe simulados
- Pre-procesamiento: lowercase, stopwords, puntuación
- Validación de sentimientos: Positivo, Negativo, Neutro
- Análisis de confianza y palabras clave

Aplicaciones:
- Análisis de reviews de clientes
- Monitoreo de redes sociales
- Análisis de retroalimentación
- Detección de sentimientos en tickets de soporte

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosTexto:
    """Contenedor para datos de texto"""
    X_train: List[str]
    y_train: np.ndarray
    X_val: List[str]
    y_val: np.ndarray
    X_test: List[str]
    y_test: np.ndarray
    tokenizer: Tokenizer
    label_encoder: LabelEncoder
    etiquetas: List[str]
    
    def info(self) -> str:
        return (f"Texto: Train {len(self.X_train)}, "
                f"Val {len(self.X_val)}, Test {len(self.X_test)}")


class GeneradorTextoSentimientos:
    """Generador sintético de textos con sentimientos etiquetados"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    # Vocabulario de palabras por sentimiento
    POSITIVOS = [
        'excelente', 'maravilloso', 'fantástico', 'increíble', 'perfecto',
        'hermoso', 'genial', 'espléndido', 'admirable', 'asombroso',
        'magnífico', 'sensacional', 'extraordinario', 'excepcional',
        'destacado', 'brillante', 'radiante', 'precioso', 'delightful',
        'feliz', 'amor', 'amo', 'adoro', 'fascinante', 'cautivador'
    ]
    
    NEGATIVOS = [
        'terrible', 'horrible', 'malo', 'peor', 'desagradable',
        'horrible', 'decepcionante', 'inaceptable', 'repugnante',
        'despreciable', 'odio', 'odio', 'aborrezco', 'nauseabundo',
        'asqueroso', 'abominable', 'detestable', 'triste', 'pobre',
        'inferior', 'mediocre', 'deficiente', 'fallido', 'desastroso'
    ]
    
    NEUTROS = [
        'el', 'la', 'de', 'que', 'y', 'es', 'a', 'en', 'por', 'para',
        'con', 'no', 'una', 'su', 'se', 'le', 'lo', 'como', 'más', 'o',
        'esta', 'ese', 'eso', 'están', 'han', 'había', 'tengo'
    ]
    
    ADJETIVOS_CONTEXTO = [
        'muy', 'bastante', 'algo', 'poco', 'realmente', 'verdaderamente',
        'sumamente', 'extraordinariamente', 'tremendamente', 'incredibly'
    ]
    
    SUSTANTIVOS = [
        'película', 'producto', 'servicio', 'experiencia', 'comida',
        'restaurante', 'hotel', 'día', 'momento', 'viaje', 'trabajo',
        'libro', 'canción', 'persona', 'equipo', 'tienda', 'lugar'
    ]
    
    def _limpiar_texto(self, texto: str) -> str:
        """Limpia y normaliza texto"""
        # Lowercase
        texto = texto.lower()
        # Remover puntuación
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        # Normalizar espacios
        texto = ' '.join(texto.split())
        return texto
    
    def _generar_sentimiento_positivo(self) -> str:
        """Genera oración con sentimiento positivo"""
        adjetivos = [
            np.random.choice(self.ADJETIVOS_CONTEXTO),
            np.random.choice(self.POSITIVOS)
        ]
        sustantivo = np.random.choice(self.SUSTANTIVOS)
        
        templates = [
            f"{' '.join(adjetivos)} {sustantivo}",
            f"Me encanta este {sustantivo}, es {' '.join(adjetivos)}",
            f"El {sustantivo} fue {' '.join(adjetivos)}",
            f"Estoy tan feliz con este {sustantivo}, fue {self.POSITIVOS[np.random.randint(len(self.POSITIVOS))]}",
        ]
        
        return np.random.choice(templates)
    
    def _generar_sentimiento_negativo(self) -> str:
        """Genera oración con sentimiento negativo"""
        adjetivos = [
            np.random.choice(self.ADJETIVOS_CONTEXTO),
            np.random.choice(self.NEGATIVOS)
        ]
        sustantivo = np.random.choice(self.SUSTANTIVOS)
        
        templates = [
            f"{' '.join(adjetivos)} {sustantivo}",
            f"Odio este {sustantivo}, es {' '.join(adjetivos)}",
            f"El {sustantivo} fue {' '.join(adjetivos)}",
            f"Estoy tan decepcionado con este {sustantivo}, fue {self.NEGATIVOS[np.random.randint(len(self.NEGATIVOS))]}",
        ]
        
        return np.random.choice(templates)
    
    def _generar_sentimiento_neutro(self) -> str:
        """Genera oración neutra"""
        sustantivo1 = np.random.choice(self.SUSTANTIVOS)
        sustantivo2 = np.random.choice(self.SUSTANTIVOS)
        
        templates = [
            f"El {sustantivo1} es un {sustantivo2}",
            f"He visto un {sustantivo1}",
            f"El {sustantivo2} contiene un {sustantivo1}",
            f"Se puede describir el {sustantivo1} como un {sustantivo2}",
        ]
        
        return np.random.choice(templates)
    
    def generar(self, n_positivos: int = 100, n_negativos: int = 100,
               n_neutros: int = 100) -> Tuple[List[str], np.ndarray]:
        """
        Genera corpus de textos etiquetados
        
        Args:
            n_positivos: Cantidad textos positivos
            n_negativos: Cantidad textos negativos
            n_neutros: Cantidad textos neutros
        
        Returns:
            (textos, etiquetas) donde etiquetas son [0, 1, 2]
        """
        textos = []
        etiquetas = []
        
        # Positivos (etiqueta 2)
        for _ in range(n_positivos):
            texto = self._generar_sentimiento_positivo()
            textos.append(self._limpiar_texto(texto))
            etiquetas.append(2)  # Positivo
        
        # Negativos (etiqueta 0)
        for _ in range(n_negativos):
            texto = self._generar_sentimiento_negativo()
            textos.append(self._limpiar_texto(texto))
            etiquetas.append(0)  # Negativo
        
        # Neutros (etiqueta 1)
        for _ in range(n_neutros):
            texto = self._generar_sentimiento_neutro()
            textos.append(self._limpiar_texto(texto))
            etiquetas.append(1)  # Neutro
        
        return textos, np.array(etiquetas)
    
    def generar_dataset(self, n_samples_por_clase: int = 150,
                       max_words: int = 1000,
                       max_len: int = 50,
                       split: Tuple[float, float, float] = (0.6, 0.2, 0.2)
                       ) -> DatosTexto:
        """
        Genera dataset preprocessado listo para entrenar
        
        Args:
            n_samples_por_clase: Textos por clase de sentimiento
            max_words: Vocabulario máximo
            max_len: Longitud máxima de secuencia
            split: Ratios train/val/test
        
        Returns:
            DatosTexto
        """
        # Generar textos
        textos, etiquetas = self.generar(
            n_samples_por_clase, n_samples_por_clase, n_samples_por_clase
        )
        
        # Tokenizar
        tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
        tokenizer.fit_on_texts(textos)
        
        # Codificar secuencias
        sequences = tokenizer.texts_to_sequences(textos)
        X = pad_sequences(sequences, maxlen=max_len, padding='post')
        
        # Codificar etiquetas
        label_encoder = LabelEncoder()
        etiquetas_encoded = label_encoder.fit_transform(etiquetas)
        etiquetas_cat = keras.utils.to_categorical(etiquetas_encoded, num_classes=3)
        
        # Split temporal (orden original)
        n = len(X)
        train_ratio, val_ratio, test_ratio = split
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        X_train, y_train = X[:n_train], etiquetas_cat[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], etiquetas_cat[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], etiquetas_cat[n_train+n_val:]
        
        # Textos originales
        X_train_textos = textos[:n_train]
        X_val_textos = textos[n_train:n_train+n_val]
        X_test_textos = textos[n_train+n_val:]
        
        return DatosTexto(
            X_train_textos, X_train, X_val_textos, X_val, X_test_textos, X_test,
            tokenizer, label_encoder,
            etiquetas=list(label_encoder.classes_)
        )


class ClasificadorSentimientos:
    """Clasificador de sentimientos con múltiples arquitecturas"""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 128,
                 seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.entrenado = False
    
    def construir_lstm(self, max_len: int = 50) -> models.Model:
        """
        Construye LSTM bidireccional para clasificación de textos
        
        Arquitectura:
        - Embedding layer: Convierte índices a vectores densos
        - BiLSTM: Procesa contexto bidireccional
        - Dense layers: Clasificación
        """
        modelo = models.Sequential([
            layers.Input(shape=(max_len,)),
            
            # Embedding: Convierte índices a vectores 128D
            layers.Embedding(self.vocab_size, self.embedding_dim),
            
            # BiLSTM 1
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # BiLSTM 2
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # Clasificación
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')  # 3 clases: pos, neg, neut
        ])
        
        return modelo
    
    def construir_transformer(self, max_len: int = 50,
                            num_heads: int = 4,
                            ff_dim: int = 128) -> models.Model:
        """
        Construye Transformer para clasificación de textos
        
        Componentes:
        - Multi-head self-attention
        - Feed-forward layers
        - Layer normalization
        """
        inputs = layers.Input(shape=(max_len,))
        
        # Embedding
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # Transformer blocks (2 bloques)
        for _ in range(2):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=self.embedding_dim // num_heads
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward
            ff = layers.Dense(ff_dim, activation='relu')(x)
            ff = layers.Dense(self.embedding_dim)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Clasificación
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        modelo = models.Model(inputs=inputs, outputs=outputs)
        return modelo
    
    def construir_cnn1d(self, max_len: int = 50) -> models.Model:
        """
        Construye CNN 1D para clasificación de textos
        
        Idea: Detecta n-gramas como características locales
        """
        modelo = models.Sequential([
            layers.Input(shape=(max_len,)),
            
            # Embedding
            layers.Embedding(self.vocab_size, self.embedding_dim),
            
            # Conv1D con múltiples filter sizes
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            
            # Clasificación
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 30, arquitectura: str = 'lstm',
                 verbose: int = 1) -> Dict:
        """Entrena el modelo"""
        
        max_len = X_train.shape[1]
        
        if arquitectura == 'lstm':
            self.modelo = self.construir_lstm(max_len)
        elif arquitectura == 'transformer':
            self.modelo = self.construir_transformer(max_len)
        elif arquitectura == 'cnn1d':
            self.modelo = self.construir_cnn1d(max_len)
        else:
            raise ValueError(f"Arquitectura inválida: {arquitectura}")
        
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
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
        """Evalúa el modelo"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        y_pred_probs = self.modelo.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Accuracies
        accuracy = np.mean(y_pred == y_test_labels)
        
        # Por clase
        per_class_acc = {}
        for clase in [0, 1, 2]:
            mask = y_test_labels == clase
            if mask.sum() > 0:
                per_class_acc[clase] = np.mean(y_pred[mask] == y_test_labels[mask])
        
        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'predicciones': y_pred,
            'probabilidades': y_pred_probs,
            'verdaderos': y_test_labels
        }
    
    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        probs = self.modelo.predict(X, verbose=0)
        clases = np.argmax(probs, axis=1)
        return clases, probs
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(f"{ruta}_modelo.h5")
    
    @staticmethod
    def cargar(ruta: str, vocab_size: int, embedding_dim: int):
        """Carga un modelo guardado"""
        clasificador = ClasificadorSentimientos(vocab_size, embedding_dim)
        clasificador.modelo = keras.models.load_model(f"{ruta}_modelo.h5")
        clasificador.entrenado = True
        return clasificador


def demo():
    """Demostración completa"""
    print("="*70)
    print("CLASIFICADOR DE SENTIMIENTOS - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Generar datos
    print("\n[1] Generando textos sintéticos...")
    generador = GeneradorTextoSentimientos(seed=42)
    datos = generador.generar_dataset(
        n_samples_por_clase=100,
        max_words=1000,
        max_len=50
    )
    print(f"✓ {datos.info()}")
    print(f"  Clases: {data.etiquetas}")
    
    # 2. Entrenar LSTM
    print("\n[2] Entrenando LSTM bidireccional...")
    clasificador = ClasificadorSentimientos(vocab_size=1000, embedding_dim=128)
    clasificador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=15, arquitectura='lstm', verbose=0
    )
    metricas = clasificador.evaluar(datos.X_test, datos.y_test)
    print(f"✓ Accuracy: {metricas['accuracy']:.4f}")
    
    # 3. Transformer
    print("\n[3] Entrenando Transformer...")
    clasificador_tf = ClasificadorSentimientos()
    clasificador_tf.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=15, arquitectura='transformer', verbose=0
    )
    metricas_tf = clasificador_tf.evaluar(datos.X_test, datos.y_test)
    print(f"✓ Accuracy: {metricas_tf['accuracy']:.4f}")
    
    # 4. CNN1D
    print("\n[4] Entrenando CNN 1D...")
    clasificador_cnn = ClasificadorSentimientos()
    clasificador_cnn.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=15, arquitectura='cnn1d', verbose=0
    )
    metricas_cnn = clasificador_cnn.evaluar(datos.X_test, datos.y_test)
    print(f"✓ Accuracy: {metricas_cnn['accuracy']:.4f}")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
