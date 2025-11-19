"""
Proyecto 7: Clasificador de Audio con Espectrogramas y Redes Neuronales
=========================================================================

Módulo para clasificación de audio en 3 categorías:
- Ruido (noise): Ruido ambiental
- Música (music): Música instrumental
- Voz (speech): Voz humana

Características:
- Generación sintética de audio usando numpy
- Extracción de espectrogramas (STFT)
- Arquitecturas CNN 1D/2D y LSTM
- Validación completa con métricas de clasificación

Teórico:
La Transformada de Fourier de Corta Duración (STFT) es fundamental en
procesamiento de audio. Permite representar señales no-estacionarias como
imágenes de tiempo-frecuencia. Un clasificador CNN captura patrones
locales en espectrogramas, mientras LSTM captura dependencias temporales.

"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DatosAudio:
    """Contenedor para datos de audio preprocesados"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    labels: Dict[int, str]
    
    def info(self) -> str:
        return (f"Audio Dataset: Train {self.X_train.shape}, "
                f"Val {self.X_val.shape}, Test {self.X_test.shape}")


class GeneradorAudioSintetico:
    """
    Generador de señales de audio sintéticas para 3 categorías.
    
    Categorías:
    - Ruido (noise): Ruido blanco/rosa
    - Música (music): Múltiples sinusoides con modulación
    - Voz (speech): Envolvente modulada (formantes)
    """
    
    CATEGORIAS = {
        'noise': 0,
        'music': 1,
        'speech': 2
    }
    
    def __init__(self, sr: int = 16000, seed: int = 42):
        """
        Args:
            sr: Sample rate (Hz)
            seed: Random seed para reproducibilidad
        """
        self.sr = sr
        self.seed = seed
        np.random.seed(seed)
    
    def _generar_ruido(self, duracion: float = 2.0) -> np.ndarray:
        """Genera ruido blanco/rosa"""
        n_samples = int(self.sr * duracion)
        
        # 70% ruido blanco, 30% ruido rosa
        if np.random.rand() > 0.7:
            # Ruido rosa: filtro paso-bajo en frecuencia
            blanco = np.random.randn(n_samples)
            ruido_rosa = np.zeros_like(blanco)
            ruido_rosa[0] = blanco[0]
            for i in range(1, n_samples):
                ruido_rosa[i] = 0.99 * ruido_rosa[i-1] + blanco[i]
            return ruido_rosa / np.max(np.abs(ruido_rosa))
        else:
            return np.random.randn(n_samples) * 0.5
    
    def _generar_musica(self, duracion: float = 2.0) -> np.ndarray:
        """Genera múltiples sinusoides con modulación (imitando música)"""
        n_samples = int(self.sr * duracion)
        t = np.arange(n_samples) / self.sr
        
        # 3-5 frecuencias fundamentales
        n_armonicos = np.random.randint(3, 6)
        frecuencias = np.random.choice(
            [100, 130, 165, 196, 220, 246, 261, 293, 330, 349],
            n_armonicos, replace=False
        )
        
        senial = np.zeros(n_samples)
        for f in frecuencias:
            # Modulación de amplitud
            amplitud = np.exp(-t / np.random.uniform(0.5, 2.0))
            senial += amplitud * np.sin(2 * np.pi * f * t)
        
        # Añadir vibrato
        vibrato_freq = np.random.uniform(4, 8)
        vibrato_depth = np.random.uniform(5, 20)
        senial *= (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t))
        
        return senial / np.max(np.abs(senial)) * 0.8
    
    def _generar_voz(self, duracion: float = 2.0) -> np.ndarray:
        """Genera envolvente modulada (imitando formantes de voz)"""
        n_samples = int(self.sr * duracion)
        t = np.arange(n_samples) / self.sr
        
        # Frecuencia fundamental (pitch) variable
        f0_base = np.random.uniform(80, 200)
        f0 = f0_base * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))  # Vibrato lento
        
        # Acumulación de fase para frecuencia variable
        phase = 2 * np.pi * np.cumsum(f0) / self.sr
        senial_base = np.sin(phase)
        
        # Formantes (resonancias vocales)
        formantes = [
            np.random.uniform(600, 1000),    # F1
            np.random.uniform(1200, 2000),   # F2
            np.random.uniform(2500, 3500)    # F3
        ]
        
        senial = senial_base * 0.6
        for f_form in formantes:
            # Filtro resonante simple: sinusoides amortiguadas
            amortiguamiento = np.random.uniform(0.95, 0.98)
            envolvente = np.power(amortiguamiento, t * self.sr)
            senial += 0.1 * envolvente * np.sin(2 * np.pi * f_form * t)
        
        # Modulación de envolvente (sílabas)
        n_silabas = np.random.randint(3, 6)
        duracion_silaba = duracion / n_silabas
        envolvente_silabas = np.zeros(n_samples)
        
        for i in range(n_silabas):
            inicio = int(i * duracion_silaba * self.sr)
            fin = int((i + 1) * duracion_silaba * self.sr)
            t_silaba = np.arange(fin - inicio) / (duracion_silaba * self.sr)
            env = np.sin(np.pi * t_silaba) ** 2  # Envolvente gaussiana
            envolvente_silabas[inicio:fin] = env
        
        senial = senial * envolvente_silabas
        return senial / (np.max(np.abs(senial)) + 1e-8)
    
    def generar(self, categoria: str, n_muestras: int = 100,
                duracion: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera señales de audio sintéticas
        
        Args:
            categoria: 'noise', 'music', 'speech'
            n_muestras: Número de samples
            duracion: Duración de cada audio (segundos)
        
        Returns:
            X: Array [n_samples, n_time_steps]
            y: Labels
        """
        if categoria not in self.CATEGORIAS:
            raise ValueError(f"Categoría inválida: {categoria}")
        
        generador = {
            'noise': self._generar_ruido,
            'music': self._generar_musica,
            'speech': self._generar_voz
        }[categoria]
        
        X = np.array([generador(duracion) for _ in range(n_muestras)])
        y = np.full(n_muestras, self.CATEGORIAS[categoria])
        
        return X, y
    
    def generar_dataset(self, muestras_por_clase: int = 100,
                       duracion: float = 2.0,
                       split: Tuple[float, float, float] = (0.6, 0.2, 0.2)
                       ) -> DatosAudio:
        """
        Genera dataset completo con train/val/test split
        
        Args:
            muestras_por_clase: Samples por categoría
            duracion: Duración de cada audio
            split: (train_ratio, val_ratio, test_ratio)
        
        Returns:
            DatosAudio con splits
        """
        train_ratio, val_ratio, test_ratio = split
        
        X_list = []
        y_list = []
        
        for categoria in self.CATEGORIAS.keys():
            X, y = self.generar(categoria, muestras_por_clase, duracion)
            X_list.append(X)
            y_list.append(y)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        X, y = X[indices], y[indices]
        
        # Split
        n_train = int(len(y) * train_ratio)
        n_val = int(len(y) * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        labels = {v: k for k, v in self.CATEGORIAS.items()}
        
        return DatosAudio(X_train, y_train, X_val, y_val, X_test, y_test, labels)


class ExtractorEspectrograma:
    """Extrae características de espectrograma STFT"""
    
    def __init__(self, n_fft: int = 512, hop_length: int = 128):
        """
        Args:
            n_fft: Tamaño FFT
            hop_length: Desplazamiento entre ventanas
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def _stft(self, x: np.ndarray) -> np.ndarray:
        """Calcula STFT usando ventana Hann"""
        X = np.zeros((self.n_fft // 2 + 1, 
                      (len(x) - self.n_fft) // self.hop_length + 1),
                     dtype=complex)
        
        window = np.hanning(self.n_fft)
        
        for i in range(X.shape[1]):
            inicio = i * self.hop_length
            fin = inicio + self.n_fft
            if fin > len(x):
                break
            ventana = x[inicio:fin] * window
            X[:, i] = np.fft.rfft(ventana)[:self.n_fft // 2 + 1]
        
        return X
    
    def extraer(self, X: np.ndarray, db_scale: bool = True) -> np.ndarray:
        """
        Extrae espectrogramas de señales de audio
        
        Args:
            X: Array [n_samples, n_time_steps]
            db_scale: Si True, convierte a escala dB
        
        Returns:
            Espectrogramas [n_samples, freq_bins, time_steps, 1]
        """
        espectrogramas = []
        
        for x in X:
            spec = np.abs(self._stft(x))
            
            if db_scale:
                spec = 20 * np.log10(spec + 1e-9)
                spec = np.maximum(spec, spec.max() - 80)
            
            espectrogramas.append(spec)
        
        # Padding a dimensiones iguales
        max_time = max(s.shape[1] for s in espectrogramas)
        espectrogramas_padded = []
        
        for spec in espectrogramas:
            if spec.shape[1] < max_time:
                pad_width = max_time - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)))
            espectrogramas_padded.append(spec)
        
        X_spec = np.array(espectrogramas_padded)
        
        return X_spec[:, :, :, np.newaxis]


class ClasificadorAudio:
    """Clasificador de audio con CNN 1D/2D y LSTM"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.scaler = StandardScaler()
        self.entrenado = False
    
    def _normalizar(self, X: np.ndarray) -> np.ndarray:
        """Normaliza características"""
        shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        X_norm = self.scaler.fit_transform(X_flat)
        return X_norm.reshape(shape)
    
    def construir_cnn_2d(self, input_shape: Tuple[int, ...],
                         n_clases: int = 3) -> models.Model:
        """
        Construye CNN 2D para espectrogramas
        
        Input: [freq_bins, time_steps, 1]
        """
        modelo = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Bloque 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Global pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(n_clases, activation='softmax')
        ])
        
        return modelo
    
    def construir_lstm(self, input_shape: Tuple[int, ...],
                       n_clases: int = 3) -> models.Model:
        """
        Construye LSTM para capturar dependencias temporales
        
        Input: [time_steps, features]
        """
        modelo = models.Sequential([
            layers.Input(shape=input_shape),
            
            # LSTM bidireccional
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=0.2)
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_clases, activation='softmax')
        ])
        
        return modelo
    
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 epochs: int = 30, arquitectura: str = 'cnn',
                 verbose: int = 1) -> Dict:
        """
        Entrena el modelo
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Datos de validación
            y_val: Labels de validación
            epochs: Número de épocas
            arquitectura: 'cnn' o 'lstm'
            verbose: Nivel de verbosidad
        
        Returns:
            Historial de entrenamiento
        """
        X_train_norm = self._normalizar(X_train)
        
        n_clases = len(np.unique(y_train))
        input_shape = X_train_norm.shape[1:]
        
        if arquitectura == 'cnn':
            self.modelo = self.construir_cnn_2d(input_shape, n_clases)
        elif arquitectura == 'lstm':
            # Remodelar para LSTM
            X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], 
                                                X_train_norm.shape[1], -1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
            self.modelo = self.construir_lstm((X_train_norm.shape[1], 
                                              X_train_norm.shape[2]), n_clases)
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
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
            )
        ]
        
        hist = self.modelo.fit(
            X_train_norm, y_train,
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
        
        X_test_norm = self._normalizar(X_test)
        if X_test_norm.ndim == 4 and self.modelo.input_shape[-1] != 1:
            X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], 
                                             X_test_norm.shape[1], -1)
        
        loss, accuracy = self.modelo.evaluate(X_test_norm, y_test, verbose=0)
        
        y_pred = np.argmax(self.modelo.predict(X_test_norm, verbose=0), axis=1)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def predecir(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predice clase y probabilidades"""
        if not self.entrenado:
            raise ValueError("Modelo no entrenado")
        
        X_norm = self._normalizar(X)
        if X_norm.ndim == 4 and self.modelo.input_shape[-1] != 1:
            X_norm = X_norm.reshape(X_norm.shape[0], X_norm.shape[1], -1)
        
        probs = self.modelo.predict(X_norm, verbose=0)
        clases = np.argmax(probs, axis=1)
        
        return clases, probs
    
    def guardar(self, ruta: str):
        """Guarda el modelo"""
        self.modelo.save(f"{ruta}_modelo.h5")
        with open(f"{ruta}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @staticmethod
    def cargar(ruta: str) -> 'ClasificadorAudio':
        """Carga un modelo guardado"""
        clf = ClasificadorAudio()
        clf.modelo = keras.models.load_model(f"{ruta}_modelo.h5")
        with open(f"{ruta}_scaler.pkl", 'rb') as f:
            clf.scaler = pickle.load(f)
        clf.entrenado = True
        return clf


def demo():
    """Demostración completa del clasificador"""
    print("="*70)
    print("CLASIFICADOR DE AUDIO - DEMOSTRACIÓN")
    print("="*70)
    
    # 1. Generar datos
    print("\n[1] Generando datos sintéticos...")
    generador = GeneradorAudioSintetico()
    datos = generador.generar_dataset(muestras_por_clase=50, duracion=2.0)
    print(f"✓ {datos.info()}")
    
    # 2. Extraer espectrogramas
    print("\n[2] Extrayendo espectrogramas...")
    extractor = ExtractorEspectrograma()
    X_train_spec = extractor.extraer(datos.X_train)
    X_test_spec = extractor.extraer(datos.X_test)
    print(f"✓ Shape espectrogramas: {X_train_spec.shape}")
    
    # 3. Entrenar CNN
    print("\n[3] Entrenando CNN...")
    clf_cnn = ClasificadorAudio()
    clf_cnn.entrenar(X_train_spec, datos.y_train, 
                     X_test_spec, datos.y_val,
                     epochs=10, arquitectura='cnn', verbose=0)
    metricas_cnn = clf_cnn.evaluar(X_test_spec, datos.y_test)
    print(f"✓ Accuracy CNN: {metricas_cnn['accuracy']:.4f}")
    
    # 4. Predecir
    print("\n[4] Realizando predicciones...")
    clases, probs = clf_cnn.predecir(X_test_spec[:5])
    labels = datos.labels
    for i, (clase, prob) in enumerate(zip(clases, probs)):
        print(f"  Sample {i}: {labels[clase]} ({prob[clase]:.2%})")
    
    print("\n✓ Demostración completada")


if __name__ == '__main__':
    demo()
