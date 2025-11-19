"""
Aplicación: Clasificador de Ruido Ambiental
==========================================

Caso de uso real: Clasificación de tipos de ruido (tráfico, lluvia, voces, etc.)

Características:
- Generación de espectrogramas
- Clasificación con CNN
- Análisis de frecuencias
- Detección de ambiente

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
from scipy import signal


class GeneradorEspectrogramas:
    """Generador de espectrogramas sintéticos."""
    
    @staticmethod
    def generar_sonido_ruido(duracion=2, freq_muestreo=16000):
        """Ruido blanco."""
        n_samples = int(freq_muestreo * duracion)
        return np.random.randn(n_samples)
    
    @staticmethod
    def generar_sonido_trafico(duracion=2, freq_muestreo=16000):
        """Simula ruido de tráfico."""
        n_samples = int(freq_muestreo * duracion)
        t = np.arange(n_samples) / freq_muestreo
        
        # Mezcla de frecuencias bajas (tráfico)
        sonido = (
            0.5 * np.sin(2 * np.pi * 200 * t) +
            0.3 * np.sin(2 * np.pi * 300 * t) +
            0.2 * np.random.randn(n_samples)
        )
        
        return sonido
    
    @staticmethod
    def generar_sonido_lluvia(duracion=2, freq_muestreo=16000):
        """Simula ruido de lluvia."""
        n_samples = int(freq_muestreo * duracion)
        t = np.arange(n_samples) / freq_muestreo
        
        # Ruido filtrado en altas frecuencias
        ruido = np.random.randn(n_samples)
        # Aplicar filtro pasa-altos simple
        ruido_filtrado = ruido - np.roll(ruido, 1)
        
        return ruido_filtrado * 0.5
    
    @staticmethod
    def generar_sonido_voces(duracion=2, freq_muestreo=16000):
        """Simula sonido de voces."""
        n_samples = int(freq_muestreo * duracion)
        t = np.arange(n_samples) / freq_muestreo
        
        # Mezcla de frecuencias vocales
        sonido = (
            0.4 * np.sin(2 * np.pi * 500 * t) +
            0.3 * np.sin(2 * np.pi * 800 * t) +
            0.3 * np.sin(2 * np.pi * 1200 * t) +
            0.2 * np.random.randn(n_samples)
        )
        
        return sonido
    
    @staticmethod
    def generar_espectrograma(sonido, freq_muestreo=16000):
        """Genera espectrograma STFT."""
        f, t, Sxx = signal.spectrogram(
            sonido,
            fs=freq_muestreo,
            nperseg=512,
            noverlap=256
        )
        
        # Convertir a dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Redimensionar a 64x64
        Sxx_resized = np.resize(Sxx_db, (64, 64))
        
        return Sxx_resized
    
    @staticmethod
    def generar_dataset(n_samples_por_clase=100):
        """Genera dataset completo."""
        generadores = {
            'ruido': GeneradorEspectrogramas.generar_sonido_ruido,
            'trafico': GeneradorEspectrogramas.generar_sonido_trafico,
            'lluvia': GeneradorEspectrogramas.generar_sonido_lluvia,
            'voces': GeneradorEspectrogramas.generar_sonido_voces
        }
        
        X = []
        y = []
        
        for clase_id, (clase_nombre, generador) in enumerate(generadores.items()):
            for i in range(n_samples_por_clase):
                sonido = generador()
                espectrograma = GeneradorEspectrogramas.generar_espectrograma(sonido)
                X.append(espectrograma)
                y.append(clase_id)
        
        return {
            'X': np.array(X).reshape(-1, 64, 64, 1).astype(np.float32),
            'y': np.array(y),
            'clases': list(generadores.keys())
        }


class ClasificadorRuido:
    """Clasificador de ruido ambiental."""
    
    def __init__(self, seed=42):
        """Inicializa el clasificador."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.metricas = {}
        self.clases = None
    
    def construir_modelo(self, n_clases=4):
        """Construye CNN."""
        self.modelo = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_clases, activation='softmax')
        ])
        
        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo construido para {n_clases} clases")
    
    def entrenar(self, X_train, y_train, epochs=10):
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
        self.clases = clases
        
        pérdida, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=1)
        
        self.metricas = {
            'loss': float(pérdida),
            'accuracy': float(accuracy)
        }
        
        print(f"\n📊 Métricas:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Pérdida: {pérdida:.4f}")
        
        # Matriz de confusión
        if clases:
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n   Matriz de confusión:")
            print(f"        {'   '.join(f'{c[:5]:5s}' for c in clases)}")
            for i, fila in enumerate(cm):
                print(f"   {clases[i][:5]:5s} {' '.join(f'{x:5d}' for x in fila)}")
        
        return self.metricas
    
    def clasificar_audio(self, espectrograma, clases=None):
        """Clasifica un audio individual."""
        if len(espectrograma.shape) == 2:
            espectrograma = espectrograma.reshape(64, 64, 1)
        
        # Normalizar
        espectrograma = (espectrograma - espectrograma.mean()) / (espectrograma.std() + 1e-8)
        
        batch = espectrograma.reshape(1, 64, 64, 1).astype(np.float32)
        
        probabilidades = self.modelo.predict(batch, verbose=0)[0]
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
    print("🔊 CLASIFICADOR DE RUIDO AMBIENTAL - CNN")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando espectrogramas de audio...")
    datos = GeneradorEspectrogramas.generar_dataset(n_samples_por_clase=100)
    
    X = datos['X']
    y = datos['y']
    clases = datos['clases']
    
    print(f"✅ Dataset generado: {X.shape}")
    print(f"   Clases: {clases}")
    print(f"   Muestras por clase: {[(y==i).sum() for i in range(len(clases))]}")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(X_train.shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Paso 3: Construir
    print("\n[3] Construyendo modelo...")
    clasificador = ClasificadorRuido()
    clasificador.construir_modelo(n_clases=len(clases))
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando...")
    clasificador.entrenar(X_train, y_train, epochs=10)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    clasificador.evaluar(X_test, y_test, clases=clases)
    
    # Paso 6: Clasificar audios
    print("\n[6] Clasificando audios individuales:")
    for i in range(min(3, len(X_test))):
        resultado = clasificador.clasificar_audio(X_test[i], clases=clases)
        print(f"\n   Audio {i+1}:")
        print(f"     Clase: {resultado.get('clase_nombre', f'Clase {resultado['clase']}')} ({resultado['confianza']:.2%})")
    
    # Paso 7: Reporte
    print("\n[7] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'CNN Clasificador de Ruido',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'clases': clases,
        'metricas': clasificador.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
