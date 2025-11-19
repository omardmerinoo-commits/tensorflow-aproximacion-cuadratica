"""
Aplicación: Segmentador Semántico
================================

Caso de uso real: Segmentación pixel-por-pixel de imágenes

Características:
- Generación de imágenes segmentadas
- U-Net para segmentación semántica
- Análisis por píxel
- Métricas de segmentación

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json


class GeneradorMascaras:
    """Generador de imágenes con máscaras de segmentación."""
    
    @staticmethod
    def crear_mascara_vacia(tamaño=64):
        """Crea máscara vacía."""
        return np.zeros((tamaño, tamaño), dtype=np.uint8)
    
    @staticmethod
    def generar_imagen_fondo(tamaño=64, num_canales=3):
        """Crea imagen de fondo."""
        return np.random.rand(tamaño, tamaño, num_canales) * 0.2
    
    @staticmethod
    def generar_dataset_segmentacion(n_samples=200, tamaño=64):
        """Genera dataset de segmentación."""
        X = []
        y = []
        
        for i in range(n_samples):
            # Imagen de fondo
            imagen = GeneradorMascaras.generar_imagen_fondo(tamaño, num_canales=3)
            mascara = GeneradorMascaras.crear_mascara_vacia(tamaño)
            
            # Región 1: Cuadrado (clase 1)
            x1 = np.random.randint(5, tamaño - 30)
            y1 = np.random.randint(5, tamaño - 30)
            lado1 = np.random.randint(15, 25)
            
            imagen[y1:y1+lado1, x1:x1+lado1] = [0.8, 0.2, 0.2]  # Rojo
            mascara[y1:y1+lado1, x1:x1+lado1] = 1
            
            # Región 2: Círculo (clase 2)
            cx = np.random.randint(15, tamaño - 15)
            cy = np.random.randint(15, tamaño - 15)
            radio = np.random.randint(8, 15)
            
            y_circle, x_circle = np.ogrid[:tamaño, :tamaño]
            mascara_circle = (x_circle - cx)**2 + (y_circle - cy)**2 <= radio**2
            
            imagen[mascara_circle] = [0.2, 0.8, 0.2]  # Verde
            mascara[mascara_circle] = 2
            
            # Región 3: Triángulo (clase 3)
            x_tri = np.random.randint(20, tamaño - 20)
            y_tri = np.random.randint(20, tamaño - 20)
            size_tri = np.random.randint(8, 15)
            
            # Puntos del triángulo
            pts = np.array([
                [x_tri - size_tri, y_tri + size_tri],
                [x_tri + size_tri, y_tri + size_tri],
                [x_tri, y_tri - size_tri]
            ], dtype=np.int32)
            
            # Llenar triángulo (aproximado)
            y_min = max(0, np.min(pts[:, 1]))
            y_max = min(tamaño, np.max(pts[:, 1]))
            x_min = max(0, np.min(pts[:, 0]))
            x_max = min(tamaño, np.max(pts[:, 0]))
            
            imagen[y_min:y_max, x_min:x_max] = [0.2, 0.2, 0.8]  # Azul
            mascara[y_min:y_max, x_min:x_max] = 3
            
            X.append(imagen)
            y.append(mascara)
        
        return {
            'X': np.array(X).astype(np.float32),
            'y': np.array(y).astype(np.uint8),
            'clases': ['fondo', 'cuadrado', 'círculo', 'triángulo'],
            'colores': [[0, 0, 0], [204, 51, 51], [51, 204, 51], [51, 51, 204]]
        }
    
    @staticmethod
    def aplicar_colormap(mascara, colores=None):
        """Convierte máscara a imagen RGB."""
        if colores is None:
            colores = [
                [0, 0, 0], [204, 51, 51], [51, 204, 51], [51, 51, 204]
            ]
        
        imagen_rgb = np.zeros((mascara.shape[0], mascara.shape[1], 3), dtype=np.uint8)
        
        for clase_id, color in enumerate(colores):
            imagen_rgb[mascara == clase_id] = color
        
        return imagen_rgb


class SegmentadorSemantico:
    """Segmentador semántico con U-Net."""
    
    def __init__(self, seed=42):
        """Inicializa el segmentador."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.metricas = {}
        self.num_clases = 4
    
    def construir_modelo(self, tamaño=64, num_clases=4):
        """Construye U-Net simplificado."""
        self.num_clases = num_clases
        
        inputs = keras.Input(shape=(tamaño, tamaño, 3))
        
        # Encoder
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        # Bottleneck
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
        
        # Decoder
        u1 = layers.UpSampling2D((2, 2))(c3)
        u1 = layers.concatenate([u1, c2])
        d1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
        
        u2 = layers.UpSampling2D((2, 2))(d1)
        u2 = layers.concatenate([u2, c1])
        d2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
        
        # Output
        outputs = layers.Conv2D(num_clases, (1, 1), activation='softmax')(d2)
        
        self.modelo = keras.Model(inputs=inputs, outputs=outputs)
        
        self.modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ U-Net construido para {num_clases} clases")
    
    def entrenar(self, X_train, y_train, epochs=10):
        """Entrena el modelo."""
        self.modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test, y_test):
        """Evalúa el modelo."""
        pérdida, accuracy = self.modelo.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(self.modelo.predict(X_test, verbose=0), axis=-1)
        
        # IoU (Intersection over Union) por clase
        ious = []
        for clase_id in range(self.num_clases):
            interseccion = np.sum((y_pred == clase_id) & (y_test == clase_id))
            union = np.sum((y_pred == clase_id) | (y_test == clase_id))
            iou = interseccion / (union + 1e-8)
            ious.append(iou)
        
        self.metricas = {
            'loss': float(pérdida),
            'accuracy': float(accuracy),
            'mean_iou': float(np.mean(ious)),
            'iou_por_clase': [float(i) for i in ious]
        }
        
        print(f"\n📊 Métricas:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Mean IoU: {np.mean(ious):.4f}")
        
        for i, iou in enumerate(ious):
            print(f"   IoU Clase {i}: {iou:.4f}")
        
        return self.metricas
    
    def segmentar(self, imagen):
        """Segmenta una imagen."""
        batch = imagen.reshape(1, *imagen.shape).astype(np.float32)
        
        prediccion = self.modelo.predict(batch, verbose=0)[0]
        mascara = np.argmax(prediccion, axis=-1)
        
        # Confianza por píxel
        confianza = np.max(prediccion, axis=-1)
        
        return {
            'mascara': mascara.astype(int),
            'confianza_promedio': float(np.mean(confianza)),
            'confianza_min': float(np.min(confianza)),
            'confianza_max': float(np.max(confianza))
        }


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🎨 SEGMENTADOR SEMÁNTICO - U-NET")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando imágenes segmentadas...")
    datos = GeneradorMascaras.generar_dataset_segmentacion(n_samples=200, tamaño=64)
    
    X = datos['X']
    y = datos['y']
    clases = datos['clases']
    
    print(f"✅ Dataset generado: {X.shape}, Máscaras: {y.shape}")
    print(f"   Clases: {clases}")
    
    # Contar píxeles por clase
    for i, clase in enumerate(clases):
        count = np.sum(y == i)
        print(f"   {clase}: {count} píxeles")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalizar
    X_train = X_train / 1.0
    X_test = X_test / 1.0
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Paso 3: Construir
    print("\n[3] Construyendo U-Net...")
    segmentador = SegmentadorSemantico()
    segmentador.construir_modelo(tamaño=64, num_clases=len(clases))
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando...")
    segmentador.entrenar(X_train, y_train, epochs=10)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    segmentador.evaluar(X_test, y_test)
    
    # Paso 6: Segmentar
    print("\n[6] Segmentando imágenes:")
    for i in range(min(3, len(X_test))):
        resultado = segmentador.segmentar(X_test[i])
        print(f"\n   Imagen {i+1}:")
        print(f"     Confianza promedio: {resultado['confianza_promedio']:.4f}")
        print(f"     Rango confianza: [{resultado['confianza_min']:.4f}, {resultado['confianza_max']:.4f}]")
    
    # Paso 7: Reporte
    print("\n[7] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'U-Net Segmentador Semántico',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'clases': clases,
        'metricas': segmentador.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
