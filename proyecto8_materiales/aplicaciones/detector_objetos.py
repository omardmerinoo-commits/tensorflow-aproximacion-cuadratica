"""
Aplicación: Detector de Objetos
==============================

Caso de uso real: Detección y localización de objetos en imágenes

Características:
- Generación de imágenes sintéticas con objetos
- Detección con YOLO-style arquitectura
- Localización con bounding boxes
- Análisis de confianza

Autor: Proyecto TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from pathlib import Path
import json


class GeneradorObjetos:
    """Generador de imágenes con objetos."""
    
    @staticmethod
    def crear_imagen_fondo(tamaño=128):
        """Crea imagen de fondo."""
        return np.random.rand(tamaño, tamaño, 3) * 0.3  # Fondo oscuro
    
    @staticmethod
    def dibujar_rectangulo(imagen, x1, y1, x2, y2, color):
        """Dibuja un rectángulo."""
        imagen[int(y1):int(y2), int(x1):int(x2)] = color
    
    @staticmethod
    def generar_imagen_con_circulo(tamaño=128):
        """Genera imagen con círculo."""
        imagen = GeneradorObjetos.crear_imagen_fondo(tamaño)
        
        # Círculo aleatorio
        cx = np.random.randint(30, tamaño - 30)
        cy = np.random.randint(30, tamaño - 30)
        radio = np.random.randint(15, 30)
        
        color = np.random.rand(3)
        
        y, x = np.ogrid[:tamaño, :tamaño]
        mascara = (x - cx)**2 + (y - cy)**2 <= radio**2
        imagen[mascara] = color
        
        # Bbox normalizado [cx, cy, w, h] en rango [0, 1]
        bbox = [cx/tamaño, cy/tamaño, (2*radio)/tamaño, (2*radio)/tamaño]
        
        return imagen, bbox, 'circle'
    
    @staticmethod
    def generar_imagen_con_cuadrado(tamaño=128):
        """Genera imagen con cuadrado."""
        imagen = GeneradorObjetos.crear_imagen_fondo(tamaño)
        
        # Cuadrado aleatorio
        lado = np.random.randint(20, 50)
        x1 = np.random.randint(10, tamaño - lado - 10)
        y1 = np.random.randint(10, tamaño - lado - 10)
        x2 = x1 + lado
        y2 = y1 + lado
        
        color = np.random.rand(3)
        GeneradorObjetos.dibujar_rectangulo(imagen, x1, y1, x2, y2, color)
        
        # Bbox normalizado
        bbox = [(x1+x2)/(2*tamaño), (y1+y2)/(2*tamaño), lado/tamaño, lado/tamaño]
        
        return imagen, bbox, 'square'
    
    @staticmethod
    def generar_imagen_con_triangulo(tamaño=128):
        """Genera imagen con triángulo."""
        imagen = GeneradorObjetos.crear_imagen_fondo(tamaño)
        
        # Triángulo aleatorio
        x_base = np.random.randint(30, tamaño - 30)
        y_base = np.random.randint(30, tamaño - 30)
        lado = np.random.randint(20, 40)
        
        # Puntos del triángulo
        pts = np.array([
            [x_base - lado//2, y_base + lado],
            [x_base + lado//2, y_base + lado],
            [x_base, y_base - lado//2]
        ], dtype=np.int32)
        
        # Llenar triángulo
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        
        color = np.random.rand(3)
        GeneradorObjetos.dibujar_rectangulo(imagen, x_min, y_min, x_max, y_max, color)
        
        # Bbox normalizado
        bbox = [(x_min+x_max)/(2*tamaño), (y_min+y_max)/(2*tamaño), 
                (x_max-x_min)/tamaño, (y_max-y_min)/tamaño]
        
        return imagen, bbox, 'triangle'
    
    @staticmethod
    def generar_dataset(n_samples=300, tamaño=128):
        """Genera dataset completo."""
        generadores = [
            GeneradorObjetos.generar_imagen_con_circulo,
            GeneradorObjetos.generar_imagen_con_cuadrado,
            GeneradorObjetos.generar_imagen_con_triangulo
        ]
        
        X = []
        y_bbox = []
        y_clase = []
        
        for i in range(n_samples):
            gen = np.random.choice(generadores)
            imagen, bbox, clase = gen(tamaño)
            
            X.append(imagen)
            y_bbox.append(bbox)
            y_clase.append(['circle', 'square', 'triangle'].index(clase))
        
        return {
            'X': np.array(X).astype(np.float32),
            'y_bbox': np.array(y_bbox).astype(np.float32),
            'y_clase': np.array(y_clase),
            'clases': ['círculo', 'cuadrado', 'triángulo']
        }


class DetectorObjetos:
    """Detector de objetos."""
    
    def __init__(self, seed=42):
        """Inicializa el detector."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.modelo = None
        self.metricas = {}
    
    def construir_modelo(self, tamaño_imagen=128):
        """Construye CNN para detección."""
        inputs = keras.Input(shape=(tamaño_imagen, tamaño_imagen, 3))
        
        # Backbone CNN
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Ramas de salida
        # Rama 1: Bbox (cx, cy, w, h)
        bbox_branch = layers.Dense(64, activation='relu')(x)
        bbox_out = layers.Dense(4, activation='sigmoid', name='bbox')(bbox_branch)
        
        # Rama 2: Clasificación de objetos
        class_branch = layers.Dense(64, activation='relu')(x)
        class_out = layers.Dense(3, activation='softmax', name='class')(class_branch)
        
        self.modelo = keras.Model(inputs=inputs, outputs=[bbox_out, class_out])
        
        self.modelo.compile(
            optimizer='adam',
            loss={'bbox': 'mse', 'class': 'sparse_categorical_crossentropy'},
            metrics={'bbox': 'mae', 'class': 'accuracy'}
        )
        
        print(f"✅ Modelo de detección construido")
    
    def entrenar(self, X_train, y_bbox_train, y_clase_train, epochs=10):
        """Entrena el modelo."""
        self.modelo.fit(
            X_train,
            {'bbox': y_bbox_train, 'class': y_clase_train},
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado")
    
    def evaluar(self, X_test, y_bbox_test, y_clase_test):
        """Evalúa el modelo."""
        bbox_pred, clase_pred = self.modelo.predict(X_test, verbose=0)
        
        # Error de bbox
        bbox_mae = np.mean(np.abs(bbox_pred - y_bbox_test))
        
        # Accuracy de clasificación
        clase_predicha = np.argmax(clase_pred, axis=1)
        class_acc = np.mean(clase_predicha == y_clase_test)
        
        self.metricas = {
            'bbox_mae': float(bbox_mae),
            'class_accuracy': float(class_acc)
        }
        
        print(f"\n📊 Métricas:")
        print(f"   Bbox MAE: {bbox_mae:.4f}")
        print(f"   Precisión clasificación: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        return self.metricas
    
    def detectar(self, imagen, clases=None):
        """Detecta objetos en una imagen."""
        batch = imagen.reshape(1, *imagen.shape).astype(np.float32)
        
        bbox, clase_probs = self.modelo.predict(batch, verbose=0)
        
        bbox = bbox[0]
        clase_predicha = np.argmax(clase_probs[0])
        confianza = float(clase_probs[0][clase_predicha])
        
        resultado = {
            'bbox': {
                'cx': float(bbox[0]),
                'cy': float(bbox[1]),
                'w': float(bbox[2]),
                'h': float(bbox[3])
            },
            'clase': int(clase_predicha),
            'confianza': confianza,
            'probabilidades': {
                f"{clases[i] if clases else f'Objeto {i}'}": float(p)
                for i, p in enumerate(clase_probs[0])
            }
        }
        
        if clases:
            resultado['clase_nombre'] = clases[clase_predicha]
        
        return resultado


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🎯 DETECTOR DE OBJETOS - CNN")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando imágenes con objetos...")
    datos = GeneradorObjetos.generar_dataset(n_samples=300, tamaño=128)
    
    X = datos['X']
    y_bbox = datos['y_bbox']
    y_clase = datos['y_clase']
    clases = datos['clases']
    
    print(f"✅ Dataset generado: {X.shape}")
    print(f"   Clases: {clases}")
    print(f"   Muestras por clase: {[(y_clase==i).sum() for i in range(len(clases))]}")
    
    # Paso 2: Split
    print("\n[2] División train/test...")
    X_train, X_test, y_bbox_train, y_bbox_test, y_clase_train, y_clase_test = train_test_split(
        X, y_bbox, y_clase, test_size=0.2, random_state=42, stratify=y_clase
    )
    
    # Normalizar imágenes
    X_train = X_train / 1.0
    X_test = X_test / 1.0
    
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Paso 3: Construir
    print("\n[3] Construyendo modelo...")
    detector = DetectorObjetos()
    detector.construir_modelo(tamaño_imagen=128)
    
    # Paso 4: Entrenar
    print("\n[4] Entrenando...")
    detector.entrenar(X_train, y_bbox_train, y_clase_train, epochs=10)
    
    # Paso 5: Evaluar
    print("\n[5] Evaluando...")
    detector.evaluar(X_test, y_bbox_test, y_clase_test)
    
    # Paso 6: Detectar
    print("\n[6] Detectando objetos:")
    for i in range(min(3, len(X_test))):
        resultado = detector.detectar(X_test[i], clases=clases)
        print(f"\n   Imagen {i+1}:")
        print(f"     Objeto: {resultado.get('clase_nombre', f'Objeto {resultado['clase']}')}")
        print(f"     Confianza: {resultado['confianza']:.2%}")
        print(f"     Bbox: ({resultado['bbox']['cx']:.3f}, {resultado['bbox']['cy']:.3f}, "
              f"{resultado['bbox']['w']:.3f}, {resultado['bbox']['h']:.3f})")
    
    # Paso 7: Reporte
    print("\n[7] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'CNN Detector de Objetos',
        'dataset': f"{len(X_train)} entrenamientos, {len(X_test)} tests",
        'clases': clases,
        'metricas': detector.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
