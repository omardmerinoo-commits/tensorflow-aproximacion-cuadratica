"""
Aplicación: Compresión de Imágenes con PCA
==========================================

Caso de uso real: Reducción de dimensionalidad para compresión de imágenes y visualización.

Características:
- Compresión de imágenes
- Reconstrucción con pérdida controlada
- Varianza explicada
- Visualización 2D

Autor: Proyecto TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import json


class GeneradorImagenesComun:
    """Generador de imágenes sintéticas comunes."""
    
    @staticmethod
    def generar_imagen_ruido(tamaño=64, complejidad=0.5, seed=42):
        """Genera imagen de ruido Perlin simulado."""
        np.random.seed(seed)
        # Ruido Gaussiano filtrado
        img = np.random.randn(tamaño, tamaño)
        
        # Aplicar promedio móvil para suavizar
        for _ in range(int(complejidad * 3)):
            img = (img + np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0)) / 3
            img = (img + np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)) / 3
        
        # Normalizar a [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    @staticmethod
    def generar_imagen_gradiente(tamaño=64, tipo='horizontal'):
        """Genera imagen con gradiente."""
        if tipo == 'horizontal':
            img = np.linspace(0, 1, tamaño)[np.newaxis, :].repeat(tamaño, axis=0)
        else:  # vertical
            img = np.linspace(0, 1, tamaño)[:, np.newaxis].repeat(tamaño, axis=1)
        return img
    
    @staticmethod
    def generar_imagen_formas(tamaño=64):
        """Genera imagen con formas geométricas."""
        img = np.zeros((tamaño, tamaño))
        
        # Círculo
        y, x = np.ogrid[:tamaño, :tamaño]
        circle = (x - tamaño//4) ** 2 + (y - tamaño//4) ** 2 <= (tamaño//8) ** 2
        img[circle] = 1
        
        # Rectángulo
        img[tamaño//2:3*tamaño//4, tamaño//2:3*tamaño//4] = 0.5
        
        return img


class CompresorImagenPCA:
    """Compresor de imágenes usando PCA."""
    
    def __init__(self, n_components=None, seed=42):
        """Inicializa el compresor."""
        np.random.seed(seed)
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.metricas = {}
    
    def generar_dataset(self, n_imagenes=100, tamaño_img=32):
        """
        Genera dataset de imágenes.
        
        Args:
            n_imagenes: Número de imágenes
            tamaño_img: Tamaño de cada imagen
        
        Returns:
            Array (n_imagenes, tamaño_img * tamaño_img)
        """
        generador = GeneradorImagenesComun()
        X = []
        
        for i in range(n_imagenes):
            # Alternar tipos de imágenes
            tipo = i % 3
            
            if tipo == 0:
                img = generador.generar_imagen_ruido(tamaño_img, seed=42+i)
            elif tipo == 1:
                img = generador.generar_imagen_gradiente(tamaño_img, tipo='horizontal')
            else:
                img = generador.generar_imagen_formas(tamaño_img)
            
            X.append(img.flatten())
        
        return np.array(X)
    
    def fit(self, X):
        """
        Entrena PCA.
        
        Args:
            X: Imágenes aplanadas (n_samples, n_features)
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Si n_components no se especifica, usar 95% varianza
        if self.n_components is None:
            self.pca = PCA(n_components=0.95)
        else:
            self.pca = PCA(n_components=self.n_components)
        
        self.pca.fit(X_scaled)
        
        print(f"✅ PCA entrenado")
        print(f"   Componentes: {self.pca.n_components_}")
        print(f"   Varianza explicada acumulada: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def transformar(self, X):
        """Transforma a espacio comprimido."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def reconstruir(self, X):
        """Reconstruye imágenes desde espacio comprimido."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_reconstructed)
    
    def evaluar_compresion(self, X_original, X_reconstruida):
        """Evalúa compresión."""
        # Error cuadrático medio
        mse = np.mean((X_original - X_reconstruida) ** 2)
        
        # Error máximo absoluto
        mae = np.mean(np.abs(X_original - X_reconstruida))
        
        # Ratio de compresión
        original_dim = X_original.shape[1]
        compressed_dim = self.pca.n_components_
        ratio = original_dim / compressed_dim
        
        self.metricas = {
            'mse': float(mse),
            'mae': float(mae),
            'compression_ratio': float(ratio),
            'original_dimensions': int(original_dim),
            'compressed_dimensions': int(compressed_dim),
            'variance_explained': float(self.pca.explained_variance_ratio_.sum())
        }
        
        print(f"\n📊 Métricas de compresión:")
        print(f"   Dimensiones originales: {original_dim}")
        print(f"   Dimensiones comprimidas: {compressed_dim}")
        print(f"   Ratio de compresión: {ratio:.2f}x")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   Varianza explicada: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return self.metricas
    
    def obtener_varianza_explicada(self):
        """Obtiene varianza explicada acumulada."""
        varianza_acumulada = np.cumsum(self.pca.explained_variance_ratio_)
        
        print(f"\n📊 Varianza explicada acumulada:")
        for i in [1, 5, 10, 20, 50]:
            if i <= len(varianza_acumulada):
                print(f"   Primeros {i:2d} componentes: {varianza_acumulada[i-1]:.4f}")
        
        return varianza_acumulada
    
    def obtener_componentes_principales(self, n=5):
        """Visualiza componentes principales."""
        print(f"\n📊 Primeros {n} componentes principales:")
        
        componentes = self.pca.components_[:n]
        
        for i, comp in enumerate(componentes):
            norm = np.linalg.norm(comp)
            varianza = self.pca.explained_variance_ratio_[i]
            print(f"   Componente {i+1}: Varianza={varianza:.4f}, Norma={norm:.4f}")
        
        return componentes


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("🖼️  COMPRESIÓN DE IMÁGENES CON PCA - REDUCCIÓN DIMENSIONAL")
    print("="*80)
    
    # Paso 1: Generar imágenes
    print("\n[1] Generando dataset de imágenes...")
    compresor = CompresorImagenPCA()
    X = compresor.generar_dataset(n_imagenes=100, tamaño_img=32)
    
    print(f"✅ Dataset generado: {X.shape}")
    print(f"   Imágenes: {X.shape[0]}")
    print(f"   Píxeles por imagen: {X.shape[1]} ({int(np.sqrt(X.shape[1]))}×{int(np.sqrt(X.shape[1]))})")
    
    # Paso 2: Entrenar PCA con diferentes números de componentes
    print("\n[2] Comparando diferentes números de componentes...")
    
    ratios = []
    mses = []
    
    for n_comp in [5, 10, 20, 50, 100]:
        compresor_temp = CompresorImagenPCA(n_components=n_comp)
        compresor_temp.fit(X)
        
        X_reconstructed = compresor_temp.reconstruir(X)
        mse = np.mean((X - X_reconstructed) ** 2)
        
        ratio = X.shape[1] / n_comp
        ratios.append(ratio)
        mses.append(mse)
        
        print(f"   {n_comp:3d} componentes: Ratio={ratio:.2f}x, MSE={mse:.6f}")
    
    # Paso 3: Entrenar con 95% varianza
    print("\n[3] Entrenando PCA (95% varianza)...")
    compresor.fit(X)
    
    # Paso 4: Evaluar
    print("\n[4] Evaluando compresión...")
    X_reconstructed = compresor.reconstruir(X)
    compresor.evaluar_compresion(X, X_reconstructed)
    
    # Paso 5: Varianza explicada
    print("\n[5] Analizando varianza explicada...")
    compresor.obtener_varianza_explicada()
    
    # Paso 6: Componentes principales
    print("\n[6] Analizando componentes principales...")
    compresor.obtener_componentes_principales(n=5)
    
    # Paso 7: Visualizar imágenes comprimidas
    print("\n[7] Generando visualizaciones...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Compresión de Imágenes con PCA')
    
    # Mostrar 3 imágenes originales vs reconstruidas
    for i in range(3):
        # Original
        img_orig = X[i].reshape(32, 32)
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruida
        img_recon = X_reconstructed[i].reshape(32, 32)
        axes[1, i].imshow(img_recon, cmap='gray')
        axes[1, i].set_title(f'Reconstruida {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"compresion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=100)
    plt.close()
    
    print(f"✅ Visualización guardada")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'PCA Compresión de Imágenes',
        'muestras': len(X),
        'metricas': compresor.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
