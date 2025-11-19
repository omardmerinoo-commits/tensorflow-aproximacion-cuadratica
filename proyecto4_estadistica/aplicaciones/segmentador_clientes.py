"""
Aplicación: Segmentación de Clientes
====================================

Caso de uso real: Clustering de clientes por comportamiento de compra para marketing dirigido.

Características:
- Segmentación automática
- Perfiles de clientes
- Métricas de clustering
- Estrategias de marketing

Autor: Proyecto TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from datetime import datetime
from pathlib import Path
import json


class GeneradorDatosClientes:
    """Generador de datos de clientes."""
    
    def __init__(self, seed=42):
        """Inicializa el generador."""
        np.random.seed(seed)
        self.seed = seed
    
    def generar_dataset(self, n_samples=300):
        """
        Genera dataset de comportamiento de clientes.
        
        Args:
            n_samples: Número de clientes
        
        Returns:
            DataFrame con gasto anual y frecuencia de compra
        """
        # Crear 3 segmentos naturales
        n_per_segment = n_samples // 3
        
        # Segmento 1: Clientes ocasionales, bajo gasto
        seg1 = np.random.normal([500, 5], [200, 2], (n_per_segment, 2))
        
        # Segmento 2: Clientes regulares, gasto medio
        seg2 = np.random.normal([2000, 20], [400, 5], (n_per_segment, 2))
        
        # Segmento 3: VIP, alto gasto
        seg3 = np.random.normal([5000, 50], [1000, 10], (n_per_segment + n_samples % 3, 2))
        
        # Combinar
        X = np.vstack([seg1, seg2, seg3])
        X = np.abs(X)  # Asegurar valores positivos
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X = X[idx]
        
        return {
            'X': X,
            'features': ['Gasto Anual ($)', 'Frecuencia Compra (veces/año)']
        }


class SegmentadorClientes:
    """Segmentador de clientes usando K-Means."""
    
    def __init__(self, n_clusters=3, seed=42):
        """Inicializa el segmentador."""
        np.random.seed(seed)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        self.labels = None
        self.centros = None
        self.metricas = {}
    
    def fit(self, X):
        """
        Entrena modelo K-Means.
        
        Args:
            X: Features (n_samples, n_features)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        self.labels = self.kmeans.labels_
        
        # Obtener centros en escala original
        self.centros = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        print(f"✅ Segmentación completada ({self.n_clusters} clusters)")
        print(f"   Inercia: {self.kmeans.inertia_:.2f}")
    
    def evaluar(self, X):
        """Evalúa la segmentación."""
        X_scaled = self.scaler.transform(X)
        
        # Silhueta (mayor es mejor, rango: [-1, 1])
        silhueta = silhouette_score(X_scaled, self.labels)
        
        # Davies-Bouldin (menor es mejor)
        davies_bouldin = davies_bouldin_score(X_scaled, self.labels)
        
        self.metricas = {
            'silhouette_score': float(silhueta),
            'davies_bouldin_score': float(davies_bouldin),
            'inertia': float(self.kmeans.inertia_)
        }
        
        print(f"\n📊 Métricas de clustering:")
        print(f"   Silhueta:      {silhueta:.4f}")
        print(f"   Davies-Bouldin: {davies_bouldin:.4f}")
        print(f"   Inercia:       {self.kmeans.inertia_:.2f}")
        
        return self.metricas
    
    def obtener_perfiles(self, X):
        """
        Obtiene perfil de cada cluster.
        
        Args:
            X: Features originales
        
        Returns:
            Dict con características de cada cluster
        """
        perfiles = {}
        
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_data = X[mask]
            
            perfiles[f'Segmento {i}'] = {
                'tamaño': int(mask.sum()),
                'porcentaje': float(100 * mask.sum() / len(X)),
                'gasto_promedio': float(cluster_data[:, 0].mean()),
                'frecuencia_promedio': float(cluster_data[:, 1].mean()),
                'centro': {
                    'gasto': float(self.centros[i, 0]),
                    'frecuencia': float(self.centros[i, 1])
                }
            }
        
        print(f"\n👥 Perfiles de segmentos:")
        for seg, prof in perfiles.items():
            print(f"\n   {seg}:")
            print(f"     Tamaño: {prof['tamaño']} clientes ({prof['porcentaje']:.1f}%)")
            print(f"     Gasto promedio: ${prof['gasto_promedio']:.2f}")
            print(f"     Frecuencia: {prof['frecuencia_promedio']:.1f} compras/año")
        
        return perfiles
    
    def recomendar_estrategia(self, X):
        """Recomienda estrategias de marketing."""
        perfiles = self.obtener_perfiles(X)
        
        print(f"\n🎯 Estrategias de marketing recomendadas:")
        
        estrategias = []
        for i, (seg, prof) in enumerate(perfiles.items()):
            gasto = prof['gasto_promedio']
            freq = prof['frecuencia_promedio']
            
            if gasto < 1000:
                estrategia = "Promociones frecuentes, descuentos por volumen"
            elif gasto < 3500:
                estrategia = "Programa de puntos, ofertas personalizadas"
            else:
                estrategia = "Servicio VIP, ofertas exclusivas, relación personalizada"
            
            print(f"\n   {seg}:")
            print(f"     → {estrategia}")
            
            estrategias.append({
                'segmento': seg,
                'estrategia': estrategia,
                'perfil': prof
            })
        
        return estrategias
    
    def asignar_cliente(self, gasto, frecuencia):
        """
        Asigna nuevo cliente a un segmento.
        
        Args:
            gasto: Gasto estimado
            frecuencia: Frecuencia de compra estimada
        
        Returns:
            ID del segmento
        """
        X_new = np.array([[gasto, frecuencia]])
        X_new_scaled = self.scaler.transform(X_new)
        
        segmento = self.kmeans.predict(X_new_scaled)[0]
        distancia = np.min(np.linalg.norm(X_new_scaled - self.kmeans.cluster_centers_, axis=1))
        
        return {
            'cliente': {'gasto': gasto, 'frecuencia': frecuencia},
            'segmento': int(segmento),
            'distancia_centro': float(distancia)
        }


def encontrar_k_optimo(X, k_range=range(2, 11)):
    """Encuentra número óptimo de clusters."""
    print(f"\n🔍 Encontrando k óptimo...")
    
    silhuetas = []
    db_scores = []
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhueta = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        
        silhuetas.append(silhueta)
        db_scores.append(db)
        
        print(f"   k={k}: Silhueta={silhueta:.4f}, DB={db:.4f}")
    
    k_optimo = list(k_range)[np.argmax(silhuetas)]
    print(f"\n   ✅ k óptimo: {k_optimo}")
    
    return k_optimo


def main():
    """Demostración."""
    print("\n" + "="*80)
    print("👥 SEGMENTACIÓN DE CLIENTES - K-MEANS CLUSTERING")
    print("="*80)
    
    # Paso 1: Generar datos
    print("\n[1] Generando datos de clientes...")
    generador = GeneradorDatosClientes(seed=42)
    datos = generador.generar_dataset(n_samples=300)
    
    X = datos['X']
    print(f"✅ Dataset generado: {len(X)} clientes")
    print(f"   Gasto: [${X[:, 0].min():.0f}, ${X[:, 0].max():.0f}]")
    print(f"   Frecuencia: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")
    
    # Paso 2: Encontrar k óptimo
    print("\n[2] Buscando número óptimo de clusters...")
    k_optimo = encontrar_k_optimo(X, k_range=range(2, 8))
    
    # Paso 3: Segmentar
    print("\n[3] Segmentando clientes...")
    segmentador = SegmentadorClientes(n_clusters=k_optimo)
    segmentador.fit(X)
    
    # Paso 4: Evaluar
    print("\n[4] Evaluando segmentación...")
    segmentador.evaluar(X)
    
    # Paso 5: Perfiles
    print("\n[5] Analizando perfiles...")
    segmentador.obtener_perfiles(X)
    
    # Paso 6: Estrategias
    print("\n[6] Recomendando estrategias...")
    estrategias = segmentador.recomendar_estrategia(X)
    
    # Paso 7: Asignar nuevos clientes
    print("\n[7] Asignando nuevos clientes:")
    nuevos_clientes = [
        (300, 3),      # Cliente ocasional
        (1500, 18),    # Cliente regular
        (4500, 45),    # VIP
    ]
    
    for gasto, freq in nuevos_clientes:
        resultado = segmentador.asignar_cliente(gasto, freq)
        print(f"\n   Cliente: ${resultado['cliente']['gasto']}, {resultado['cliente']['frecuencia']} compras")
        print(f"   → Asignado a: Segmento {resultado['segmento']}")
    
    # Paso 8: Reporte
    print("\n[8] Generando reporte...")
    output_dir = Path(__file__).parent / 'reportes'
    output_dir.mkdir(exist_ok=True)
    
    reporte = {
        'fecha': datetime.now().isoformat(),
        'modelo': 'K-Means Clustering',
        'muestras': len(X),
        'clusters': k_optimo,
        'metricas': segmentador.metricas
    }
    
    with open(output_dir / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"✅ Reporte generado")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
