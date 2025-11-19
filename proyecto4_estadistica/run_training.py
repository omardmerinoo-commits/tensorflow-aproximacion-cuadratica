"""
Script de Entrenamiento Completo - Proyecto 4: Análisis Estadístico
====================================================================

Este script ejecuta un flujo completo de 9 pasos demostrando todas
las capacidades del AnalizadorEstadistico.

Pasos:
1. Generar datos sintéticos con clusters
2. Cargar y explorar estadísticas
3. PCA: Reducción de dimensionalidad
4. K-Means: Clustering de particionamiento
5. Clustering Jerárquico: Análisis de similitud
6. GMM: Modelado probabilístico
7. Autoencoder: Reducción neural
8. Evaluación: Métricas de validación
9. Persistencia: Guardar/cargar modelos
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from analizador_estadistico import AnalizadorEstadistico


def linea_separador(titulo=""):
    """Imprime línea separadora con título."""
    if titulo:
        print(f"\n{'='*70}")
        print(f"  {titulo}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def paso_1_generar_datos():
    """Paso 1: Generar datos sintéticos con clusters."""
    linea_separador("PASO 1: GENERAR DATOS SINTÉTICOS")
    
    from sklearn.datasets import make_blobs
    
    print("Generando 300 muestras con 15 características y 3 clusters...")
    
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=15,
        cluster_std=0.6,
        random_state=42
    )
    
    print(f"✓ Datos generados: {X.shape}")
    print(f"  - Muestras: {X.shape[0]}")
    print(f"  - Características: {X.shape[1]}")
    print(f"  - Clusters reales: {len(np.unique(y_true))}")
    print(f"  - Rango de valores: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y_true


def paso_2_estadisticas(analizador, X):
    """Paso 2: Cargar datos y explorar estadísticas."""
    linea_separador("PASO 2: ESTADÍSTICAS DESCRIPTIVAS")
    
    print("Cargando datos y estandarizando...")
    X_orig, X_est = analizador.cargar_datos(X)
    print("✓ Datos cargados y estandarizados")
    
    print("\nCalculando estadísticas descriptivas...")
    stats = analizador.estadisticas_descriptivas()
    
    print("\nEjemplos de estadísticas (primeras 5 características):")
    for i in range(min(5, len(stats))):
        stat = stats[i]
        print(f"  Característica {i+1}:")
        print(f"    Media: {stat['media']:.4f}")
        print(f"    Std: {stat['std']:.4f}")
        print(f"    Min: {stat['min']:.4f}")
        print(f"    Max: {stat['max']:.4f}")
    
    print("\nCalculando matriz de correlación...")
    corr = analizador.matriz_correlacion()
    print(f"✓ Matriz de correlación {corr.shape}")
    print(f"  Correlación máxima fuera diagonal: {np.max(np.abs(corr - np.eye(corr.shape[0]))):.4f}")
    
    print("\nDetectando outliers...")
    outliers = analizador.deteccion_outliers(metodo='zscore', umbral=3)
    print(f"✓ Outliers detectados: {len(outliers)} ({100*len(outliers)/len(X):.2f}%)")


def paso_3_pca(analizador):
    """Paso 3: PCA - Reducción de dimensionalidad."""
    linea_separador("PASO 3: PCA - ANÁLISIS DE COMPONENTES PRINCIPALES")
    
    print("Seleccionando número óptimo de componentes mediante método del codo...")
    n_opt = analizador.codo_pca()
    print(f"✓ Componentes óptimos: {n_opt}")
    
    print(f"\nAplicando PCA con {n_opt} componentes...")
    X_pca, var_exp, var_acum = analizador.pca(n_componentes=n_opt)
    print(f"✓ PCA aplicado: {X_pca.shape}")
    
    print(f"\nVarianza explicada por componente:")
    for i, v in enumerate(var_exp):
        barra = '█' * int(30 * v / var_exp.max())
        print(f"  PC{i+1}: {v:.4f} {barra}")
    
    print(f"\nVarianza acumulada:")
    for i, v in enumerate(var_acum):
        print(f"  Primeros {i+1} componentes: {v:.4f} ({100*v:.2f}%)")


def paso_4_kmeans(analizador, X):
    """Paso 4: K-Means clustering."""
    linea_separador("PASO 4: K-MEANS CLUSTERING")
    
    print("Evaluando diferentes valores de k mediante método del codo...")
    inercias = analizador.metodo_codo(k_max=8)
    
    print("\nInercia para diferentes k:")
    for k, inercia in enumerate(inercias, 1):
        barra = '█' * int(30 * inercia / inercias[0])
        print(f"  k={k}: {inercia:.2f} {barra}")
    
    print(f"\nAplicando K-Means con k=3...")
    etiquetas_km, centros, inercia = analizador.kmeans(n_clusters=3)
    print(f"✓ K-Means aplicado")
    print(f"  - Inercia final: {inercia:.2f}")
    print(f"  - Centros: {centros.shape}")
    
    print(f"\nDistribución de muestras por cluster:")
    for k in range(3):
        n_muestras = np.sum(etiquetas_km == k)
        porcentaje = 100 * n_muestras / len(etiquetas_km)
        barra = '█' * int(30 * n_muestras / len(etiquetas_km))
        print(f"  Cluster {k}: {n_muestras:3d} muestras ({porcentaje:5.2f}%) {barra}")
    
    # Evaluar
    silhueta = analizador.score_silhueta(etiquetas_km)
    db = analizador.indice_davies_bouldin(etiquetas_km)
    print(f"\nMétricas de validación:")
    print(f"  Silhueta: {silhueta:.4f} (rango: -1 a 1)")
    print(f"  Davies-Bouldin: {db:.4f} (menor es mejor)")


def paso_5_jerarquico(analizador):
    """Paso 5: Clustering jerárquico."""
    linea_separador("PASO 5: CLUSTERING JERÁRQUICO")
    
    print("Aplicando clustering jerárquico con enlace Ward...")
    inicio = time.time()
    etiquetas_jq, Z = analizador.clustering_jerarquico(metodo='ward')
    tiempo = time.time() - inicio
    
    print(f"✓ Clustering jerárquico aplicado en {tiempo:.4f}s")
    print(f"  - Matriz de enlaces: {Z.shape}")
    print(f"  - Etiquetas únicas: {len(np.unique(etiquetas_jq))}")
    
    print(f"\nDistribución por cluster:")
    for k in np.unique(etiquetas_jq):
        n_muestras = np.sum(etiquetas_jq == k)
        print(f"  Cluster {k}: {n_muestras:3d} muestras")


def paso_6_gmm(analizador):
    """Paso 6: Gaussian Mixture Model."""
    linea_separador("PASO 6: GAUSSIAN MIXTURE MODEL (GMM)")
    
    print("Seleccionando número óptimo de componentes...")
    n_opt = analizador.seleccionar_componentes_gmm(n_max=6)
    print(f"✓ Componentes óptimos: {n_opt}")
    
    print(f"\nAplicando GMM con {n_opt} componentes...")
    etiquetas_gmm, probs, bic = analizador.gmm(n_componentes=n_opt)
    print(f"✓ GMM aplicado")
    print(f"  - Probabilidades: {probs.shape}")
    print(f"  - BIC: {bic:.4f}")
    
    print(f"\nEjemplos de probabilidades de componentes (primeras 5 muestras):")
    for i in range(5):
        print(f"  Muestra {i+1}: {probs[i]}")
    
    print(f"\nDistribución por componente:")
    for k in range(n_opt):
        n_muestras = np.sum(etiquetas_gmm == k)
        print(f"  Componente {k}: {n_muestras:3d} muestras")


def paso_7_autoencoder(analizador, X):
    """Paso 7: Autoencoder - Reducción neural."""
    linea_separador("PASO 7: AUTOENCODER - REDUCCIÓN NEURAL")
    
    print("Construyendo arquitectura del autoencoder...")
    modelo = analizador.construir_autoencoder(
        dim_entrada=15,
        dim_latente=5,
        capas_ocultas=[32, 16]
    )
    print(f"✓ Autoencoder construido")
    print(f"  Arquitectura:")
    print(f"    - Entrada: 15 características")
    print(f"    - Capa oculta 1: 32 unidades")
    print(f"    - Capa oculta 2: 16 unidades")
    print(f"    - Espacio latente: 5 dimensiones")
    print(f"    - Salida: 15 características (reconstrucción)")
    
    print(f"\nEntrenando autoencoder (20 épocas)...")
    inicio = time.time()
    historial = analizador.entrenar_autoencoder(
        epochs=20,
        batch_size=32,
        verbose=0
    )
    tiempo = time.time() - inicio
    print(f"✓ Entrenamiento completado en {tiempo:.2f}s")
    
    print(f"\nProgreso del entrenamiento:")
    print(f"  Pérdida inicial: {historial['loss'][0]:.4f}")
    print(f"  Pérdida final: {historial['loss'][-1]:.4f}")
    print(f"  Reducción: {100*(1 - historial['loss'][-1]/historial['loss'][0]):.2f}%")
    
    print(f"\nCodificando datos al espacio latente...")
    X_latente = analizador.codificar()
    print(f"✓ Datos codificados: {X_latente.shape}")
    print(f"  Media del espacio latente: {np.mean(X_latente, axis=0)}")


def paso_8_evaluacion(analizador):
    """Paso 8: Evaluación completa."""
    linea_separador("PASO 8: EVALUACIÓN COMPLETA")
    
    print("Re-aplicando K-Means para evaluación final...")
    etiquetas, _, _ = analizador.kmeans(n_clusters=3)
    
    print("\nCalculando métricas de validación:")
    
    silhueta = analizador.score_silhueta(etiquetas)
    print(f"  Índice de Silhueta: {silhueta:.4f}")
    if silhueta < 0.2:
        calidad = "Pobre"
    elif silhueta < 0.5:
        calidad = "Moderada"
    elif silhueta < 0.7:
        calidad = "Buena"
    else:
        calidad = "Excelente"
    print(f"    → Calidad de clustering: {calidad}")
    
    db = analizador.indice_davies_bouldin(etiquetas)
    print(f"\n  Davies-Bouldin Index: {db:.4f}")
    if db < 1:
        calidad_db = "Excelente"
    elif db < 1.5:
        calidad_db = "Buena"
    else:
        calidad_db = "Moderada"
    print(f"    → Calidad de clustering: {calidad_db}")


def paso_9_persistencia(analizador):
    """Paso 9: Persistencia - Guardar/cargar."""
    linea_separador("PASO 9: PERSISTENCIA - GUARDAR/CARGAR MODELOS")
    
    ruta_modelo = Path("./modelos_proyecto4")
    ruta_modelo.mkdir(exist_ok=True)
    ruta_completa = ruta_modelo / "analizador_estadistico"
    
    print(f"Guardando modelos en: {ruta_completa}")
    resultado = analizador.guardar_modelo(str(ruta_completa))
    
    if resultado:
        print("✓ Modelos guardados exitosamente")
        
        # Listar archivos
        archivos = list(ruta_modelo.glob("*"))
        print(f"\nArchivos guardados:")
        for archivo in archivos:
            tamaño = archivo.stat().st_size
            print(f"  - {archivo.name} ({tamaño/1024:.1f} KB)")
        
        print(f"\nCargando modelos desde disco...")
        analizador_cargado = AnalizadorEstadistico.cargar_modelo(str(ruta_completa))
        print("✓ Modelos cargados exitosamente")
    else:
        print("✗ Error al guardar modelos")


def main():
    """Ejecutar flujo completo de 9 pasos."""
    
    print("\n" + "="*70)
    print("ANÁLISIS ESTADÍSTICO MULTIVARIADO - FLUJO COMPLETO")
    print("="*70)
    
    # Paso 1: Generar datos
    X, y_true = paso_1_generar_datos()
    
    # Crear analizador
    analizador = AnalizadorEstadistico(seed=42)
    
    # Paso 2: Estadísticas
    paso_2_estadisticas(analizador, X)
    
    # Paso 3: PCA
    paso_3_pca(analizador)
    
    # Paso 4: K-Means
    paso_4_kmeans(analizador, X)
    
    # Paso 5: Clustering Jerárquico
    paso_5_jerarquico(analizador)
    
    # Paso 6: GMM
    paso_6_gmm(analizador)
    
    # Paso 7: Autoencoder
    paso_7_autoencoder(analizador, X)
    
    # Paso 8: Evaluación
    paso_8_evaluacion(analizador)
    
    # Paso 9: Persistencia
    paso_9_persistencia(analizador)
    
    # Resumen final
    linea_separador("RESUMEN FINAL")
    print("✓ Flujo completo ejecutado exitosamente")
    print(f"\nTécnicas demostradas:")
    print(f"  1. ✓ Estadísticas descriptivas y correlaciones")
    print(f"  2. ✓ PCA con selección automática de componentes")
    print(f"  3. ✓ K-Means con método del codo")
    print(f"  4. ✓ Clustering jerárquico")
    print(f"  5. ✓ GMM con selección de componentes")
    print(f"  6. ✓ Autoencoder profundo")
    print(f"  7. ✓ Métricas de validación (Silhueta, Davies-Bouldin)")
    print(f"  8. ✓ Persistencia de modelos")
    print(f"\n¡Listo para análisis estadístico en producción!")


if __name__ == '__main__':
    main()
