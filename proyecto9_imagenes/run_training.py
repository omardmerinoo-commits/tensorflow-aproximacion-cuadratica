"""
Script de Entrenamiento - Proyecto 9: Clasificador de Imágenes CIFAR-10
========================================================================

Flujo completo de 7 pasos:
1. Cargar dataset CIFAR-10
2. Entrenar CNN personalizada
3. Entrenar Transfer Learning (MobileNetV2)
4. Comparar arquitecturas
5. Análisis por clase
6. Visualizar ejemplos de predicción
7. Análisis de errores

"""

import numpy as np
import time

from clasificador_imagenes import GeneradorCIFAR10, ClasificadorImagenes


def linea(titulo=""):
    if titulo:
        print(f"\n{'='*70}")
        print(f"  {titulo}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def main():
    print("\n" + "="*70)
    print("CLASIFICADOR DE IMÁGENES CIFAR-10 - FLUJO COMPLETO")
    print("="*70)
    
    # 1. Cargar dataset
    linea("PASO 1: CARGAR DATASET CIFAR-10")
    print("Descargando CIFAR-10 (50k train, 10k test)...")
    
    generador = GeneradorCIFAR10(seed=42)
    inicio = time.time()
    datos = generador.cargar_datos(validacion_split=0.2)
    tiempo = time.time() - inicio
    
    print(f"✓ Dataset cargado en {tiempo:.2f}s")
    print(f"  {datos.info()}")
    print(f"  Clases: {', '.join(datos.clases[:5])}...")
    print(f"  Rango de píxeles: [{datos.X_train.min():.2f}, {datos.X_train.max():.2f}]")
    
    # 2. Entrenar CNN
    linea("PASO 2: ENTRENAR CNN PERSONALIZADA")
    print("Entrenando CNN con data augmentation...")
    
    clf_cnn = ClasificadorImagenes(seed=42)
    inicio = time.time()
    hist_cnn = clf_cnn.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='cnn',
        usar_augmentacion=True,
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_cnn = clf_cnn.evaluar(datos.X_test, datos.y_test)
    
    print(f"✓ CNN entrenado en {tiempo:.2f}s")
    print(f"  Loss inicial: {hist_cnn['loss'][0]:.4f}")
    print(f"  Loss final: {hist_cnn['loss'][-1]:.4f}")
    print(f"  Accuracy test: {metricas_cnn['accuracy']:.4f}")
    print(f"  Precision: {metricas_cnn['precision']:.4f}")
    print(f"  Recall: {metricas_cnn['recall']:.4f}")
    
    # 3. Entrenar Transfer Learning
    linea("PASO 3: ENTRENAR CON TRANSFER LEARNING (MobileNetV2)")
    print("Entrenando con pesos pre-entrenados en ImageNet...")
    
    clf_tl = ClasificadorImagenes(seed=42)
    inicio = time.time()
    hist_tl = clf_tl.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='transfer',
        usar_augmentacion=True,
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_tl = clf_tl.evaluar(datos.X_test, datos.y_test)
    
    print(f"✓ Transfer Learning entrenado en {tiempo:.2f}s")
    print(f"  Loss inicial: {hist_tl['loss'][0]:.4f}")
    print(f"  Loss final: {hist_tl['loss'][-1]:.4f}")
    print(f"  Accuracy test: {metricas_tl['accuracy']:.4f}")
    print(f"  Precision: {metricas_tl['precision']:.4f}")
    print(f"  Recall: {metricas_tl['recall']:.4f}")
    
    # 4. Comparar
    linea("PASO 4: COMPARACIÓN DE ARQUITECTURAS")
    print("Comparando CNN vs Transfer Learning:\n")
    
    print(f"  Métrica           │ CNN        │ Transfer Learning")
    print(f"  ──────────────────┼────────────┼──────────────────")
    print(f"  Accuracy          │ {metricas_cnn['accuracy']:.4f}    │ {metricas_tl['accuracy']:.4f}")
    print(f"  Precision         │ {metricas_cnn['precision']:.4f}    │ {metricas_tl['precision']:.4f}")
    print(f"  Recall            │ {metricas_cnn['recall']:.4f}    │ {metricas_tl['recall']:.4f}")
    print(f"  F1-Score          │ {metricas_cnn['f1_score']:.4f}    │ {metricas_tl['f1_score']:.4f}")
    
    mejor = "Transfer Learning" if metricas_tl['accuracy'] > metricas_cnn['accuracy'] else "CNN"
    mejora = abs(metricas_tl['accuracy'] - metricas_cnn['accuracy'])
    print(f"\n  ✓ Mejor: {mejor} (+{mejora:.4f} accuracy)")
    
    # 5. Análisis por clase
    linea("PASO 5: ANÁLISIS POR CLASE")
    print("Desempeño de CNN en cada clase:\n")
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    y_test = datos.y_test
    y_pred = metricas_cnn['predicciones']
    
    print(f"  {'Clase':<12} │ Precision │ Recall │ F1-Score │ Support")
    print(f"  {'-'*62}")
    
    for i, clase in enumerate(datos.clases):
        y_test_bin = (y_test == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        
        precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
        support = np.sum(y_test_bin)
        
        print(f"  {clase:<12} │ {precision:.3f}      │ {recall:.3f}   │ {f1:.3f}    │ {support:4}")
    
    # 6. Predicciones ejemplo
    linea("PASO 6: EJEMPLOS DE PREDICCIÓN")
    print("Predicciones de CNN en primeras 10 imágenes de test:\n")
    
    clases_cnn, probs_cnn = clf_cnn.predecir(datos.X_test[:10])
    clases_tl, probs_tl = clf_tl.predecir(datos.X_test[:10])
    
    print(f"  Idx │ Real      │ CNN       │ TL        │ Acuerdo")
    print(f"  ────┼───────────┼───────────┼───────────┼────────")
    
    for i in range(10):
        real = datos.clases[datos.y_test[i]]
        pred_cnn = datos.clases[clases_cnn[i]]
        pred_tl = datos.clases[clases_tl[i]]
        acuerdo = "✓" if clases_cnn[i] == clases_tl[i] else "✗"
        
        print(f"  {i:3} │ {real:<9} │ {pred_cnn:<9} │ {pred_tl:<9} │ {acuerdo}")
    
    # 7. Análisis de errores
    linea("PASO 7: ANÁLISIS DE ERRORES")
    print("Clases más confundidas:\n")
    
    from sklearn.metrics import confusion_matrix
    
    cm = metricas_cnn['confusion_matrix']
    
    # Encontrar pares más confundidos
    errores = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                errores.append((cm[i, j], datos.clases[i], datos.clases[j]))
    
    errores.sort(reverse=True)
    
    print("  Top 5 confusiones:")
    for rank, (count, clase_real, clase_pred) in enumerate(errores[:5], 1):
        print(f"  {rank}. {clase_real} → {clase_pred}: {count} casos")
    
    # Resumen final
    linea("RESUMEN FINAL")
    print("✓ Flujo completo ejecutado exitosamente\n")
    print("Capacidades demostradas:")
    print("  1. ✓ Carga y preprocesamiento de CIFAR-10")
    print("  2. ✓ CNN personalizada con 5 bloques convolucionales")
    print("  3. ✓ Transfer learning con MobileNetV2 pre-entrenada")
    print("  4. ✓ Data augmentation (rotación, zoom, flip, desplazamiento)")
    print("  5. ✓ Técnicas avanzadas (BatchNorm, Dropout, L2)")
    print("  6. ✓ Evaluación completa (accuracy, precision, recall, F1)")
    print("  7. ✓ Análisis por clase y matriz de confusión")
    
    print(f"\n✓ CNN Accuracy: {metricas_cnn['accuracy']:.2%}")
    print(f"✓ Transfer Learning Accuracy: {metricas_tl['accuracy']:.2%}")
    print("✓ Listo para clasificación de imágenes en producción!")


if __name__ == '__main__':
    main()
