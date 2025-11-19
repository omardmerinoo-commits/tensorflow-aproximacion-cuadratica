"""
Script de Entrenamiento - Proyecto 7: Clasificador de Audio
===========================================================

Flujo completo de 6 pasos:
1. Generar datos sintéticos para 3 categorías
2. Extraer espectrogramas con STFT
3. Entrenar CNN 2D
4. Entrenar LSTM
5. Comparar resultados
6. Realizar predicciones

"""

import numpy as np
import time

from clasificador_audio import (
    GeneradorAudioSintetico, ExtractorEspectrograma, ClasificadorAudio
)


def linea(titulo=""):
    if titulo:
        print(f"\n{'='*70}")
        print(f"  {titulo}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def main():
    print("\n" + "="*70)
    print("CLASIFICADOR DE AUDIO CON ESPECTROGRAMAS - FLUJO COMPLETO")
    print("="*70)
    
    # 1. Generar datos sintéticos
    linea("PASO 1: GENERAR DATOS SINTÉTICOS")
    print("Generando 100 audios por categoría...")
    
    generador = GeneradorAudioSintetico(sr=16000)
    inicio = time.time()
    datos = generador.generar_dataset(muestras_por_clase=100, duracion=2.0)
    tiempo = time.time() - inicio
    
    print(f"✓ Datos generados en {tiempo:.2f}s")
    print(f"  Train: {datos.X_train.shape[0]} audios")
    print(f"  Val: {datos.X_val.shape[0]} audios")
    print(f"  Test: {datos.X_test.shape[0]} audios")
    print(f"  Categorías: {list(datos.labels.values())}")
    
    # 2. Extraer espectrogramas
    linea("PASO 2: EXTRAER ESPECTROGRAMAS CON STFT")
    print("Calculando STFT y espectrogramas...")
    
    extractor = ExtractorEspectrograma(n_fft=512, hop_length=128)
    
    inicio = time.time()
    X_train_spec = extractor.extraer(datos.X_train, db_scale=True)
    X_val_spec = extractor.extraer(datos.X_val, db_scale=True)
    X_test_spec = extractor.extraer(datos.X_test, db_scale=True)
    tiempo = time.time() - inicio
    
    print(f"✓ Espectrogramas extraídos en {tiempo:.2f}s")
    print(f"  Shape: {X_train_spec.shape}")
    print(f"  Frecuencias: {X_train_spec.shape[1]} bins")
    print(f"  Ventanas temporales: {X_train_spec.shape[2]}")
    
    # 3. Entrenar CNN 2D
    linea("PASO 3: ENTRENAR CNN 2D")
    print("Entrenando modelo CNN con espectrogramas...")
    
    clf_cnn = ClasificadorAudio(seed=42)
    inicio = time.time()
    hist_cnn = clf_cnn.entrenar(
        X_train_spec, datos.y_train,
        X_val_spec, datos.y_val,
        epochs=20,
        arquitectura='cnn',
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_cnn = clf_cnn.evaluar(X_test_spec, datos.y_test)
    
    print(f"✓ CNN entrenado en {tiempo:.2f}s")
    print(f"  Loss final: {hist_cnn['loss'][-1]:.6f}")
    print(f"  Accuracy train: {hist_cnn['accuracy'][-1]:.4f}")
    print(f"  Accuracy test: {metricas_cnn['accuracy']:.4f}")
    
    # 4. Entrenar LSTM
    linea("PASO 4: ENTRENAR LSTM BIDIRECCIONAL")
    print("Entrenando modelo LSTM para secuencias...")
    
    clf_lstm = ClasificadorAudio(seed=42)
    inicio = time.time()
    hist_lstm = clf_lstm.entrenar(
        X_train_spec, datos.y_train,
        X_val_spec, datos.y_val,
        epochs=20,
        arquitectura='lstm',
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_lstm = clf_lstm.evaluar(X_test_spec, datos.y_test)
    
    print(f"✓ LSTM entrenado en {tiempo:.2f}s")
    print(f"  Loss final: {hist_lstm['loss'][-1]:.6f}")
    print(f"  Accuracy train: {hist_lstm['accuracy'][-1]:.4f}")
    print(f"  Accuracy test: {metricas_lstm['accuracy']:.4f}")
    
    # 5. Comparar arquitecturas
    linea("PASO 5: COMPARACIÓN CNN vs LSTM")
    print("Comparando resultados:\n")
    print(f"  Métrica           │ CNN 2D     │ LSTM")
    print(f"  ──────────────────┼────────────┼──────────")
    print(f"  Accuracy          │ {metricas_cnn['accuracy']:0.4f}    │ {metricas_lstm['accuracy']:0.4f}")
    print(f"  Loss              │ {metricas_cnn['loss']:0.6f}    │ {metricas_lstm['loss']:0.6f}")
    
    mejor = "CNN" if metricas_cnn['accuracy'] > metricas_lstm['accuracy'] else "LSTM"
    print(f"\n  ✓ Mejor arquitectura: {mejor}")
    
    # Análisis de matriz de confusión
    print("\n  Matriz de confusión (CNN):")
    cm = metricas_cnn['confusion_matrix']
    categorias = list(datos.labels.values())
    print(f"\n           {''.join([f'{c:>8}' for c in categorias])}")
    for i, fila in enumerate(cm):
        print(f"  {categorias[i]:>7} {' '.join([f'{v:8}' for v in fila])}")
    
    # 6. Predecir y visualizar
    linea("PASO 6: PREDICCIONES EN DATOS NUEVOS")
    print("Predicciones de CNN en primeros 10 test samples:\n")
    
    clases, probs = clf_cnn.predecir(X_test_spec[:10])
    
    print(f"  Índice │ Predicción │ Confianza │ Real")
    print(f"  ───────┼────────────┼───────────┼──────────")
    for i, (pred_clase, prob_vector) in enumerate(zip(clases, probs)):
        confianza = prob_vector[pred_clase]
        real_clase = datos.y_test[i]
        correcto = "✓" if pred_clase == real_clase else "✗"
        
        print(f"  {i:6} │ {datos.labels[pred_clase]:>10} │ {confianza:0.2%}    │ "
              f"{datos.labels[real_clase]:>8} {correcto}")
    
    # 7. Análisis por categoría
    linea("PASO 7: ANÁLISIS POR CATEGORÍA")
    print("Desempeño por categoría de audio:\n")
    
    print("  Categoría │ Precisión │ Recall │ F1-Score")
    print("  ──────────┼───────────┼────────┼──────────")
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    y_pred = np.argmax(clf_cnn.modelo.predict(X_test_spec, verbose=0), axis=1)
    
    for clase_idx in range(3):
        clase_name = datos.labels[clase_idx]
        
        y_test_binary = (datos.y_test == clase_idx).astype(int)
        y_pred_binary = (y_pred == clase_idx).astype(int)
        
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        print(f"  {clase_name:>9} │ {precision:0.2%}      │ {recall:0.2%}   │ {f1:0.2%}")
    
    # Resumen
    linea("RESUMEN FINAL")
    print("✓ Flujo completo ejecutado exitosamente\n")
    print("Capacidades demostradas:")
    print("  1. ✓ Generación realista de audio sintético")
    print("  2. ✓ Extracción de STFT y espectrogramas (escala dB)")
    print("  3. ✓ Arquitectura CNN 2D para imágenes de tiempo-frecuencia")
    print("  4. ✓ Arquitectura LSTM bidireccional para secuencias")
    print("  5. ✓ Entrenamiento con regularización avanzada")
    print("  6. ✓ Evaluación con matriz de confusión")
    print("  7. ✓ Predicción con probabilidades por clase")
    print("  8. ✓ Análisis detallado de desempeño")
    
    print("\n✓ Listo para clasificación de audio en producción!")


if __name__ == '__main__':
    main()
