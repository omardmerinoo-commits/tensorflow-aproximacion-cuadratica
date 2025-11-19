"""
Run Training: Clasificador de Sentimientos
==========================================

Demostración completa del sistema de análisis de sentimientos.

Pasos:
1. Generar corpus de textos balanceado
2. Entrenar LSTM bidireccional
3. Entrenar Transformer
4. Entrenar CNN 1D
5. Comparar arquitecturas
6. Análisis por clase
7. Ejemplos de predicción
"""

import numpy as np
from clasificador_sentimientos import (
    GeneradorTextoSentimientos,
    ClasificadorSentimientos
)
import time


def print_section(titulo):
    print("\n" + "="*70)
    print(titulo)
    print("="*70)


def step1_generar_datos():
    """Paso 1: Generación de datos"""
    print_section("[1] GENERACIÓN DE CORPUS BALANCEADO")
    
    generador = GeneradorTextoSentimientos(seed=42)
    
    n_por_clase = 100
    datos = generador.generar_dataset(
        n_samples_por_clase=n_por_clase,
        max_words=1000,
        max_len=50,
        split=(0.6, 0.2, 0.2)
    )
    
    print(f"\n✓ Dataset generado")
    print(f"  - Muestras por clase: {n_por_clase}")
    print(f"  - Total: {n_por_clase * 3} textos")
    print(f"  - Clases: Negativo, Neutro, Positivo")
    print(f"\n  Distribución:")
    print(f"  - Train: {len(datos.X_train)} textos")
    print(f"  - Val:   {len(datos.X_val)} textos")
    print(f"  - Test:  {len(datos.X_test)} textos")
    
    print(f"\n  Parámetros:")
    print(f"  - Vocabulario: 1000 palabras")
    print(f"  - Max length: 50 tokens")
    print(f"  - Embedding: 128 dimensiones")
    
    # Ejemplos
    print(f"\n  Ejemplos por clase:")
    for i, sentimiento in enumerate(['Negativo', 'Neutro', 'Positivo']):
        idx = i * len(datos.X_train) // 3
        if idx < len(datos.X_train):
            print(f"    {sentimiento}: '{datos.X_train[idx][:50]}...'")
    
    return datos


def step2_entrenar_lstm(datos):
    """Paso 2: Entrenar LSTM"""
    print_section("[2] ENTRENAMIENTO LSTM BIDIRECCIONAL")
    
    print("\n  Inicializando modelo...")
    print("  - Arquitectura: BiLSTM(64) → BiLSTM(32) → Dense → Softmax")
    
    clasificador = ClasificadorSentimientos(vocab_size=1000, embedding_dim=128)
    
    print("  Entrenando (20 épocas)...")
    t_inicio = time.time()
    
    hist = clasificador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='lstm',
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico:")
    print(f"  - Epoch 1:  Loss={hist['loss'][0]:.4f}, Acc={hist['accuracy'][0]:.4f}")
    print(f"  - Epoch 10: Loss={hist['loss'][9]:.4f}, Acc={hist['accuracy'][9]:.4f}")
    print(f"  - Epoch 20: Loss={hist['loss'][-1]:.4f}, Acc={hist['accuracy'][-1]:.4f}")
    
    # Evaluación
    print(f"\n  Evaluando en test...")
    metricas = clasificador.evaluar(datos.X_test, datos.y_test)
    
    print(f"\n  ✓ Resultados LSTM:")
    print(f"    - Accuracy: {metricas['accuracy']:.4f}")
    print(f"    - Per-class:")
    for clase_id, acc in metricas['per_class_accuracy'].items():
        nombres = ['Negativo', 'Neutro', 'Positivo']
        print(f"      {nombres[clase_id]}: {acc:.4f}")
    
    return clasificador, metricas


def step3_entrenar_transformer(datos):
    """Paso 3: Entrenar Transformer"""
    print_section("[3] ENTRENAMIENTO TRANSFORMER")
    
    print("\n  Inicializando modelo...")
    print("  - Arquitectura: MultiHeadAttention(4 heads) × 2")
    
    clasificador = ClasificadorSentimientos(vocab_size=1000, embedding_dim=128)
    
    print("  Entrenando (20 épocas)...")
    t_inicio = time.time()
    
    hist = clasificador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='transformer',
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico:")
    print(f"  - Epoch 1:  Loss={hist['loss'][0]:.4f}, Acc={hist['accuracy'][0]:.4f}")
    print(f"  - Epoch 10: Loss={hist['loss'][9]:.4f}, Acc={hist['accuracy'][9]:.4f}")
    print(f"  - Epoch 20: Loss={hist['loss'][-1]:.4f}, Acc={hist['accuracy'][-1]:.4f}")
    
    # Evaluación
    print(f"\n  Evaluando en test...")
    metricas = clasificador.evaluar(datos.X_test, datos.y_test)
    
    print(f"\n  ✓ Resultados Transformer:")
    print(f"    - Accuracy: {metricas['accuracy']:.4f}")
    
    return clasificador, metricas


def step4_entrenar_cnn1d(datos):
    """Paso 4: Entrenar CNN 1D"""
    print_section("[4] ENTRENAMIENTO CNN 1D")
    
    print("\n  Inicializando modelo...")
    print("  - Arquitectura: Conv1D(64) → MaxPool → Conv1D(32) → GlobalPool")
    
    clasificador = ClasificadorSentimientos(vocab_size=1000, embedding_dim=128)
    
    print("  Entrenando (20 épocas)...")
    t_inicio = time.time()
    
    hist = clasificador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='cnn1d',
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico:")
    print(f"  - Epoch 1:  Loss={hist['loss'][0]:.4f}, Acc={hist['accuracy'][0]:.4f}")
    print(f"  - Epoch 10: Loss={hist['loss'][9]:.4f}, Acc={hist['accuracy'][9]:.4f}")
    print(f"  - Epoch 20: Loss={hist['loss'][-1]:.4f}, Acc={hist['accuracy'][-1]:.4f}")
    
    # Evaluación
    print(f"\n  Evaluando en test...")
    metricas = clasificador.evaluar(datos.X_test, datos.y_test)
    
    print(f"\n  ✓ Resultados CNN 1D:")
    print(f"    - Accuracy: {metricas['accuracy']:.4f}")
    
    return clasificador, metricas


def step5_comparar_arquitecturas(m_lstm, m_tf, m_cnn):
    """Paso 5: Comparar arquitecturas"""
    print_section("[5] COMPARACIÓN DE ARQUITECTURAS")
    
    print("\n┌──────────────┬────────────────┬────────────────┬────────────────┐")
    print("│ Métrica      │ LSTM           │ Transformer    │ CNN 1D         │")
    print("├──────────────┼────────────────┼────────────────┼────────────────┤")
    print(f"│ Accuracy     │ {m_lstm['accuracy']:>14.4f} │ {m_tf['accuracy']:>14.4f} │ {m_cnn['accuracy']:>14.4f} │")
    print("└──────────────┴────────────────┴────────────────┴────────────────┘")
    
    # Ranking
    modelos = [
        ('LSTM', m_lstm['accuracy']),
        ('Transformer', m_tf['accuracy']),
        ('CNN 1D', m_cnn['accuracy'])
    ]
    modelos_sorted = sorted(modelos, key=lambda x: x[1], reverse=True)
    
    print(f"\n  Ranking por Accuracy:")
    for i, (nombre, acc) in enumerate(modelos_sorted):
        print(f"  {i+1}. {nombre}: {acc:.4f}")


def step6_analisis_por_clase(clasificador, datos):
    """Paso 6: Análisis por clase"""
    print_section("[6] ANÁLISIS POR CLASE")
    
    metricas = clasificador.evaluar(datos.X_test, datos.y_test)
    
    nombres = ['Negativo', 'Neutro', 'Positivo']
    print("\n  Precisión por sentimiento:")
    print("  ┌─────────────┬────────────────┬─────────┐")
    print("  │ Clase       │ Accuracy       │ Samples │")
    print("  ├─────────────┼────────────────┼─────────┤")
    
    verdaderos = metricas['verdaderos']
    for clase_id, nombre in enumerate(nombres):
        mask = verdaderos == clase_id
        n_samples = np.sum(mask)
        acc = metricas['per_class_accuracy'].get(clase_id, 0.0)
        print(f"  │ {nombre:<11} │ {acc:>14.4f} │ {n_samples:>7} │")
    
    print("  └─────────────┴────────────────┴─────────┘")


def step7_predicciones_ejemplo(clasificador, datos):
    """Paso 7: Ejemplos de predicción"""
    print_section("[7] EJEMPLOS DE PREDICCIÓN")
    
    clases, probs = clasificador.predecir(datos.X_test[:10])
    
    nombres = ['Negativo', 'Neutro', 'Positivo']
    print("\n  Primeros 10 predicciones:")
    print("\n  Index │ Texto (primeros 40 chars)      │ Predicción │ Confianza")
    print("  ──────┼───────────────────────────────┼────────────┼──────────")
    
    for i in range(10):
        texto = datos.X_test[i][:40]
        clase = clases[i]
        confianza = probs[i, clase]
        nombre_clase = nombres[clase]
        
        print(f"  {i:>5} │ {texto:<30} │ {nombre_clase:<10} │ {confianza:>7.2%}")


def step8_distribucion_confianza(clasificador, datos):
    """Paso 8: Distribución de confianza"""
    print_section("[8] ANÁLISIS DE CONFIANZA")
    
    _, probs = clasificador.predecir(datos.X_test)
    confianzas = np.max(probs, axis=1)
    
    print("\n  Estadísticas de confianza del modelo:")
    print(f"  - Media:     {np.mean(confianzas):.4f}")
    print(f"  - Std Dev:   {np.std(confianzas):.4f}")
    print(f"  - Mínima:    {np.min(confianzas):.4f}")
    print(f"  - Máxima:    {np.max(confianzas):.4f}")
    
    # Percentiles
    p25 = np.percentile(confianzas, 25)
    p50 = np.percentile(confianzas, 50)
    p75 = np.percentile(confianzas, 75)
    
    print(f"\n  Percentiles:")
    print(f"  - Q1 (25%):  {p25:.4f}")
    print(f"  - Mediana:   {p50:.4f}")
    print(f"  - Q3 (75%):  {p75:.4f}")
    
    # Distribución
    high_conf = (confianzas > 0.8).sum()
    med_conf = ((confianzas > 0.6) & (confianzas <= 0.8)).sum()
    low_conf = (confianzas <= 0.6).sum()
    
    print(f"\n  Distribución de confianza:")
    print(f"  - Alta (>0.8):     {high_conf} ({high_conf/len(confianzas)*100:.1f}%)")
    print(f"  - Media (0.6-0.8): {med_conf} ({med_conf/len(confianzas)*100:.1f}%)")
    print(f"  - Baja (≤0.6):     {low_conf} ({low_conf/len(confianzas)*100:.1f}%)")


def main():
    """Ejecuta demostración completa"""
    print("\n" + "="*70)
    print("CLASIFICADOR DE SENTIMIENTOS CON NLP")
    print("LSTM + Transformer + CNN 1D")
    print("="*70)
    
    # Paso 1: Generar datos
    datos = step1_generar_datos()
    
    # Paso 2: LSTM
    lstm, m_lstm = step2_entrenar_lstm(datos)
    
    # Paso 3: Transformer
    tf, m_tf = step3_entrenar_transformer(datos)
    
    # Paso 4: CNN 1D
    cnn, m_cnn = step4_entrenar_cnn1d(datos)
    
    # Paso 5: Comparar
    step5_comparar_arquitecturas(m_lstm, m_tf, m_cnn)
    
    # Paso 6: Análisis por clase
    step6_analisis_por_clase(lstm, datos)
    
    # Paso 7: Predicciones
    step7_predicciones_ejemplo(lstm, datos)
    
    # Paso 8: Confianza
    step8_distribucion_confianza(lstm, datos)
    
    # Resumen
    print_section("RESUMEN")
    print("\n✓ Demostración completada exitosamente")
    print("\n  Componentes probados:")
    print("    ✓ Generación de corpus sintético balanceado")
    print("    ✓ Pre-procesamiento: tokenización y padding")
    print("    ✓ Arquitectura LSTM bidireccional")
    print("    ✓ Arquitectura Transformer con Multi-Head Attention")
    print("    ✓ Arquitectura CNN 1D para n-gramas")
    print("    ✓ Evaluación y análisis por clase")
    print("    ✓ Análisis de confianza de predicciones")
    
    print("\n" + "="*70)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
