"""
Run Training: Pronosticador de Series Temporales
=================================================

Script de demostración completo del sistema de pronóstico de series temporales.

Pasos:
1. Generar serie con tendencia + estacionalidad + ruido
2. Entrenar LSTM bidireccional
3. Entrenar CNN-LSTM híbrido
4. Comparar arquitecturas
5. Análisis de rendimiento por variable
6. Predicciones ejemplo
7. Análisis de residuos
"""

import numpy as np
from pronosticador_series import (
    GeneradorSeriesTemporales,
    PronostadorSeriesTemporales
)
import time


def print_section(titulo):
    print("\n" + "="*70)
    print(titulo)
    print("="*70)


def step1_generar_datos():
    """Paso 1: Generación de datos"""
    print_section("[1] GENERACIÓN DE DATOS SINTÉTICOS")
    
    generador = GeneradorSeriesTemporales(seed=42)
    
    # Generar serie multivariada
    n_puntos = 500
    n_series = 2
    ventana = 10
    
    datos = generador.generar_dataset(
        n_puntos=n_puntos,
        n_series=n_series,
        ventana=ventana,
        split=(0.6, 0.2, 0.2)
    )
    
    print(f"\n✓ Dataset generado")
    print(f"  - Total puntos: {n_puntos}")
    print(f"  - Series/Variables: {n_series} ({', '.join(datos.nombres_features)})")
    print(f"  - Ventana temporal: {ventana} pasos")
    print(f"\n  - X_train: {datos.X_train.shape}")
    print(f"  - X_val:   {datos.X_val.shape}")
    print(f"  - X_test:  {datos.X_test.shape}")
    print(f"  - y_train: {datos.y_train.shape}")
    
    # Estadísticas de la serie original
    print(f"\n  Estadísticas serie original:")
    print(f"  - Media: {np.mean(datos.serie_original):.3f}")
    print(f"  - Std:   {np.std(datos.serie_original):.3f}")
    print(f"  - Min:   {np.min(datos.serie_original):.3f}")
    print(f"  - Max:   {np.max(datos.serie_original):.3f}")
    
    return datos


def step2_entrenar_lstm(datos):
    """Paso 2: Entrenar LSTM bidireccional"""
    print_section("[2] ENTRENAMIENTO LSTM BIDIRECCIONAL")
    
    print("\n  Inicializando modelo...")
    pronosticador = PronostadorSeriesTemporales(seed=42)
    
    print("  Entrenando (20 épocas)...")
    t_inicio = time.time()
    
    hist = pronosticador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='lstm',
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico pérdida:")
    print(f"  - Inicial:  Loss={hist['loss'][0]:.6f}")
    print(f"  - Épocas 5: Loss={hist['loss'][4]:.6f}")
    print(f"  - Final:    Loss={hist['loss'][-1]:.6f}")
    
    # Evaluación
    print(f"\n  Evaluando en test set...")
    metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
    
    print(f"\n  ✓ Métricas LSTM:")
    print(f"    - RMSE:     {metricas['rmse']:.4f}")
    print(f"    - MAE:      {metricas['mae']:.4f}")
    print(f"    - MAPE:     {metricas['mape']:.2f}%")
    if isinstance(metricas['r2_score'], np.ndarray):
        print(f"    - R² Score: {np.mean(metricas['r2_score']):.4f}")
    else:
        print(f"    - R² Score: {metricas['r2_score']:.4f}")
    
    return pronosticador, metricas


def step3_entrenar_cnn_lstm(datos):
    """Paso 3: Entrenar CNN-LSTM híbrido"""
    print_section("[3] ENTRENAMIENTO CNN-LSTM HÍBRIDO")
    
    print("\n  Inicializando modelo...")
    pronosticador = PronostadorSeriesTemporales(seed=42)
    
    print("  Entrenando (20 épocas)...")
    t_inicio = time.time()
    
    hist = pronosticador.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=20,
        arquitectura='cnn_lstm',
        verbose=0
    )
    
    t_duracion = time.time() - t_inicio
    
    print(f"✓ Entrenamiento completado en {t_duracion:.2f}s")
    print(f"\n  Histórico pérdida:")
    print(f"  - Inicial:  Loss={hist['loss'][0]:.6f}")
    print(f"  - Épocas 5: Loss={hist['loss'][4]:.6f}")
    print(f"  - Final:    Loss={hist['loss'][-1]:.6f}")
    
    # Evaluación
    print(f"\n  Evaluando en test set...")
    metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
    
    print(f"\n  ✓ Métricas CNN-LSTM:")
    print(f"    - RMSE:     {metricas['rmse']:.4f}")
    print(f"    - MAE:      {metricas['mae']:.4f}")
    print(f"    - MAPE:     {metricas['mape']:.2f}%")
    if isinstance(metricas['r2_score'], np.ndarray):
        print(f"    - R² Score: {np.mean(metricas['r2_score']):.4f}")
    else:
        print(f"    - R² Score: {metricas['r2_score']:.4f}")
    
    return pronosticador, metricas


def step4_comparar_arquitecturas(metricas_lstm, metricas_cnn):
    """Paso 4: Comparar arquitecturas"""
    print_section("[4] COMPARACIÓN DE ARQUITECTURAS")
    
    print("\n┌─────────────────┬────────────────┬────────────────┐")
    print("│ Métrica         │ LSTM           │ CNN-LSTM       │")
    print("├─────────────────┼────────────────┼────────────────┤")
    print(f"│ RMSE            │ {metricas_lstm['rmse']:>14.4f} │ {metricas_cnn['rmse']:>14.4f} │")
    print(f"│ MAE             │ {metricas_lstm['mae']:>14.4f} │ {metricas_cnn['mae']:>14.4f} │")
    print(f"│ MAPE            │ {metricas_lstm['mape']:>14.2f}% │ {metricas_cnn['mape']:>14.2f}% │")
    print("└─────────────────┴────────────────┴────────────────┘")
    
    # Diferencia relativa
    diff_rmse = ((metricas_lstm['rmse'] - metricas_cnn['rmse']) / 
                 metricas_lstm['rmse'] * 100)
    
    if diff_rmse > 0:
        print(f"\n✓ CNN-LSTM es {abs(diff_rmse):.1f}% mejor en RMSE")
    else:
        print(f"\n✓ LSTM es {abs(diff_rmse):.1f}% mejor en RMSE")


def step5_analisis_rendimiento(pronosticador, datos):
    """Paso 5: Análisis de rendimiento por variable"""
    print_section("[5] ANÁLISIS POR VARIABLE")
    
    y_pred = pronosticador.predecir(datos.X_test)
    
    print("\n  Error por variable:")
    for i in range(y_pred.shape[1] if y_pred.ndim > 1 else 1):
        if y_pred.ndim > 1:
            y_true_i = datos.y_test[:, i]
            y_pred_i = y_pred[:, i]
            var_name = datos.nombres_features[i]
        else:
            y_true_i = datos.y_test
            y_pred_i = y_pred
            var_name = "Variable_1"
        
        mae_i = np.mean(np.abs(y_true_i - y_pred_i))
        rmse_i = np.sqrt(np.mean((y_true_i - y_pred_i)**2))
        
        print(f"    {var_name}:")
        print(f"      - MAE:  {mae_i:.4f}")
        print(f"      - RMSE: {rmse_i:.4f}")


def step6_predicciones_ejemplo(pronosticador, datos):
    """Paso 6: Ejemplos de predicción"""
    print_section("[6] PREDICCIONES EJEMPLO")
    
    y_pred = pronosticador.predecir(datos.X_test[:10])
    
    print("\n  Primeros 10 pasos del conjunto test:")
    print("\n  Index │ Real Value    │ Predicción    │ Error Abs")
    print("  ──────┼───────────────┼───────────────┼──────────")
    
    for i in range(10):
        real = datos.y_test[i, 0] if datos.y_test.ndim > 1 else datos.y_test[i]
        pred = y_pred[i, 0] if y_pred.ndim > 1 else y_pred[i]
        error = abs(real - pred)
        print(f"  {i:>5d} │ {real:>13.4f} │ {pred:>13.4f} │ {error:>8.4f}")


def step7_analisis_residuos(pronosticador, datos):
    """Paso 7: Análisis de residuos"""
    print_section("[7] ANÁLISIS DE RESIDUOS")
    
    metricas = pronosticador.evaluar(datos.X_test, datos.y_test)
    residuos = metricas['residuos']
    
    if residuos.ndim > 1:
        residuos_flat = residuos.flatten()
    else:
        residuos_flat = residuos
    
    print("\n  Estadísticas de residuos:")
    print(f"    - Media:      {np.mean(residuos_flat):>10.6f}")
    print(f"    - Std Dev:    {np.std(residuos_flat):>10.6f}")
    print(f"    - Min:        {np.min(residuos_flat):>10.6f}")
    print(f"    - Máx:        {np.max(residuos_flat):>10.6f}")
    print(f"    - Mediana:    {np.median(residuos_flat):>10.6f}")
    
    # Percentiles
    p25 = np.percentile(residuos_flat, 25)
    p75 = np.percentile(residuos_flat, 75)
    iqr = p75 - p25
    
    print(f"\n  Cuartiles:")
    print(f"    - Q1 (25%):   {p25:>10.6f}")
    print(f"    - Q3 (75%):   {p75:>10.6f}")
    print(f"    - IQR:        {iqr:>10.6f}")
    
    # Porcentaje de residuos pequeños
    small_residuos = (np.abs(residuos_flat) < np.std(residuos_flat)).sum()
    percent_small = (small_residuos / len(residuos_flat)) * 100
    
    print(f"\n  Distribución de errores:")
    print(f"    - Residuos < 1σ: {percent_small:.1f}%")


def main():
    """Ejecuta demostración completa"""
    print("\n" + "="*70)
    print("SISTEMA DE PRONÓSTICO DE SERIES TEMPORALES")
    print("ARIMA + LSTM + CNN-LSTM")
    print("="*70)
    
    # Paso 1: Generar datos
    datos = step1_generar_datos()
    
    # Paso 2: Entrenar LSTM
    pronosticador_lstm, metricas_lstm = step2_entrenar_lstm(datos)
    
    # Paso 3: Entrenar CNN-LSTM
    pronosticador_cnn, metricas_cnn = step3_entrenar_cnn_lstm(datos)
    
    # Paso 4: Comparar
    step4_comparar_arquitecturas(metricas_lstm, metricas_cnn)
    
    # Paso 5: Análisis de rendimiento
    step5_analisis_rendimiento(pronosticador_lstm, datos)
    
    # Paso 6: Predicciones ejemplo
    step6_predicciones_ejemplo(pronosticador_lstm, datos)
    
    # Paso 7: Residuos
    step7_analisis_residuos(pronosticador_lstm, datos)
    
    # Resumen final
    print_section("RESUMEN")
    print("\n✓ Demostración completada exitosamente")
    print("\n  Componentes probados:")
    print("    ✓ Generación de series temporales multivariadas")
    print("    ✓ Arquitectura LSTM bidireccional")
    print("    ✓ Arquitectura CNN-LSTM híbrida")
    print("    ✓ Normalización y desnormalización")
    print("    ✓ Evaluación con múltiples métricas")
    print("    ✓ Análisis de residuos")
    
    print("\n" + "="*70)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
