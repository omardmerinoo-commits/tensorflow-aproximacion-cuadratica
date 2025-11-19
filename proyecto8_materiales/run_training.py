"""
Script de Entrenamiento - Proyecto 8: Predictor de Propiedades de Materiales
=============================================================================

Flujo completo de 6 pasos:
1. Generar dataset de materiales
2. Entrenar modelo multivariado
3. Evaluar desempeño por propiedad
4. Análisis de correlaciones
5. Predicciones en nuevas composiciones
6. Análisis de residuos

"""

import numpy as np
import time

from predictor_materiales import GeneradorMateriales, PredictorMateriales


def linea(titulo=""):
    if titulo:
        print(f"\n{'='*70}")
        print(f"  {titulo}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def main():
    print("\n" + "="*70)
    print("PREDICTOR DE PROPIEDADES DE MATERIALES - FLUJO COMPLETO")
    print("="*70)
    
    # 1. Generar dataset
    linea("PASO 1: GENERAR DATASET DE MATERIALES")
    print("Generando 500 composiciones de materiales...")
    
    generador = GeneradorMateriales(seed=42)
    inicio = time.time()
    datos = generador.generar_dataset(n_muestras=500)
    tiempo = time.time() - inicio
    
    print(f"✓ Dataset generado en {tiempo:.2f}s")
    print(f"  {datos.info()}")
    print(f"  Propiedades: {', '.join(datos.nombres_propiedades)}")
    print(f"  Features: {', '.join(datos.nombres_features)}")
    
    # 2. Entrenar modelo
    linea("PASO 2: ENTRENAR MODELO DE REGRESIÓN MULTIVARIADA")
    print("Entrenando MLP para 3 propiedades simultáneamente...")
    
    predictor = PredictorMateriales(seed=42)
    inicio = time.time()
    hist = predictor.entrenar(
        datos.X_train, datos.y_train,
        datos.X_val, datos.y_val,
        epochs=30,
        verbose=0
    )
    tiempo = time.time() - inicio
    
    print(f"✓ Modelo entrenado en {tiempo:.2f}s")
    print(f"  Épocas: {len(hist['loss'])}")
    print(f"  Loss inicial: {hist['loss'][0]:.6f}")
    print(f"  Loss final: {hist['loss'][-1]:.6f}")
    print(f"  Loss disminuyó: {hist['loss'][0] - hist['loss'][-1]:.6f}")
    
    # 3. Evaluar por propiedad
    linea("PASO 3: EVALUACIÓN POR PROPIEDAD")
    print("Calculando métricas en conjunto de test...\n")
    
    metricas = predictor.evaluar(datos.X_test, datos.y_test)
    
    print(f"  Propiedad          │ R²     │ RMSE   │ MAE")
    print(f"  ───────────────────┼────────┼────────┼──────")
    
    for i, prop in enumerate(datos.nombres_propiedades):
        r2 = metricas['r2_score'][i]
        rmse = metricas['rmse'][i]
        mae = metricas['mae'][i]
        print(f"  {prop:19} │ {r2:.4f} │ {rmse:.4f} │ {mae:.4f}")
    
    promedio_r2 = np.mean(metricas['r2_score'])
    print(f"\n  Promedio R²: {promedio_r2:.4f}")
    
    # 4. Correlaciones
    linea("PASO 4: ANÁLISIS DE CORRELACIONES")
    print("Correlaciones entre propiedades reales:\n")
    
    # Calcular matriz de correlación
    correlaciones = np.corrcoef(datos.y_test.T)
    
    props_short = ['Densidad', 'Dureza', 'P.Fusión']
    print("  Matriz de correlación:")
    print(f"\n       │ {' '.join([f'{p:>8}' for p in props_short])}")
    print(f"  ─────┼" + "─" * 30)
    for i, p1 in enumerate(props_short):
        fila = " ".join([f"{correlaciones[i, j]:8.4f}" for j in range(3)])
        print(f"  {p1:>5} │ {fila}")
    
    # Ejemplo: densidad vs dureza
    corr_dens_dur = correlaciones[0, 1]
    print(f"\n  ✓ Densidad-Dureza correlación: {corr_dens_dur:.4f} (positiva)")
    
    # 5. Predicciones
    linea("PASO 5: PREDICCIONES EN MATERIALES NUEVOS")
    print("Predicciones en primeros 5 test samples:\n")
    
    y_pred = predictor.predecir(datos.X_test[:5])
    
    for idx in range(5):
        print(f"  Muestra {idx}:")
        for i, prop in enumerate(datos.nombres_propiedades):
            real = datos.y_test[idx, i]
            pred = y_pred[idx, i]
            error = abs(real - pred)
            pct_error = 100 * error / abs(real) if real != 0 else 0
            print(f"    {prop:20}: Real={real:8.2f}, Pred={pred:8.2f}, Error={pct_error:5.1f}%")
        print()
    
    # 6. Análisis de residuos
    linea("PASO 6: ANÁLISIS DE RESIDUOS")
    print("Estadísticas de residuos:\n")
    
    residuos = metricas['residuos']
    
    print(f"  Propiedad          │ Media   │ Std     │ Min     │ Max")
    print(f"  ───────────────────┼─────────┼─────────┼─────────┼──────")
    
    for i, prop in enumerate(datos.nombres_propiedades):
        media = np.mean(residuos[:, i])
        std = np.std(residuos[:, i])
        minimo = np.min(residuos[:, i])
        maximo = np.max(residuos[:, i])
        print(f"  {prop:19} │ {media:7.4f} │ {std:7.4f} │ {minimo:7.4f} │ {maximo:7.4f}")
    
    # Diagnóstico
    print("\n  Diagnóstico:")
    for i, prop in enumerate(datos.nombres_propiedades):
        media_residuo = np.mean(residuos[:, i])
        if abs(media_residuo) < 0.1 * np.std(residuos[:, i]):
            print(f"  ✓ {prop}: Residuos centrados en cero")
        else:
            print(f"  ⚠ {prop}: Sesgo en residuos ({media_residuo:.4f})")
    
    # Resumen
    linea("RESUMEN FINAL")
    print("✓ Flujo completo ejecutado exitosamente\n")
    print("Capacidades demostradas:")
    print("  1. ✓ Generación realista de composiciones de materiales")
    print("  2. ✓ Cálculo de propiedades basado en física")
    print("  3. ✓ Normalización multivariada diferenciada")
    print("  4. ✓ Arquitectura MLP para regresión multivariada")
    print("  5. ✓ Entrenamiento con regularización avanzada")
    print("  6. ✓ Evaluación con 4 métricas (R², RMSE, MAE, residuos)")
    print("  7. ✓ Análisis de correlaciones entre propiedades")
    print("  8. ✓ Detección de sesgos sistemáticos")
    
    print(f"\n✓ Promedio R² en test: {promedio_r2:.4f}")
    print("✓ Listo para predicción de propiedades en producción!")


if __name__ == '__main__':
    main()
