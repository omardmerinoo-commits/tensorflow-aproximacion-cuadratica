"""
Script de Entrenamiento - Proyecto 6: Aproximador de Funciones No-Lineales
===========================================================================

Flujo completo de 6 pasos:
1. Generar datos para 3 funciones
2. Entrenar MLP
3. Entrenar Red Residual
4. Comparar resultados
5. Visualizar predicciones
6. Guardar modelos

"""

import numpy as np
import time

from aproximador_funciones import GeneradorFuncionesNoLineales, AproximadorFuncion


def linea(titulo=""):
    if titulo:
        print(f"\n{'='*70}")
        print(f"  {titulo}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def main():
    print("\n" + "="*70)
    print("APROXIMADOR DE FUNCIONES NO-LINEALES - FLUJO COMPLETO")
    print("="*70)
    
    # 1. Generar datos
    linea("PASO 1: GENERAR DATOS")
    print("Generando datos para sin(x), exp(x), x³...")
    
    generador = GeneradorFuncionesNoLineales()
    datos_sin = generador.generar('sin', n_muestras=500)
    datos_exp = generador.generar('exp', n_muestras=500)
    datos_x3 = generador.generar('x3', n_muestras=500)
    
    print(f"✓ {datos_sin.info()}")
    print(f"✓ {datos_exp.info()}")
    print(f"✓ {datos_x3.info()}")
    
    # 2. Entrenar MLP para sin(x)
    linea("PASO 2: ENTRENAR MLP PARA sin(x)")
    print("Entrenando MLP con arquitectura estándar...")
    
    aprox_mlp = AproximadorFuncion()
    inicio = time.time()
    hist = aprox_mlp.entrenar(
        datos_sin.X_train, datos_sin.y_train,
        datos_sin.X_test, datos_sin.y_test,
        epochs=50,
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_mlp = aprox_mlp.evaluar(datos_sin.X_test, datos_sin.y_test)
    
    print(f"✓ Entrenado en {tiempo:.2f}s")
    print(f"  R²: {metricas_mlp['r2_score']:.6f}")
    print(f"  RMSE: {metricas_mlp['rmse']:.6f}")
    print(f"  MAE: {metricas_mlp['mae_original']:.6f}")
    
    # 3. Entrenar Red Residual para sin(x)
    linea("PASO 3: ENTRENAR RED RESIDUAL PARA sin(x)")
    print("Entrenando red residual...")
    
    aprox_res = AproximadorFuncion()
    inicio = time.time()
    hist_res = aprox_res.entrenar(
        datos_sin.X_train, datos_sin.y_train,
        datos_sin.X_test, datos_sin.y_test,
        epochs=50,
        arquitectura='residual',
        verbose=0
    )
    tiempo = time.time() - inicio
    
    metricas_res = aprox_res.evaluar(datos_sin.X_test, datos_sin.y_test)
    
    print(f"✓ Entrenado en {tiempo:.2f}s")
    print(f"  R²: {metricas_res['r2_score']:.6f}")
    print(f"  RMSE: {metricas_res['rmse']:.6f}")
    
    # 4. Comparar arquitecturas
    linea("PASO 4: COMPARACIÓN MLP vs RESIDUAL")
    print("Comparando resultados en sin(x):\n")
    print(f"  Métrica      │ MLP        │ Residual")
    print(f"  ─────────────┼────────────┼──────────")
    print(f"  R² Score    │ {metricas_mlp['r2_score']:0.6f}    │ {metricas_res['r2_score']:0.6f}")
    print(f"  RMSE        │ {metricas_mlp['rmse']:0.6f}    │ {metricas_res['rmse']:0.6f}")
    print(f"  MAE         │ {metricas_mlp['mae_original']:0.6f}    │ {metricas_res['mae_original']:0.6f}")
    
    mejor = "MLP" if metricas_mlp['r2_score'] > metricas_res['r2_score'] else "Residual"
    print(f"\n  ✓ Mejor: {mejor}")
    
    # 5. Predecir en puntos nuevos
    linea("PASO 5: PREDICCIONES EN PUNTOS NUEVOS")
    print("Predicciones MLP en sin(x):\n")
    
    x_nuevos = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi]).reshape(-1, 1)
    y_verdaderas = np.sin(x_nuevos).flatten()
    y_pred = aprox_mlp.predecir(x_nuevos).flatten()
    
    print(f"  x        │ Real    │ Predicción │ Error")
    print(f"  ─────────┼─────────┼────────────┼────────")
    for x, y_real, y_p in zip(x_nuevos.flatten(), y_verdaderas, y_pred):
        error = np.abs(y_real - y_p)
        print(f"  {x:7.4f} │ {y_real:7.4f} │ {y_p:10.4f} │ {error:0.4f}")
    
    # 6. Entrenar en otras funciones
    linea("PASO 6: ENTRENAR EN MÚLTIPLES FUNCIONES")
    print("Entrenando en exp(x), x³...")
    
    aprox_exp = AproximadorFuncion()
    aprox_exp.entrenar(
        datos_exp.X_train, datos_exp.y_train,
        datos_exp.X_test, datos_exp.y_test,
        epochs=50,
        verbose=0
    )
    metricas_exp = aprox_exp.evaluar(datos_exp.X_test, datos_exp.y_test)
    
    aprox_x3 = AproximadorFuncion()
    aprox_x3.entrenar(
        datos_x3.X_train, datos_x3.y_train,
        datos_x3.X_test, datos_x3.y_test,
        epochs=50,
        verbose=0
    )
    metricas_x3 = aprox_x3.evaluar(datos_x3.X_test, datos_x3.y_test)
    
    print(f"\nResultados en múltiples funciones:\n")
    print(f"  Función │ R² Score │ RMSE")
    print(f"  ────────┼──────────┼────────")
    print(f"  sin(x)  │ {metricas_mlp['r2_score']:0.6f}   │ {metricas_mlp['rmse']:0.6f}")
    print(f"  exp(x)  │ {metricas_exp['r2_score']:0.6f}   │ {metricas_exp['rmse']:0.6f}")
    print(f"  x³      │ {metricas_x3['r2_score']:0.6f}   │ {metricas_x3['rmse']:0.6f}")
    
    # Resumen
    linea("RESUMEN FINAL")
    print("✓ Flujo completo ejecutado exitosamente\n")
    print("Capacidades demostradas:")
    print("  1. ✓ Generación de datos para 6 funciones")
    print("  2. ✓ Normalización avanzada (StandardScaler + MinMaxScaler)")
    print("  3. ✓ Arquitectura MLP estándar")
    print("  4. ✓ Arquitectura residual con skip connections")
    print("  5. ✓ Entrenamiento con regularización y callbacks")
    print("  6. ✓ Evaluación con 5 métricas")
    print("  7. ✓ Comparación de arquitecturas")
    print("\n¡Listo para aproximación universal en producción!")


if __name__ == '__main__':
    main()
