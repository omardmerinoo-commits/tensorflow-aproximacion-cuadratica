#!/usr/bin/env python3
"""
Script de prueba para verificar que el modelo carga correctamente.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from modelo_cuadratico import ModeloCuadratico

print("\n" + "="*60)
print("PRUEBA DE CARGA DEL MODELO ENTRENADO")
print("="*60 + "\n")

try:
    # Crear instancia del modelo
    modelo = ModeloCuadratico()
    
    # Cargar el modelo en formato Keras nativo
    print("Cargando modelo desde archivo 'modelo_entrenado.keras'...")
    modelo.cargar_modelo(path_tf="modelo_entrenado.keras")
    print("✓ Modelo cargado exitosamente\n")
    
    # Realizar pruebas de predicción
    print("Realizando pruebas de predicción...")
    ejemplos_x = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    predicciones = modelo.predecir(ejemplos_x)
    
    print(f"\n{'='*60}")
    print(f"EJEMPLOS DE PREDICCIONES")
    print(f"{'='*60}")
    print(f"{'x':>10} {'y_real':>15} {'y_pred':>15} {'error':>15}")
    print(f"{'-'*60}")
    
    for x_val, y_pred in zip(ejemplos_x, predicciones):
        y_real = x_val[0] ** 2
        error = abs(y_real - y_pred[0])
        print(f"{x_val[0]:>10.2f} {y_real:>15.6f} {y_pred[0]:>15.6f} {error:>15.6f}")
    
    print(f"{'='*60}\n")
    
    print("✓ PRUEBA COMPLETADA SIN ERRORES")
    print("El modelo cargó correctamente y las predicciones funcionan.\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
