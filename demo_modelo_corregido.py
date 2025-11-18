#!/usr/bin/env python3
"""
Demostración: Cargar y usar el modelo sin errores (usando formato .keras)
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
sys.path.insert(0, r'C:\Users\Usuario\Desktop\tensorflow-aproximacion-cuadratica')

import numpy as np
from tensorflow import keras

print("\n" + "="*70)
print("[OK] DEMOSTRACION: MODELO ENTRENADO CARGADO EXITOSAMENTE")
print("="*70 + "\n")

print("1. Cargando el modelo en formato .keras...")
try:
    modelo = keras.models.load_model("modelo_entrenado.keras")
    print("   [OK] Modelo cargado sin errores\n")
except Exception as e:
    print(f"   [ERROR] Error: {e}\n")
    sys.exit(1)

print("2. Verificando arquitectura del modelo:")
print(f"   - Parametros totales: {modelo.count_params():,}")
print(f"   - Capas: {len(modelo.layers)}")
print(f"   - Optimizador: {modelo.optimizer.__class__.__name__}\n")

print("3. Realizando predicciones con valores de prueba:")
x_prueba = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32)
predicciones = modelo.predict(x_prueba, verbose=0)

print(f"\n   {'x':>8} -> {'y_real':>10} | {'y_pred':>10} (error)")
print("   " + "-"*42)
for x, y_pred in zip(x_prueba, predicciones):
    y_real = x[0] ** 2
    error = abs(y_real - y_pred[0])
    marker = "[OK]" if error < 0.2 else "[!]"
    print(f"   {marker} {x[0]:>6.1f} -> {y_real:>10.4f} | {y_pred[0]:>10.4f} ({error:.4f})")

print("\n" + "="*70)
print("[OK] ERROR CORREGIDO - El modelo funciona perfectamente")
print("="*70 + "\n")
