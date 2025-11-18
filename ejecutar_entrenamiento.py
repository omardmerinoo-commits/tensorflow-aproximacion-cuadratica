#!/usr/bin/env python3
"""
Script wrapper para ejecutar el entrenamiento sin interactividad de terminal.
"""
import subprocess
import sys
import os

# Cambiar al directorio del proyecto
os.chdir(r'C:\Users\Usuario\Desktop\tensorflow-aproximacion-cuadratica')

# Asegurarse de que los paquetes necesarios estén instalados
print("Instalando dependencias...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'matplotlib', 'scikit-learn', 'tensorflow'], check=False)

# Ejecutar el script de entrenamiento
print("\n" + "="*60)
print("Ejecutando entrenamiento...")
print("="*60 + "\n")

resultado = subprocess.run([sys.executable, 'run_training.py'], capture_output=False, text=True)
sys.exit(resultado.returncode)
