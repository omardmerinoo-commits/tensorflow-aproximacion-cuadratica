#!/usr/bin/env python3
"""
Ejecutar tests y guardar resultados en archivo
"""

import subprocess
import sys
import os

os.chdir(r'C:\Users\Usuario\Desktop\tensorflow-aproximacion-cuadratica')

print("Ejecutando tests...")
resultado = subprocess.run([
    sys.executable, '-m', 'pytest', 'test_model.py', 
    '-v', '--tb=short', '-x'
], capture_output=True, text=True, timeout=300)

# Guardar resultados
with open('test_results.txt', 'w', encoding='utf-8', errors='ignore') as f:
    f.write("RESULTADOS DE TESTS\n")
    f.write("="*60 + "\n\n")
    f.write("STDOUT:\n")
    f.write(resultado.stdout[-2000:] if len(resultado.stdout) > 2000 else resultado.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(resultado.stderr[-1000:] if len(resultado.stderr) > 1000 else resultado.stderr)
    f.write(f"\n\nCódigo de retorno: {resultado.returncode}\n")

# Mostrar resumen
lineas = resultado.stdout.split('\n')
for linea in lineas[-20:]:
    if linea.strip():
        print(linea)

print(f"\nResultados guardados en: test_results.txt")
