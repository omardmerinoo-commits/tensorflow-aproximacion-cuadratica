#!/usr/bin/env python3
"""Test rápido de las tres nuevas aplicaciones: P10, P11, P12"""

import sys
import json
import subprocess
from pathlib import Path

def ejecutar_aplicacion(ruta_app, nombre_proyecto):
    """Ejecuta una aplicación y verifica que genere su reporte"""
    print(f"\n[TEST] {nombre_proyecto}")
    print("=" * 70)
    
    try:
        # Ejecutar la aplicación
        resultado = subprocess.run(
            [sys.executable, str(ruta_app)],
            capture_output=True,
            timeout=120,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Verificar si tuvo éxito
        if resultado.returncode == 0:
            print(f"[OK] Ejecución completada exitosamente")
            # Mostrar últimas líneas
            lineas = resultado.stdout.split('\n')
            for linea in lineas[-10:]:
                if linea.strip():
                    print(f"  {linea}")
            return True
        else:
            print(f"[ERROR] Código de retorno: {resultado.returncode}")
            if resultado.stderr:
                print(f"Stderr:\n{resultado.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Aplicación excedió 120 segundos")
        return False
    except Exception as e:
        print(f"[EXCEPCIÓN] {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "TEST DE NUEVAS APLICACIONES P10-P12")
    print("=" * 70)
    
    resultados = {}
    
    # P10: Series Temporales
    p10_path = Path("proyecto10_series/aplicaciones/predictor_series.py")
    if p10_path.exists():
        resultados['P10'] = ejecutar_aplicacion(p10_path, "P10 - Series Temporales (LSTM)")
    else:
        print(f"[ERROR] No encontrada: {p10_path}")
        resultados['P10'] = False
    
    # P11: Clasificador Sentimientos
    p11_path = Path("proyecto11_nlp/aplicaciones/clasificador_sentimientos.py")
    if p11_path.exists():
        resultados['P11'] = ejecutar_aplicacion(p11_path, "P11 - Clasificador Sentimientos (RNN+Embedding)")
    else:
        print(f"[ERROR] No encontrada: {p11_path}")
        resultados['P11'] = False
    
    # P12: Generador Imágenes
    p12_path = Path("proyecto12_generador/aplicaciones/generador_imagenes.py")
    if p12_path.exists():
        resultados['P12'] = ejecutar_aplicacion(p12_path, "P12 - Generador Imágenes (Autoencoder)")
    else:
        print(f"[ERROR] No encontrada: {p12_path}")
        resultados['P12'] = False
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    
    exitosos = sum(1 for v in resultados.values() if v)
    total = len(resultados)
    
    for nombre, exito in resultados.items():
        estado = "[OK]" if exito else "[ERROR]"
        print(f"{estado} {nombre}")
    
    print(f"\nTotal: {exitosos}/{total} aplicaciones ejecutadas correctamente")
    
    # Guardar reporte
    reporte = {
        "test_date": str(Path.cwd()),
        "resultados": {nombre: ("OK" if v else "ERROR") for nombre, v in resultados.items()},
        "exitosos": exitosos,
        "total": total
    }
    
    Path("outputs/validacion").mkdir(parents=True, exist_ok=True)
    with open("outputs/validacion/test_nuevas_aplicaciones.json", "w") as f:
        json.dump(reporte, f, indent=2)
    
    print(f"\n[REPORTE] Guardado en: outputs/validacion/test_nuevas_aplicaciones.json")
    
    return 0 if exitosos == total else 1

if __name__ == "__main__":
    sys.exit(main())
