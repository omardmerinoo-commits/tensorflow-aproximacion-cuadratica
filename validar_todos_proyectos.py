"""
Script de Validación Completa: Ejecuta y verifica todos los proyectos P0-P12
Genera un reporte de estado de cada uno
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Lista de proyectos y sus aplicaciones
PROYECTOS = [
    {"id": "P0", "path": "proyecto0_original", "app": "predictor_precios_casas.py"},
    {"id": "P1", "path": "proyecto1_oscilaciones", "app": "predictor_consumo_energia.py"},
    {"id": "P2", "path": "proyecto2_web", "app": "detector_fraude.py"},
    {"id": "P3", "path": "proyecto3_qubits", "app": "clasificador_diagnostico.py"},
    {"id": "P4", "path": "proyecto4_estadistica", "app": "segmentador_clientes.py"},
    {"id": "P5", "path": "proyecto5_clasificador", "app": "compresor_imagenes_pca.py"},
    {"id": "P6", "path": "proyecto6_funciones", "app": "reconocedor_digitos.py"},
    {"id": "P7", "path": "proyecto7_audio", "app": "clasificador_ruido.py"},
    {"id": "P8", "path": "proyecto8_materiales", "app": "detector_objetos.py"},
    {"id": "P9", "path": "proyecto9_imagenes", "app": "segmentador_semantico.py"},
    {"id": "P10", "path": "proyecto10_series", "app": "predictor_series.py"},
    {"id": "P11", "path": "proyecto11_nlp", "app": "clasificador_sentimientos.py"},
    {"id": "P12", "path": "proyecto12_generador", "app": "generador_imagenes.py"},
]

def validar_proyecto(proyecto, venv_python):
    """Valida un proyecto individual"""
    print(f"\n{'='*70}")
    print(f"  {proyecto['id']}: {proyecto['app']}")
    print(f"{'='*70}")
    
    app_path = Path(proyecto['path']) / "aplicaciones" / proyecto['app']
    
    if not app_path.exists():
        print(f"[ERROR] ARCHIVO NO ENCONTRADO: {app_path}")
        return {
            "id": proyecto['id'],
            "archivo": str(app_path),
            "estado": "ERROR",
            "razon": "Archivo no encontrado",
            "tiempo": 0
        }
    
    print(f"[FILE] Ejecutando: {app_path}")
    
    try:
        inicio = datetime.now()
        resultado = subprocess.run(
            [venv_python, str(app_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
            encoding='utf-8',
            errors='replace'
        )
        tiempo = (datetime.now() - inicio).total_seconds()
        
        if resultado.returncode == 0:
            print(f"[OK] EXITO en {tiempo:.2f}s")
            
            # Mostrar últimas líneas del output
            lineas = resultado.stdout.split('\n')
            output_resumen = '\n'.join(lineas[-10:])
            print(f"\n[OUTPUT] Salida (ultimas lineas):\n{output_resumen}")
            
            return {
                "id": proyecto['id'],
                "archivo": str(app_path),
                "estado": "ÉXITO",
                "tiempo": tiempo,
                "output_preview": output_resumen[:200]
            }
        else:
            print(f"[ERROR] Error (codigo: {resultado.returncode}) en {tiempo:.2f}s")
            
            # Mostrar error
            error_resumen = resultado.stderr[-500:] if resultado.stderr else resultado.stdout[-500:]
            print(f"\n[WARN] Error:\n{error_resumen}")
            
            return {
                "id": proyecto['id'],
                "archivo": str(app_path),
                "estado": "ERROR",
                "razon": "Ejecución fallida",
                "tiempo": tiempo,
                "error": error_resumen[:200]
            }
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] >120s")
        return {
            "id": proyecto['id'],
            "archivo": str(app_path),
            "estado": "TIMEOUT",
            "razon": "Excedio tiempo limite (120s)",
            "tiempo": 120
        }
    except Exception as e:
        print(f"[ERROR] EXCEPCION: {str(e)}")
        return {
            "id": proyecto['id'],
            "archivo": str(app_path),
            "estado": "EXCEPCION",
            "razon": str(e),
            "tiempo": 0
        }

def main():
    print("\n" + "="*70)
    print(" "*15 + "VALIDACION COMPLETA DE TODOS LOS PROYECTOS")
    print("="*70 + "\n")
    
    # Determinar ruta de Python
    venv_path = Path(".venv_py313/Scripts/python.exe")
    if not venv_path.exists():
        print(f"[WARN] Usando python del sistema (venv no encontrado)")
        python_exe = sys.executable
    else:
        python_exe = str(venv_path.absolute())
    
    print(f"[PYTHON] {python_exe}\n")
    
    # Validar todos los proyectos
    resultados = []
    exitosos = 0
    fallidos = 0
    
    for proyecto in PROYECTOS:
        resultado = validar_proyecto(proyecto, python_exe)
        resultados.append(resultado)
        
        if resultado['estado'] == 'ÉXITO':
            exitosos += 1
        else:
            fallidos += 1
    
    # Reporte final
    print("\n" + "="*70)
    print(" "*20 + "REPORTE FINAL")
    print("="*70)
    
    print(f"\n[RESUMEN]:")
    print(f"  OK       : {exitosos}/{len(PROYECTOS)}")
    print(f"  FALLIDO  : {fallidos}/{len(PROYECTOS)}")
    print(f"  Porcentaje: {(exitosos/len(PROYECTOS)*100):.1f}%")
    
    # Tabla de resultados
    print(f"\n[TABLA]\n")
    print(f"{'ID':<5} {'Estado':<10} {'Tiempo':<10} {'Nota':<40}")
    print("-" * 70)
    
    for resultado in resultados:
        estado = resultado['estado']
        tiempo = f"{resultado['tiempo']:.2f}s" if resultado.get('tiempo') else "N/A"
        razon = resultado.get('razon', '')[:40]
        print(f"{resultado['id']:<5} {estado:<10} {tiempo:<10} {razon:<40}")
    
    # Guardar reporte JSON
    reporte = {
        "titulo": "Validación Completa de Proyectos P0-P12",
        "fecha": datetime.now().isoformat(),
        "resumen": {
            "total": len(PROYECTOS),
            "exitosos": exitosos,
            "fallidos": fallidos,
            "porcentaje": (exitosos/len(PROYECTOS)*100)
        },
        "resultados": resultados
    }
    
    os.makedirs('outputs/validacion', exist_ok=True)
    with open('outputs/validacion/reporte_validacion.json', 'w') as f:
        json.dump(reporte, f, indent=2)
    
    print(f"\n[OK] Reporte guardado en: outputs/validacion/reporte_validacion.json")
    
    # Detalles de fallos
    fallos = [r for r in resultados if r['estado'] != 'ÉXITO']
    if fallos:
        print(f"\n[FALLOS]\n")
        for fallo in fallos:
            print(f"  {fallo['id']}: {fallo.get('razon', 'Error desconocido')}")
            if 'error' in fallo:
                print(f"    -> {fallo['error'][:100]}")
    
    print("\n" + "="*70)
    if exitosos == len(PROYECTOS):
        print("[OK] TODOS LOS PROYECTOS FUNCIONAN CORRECTAMENTE")
    else:
        print(f"[WARN] {fallidos} proyecto(s) con problemas - revisar arriba")
    print("="*70 + "\n")
    
    return 0 if fallidos == 0 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
