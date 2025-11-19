"""
Script rápido: Verificación de integridad de archivos de proyectos P0-P12
Solo verifica que los archivos existan sin ejecutarlos
"""

import os
import json
from pathlib import Path
from datetime import datetime

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

def main():
    print("\n" + "="*70)
    print(" "*15 + "VERIFICACION DE INTEGRIDAD DE PROYECTOS P0-P12")
    print("="*70 + "\n")
    
    existentes = 0
    faltantes = 0
    detalles = []
    
    print("[VERIFICANDO ARCHIVOS]\n")
    print(f"{'ID':<5} {'Estado':<10} {'Ruta Archivo':<50}")
    print("-" * 70)
    
    for proyecto in PROYECTOS:
        app_path = Path(proyecto['path']) / "aplicaciones" / proyecto['app']
        
        if app_path.exists():
            print(f"{proyecto['id']:<5} {'OK':<10} {str(app_path):<50}")
            existentes += 1
            detalles.append({
                "id": proyecto['id'],
                "estado": "EXISTE",
                "ruta": str(app_path),
                "tamanio": app_path.stat().st_size,
                "fecha_mod": datetime.fromtimestamp(app_path.stat().st_mtime).isoformat()
            })
        else:
            print(f"{proyecto['id']:<5} {'FALTA':<10} {str(app_path):<50}")
            faltantes += 1
            detalles.append({
                "id": proyecto['id'],
                "estado": "FALTA",
                "ruta": str(app_path)
            })
    
    print("\n" + "="*70)
    print("[RESUMEN]")
    print("="*70)
    print(f"Total proyectos: {len(PROYECTOS)}")
    print(f"Archivos OK:     {existentes}")
    print(f"Archivos falta:  {faltantes}")
    print(f"Porcentaje:      {(existentes/len(PROYECTOS)*100):.1f}%")
    
    # Guardar reporte
    reporte = {
        "titulo": "Verificacion de Integridad - Proyectos P0-P12",
        "fecha": datetime.now().isoformat(),
        "resumen": {
            "total": len(PROYECTOS),
            "existentes": existentes,
            "faltantes": faltantes,
            "porcentaje": (existentes/len(PROYECTOS)*100)
        },
        "detalles": detalles
    }
    
    os.makedirs('outputs/validacion', exist_ok=True)
    with open('outputs/validacion/verificacion_integridad.json', 'w') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"\n[ARCHIVO] Reporte guardado: outputs/validacion/verificacion_integridad.json")
    
    print("\n" + "="*70)
    if faltantes == 0:
        print("[OK] TODOS LOS ARCHIVOS DE PROYECTOS EXISTEN")
    else:
        print(f"[WARN] {faltantes} archivo(s) faltante(s)")
    print("="*70 + "\n")
    
    return 0 if faltantes == 0 else 1

if __name__ == "__main__":
    main()
