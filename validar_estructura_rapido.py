#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VALIDACIÓN RÁPIDA - REPORTES Y ANÁLISIS
========================================
Validación sin ejecutar entrenamientos largos
- Análisis estático de código
- Reporte de estructura
- Verificación de imports y dependencias
- Estado del repositorio git
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

WORKSPACE = Path(__file__).parent

def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analiza un archivo Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        return {
            "path": str(filepath.relative_to(WORKSPACE)),
            "lines": len(lines),
            "has_type_hints": bool(re.search(r'def\s+\w+\([^)]*\)\s*->', content)),
            "has_docstrings": bool(re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', content, re.DOTALL)),
            "has_tests": "test_" in filepath.name,
            "imports": len(re.findall(r'^\s*(import|from)\s', content, re.MULTILINE)),
            "classes": len(re.findall(r'^class\s+\w+', content, re.MULTILINE)),
            "functions": len(re.findall(r'^def\s+\w+', content, re.MULTILINE)),
            "size_kb": filepath.stat().st_size / 1024
        }
    except Exception as e:
        return {"path": str(filepath.relative_to(WORKSPACE)), "error": str(e)}

def analyze_project(project_dir: Path) -> Dict[str, Any]:
    """Analiza un proyecto completo"""
    py_files = list(project_dir.glob("**/*.py"))
    
    analysis = {
        "project": project_dir.name,
        "num_files": len(py_files),
        "files": [],
        "total_lines": 0,
        "total_classes": 0,
        "total_functions": 0,
        "files_with_type_hints": 0,
        "files_with_docstrings": 0,
        "test_files": 0,
        "files": []
    }
    
    for py_file in py_files:
        file_analysis = analyze_file(py_file)
        if "error" not in file_analysis:
            analysis["files"].append(file_analysis)
            analysis["total_lines"] += file_analysis.get("lines", 0)
            analysis["total_classes"] += file_analysis.get("classes", 0)
            analysis["total_functions"] += file_analysis.get("functions", 0)
            
            if file_analysis.get("has_type_hints"):
                analysis["files_with_type_hints"] += 1
            if file_analysis.get("has_docstrings"):
                analysis["files_with_docstrings"] += 1
            if file_analysis.get("has_tests"):
                analysis["test_files"] += 1
    
    return analysis

def check_dependencies() -> Dict[str, List[str]]:
    """Verifica las dependencias importadas en el proyecto"""
    imports = {}
    
    for py_file in WORKSPACE.glob("**/*.py"):
        if ".venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extraer imports
            for match in re.finditer(r'^\s*(?:from|import)\s+(\w+)', content, re.MULTILINE):
                lib = match.group(1)
                if lib not in imports:
                    imports[lib] = []
                imports[lib].append(str(py_file.relative_to(WORKSPACE)))
        except:
            pass
    
    return imports

def main():
    """Función principal"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*18 + "ANÁLISIS RÁPIDO - VALIDACIÓN DE ESTRUCTURA" + " "*17 + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " "*58 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Analizar todos los proyectos
    project_dirs = sorted([d for d in WORKSPACE.glob("proyecto*") if d.is_dir()])
    
    print(f"📁 Encontrados {len(project_dirs)} proyectos\n")
    
    projects_analysis = []
    total_lines = 0
    total_files = 0
    total_classes = 0
    total_functions = 0
    
    for proj_dir in project_dirs:
        print(f"Analizando: {proj_dir.name}...", end=" ")
        analysis = analyze_project(proj_dir)
        projects_analysis.append(analysis)
        
        total_lines += analysis["total_lines"]
        total_files += analysis["num_files"]
        total_classes += analysis["total_classes"]
        total_functions += analysis["total_functions"]
        
        status = ""
        if analysis["num_files"] > 0:
            status = f"✓ ({analysis['num_files']} archivos, {analysis['total_lines']} líneas)"
        print(status)
    
    print("\n" + "="*80)
    print("RESUMEN GENERAL")
    print("="*80)
    
    print(f"✓ Proyectos: {len(project_dirs)}")
    print(f"✓ Archivos Python: {total_files}")
    print(f"✓ Líneas de código: {total_lines:,}")
    print(f"✓ Clases: {total_classes}")
    print(f"✓ Funciones: {total_functions}")
    
    # Analizar dependencias
    print("\n" + "="*80)
    print("LIBRERÍAS IMPORTADAS")
    print("="*80)
    
    dependencies = check_dependencies()
    external_libs = {k: v for k, v in dependencies.items() if k not in ['sys', 'os', 'json', 'numpy', 'matplotlib', 'tensorflow', 'keras', 'scipy', 'sklearn', 'qutip', 'librosa', 'cv2', 'pandas', 're', 'pathlib', 'subprocess', 'datetime', 'typing', 'pickle', 'warnings', 'collections', 'math', 'random']}
    
    print("\nLibrerías principales detectadas:")
    major_libs = ['tensorflow', 'keras', 'numpy', 'matplotlib', 'scipy', 'qutip', 'sklearn', 'cv2', 'librosa', 'pandas']
    for lib in major_libs:
        if lib in dependencies:
            print(f"  ✓ {lib:15s} ({len(set(dependencies[lib]))} archivos)")
    
    # Verificar archivos README
    print("\n" + "="*80)
    print("DOCUMENTACIÓN")
    print("="*80)
    
    readme_files = list(WORKSPACE.glob("*/README.md")) + [WORKSPACE / "README.md"]
    readme_files = [f for f in readme_files if f.exists()]
    print(f"✓ Archivos README encontrados: {len(readme_files)}")
    
    for readme in readme_files:
        size = readme.stat().st_size
        print(f"  - {readme.relative_to(WORKSPACE)} ({size} bytes)")
    
    # Verificar test files
    print("\n" + "="*80)
    print("TESTS UNITARIOS")
    print("="*80)
    
    test_files = list(WORKSPACE.glob("*/test_*.py")) + [WORKSPACE / "test_model.py"]
    test_files = [f for f in test_files if f.exists()]
    print(f"✓ Archivos de test encontrados: {len(test_files)}")
    
    for test_file in sorted(test_files):
        lines = len(open(test_file).readlines())
        print(f"  - {test_file.relative_to(WORKSPACE)} ({lines} líneas)")
    
    # Reporte JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "workspace": str(WORKSPACE),
        "proyectos": len(project_dirs),
        "estadisticas": {
            "total_archivos": total_files,
            "total_lineas": total_lines,
            "total_clases": total_classes,
            "total_funciones": total_functions,
            "archivos_readme": len(readme_files),
            "archivos_test": len(test_files)
        },
        "proyectos_detalle": projects_analysis,
        "librerías": {
            "principales": {k: len(set(v)) for k, v in dependencies.items() if k in major_libs},
            "total_únicas": len(dependencies)
        }
    }
    
    # Guardar reporte
    report_path = WORKSPACE / "VALIDACION_ESTRUCTURA_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Reporte guardado en: {report_path}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
