#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VALIDACIÓN COMPLETA DE TODOS LOS PROYECTOS
==========================================
Script para ejecutar la validación integral de los 12 proyectos.
- Tests unitarios
- Cobertura de código
- Ejecución de proyectos
- Generación de reportes
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

WORKSPACE = Path(__file__).parent
PYTHON_EXE = WORKSPACE / ".venv_py313" / "Scripts" / "python.exe"

# 12 Proyectos a validar
PROYECTOS = {
    "proyecto0_original": "Aproximación Cuadrática",
    "proyecto1_oscilaciones": "Oscilaciones Amortiguadas",
    "proyecto2_web": "API REST Web",
    "proyecto3_qubit": "Simulador de Qubit",
    "proyecto4_estadistica": "Análisis Estadístico",
    "proyecto5_clasificacion_fases": "Clasificación de Fases",
    "proyecto6_funciones_nolineales": "Aproximador de Funciones",
    "proyecto7_materiales": "Predicción de Materiales",
    "proyecto8_clasificacion_musica": "Clasificación Música",
    "proyecto9_vision_computacional": "Conteo de Objetos",
    "proyecto10_qutip_basico": "Simulador QuTiP Básico",
    "proyecto11_decoherencia": "Decoherencia Cuántica",
    "proyecto12_qubits_entrelazados": "Qubits Entrelazados"
}

def run_command(cmd: List[str], cwd: str = None, verbose: bool = True) -> tuple[int, str, str]:
    """Ejecuta un comando y retorna código de salida, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "TIMEOUT: Comando excedió 300 segundos"
    except Exception as e:
        return 1, "", str(e)

def run_pytest() -> Dict[str, Any]:
    """Ejecuta pytest con cobertura"""
    print("\n" + "="*80)
    print("EJECUTANDO PYTEST CON COBERTURA")
    print("="*80)
    
    cmd = [
        str(PYTHON_EXE), "-m", "pytest", 
        str(WORKSPACE),
        "-v", "--tb=short", "--cov=.", "--cov-report=json", "--cov-report=term-missing"
    ]
    
    returncode, stdout, stderr = run_command(cmd)
    
    # Parsear resultados
    results = {
        "timestamp": datetime.now().isoformat(),
        "success": returncode == 0,
        "return_code": returncode,
        "output": stdout,
        "errors": stderr
    }
    
    # Contar tests
    import re
    passed = len(re.findall(r"PASSED", stdout))
    failed = len(re.findall(r"FAILED", stdout))
    errors = len(re.findall(r"ERROR", stdout))
    
    results["tests_passed"] = passed
    results["tests_failed"] = failed
    results["tests_errors"] = errors
    results["tests_total"] = passed + failed + errors
    
    print(f"\n✓ Tests Passed: {passed}")
    print(f"✗ Tests Failed: {failed}")
    print(f"⚠ Tests Errors: {errors}")
    print(f"Total: {passed + failed + errors}")
    
    return results

def run_project_execution() -> Dict[str, Any]:
    """Ejecuta los proyectos principal (Proyecto 0)"""
    print("\n" + "="*80)
    print("EJECUTANDO PROYECTOS PRINCIPALES")
    print("="*80)
    
    # Ejecutar proyecto 0 (modelo cuadrático)
    run_file = WORKSPACE / "run_training.py"
    if run_file.exists():
        print(f"\nEjecutando: {run_file}")
        returncode, stdout, stderr = run_command([str(PYTHON_EXE), str(run_file)])
        
        if returncode == 0:
            print("✓ Proyecto 0 ejecutado exitosamente")
            return {
                "success": True,
                "proyecto": "proyecto0_original",
                "output": stdout[:500]  # Primeros 500 caracteres
            }
        else:
            print(f"✗ Error en Proyecto 0: {stderr[:200]}")
            return {
                "success": False,
                "proyecto": "proyecto0_original",
                "error": stderr[:500]
            }
    
    return {"success": False, "error": "run_training.py no encontrado"}

def validate_code_quality() -> Dict[str, Any]:
    """Valida calidad de código: type hints, docstrings, PEP 8"""
    print("\n" + "="*80)
    print("VALIDANDO CALIDAD DE CÓDIGO")
    print("="*80)
    
    results = {
        "files_checked": 0,
        "type_hints_coverage": 0.0,
        "docstring_coverage": 0.0,
        "files": []
    }
    
    # Buscar todos los archivos .py de proyectos
    py_files = list(WORKSPACE.glob("proyecto*/**.py")) + list(WORKSPACE.glob("*.py"))
    
    files_with_hints = 0
    files_with_docstrings = 0
    
    for py_file in py_files:
        if "__pycache__" in str(py_file) or ".venv" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                has_hints = "def " in content and "->" in content
                has_docstrings = '"""' in content or "'''" in content
                
                if has_hints:
                    files_with_hints += 1
                if has_docstrings:
                    files_with_docstrings += 1
                
                results["files"].append({
                    "file": str(py_file.relative_to(WORKSPACE)),
                    "type_hints": has_hints,
                    "docstrings": has_docstrings
                })
                results["files_checked"] += 1
        except Exception as e:
            print(f"Error procesando {py_file}: {e}")
    
    if results["files_checked"] > 0:
        results["type_hints_coverage"] = files_with_hints / results["files_checked"]
        results["docstring_coverage"] = files_with_docstrings / results["files_checked"]
    
    print(f"✓ Archivos analizados: {results['files_checked']}")
    print(f"✓ Type hints: {files_with_hints}/{results['files_checked']} ({results['type_hints_coverage']*100:.1f}%)")
    print(f"✓ Docstrings: {files_with_docstrings}/{results['files_checked']} ({results['docstring_coverage']*100:.1f}%)")
    
    return results

def check_git_status() -> Dict[str, Any]:
    """Verifica estado del repositorio git"""
    print("\n" + "="*80)
    print("VERIFICANDO INTEGRIDAD GIT")
    print("="*80)
    
    os.chdir(str(WORKSPACE))
    
    # Número de commits
    returncode, stdout, _ = run_command(["git", "rev-list", "--count", "HEAD"])
    commit_count = stdout.strip() if returncode == 0 else "unknown"
    
    # Último commit
    returncode, stdout, _ = run_command(["git", "log", "-1", "--pretty=format:%H %s"])
    last_commit = stdout.strip() if returncode == 0 else "unknown"
    
    # Estado del repo
    returncode, stdout, _ = run_command(["git", "status", "--porcelain"])
    uncommitted = len(stdout.strip().split('\n')) if stdout.strip() else 0
    
    # Rama actual
    returncode, stdout, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    current_branch = stdout.strip() if returncode == 0 else "unknown"
    
    print(f"✓ Rama: {current_branch}")
    print(f"✓ Total commits: {commit_count}")
    print(f"✓ Último commit: {last_commit[:50]}")
    print(f"✓ Cambios sin commitear: {uncommitted}")
    
    return {
        "branch": current_branch,
        "total_commits": commit_count,
        "last_commit": last_commit,
        "uncommitted_changes": uncommitted,
        "status_clean": uncommitted == 0
    }

def generate_final_report(test_results, exec_results, quality_results, git_results) -> str:
    """Genera reporte final en JSON"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "workspace": str(WORKSPACE),
        "proyectos_totales": len(PROYECTOS),
        "tests": test_results,
        "execution": exec_results,
        "code_quality": quality_results,
        "git": git_results,
        "summary": {
            "all_tests_passed": test_results.get("tests_failed", 0) == 0 and test_results.get("tests_errors", 0) == 0,
            "code_quality_ok": quality_results.get("type_hints_coverage", 0) >= 0.8,
            "repository_clean": git_results.get("status_clean", False)
        }
    }
    
    return json.dumps(report, indent=2, ensure_ascii=False)

def main():
    """Función principal"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "VALIDACIÓN COMPLETA DEL PROYECTO" + " "*25 + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " "*58 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # 1. Ejecutar tests
    print("FASE 1/5: Tests unitarios...")
    test_results = run_pytest()
    
    # 2. Ejecutar proyectos
    print("\nFASE 2/5: Ejecución de proyectos...")
    exec_results = run_project_execution()
    
    # 3. Validar calidad de código
    print("\nFASE 3/5: Calidad de código...")
    quality_results = validate_code_quality()
    
    # 4. Verificar git
    print("\nFASE 4/5: Integridad Git...")
    git_results = check_git_status()
    
    # 5. Generar reporte
    print("\nFASE 5/5: Generando reporte...")
    report = generate_final_report(test_results, exec_results, quality_results, git_results)
    
    # Guardar reporte
    report_path = WORKSPACE / "VALIDACION_COMPLETA_REPORT.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Reporte guardado en: {report_path}")
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    summary = json.loads(report)["summary"]
    print(f"✓ Tests: {'TODOS PASARON ✓' if summary['all_tests_passed'] else 'ALGUNOS FALLARON ✗'}")
    print(f"✓ Calidad: {'OK ✓' if summary['code_quality_ok'] else 'MEJORAR ⚠'}")
    print(f"✓ Repositorio: {'LIMPIO ✓' if summary['repository_clean'] else 'CAMBIOS PENDIENTES ⚠'}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
