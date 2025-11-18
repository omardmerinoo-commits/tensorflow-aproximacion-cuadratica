#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUDITORÍA PROFESIONAL DE TODOS LOS PROYECTOS
=============================================
Revisa calidad, estructura, errores potenciales y mejoras
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any

WORKSPACE = Path(__file__).parent
PROYECTOS = [
    "proyecto1_oscilaciones",
    "proyecto2_web",
    "proyecto3_qubit",
    "proyecto4_estadistica",
    "proyecto5_clasificacion_fases",
    "proyecto6_funciones_nolineales",
    "proyecto7_materiales",
    "proyecto8_clasificacion_musica",
    "proyecto9_vision_computacional",
    "proyecto10_qutip_basico",
    "proyecto11_decoherencia",
    "proyecto12_qubits_entrelazados"
]

def audit_file(filepath: Path) -> Dict[str, Any]:
    """Audita un archivo Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax Error: {e}"}
        
        issues = []
        improvements = []
        
        # Check for bad practices
        if "except:" in content:
            issues.append("⚠️  Bare except clause (except:) - should specify exception type")
        
        if "import *" in content:
            issues.append("⚠️  Wildcard imports (import *) - bad practice")
        
        if "TODO" in content or "FIXME" in content or "HACK" in content:
            issues.append("⚠️  TODO/FIXME/HACK comments found")
        
        if "print(" in content and "logging" not in content and "DEBUG" not in filepath.name:
            issues.append("⚠️  Using print() instead of logging")
        
        if len(lines) > 500:
            improvements.append("💡 Archivo muy largo (>500 líneas), considerar módulos separados")
        
        # Count functions and classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Check docstrings
        undocumented = 0
        for func in functions:
            if not ast.get_docstring(func) and not func.name.startswith("_"):
                undocumented += 1
        
        if undocumented > 0:
            issues.append(f"⚠️  {undocumented} funciones sin docstring")
        
        # Check type hints
        functions_without_hints = 0
        for func in functions:
            if func.args.args and not func.returns:
                functions_without_hints += 1
        
        if functions_without_hints > len(functions) * 0.3:
            improvements.append(f"💡 Muchas funciones sin type hints en return ({functions_without_hints})")
        
        return {
            "file": str(filepath.relative_to(WORKSPACE)),
            "lines": len(lines),
            "classes": len(classes),
            "functions": len(functions),
            "issues": issues,
            "improvements": improvements
        }
    except Exception as e:
        return {"error": str(e)}

def audit_project(project_dir: Path) -> Dict[str, Any]:
    """Audita un proyecto completo"""
    py_files = list(project_dir.glob("**/*.py"))
    
    audit = {
        "project": project_dir.name,
        "files_checked": len(py_files),
        "total_issues": 0,
        "total_improvements": 0,
        "files": []
    }
    
    for py_file in sorted(py_files):
        file_audit = audit_file(py_file)
        if "error" not in file_audit:
            audit["files"].append(file_audit)
            audit["total_issues"] += len(file_audit.get("issues", []))
            audit["total_improvements"] += len(file_audit.get("improvements", []))
    
    return audit

def main():
    print("\n" + "="*80)
    print("AUDITORÍA PROFESIONAL - TODOS LOS PROYECTOS")
    print("="*80 + "\n")
    
    all_audits = []
    
    for proyecto in PROYECTOS:
        proj_path = WORKSPACE / proyecto
        if proj_path.exists():
            print(f"🔍 Auditando: {proyecto}...", end=" ")
            audit = audit_project(proj_path)
            all_audits.append(audit)
            
            status = "✅" if audit["total_issues"] == 0 else "⚠️"
            print(f"{status} ({audit['total_issues']} issues, {audit['total_improvements']} mejoras sugeridas)")
    
    # Resumen
    print("\n" + "-"*80)
    print("RESUMEN GENERAL")
    print("-"*80)
    
    total_issues = sum(a["total_issues"] for a in all_audits)
    total_improvements = sum(a["total_improvements"] for a in all_audits)
    
    print(f"✅ Proyectos auditados: {len(all_audits)}")
    print(f"⚠️  Issues totales: {total_issues}")
    print(f"💡 Mejoras sugeridas: {total_improvements}")
    
    # Generar reporte JSON
    report = {
        "timestamp": str(Path.ctime),
        "audits": all_audits,
        "summary": {
            "total_projects": len(all_audits),
            "total_issues": total_issues,
            "total_improvements": total_improvements,
            "status": "PASS" if total_issues == 0 else "REVIEW"
        }
    }
    
    report_path = WORKSPACE / "AUDITORIA_PROYECTOS.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Reporte guardado: {report_path}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
