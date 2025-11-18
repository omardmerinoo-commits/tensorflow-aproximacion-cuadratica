#!/usr/bin/env python3
"""
Build script para compilar y validar todos los proyectos.
Ejecuta tests, genera reportes y prepara el código para producción.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


class BuildSystem:
    """Sistema de build para todos los proyectos."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.venv_python = self.root_dir / ".venv" / "Scripts" / "python.exe"
        self.projects = self._discover_projects()
        self.results = {}
        
    def _discover_projects(self):
        """Descubrir todos los proyectos."""
        projects = []
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir() and item.name.startswith("proyecto"):
                projects.append({
                    'name': item.name,
                    'path': item,
                    'test_file': item / f"test_{item.name.split('_', 1)[1] if '_' in item.name else 'test'}.py"
                })
        return projects
    
    def run_tests(self):
        """Ejecutar tests de todos los proyectos."""
        print("=" * 80)
        print("EJECUTANDO TESTS DE TODOS LOS PROYECTOS")
        print("=" * 80)
        
        for project in self.projects:
            print(f"\n📋 Proyecto: {project['name']}")
            print("-" * 80)
            
            test_files = list(project['path'].glob("test_*.py"))
            if not test_files:
                print(f"⚠️  No se encontraron tests")
                self.results[project['name']] = {'status': 'skipped', 'reason': 'no tests found'}
                continue
            
            try:
                # Ejecutar pytest
                cmd = [
                    str(self.venv_python), "-m", "pytest",
                    str(test_files[0]), "-v", "--tb=short",
                    "--cov=" + project['path'].name,
                    "--cov-report=term-missing"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project['path']))
                
                if result.returncode == 0:
                    print(f"✅ Tests pasados")
                    self.results[project['name']] = {'status': 'passed', 'tests': 'ejecutados'}
                else:
                    print(f"❌ Tests fallidos")
                    print(result.stdout)
                    print(result.stderr)
                    self.results[project['name']] = {'status': 'failed', 'error': result.stderr[:200]}
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                self.results[project['name']] = {'status': 'error', 'error': str(e)[:200]}
    
    def validate_code_style(self):
        """Validar estilo de código."""
        print("\n" + "=" * 80)
        print("VALIDANDO ESTILO DE CÓDIGO")
        print("=" * 80)
        
        for project in self.projects:
            py_files = list(project['path'].glob("*.py"))
            for py_file in py_files:
                if py_file.name.startswith(('test_', 'run_', 'modelo_', 'predictor_', 'clasificador_', 'simulador_', 'aproximador_', 'contador_', 'generador_', 'extractor_')):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Validaciones básicas
                            if '"""' in content or "'''" in content:
                                print(f"✅ {py_file.name}: Tiene docstrings")
                            if 'import' in content:
                                print(f"✅ {py_file.name}: Tiene imports")
                    except Exception as e:
                        print(f"❌ Error al validar {py_file.name}: {e}")
    
    def generate_report(self):
        """Generar reporte de build."""
        print("\n" + "=" * 80)
        print("REPORTE DE BUILD")
        print("=" * 80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_projects': len(self.projects),
            'projects': self.projects,
            'results': self.results,
            'status': 'success' if all(r.get('status') == 'passed' for r in self.results.values()) else 'partial'
        }
        
        report_file = self.root_dir / "BUILD_REPORT.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Reporte guardado en: {report_file}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        return report
    
    def build_all(self):
        """Ejecutar build completo."""
        print("\n🔨 INICIANDO BUILD COMPLETO\n")
        self.run_tests()
        self.validate_code_style()
        report = self.generate_report()
        
        if report['status'] == 'success':
            print("\n✅ BUILD EXITOSO")
            return 0
        else:
            print("\n⚠️  BUILD COMPLETADO CON WARNINGS")
            return 1


def main():
    """Punto de entrada principal."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "build"
    
    builder = BuildSystem()
    
    if command == "test":
        builder.run_tests()
    elif command == "validate":
        builder.validate_code_style()
    elif command == "report":
        builder.generate_report()
    elif command == "build":
        return builder.build_all()
    else:
        print(f"Comando desconocido: {command}")
        print("Comandos disponibles: test, validate, report, build")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
