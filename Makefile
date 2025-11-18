.PHONY: help install test build validate clean docs

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias"
	@echo "  make test       - Ejecutar todos los tests"
	@echo "  make build      - Build completo (tests + validación)"
	@echo "  make validate   - Validar estilo de código"
	@echo "  make clean      - Limpiar archivos generados"
	@echo "  make docs       - Generar documentación"

install:
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencias instaladas"

test:
	$(PYTHON) build.py test
	@echo "✅ Tests ejecutados"

build:
	$(PYTHON) build.py build
	@echo "✅ Build completado"

validate:
	$(PYTHON) build.py validate
	@echo "✅ Validación completada"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Archivos limpios"

docs:
	@echo "Generando documentación..."
	$(PYTHON) -m pip install sphinx
	@echo "✅ Documentación lista"

run-proyecto5:
	cd proyecto5_clasificacion_fases && $(PYTHON) run_fases.py

run-proyecto6:
	cd proyecto6_funciones_nolineales && $(PYTHON) run_funciones.py

run-proyecto7:
	cd proyecto7_materiales && $(PYTHON) run_materiales.py

run-proyecto8:
	cd proyecto8_clasificacion_musica && $(PYTHON) run_musica.py

run-proyecto9:
	cd proyecto9_vision_computacional && $(PYTHON) run_vision.py

run-proyecto10:
	cd proyecto10_qutip_basico && $(PYTHON) run_qutip_basico.py

run-proyecto11:
	cd proyecto11_decoherencia && $(PYTHON) run_decoherencia.py

run-proyecto12:
	cd proyecto12_qubits_entrelazados && $(PYTHON) run_entrelazados.py

run-all-new: run-proyecto5 run-proyecto6 run-proyecto7 run-proyecto8 run-proyecto9 run-proyecto10 run-proyecto11 run-proyecto12
	@echo "✅ Todos los proyectos ejecutados"
