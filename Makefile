# Sheily AI - Makefile
# ====================

.PHONY: help install dev-install test lint format clean audit train

# Default target
help:
	@echo "Sheily AI - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Instalar dependencias completas"
	@echo "  make dev-install    - Instalar en modo desarrollo editable"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Ejecutar tests"
	@echo "  make lint           - Linter de código"
	@echo "  make format         - Formatear código"
	@echo "  make clean          - Limpiar archivos temporales"
	@echo ""
	@echo "Project Management:"
	@echo "  make audit          - Auditoría completa del proyecto"
	@echo "  make check-branches - Validar todas las ramas"
	@echo "  make list-branches  - Listar ramas disponibles"
	@echo ""
	@echo "Training:"
	@echo "  make train BRANCH=physics  - Entrenar rama específica"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

lint:
	@echo "Running Black (check)"
	black --check sheily_train/ sheily_core/ tools/
	@echo "Running isort (check-only)"
	isort --check-only sheily_train/ sheily_core/ tools/
	@echo "Running Ruff"
	ruff sheily_train/ sheily_core/ tools/
	@echo "Running Flake8"
	flake8 sheily_train/ sheily_core/ tools/
	@echo "Running MyPy"
	mypy sheily_train/ sheily_core/ --ignore-missing-imports

format:
	isort sheily_train/ sheily_core/ tools/
	black sheily_train/ sheily_core/ tools/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/
	@echo "✅ Archivos temporales eliminados"

# Project Management
audit:
	python3 tools/maintenance/audit_complete_project.py

check-branches:
	python3 tools/branch_management/check_all_datasets.py

list-branches:
	python3 sheily_train/train_branch.py --list-branches

deep-search:
	python3 tools/maintenance/deep_search_issues.py

# Training
train:
ifndef BRANCH
	@echo "❌ Error: Especifica BRANCH=nombre_rama"
	@echo "Ejemplo: make train BRANCH=physics"
else
	python3 sheily_train/train_branch.py --branch $(BRANCH) --lora
endif

# Documentation
docs:
	@echo "📚 Documentación disponible en:"
	@echo "  • README.md"
	@echo "  • docs/README.md"
	@echo "  • docs/TOOLS_GUIDE.md"
	@echo "  • docs/ARCHITECTURE.md"
	@echo "  • sheily_train/README.md"

# Version
version:
	@echo "Sheily AI v1.0.0"
	@python3 --version
	@pip --version
