# 🧪 SHEILY AI - TEST SUITE

Estructura organizada de tests para el proyecto Sheily AI.

---

## 📁 Estructura de Carpetas

```
tests/
├── conftest.py              # Configuración de pytest y fixtures compartidos
├── README.md                # Esta documentación
│
├── unit/                    # Tests unitarios (rápidos, aislados)
│   ├── test_config.py       # Tests de configuración
│   ├── test_subprocess_utils.py  # Tests de subprocess_utils
│   └── test_health.py       # Tests de health checks
│
├── integration/             # Tests de integración (con dependencias)
│   └── test_integration.py  # Tests de integración de componentes
│
├── security/                # Tests de seguridad
│   └── test_security.py     # Tests de seguridad y validación
│
├── e2e/                     # Tests end-to-end (workflow completo)
│   └── test_full_workflow.py  # Tests de workflows completos
│
└── fixtures/                # Datos y fixtures compartidos
    └── sample_data.py       # Datos de ejemplo

```

---

## 🏷️ Markers de Pytest

### Markers Disponibles

- `@pytest.mark.unit` - Tests unitarios (rápidos)
- `@pytest.mark.integration` - Tests de integración
- `@pytest.mark.security` - Tests de seguridad
- `@pytest.mark.e2e` - Tests end-to-end
- `@pytest.mark.slow` - Tests lentos
- `@pytest.mark.requires_gpu` - Tests que requieren GPU
- `@pytest.mark.requires_model` - Tests que requieren archivos de modelo

### Uso de Markers

```python
@pytest.mark.unit
def test_simple_function():
    """Test rápido y aislado"""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_complex_integration():
    """Test de integración lento"""
    pass
```

---

## 🚀 Ejecutar Tests

### Tests Básicos

```bash
# Todos los tests
pytest tests/ -v

# Solo tests unitarios (rápidos)
pytest tests/unit/ -v

# Solo tests de integración
pytest tests/integration/ -v -m integration

# Solo tests de seguridad
pytest tests/security/ -v -m security
```

### Tests por Marker

```bash
# Solo tests unitarios
pytest -m unit -v

# Solo tests de seguridad
pytest -m security -v

# Excluir tests lentos
pytest -m "not slow" -v

# Tests rápidos (unit + no slow)
pytest -m "unit and not slow" -v
```

### Tests con Coverage

```bash
# Coverage completo
pytest tests/ --cov=sheily_core --cov=sheily_train --cov-report=html

# Coverage de módulo específico
pytest tests/unit/test_subprocess_utils.py --cov=sheily_core.utils.subprocess_utils --cov-report=term

# Ver reporte HTML
firefox htmlcov/index.html
```

### Tests en Paralelo

```bash
# Usar múltiples CPUs
pytest tests/ -n auto

# Con 4 workers
pytest tests/ -n 4
```

---

## 🛠️ Fixtures Disponibles

### Path Fixtures

- `project_root` - Path al directorio raíz
- `sheily_core_path` - Path a sheily_core/
- `sheily_train_path` - Path a sheily_train/
- `branches_path` - Path a all-Branches/
- `temp_dir` - Directorio temporal

### Configuration Fixtures

- `test_env_vars` - Variables de entorno para testing
- `mock_config` - Configuración mock

### Model Fixtures

- `mock_model_path` - Path a modelo mock
- `sample_training_data` - Datos de entrenamiento de ejemplo
- `sample_branch_data` - Estructura de branch de ejemplo

### Network Fixtures

- `mock_redis` - Redis mock
- `mock_database` - Database mock

### Security Fixtures

- `safe_command_examples` - Ejemplos de comandos seguros
- `dangerous_command_examples` - Ejemplos de comandos peligrosos
- `subprocess_utils` - Importar subprocess_utils

### Utility Fixtures

- `benchmark_timer` - Timer para benchmarking
- `capture_logs` - Captura de logs

---

## 📝 Escribir Tests

### Template de Test Unitario

```python
#!/usr/bin/env python3
"""
Unit Tests: Nombre del Módulo
==============================
Descripción de qué se está testeando.
"""

import pytest


@pytest.mark.unit
class TestNombreClase:
    """Tests para FuncionalidadEspecifica"""
    
    def test_basic_functionality(self):
        """Descripción del test"""
        # Arrange
        input_data = "test"
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result == expected_output
    
    def test_edge_case(self):
        """Test de caso edge"""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Template de Test de Integración

```python
@pytest.mark.integration
class TestComponentIntegration:
    """Tests de integración entre componentes"""
    
    def test_components_work_together(self, mock_database):
        """Test de integración con dependencias"""
        # Setup
        component_a = ComponentA(mock_database)
        component_b = ComponentB()
        
        # Act
        result = component_a.process(component_b.get_data())
        
        # Assert
        assert result is not None
```

### Template de Test de Seguridad

```python
@pytest.mark.security
class TestSecurityFeature:
    """Tests de seguridad"""
    
    def test_input_validation(self, dangerous_command_examples):
        """Test de validación de inputs"""
        for dangerous_cmd in dangerous_command_examples:
            with pytest.raises(ValueError):
                validate_input(dangerous_cmd)
```

---

## 🔍 Debugging Tests

### Run con Output Detallado

```bash
# Verbose output
pytest tests/ -v

# Con print statements
pytest tests/ -v -s

# Con traceback completo
pytest tests/ -v --tb=long

# Con debugging interactivo (pdb)
pytest tests/ --pdb
```

### Run Test Específico

```bash
# Archivo específico
pytest tests/unit/test_config.py

# Clase específica
pytest tests/unit/test_config.py::TestEnterpriseConfig

# Test específico
pytest tests/unit/test_config.py::TestEnterpriseConfig::test_config_import
```

---

## 📊 Mejores Prácticas

### ✅ DO

- ✅ Usar markers apropiados (`@pytest.mark.unit`, etc.)
- ✅ Seguir patrón Arrange-Act-Assert
- ✅ Nombres descriptivos de tests
- ✅ Un assert principal por test
- ✅ Tests independientes y aislados
- ✅ Usar fixtures para setup común
- ✅ Docstrings en cada test

### ❌ DON'T

- ❌ Tests que dependen del orden de ejecución
- ❌ Tests con side effects
- ❌ Hardcodear paths absolutos
- ❌ Tests sin assertions
- ❌ Tests demasiado largos (>50 líneas)
- ❌ Múltiples concepts en un test

---

## 🎯 Coverage Goals

### Objetivos de Cobertura

- **Unit Tests**: >80% coverage
- **Integration Tests**: >60% coverage
- **Security Tests**: 100% de funciones críticas
- **Overall**: >75% coverage

### Verificar Coverage

```bash
# Generar reporte
pytest --cov=sheily_core --cov=sheily_train --cov-report=html

# Ver archivos con baja cobertura
pytest --cov=sheily_core --cov-report=term-missing

# Fallar si cobertura < 75%
pytest --cov=sheily_core --cov-fail-under=75
```

---

## 🔧 Configuración Avanzada

### pytest.ini

Configuración adicional en `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
```

### Pre-commit Hook

Tests automáticos antes de commit:

```yaml
# En .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-check
      name: pytest
      entry: pytest
      language: system
      pass_filenames: false
      always_run: false
      args: ['tests/', '-x', '--tb=short']
      stages: [push]
```

---

## 📚 Recursos

### Documentación

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)

### Herramientas

- `pytest` - Framework de testing
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Tests en paralelo
- `pytest-mock` - Mocking helpers
- `pytest-timeout` - Timeout para tests

---

## 🐛 Troubleshooting

### Tests Fallan

```bash
# Ver output completo
pytest tests/ -v -s --tb=long

# Run solo el test problemático
pytest tests/path/to/test.py::TestClass::test_method -v
```

### Import Errors

```bash
# Verificar PYTHONPATH
echo $PYTHONPATH

# Ejecutar desde raíz del proyecto
cd /path/to/sheily-pruebas-1.0-final
pytest tests/
```

### Fixtures No Encontrados

- Verificar que `conftest.py` existe
- Verificar imports en el test
- Verificar scope del fixture

---

**Última actualización**: 29 de Octubre 2025  
**Versión de Tests**: 2.0  
**Estado**: ✅ Organizados y optimizados
