# 🔄 CI/CD - Integración Continua y Despliegue

Este módulo contiene herramientas para integración continua y despliegue automatizado del sistema Sheily AI.

## 📋 Funcionalidades Principales

### `github_actions_automation.py`
Automatización de GitHub Actions para pipelines de CI/CD.

**Características:**
- ✅ Configuración automática de workflows
- ✅ Gestión de artifacts y almacenamiento
- ✅ Publicación de reportes
- ✅ Integración con sistema de auditoría
- ✅ Configuración de políticas de retención

**Uso:**
```python
from audit_2025.src.ci_cd.github_actions_automation import GitHubActionsOrchestrator

orchestrator = GitHubActionsOrchestrator()
workflows = orchestrator.generate_workflows()
```

## 🚀 Workflows de GitHub Actions

### **Audit Pipeline:**
```yaml
name: Audit Pipeline
on: [push, pull_request]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: python audit_2025/run_integrated_audit.py
```

### **Quality Gates:**
```yaml
name: Quality Gates
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests_light/ --cov
      - run: bandit -r sheily_*
      - run: mypy sheily_core
```

### **Security Scanning:**
```yaml
name: Security Scanning
on: [push]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: aquasecurity/trivy-action@master
      - uses: github/codeql-action/upload-sarif@v2
```

## 📊 Gestión de Artifacts

### **Configuración de Artifacts:**
```python
artifact_config = {
    "artifacts": [
        {"name": "coverage-report", "path": "htmlcov", "retention": 30},
        {"name": "audit-reports", "path": "audit_2025/reports", "retention": 90},
        {"name": "test-results", "path": "test-results.xml", "retention": 60}
    ]
}
```

### **Políticas de Almacenamiento:**
```python
storage_policy = {
    "policies": {
        "coverage": {"max_age_days": 90, "compression": "gzip"},
        "reports": {"max_age_days": 180, "compression": "gzip"},
        "metrics": {"max_age_days": 365, "compression": "none"}
    }
}
```

## 🔧 Configuración de CI/CD

### **Variables de Entorno:**
```bash
export CI=true
export AUDIT_ENABLED=true
export COVERAGE_REQUIRED=70
export SECURITY_SCANNING=true
export ARTIFACT_RETENTION_DAYS=90
```

### **Configuración de Workflows:**
```python
# Configuración de GitHub Actions
workflow_config = {
    "name": "Sheily AI CI/CD",
    "on": ["push", "pull_request"],
    "env": {
        "PYTHON_VERSION": "3.11",
        "AUDIT_ENABLED": "true",
        "COVERAGE_THRESHOLD": "70"
    }
}
```

## 📈 Métricas de CI/CD

### **Métricas de Pipeline:**
- **Tiempo de ejecución** - Duración total del pipeline
- **Tasa de éxito** - Porcentaje de builds exitosos
- **Tiempo de despliegue** - Time to deploy
- **Frecuencia de despliegue** - Deployments por día
- **Rollback rate** - Tasa de rollbacks necesarios

### **Métricas de Calidad:**
- **Cobertura de código** - Coverage en CI/CD
- **Tiempo de feedback** - Time to feedback para developers
- **Calidad de código** - Métricas de calidad en pipeline
- **Seguridad** - Vulnerabilidades detectadas en CI

## 🚀 Despliegue Automatizado

### **Estrategias de Despliegue:**
```python
deployment_strategies = {
    "blue_green": {
        "enabled": True,
        "validation_time": 300,
        "rollback_automatic": True
    },
    "canary": {
        "enabled": False,
        "percentage": 10,
        "monitoring_time": 600
    },
    "rolling": {
        "enabled": False,
        "batch_size": 20,
        "pause_time": 60
    }
}
```

### **Automatización de Releases:**
```bash
# Crear release automática
python scripts/organized/utils/create_release.py --auto

# Desplegar versión específica
python scripts/organized/utils/deploy_version.py --version v1.0.0

# Rollback automático
python scripts/organized/utils/rollback_deployment.py --auto
```

## 🔗 Integración con Otros Sistemas

### **Con Sistema de Auditoría:**
```python
# Auditoría automática en CI/CD
from audit_2025.src.core.advanced_audit_system import AdvancedAuditSystem

auditor = AdvancedAuditSystem()
audit_results = auditor.run_complete_audit()

# Bloquear despliegue si auditoría falla
if not audit_results["quality_passed"]:
    raise Exception("Quality gates failed - blocking deployment")
```

### **Con Sistema de Monitoreo:**
```python
# Monitoreo post-despliegue
from audit_2025.src.monitoring.monitoring_system import MonitoringService

monitor = MonitoringService()
monitor.start_post_deployment_monitoring()
```

## 📚 Referencias

- **[Sistema de Auditoría](../../../README.md)** - Sistema principal de auditoría
- **[GitHub Actions](https://docs.github.com/en/actions)** - Documentación oficial
- **[CI/CD Best Practices](../../../docs/deployment/)** - Mejores prácticas
- **[Despliegue](../../../scripts/organized/deployment/)** - Scripts de despliegue

## ⚠️ Notas de Mantenimiento

### **Para el sistema CI/CD:**
- 🔄 **Actualización semanal** de workflows
- ✅ **Monitoreo continuo** de pipelines
- 📊 **Análisis de métricas** de CI/CD
- 🧹 **Limpieza automática** de artifacts antiguos

### **Solución de problemas:**
- **Pipeline lento:** Optimizar steps y caching
- **Flaky tests:** Identificar y corregir tests inestables
- **Artifacts grandes:** Configurar compresión y retención

---

**🔄 CI/CD** - Integración continua y despliegue automatizado
**🎯 Propósito:** Automatización completa del ciclo de desarrollo
**⚡ Estado:** ✅ Configurado para despliegue enterprise
