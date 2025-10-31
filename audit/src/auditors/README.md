# 🔍 Auditors - Sistema de Auditores Específicos

Este módulo contiene auditores especializados para diferentes componentes del sistema Sheily AI.

## 📋 Auditores Disponibles

### `complete_project_audit.py`
Auditor completo del proyecto con análisis exhaustivo.

**Características:**
- ✅ Auditoría de estructura y organización
- ✅ Análisis de calidad de código
- ✅ Validación de modelos y adaptadores
- ✅ Verificación de datos y corpus
- ✅ Auditoría de sistemas MCP y Cline
- ✅ Análisis de dependencias y configuraciones
- ✅ Verificación de Docker y contenedores
- ✅ Auditoría de memoria y aprendizaje
- ✅ Análisis de seguridad y validaciones
- ✅ Verificación de rendimiento y métricas
- ✅ Auditoría de documentación y logs

**Uso:**
```python
from audit_2025.src.auditors.complete_project_audit import CompleteProjectAuditor

auditor = CompleteProjectAuditor()
results = auditor.run_complete_audit()
```

## 🚀 Ejecución de Auditores

### Auditoría Completa:
```bash
python audit_2025/src/auditors/complete_project_audit.py
```

### Auditoría Específica:
```bash
# Auditoría solo de modelos
python -c "from audit_2025.src.auditors.complete_project_audit import CompleteProjectAuditor; auditor = CompleteProjectAuditor(); auditor.audit_models_adapters()"

# Auditoría solo de seguridad
python -c "from audit_2025.src.auditors.complete_project_audit import CompleteProjectAuditor; auditor = CompleteProjectAuditor(); auditor.audit_security_validations()"
```

## 📊 Tipos de Auditoría

### **Auditoría Técnica:**
- Análisis de estructura de archivos
- Verificación de sintaxis Python
- Validación de imports y dependencias
- Análisis de complejidad de código
- Verificación de formatos de datos

### **Auditoría de Calidad:**
- Cobertura de tests
- Calidad de código
- Documentación presente
- Manejo de errores
- Performance del sistema

### **Auditoría de Seguridad:**
- Detección de secretos hardcoded
- Validación de input sanitization
- Verificación de permisos
- Análisis de vulnerabilidades
- Cumplimiento de estándares

### **Auditoría de Performance:**
- Tiempos de respuesta
- Uso de memoria
- Eficiencia de código
- Optimización de recursos
- Benchmarks de velocidad

## 🔧 Configuración de Auditores

### **Umbrales Configurables:**
```python
audit_config = {
    "quality_gates": {
        "code_coverage": {"target": 70, "critical": 50},
        "security_issues": {"target": 0, "critical": 5},
        "test_pass_rate": {"target": 100, "critical": 95},
        "code_quality": {"target": 8.0, "critical": 6.0}
    }
}
```

### **Auditoría Personalizada:**
```python
# Configurar auditoría específica
auditor = CompleteProjectAuditor()
auditor.audit_results["categories"]["custom"] = {"checks": [], "status": "pending"}

# Ejecutar checks personalizados
custom_results = auditor.execute_checks(custom_checks, "custom")
```

## 📈 Reportes de Auditoría

### **Reporte JSON:**
```json
{
  "audit_id": "audit_20251025_120000",
  "timestamp": "2025-10-25T12:00:00Z",
  "overall_status": "PRODUCTION READY",
  "pass_rate": 87.5,
  "categories": {
    "structure": {"status": "EXCELLENT", "score": 95},
    "code_quality": {"status": "GOOD", "score": 85},
    "security": {"status": "EXCELLENT", "score": 100}
  }
}
```

### **Reporte de Texto:**
```
AUDITORÍA COMPLETA DEL PROYECTO SHEILY-AI
========================================

Estado General: PRODUCTION READY
Tasa de Aprobación: 87.5%
Duración: 45.2s

CATEGORÍAS AUDITADAS:
✅ Structure: EXCELLENT (95%)
✅ Code Quality: GOOD (85%)
✅ Security: EXCELLENT (100%)
✅ Models: GOOD (78%)
✅ Documentation: EXCELLENT (92%)
```

## 🔗 Integración con Otros Sistemas

### **Con Sistema de Reportes:**
```python
# Generar reporte automático
auditor = CompleteProjectAuditor()
results = auditor.run_complete_audit()

# Guardar en sistema de reportes
save_audit_report(results, "reports/audit/")
```

### **Con Sistema de Logs:**
```python
# Logging automático de auditoría
import logging

logging.basicConfig(filename='logs/audit_complete.log')
logger = logging.getLogger('audit')

auditor = CompleteProjectAuditor()
results = auditor.run_complete_audit()
logger.info(f"Auditoría completada: {results['overall_status']}")
```

## 📚 Referencias

- **[Sistema de Auditoría Principal](../../../README.md)** - Documentación general
- **[Core](../../../src/core/)** - Motor de auditoría principal
- **[Reportes](../../../reports/)** - Sistema de reportes
- **[Logs](../../../logs/)** - Sistema de logs

## ⚠️ Notas de Mantenimiento

### **Para el sistema de auditores:**
- 🔄 **Actualización semanal** de criterios de auditoría
- ✅ **Validación continua** de auditores
- 📊 **Monitoreo de performance** de auditorías
- 🧹 **Limpieza automática** de reportes antiguos

### **Solución de problemas:**
- **Auditoría muy lenta:** Optimizar queries de análisis
- **Auditoría incompleta:** Verificar dependencias
- **Reportes corruptos:** Validar generación de reportes

---

**🔍 Auditors** - Sistema de auditores especializados
**🎯 Propósito:** Auditoría exhaustiva de todos los componentes del sistema
**⚡ Estado:** ✅ Completamente funcional y optimizado
