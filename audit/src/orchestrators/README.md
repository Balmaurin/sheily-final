# 🎼 Orchestrators - Orquestadores del Sistema

Este módulo contiene los orquestadores principales para coordinación de sistemas del sistema Sheily AI.

## 📋 Funcionalidades Principales

### `run_integrated_audit.py`
Orquestador principal del sistema de auditoría integrado.

**Características:**
- ✅ Coordinación de todos los sistemas de auditoría
- ✅ Ejecución secuencial de fases de auditoría
- ✅ Integración con dashboard y monitoreo
- ✅ Generación de reportes consolidados
- ✅ Manejo de errores y recuperación

**Uso:**
```python
from audit_2025.src.orchestrators.run_integrated_audit import IntegratedAuditOrchestrator

orchestrator = IntegratedAuditOrchestrator()
results = orchestrator.run_full_audit()
```

## 🚀 Fases de Orquestación

### **Pipeline de Auditoría Completa:**
```text
Phase 1: Advanced Analysis
    ↓
Phase 2: Real-Time Dashboard
    ↓
Phase 3: Alert Checking
    ↓
Phase 4: Trend Analysis
    ↓
Phase 5: Compliance Report
    ↓
Phase 6: Compliance Certificate
    ↓
Phase 7: Final Summary
```

### **Ejecución de Fases:**
```python
# Ejecutar fase específica
orchestrator = IntegratedAuditOrchestrator()
orchestrator.run_advanced_analysis()
orchestrator.run_dashboard_display()
orchestrator.run_alert_checking()
```

## 📊 Coordinación de Sistemas

### **Integración con Core:**
```python
# Orquestación con sistema core
from audit_2025.src.core.advanced_audit_system import AdvancedAuditSystem

core_audit = AdvancedAuditSystem()
core_results = core_audit.run_complete_audit()

orchestrator = IntegratedAuditOrchestrator()
orchestrator.integrate_core_results(core_results)
```

### **Integración con Dashboard:**
```python
# Orquestación con dashboard
from audit_2025.src.dashboard.realtime_audit_dashboard import RealTimeAuditDashboard

dashboard = RealTimeAuditDashboard()
dashboard_results = dashboard.display_dashboard(metrics)

orchestrator = IntegratedAuditOrchestrator()
orchestrator.integrate_dashboard_results(dashboard_results)
```

### **Integración con Monitoreo:**
```python
# Orquestación con monitoreo
from audit_2025.src.monitoring.monitoring_system import MonitoringService

monitor = MonitoringService()
monitoring_results = monitor.collect_and_analyze()

orchestrator = IntegratedAuditOrchestrator()
orchestrator.integrate_monitoring_results(monitoring_results)
```

## 🔧 Configuración de Orquestación

### **Configuración de Pipeline:**
```python
orchestration_config = {
    "phases": [
        "advanced_analysis",
        "dashboard_display",
        "alert_checking",
        "trend_analysis",
        "compliance_report",
        "compliance_certificate",
        "final_summary"
    ],
    "error_handling": "continue_on_error",
    "timeout_per_phase": 300,
    "retry_attempts": 3
}
```

### **Configuración de Integración:**
```python
integration_config = {
    "core_integration": True,
    "dashboard_integration": True,
    "monitoring_integration": True,
    "reporting_integration": True,
    "alerting_integration": True
}
```

## 📈 Métricas de Orquestación

### **Métricas de Performance:**
- **Tiempo total de orquestación** - Duración completa del pipeline
- **Tiempo por fase** - Performance de cada fase
- **Tasa de éxito** - Éxito de cada fase
- **Errores por fase** - Errores encontrados
- **Tiempo de recuperación** - Time to recovery

### **Métricas de Integración:**
- **Sistemas integrados** - Número de sistemas coordinados
- **Datos transferidos** - Volumen de datos procesados
- **Consistencia de datos** - Integridad de información
- **Latencia de integración** - Tiempo de integración

## 🚀 Ejecución de Orquestadores

### **Orquestación Completa:**
```bash
# Ejecutar orquestador completo
python audit_2025/src/orchestrators/run_integrated_audit.py

# Con configuración personalizada
python -c "
from audit_2025.src.orchestrators.run_integrated_audit import IntegratedAuditOrchestrator
orchestrator = IntegratedAuditOrchestrator()
orchestrator.run_full_audit()
"
```

### **Orquestación Específica:**
```python
# Solo análisis avanzado
orchestrator = IntegratedAuditOrchestrator()
orchestrator.run_advanced_analysis()

# Solo dashboard
orchestrator.run_dashboard_display()

# Solo alertas
orchestrator.run_alert_checking()
```

## 🔗 Integración con Otros Sistemas

### **Con Sistema de Auditoría:**
```python
# Orquestación de auditoría completa
from audit_2025.src.core.advanced_audit_system import AdvancedAuditSystem

auditor = AdvancedAuditSystem()
audit_results = auditor.run_complete_audit()

orchestrator = IntegratedAuditOrchestrator()
final_results = orchestrator.orchestrate_audit_results(audit_results)
```

### **Con Sistema de Reportes:**
```python
# Generación de reportes orquestados
from audit_2025.src.orchestrators.run_integrated_audit import IntegratedAuditOrchestrator

orchestrator = IntegratedAuditOrchestrator()
results = orchestrator.run_full_audit()

# Guardar resultados consolidados
save_consolidated_report(results)
```

## 📚 Referencias

- **[Sistema de Auditoría](../../../README.md)** - Sistema principal de auditoría
- **[Core](../../../src/core/)** - Motor de auditoría principal
- **[Dashboard](../../../src/dashboard/)** - Visualización de métricas
- **[Monitoreo](../../../src/monitoring/)** - Sistema de monitoreo

## ⚠️ Notas de Mantenimiento

### **Para el sistema de orquestadores:**
- 🔄 **Actualización semanal** de lógica de orquestación
- ✅ **Validación continua** de integración
- 📊 **Monitoreo de performance** de orquestación
- 🧹 **Limpieza automática** de datos temporales

### **Solución de problemas:**
- **Fase fallida:** Verificar logs de esa fase específica
- **Integración rota:** Verificar compatibilidad de versiones
- **Performance pobre:** Optimizar orden de fases

---

**🎼 Orchestrators** - Orquestadores del sistema de auditoría
**🎯 Propósito:** Coordinación y orquestación de sistemas
**⚡ Estado:** ✅ Completamente funcional y optimizado
