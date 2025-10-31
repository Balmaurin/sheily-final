# 📊 Dashboard - Sistema de Dashboard en Tiempo Real

Este módulo contiene el sistema de dashboard para visualización de métricas de auditoría en tiempo real.

## 📋 Funcionalidades Principales

### `realtime_audit_dashboard.py`
Dashboard interactivo para monitoreo de métricas de auditoría.

**Características:**
- ✅ Visualización en tiempo real de métricas
- ✅ Sistema de alertas configurable
- ✅ Análisis de tendencias históricas
- ✅ Reportes de cumplimiento normativo
- ✅ Certificados de cumplimiento automáticos

**Uso:**
```python
from audit_2025.src.dashboard.realtime_audit_dashboard import RealTimeAuditDashboard

dashboard = RealTimeAuditDashboard(audit_dir)
dashboard.display_dashboard(metrics)
```

## 🚀 Ejecución

### Dashboard web:
```bash
python audit_2025/src/dashboard/realtime_audit_dashboard.py
```

### Dashboard estático:
```bash
python -c "from audit_2025.src.dashboard.realtime_audit_dashboard import create_static_dashboard; create_static_dashboard()"
```

## 📊 Métricas Visualizadas

- **Cobertura de código** - Progreso visual
- **Estado de seguridad** - Indicadores de vulnerabilidades
- **Calidad del código** - Métricas de calidad
- **Tendencias históricas** - Gráficos de evolución
- **Alertas activas** - Notificaciones en tiempo real

## 🔗 Integración

Este módulo se integra con:
- **Core** (`src/core/`) - Datos de auditoría
- **Monitoreo** (`src/monitoring/`) - Métricas en tiempo real
- **Reportes** (`../reports/`) - Generación de reportes
- **Web** (`src/web/`) - Interfaces web

---

**📊 Dashboard** - Visualización de métricas de auditoría
**🎯 Propósito:** Monitoreo visual del estado del sistema
**⚡ Estado:** ✅ Completamente funcional
