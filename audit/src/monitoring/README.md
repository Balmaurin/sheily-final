# 🛠️ Monitoring - Sistema de Monitoreo Continuo

Este módulo contiene el sistema de monitoreo para recolección continua de métricas del sistema.

## 📋 Funcionalidades Principales

### `monitoring_system.py`
Sistema de monitoreo en tiempo real con detección de anomalías.

**Características:**
- ✅ Recolección automática de métricas del sistema
- ✅ Detección de anomalías con IA
- ✅ Alertas inteligentes por email/console
- ✅ Health checks automáticos
- ✅ Reportes de performance detallados

**Uso:**
```python
from audit_2025.src.monitoring.monitoring_system import MonitoringService

monitor = MonitoringService()
monitor.start()  # Iniciar monitoreo continuo
```

## 🚀 Ejecución

### Monitoreo continuo:
```bash
python audit_2025/src/monitoring/monitoring_system.py
```

### Health check único:
```python
from audit_2025.src.monitoring.monitoring_system import MonitoringService
monitor = MonitoringService()
result = monitor.collect_and_analyze()
```

## 📊 Métricas Monitoreadas

- **CPU y Memoria** - Uso de recursos del sistema
- **Cobertura de Tests** - Calidad del código
- **Problemas de Seguridad** - Vulnerabilidades detectadas
- **Dependencias** - Paquetes actualizados
- **Performance** - Velocidad de ejecución

## 🔗 Integración

Este módulo se integra con:
- **Core** (`src/core/`) - Datos de auditoría
- **Dashboard** (`src/dashboard/`) - Visualización
- **Utils** (`../utils/`) - Funciones de soporte

---

**🛠️ Monitoring** - Monitoreo continuo del sistema
**🎯 Propósito:** Seguimiento en tiempo real del estado del sistema
**⚡ Estado:** ✅ Completamente funcional
