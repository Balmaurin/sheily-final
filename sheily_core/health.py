#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Check System para Sheily AI
===================================
Sistema de health checks y monitoring para producción.
"""

import logging
import platform
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Estado de salud de un componente"""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)


@dataclass
class SystemHealth:
    """Estado de salud del sistema completo"""
    status: HealthStatus
    components: List[ComponentHealth]
    system_info: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        """Verificar si el sistema está saludable"""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def unhealthy_components(self) -> List[str]:
        """Obtener lista de componentes no saludables"""
        return [
            c.name for c in self.components 
            if c.status == HealthStatus.UNHEALTHY
        ]


class HealthChecker:
    """Sistema de health checks"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Registrar checks por defecto"""
        self.register_check("system", self._check_system)
        self.register_check("disk", self._check_disk)
        self.register_check("memory", self._check_memory)
    
    def register_check(self, name: str, check_fn: callable):
        """Registrar un nuevo health check"""
        self.checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")
    
    def _check_system(self) -> ComponentHealth:
        """Verificar sistema operativo"""
        try:
            start = time.time()
            
            system_info = {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "hostname": platform.node()
            }
            
            latency = (time.time() - start) * 1000
            
            return ComponentHealth(
                name="system",
                status=HealthStatus.HEALTHY,
                message="System operational",
                latency_ms=latency,
                metadata=system_info
            )
        except Exception as e:
            logger.error(f"System check failed: {e}")
            return ComponentHealth(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )
    
    def _check_disk(self) -> ComponentHealth:
        """Verificar espacio en disco"""
        try:
            start = time.time()
            
            # Verificar directorio de trabajo
            disk = psutil.disk_usage('/')
            percent_used = disk.percent
            
            latency = (time.time() - start) * 1000
            
            # Determinar estado
            if percent_used > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical: {percent_used}%"
            elif percent_used > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {percent_used}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {percent_used}%"
            
            return ComponentHealth(
                name="disk",
                status=status,
                message=message,
                latency_ms=latency,
                metadata={
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent_used": percent_used
                }
            )
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )
    
    def _check_memory(self) -> ComponentHealth:
        """Verificar memoria RAM"""
        try:
            start = time.time()
            
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            
            latency = (time.time() - start) * 1000
            
            # Determinar estado
            if percent_used > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {percent_used}%"
            elif percent_used > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {percent_used}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {percent_used}%"
            
            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                latency_ms=latency,
                metadata={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": percent_used
                }
            )
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )
    
    def check_all(self) -> SystemHealth:
        """Ejecutar todos los health checks"""
        components = []
        
        for name, check_fn in self.checks.items():
            try:
                component_health = check_fn()
                components.append(component_health)
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}"
                ))
        
        # Determinar estado general
        if all(c.status == HealthStatus.HEALTHY for c in components):
            overall_status = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Información del sistema
        system_info = {
            "version": "1.0.0",
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "uptime_seconds": time.time() - psutil.boot_time()
        }
        
        return SystemHealth(
            status=overall_status,
            components=components,
            system_info=system_info
        )
    
    def to_dict(self, health: SystemHealth) -> Dict:
        """Convertir SystemHealth a diccionario"""
        return {
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "components": {
                c.name: {
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": round(c.latency_ms, 2),
                    "metadata": c.metadata,
                    "checked_at": c.checked_at.isoformat()
                }
                for c in health.components
            },
            "system": health.system_info
        }


# Instancia global
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Obtener instancia global del health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def check_health() -> Dict:
    """Ejecutar health check y devolver resultado"""
    checker = get_health_checker()
    health = checker.check_all()
    return checker.to_dict(health)


# Exports
__all__ = [
    'HealthStatus',
    'ComponentHealth',
    'SystemHealth',
    'HealthChecker',
    'get_health_checker',
    'check_health',
]
