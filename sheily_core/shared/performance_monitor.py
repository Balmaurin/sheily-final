#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PERFORMANCE MONITOR - MONITOREO UNIFICADO DE RENDIMIENTO
======================================================

Sistema de monitoreo que proporciona métricas en tiempo real de:
- Tiempo de respuesta de consultas
- Uso de memoria y CPU
- Tasa de aciertos de caché
- Salud de componentes
- Métricas de embeddings y memoria
- Estadísticas de uso del sistema
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
from sheily_config import get_config


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento de una operación"""

    operation: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calcular duración de la operación"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def complete(self, success: bool = True, error: Optional[str] = None):
        """Completar métricas de la operación"""
        self.end_time = time.time()
        self.success = success
        if error:
            self.error_message = error


@dataclass
class SystemHealth:
    """Estado de salud del sistema"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_connections: int
    open_files: int
    thread_count: int
    process_memory_mb: float


class PerformanceMonitor:
    """Monitor de rendimiento unificado"""

    def __init__(self):
        self.config = get_config()
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_health_history: List[SystemHealth] = []
        self.max_history_size = 1000
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()

        # Métricas agregadas
        self.aggregated_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_response_time": 0.0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage_peak": 0.0,
        }

    def start_monitoring(self):
        """Iniciar monitoreo automático"""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Capturar salud del sistema
                    self._capture_system_health()

                    # Limpiar métricas antiguas (mantener solo últimas 24 horas)
                    self._cleanup_old_metrics()

                    time.sleep(self.config.performance.health_check_interval)
                except Exception as e:
                    print(f"Error en monitoreo: {e}")
                    time.sleep(self.config.performance.health_check_interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Parar monitoreo automático"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def record_operation(
        self,
        operation: str,
        success: bool = True,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """Registrar métricas de una operación"""
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            success=success,
            error_message=error,
            metadata=metadata or {},
        )

        if duration:
            metrics.end_time = metrics.start_time + duration
            metrics.success = success
            if error:
                metrics.error_message = error

        with self._lock:
            self.metrics_history.append(metrics)

            # Actualizar métricas agregadas
            self.aggregated_metrics["total_operations"] += 1
            if success:
                self.aggregated_metrics["successful_operations"] += 1
                if duration:
                    self.aggregated_metrics["total_response_time"] += duration
            else:
                self.aggregated_metrics["failed_operations"] += 1

            # Recalcular promedio
            if self.aggregated_metrics["successful_operations"] > 0:
                self.aggregated_metrics["avg_response_time"] = (
                    self.aggregated_metrics["total_response_time"] / self.aggregated_metrics["successful_operations"]
                )

            # Mantener tamaño máximo
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size :]

        return metrics

    def _capture_system_health(self):
        """Capturar métricas de salud del sistema"""
        try:
            health = SystemHealth(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage("/").percent,
                network_connections=len(psutil.net_connections()),
                open_files=len(psutil.Process().open_files()),
                thread_count=threading.active_count(),
                process_memory_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            )

            with self._lock:
                self.system_health_history.append(health)

                # Actualizar pico de memoria
                if health.process_memory_mb > self.aggregated_metrics["memory_usage_peak"]:
                    self.aggregated_metrics["memory_usage_peak"] = health.process_memory_mb

                # Mantener tamaño máximo
                if len(self.system_health_history) > 100:  # 100 muestras de salud
                    self.system_health_history = self.system_health_history[-100:]

        except Exception as e:
            print(f"Error capturando salud del sistema: {e}")

    def _cleanup_old_metrics(self):
        """Limpiar métricas antiguas"""
        cutoff_time = time.time() - 86400  # 24 horas

        with self._lock:
            self.metrics_history = [m for m in self.metrics_history if m.start_time > cutoff_time]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generar reporte completo de rendimiento"""
        with self._lock:
            # Calcular métricas por operación
            operation_stats = {}
            for metric in self.metrics_history:
                op = metric.operation
                if op not in operation_stats:
                    operation_stats[op] = {
                        "count": 0,
                        "total_time": 0.0,
                        "success_count": 0,
                        "error_count": 0,
                        "avg_time": 0.0,
                    }

                operation_stats[op]["count"] += 1
                if metric.end_time:
                    operation_stats[op]["total_time"] += metric.duration
                    operation_stats[op]["avg_time"] = operation_stats[op]["total_time"] / operation_stats[op]["count"]

                if metric.success:
                    operation_stats[op]["success_count"] += 1
                else:
                    operation_stats[op]["error_count"] += 1

            # Salud del sistema más reciente
            current_health = self.system_health_history[-1] if self.system_health_history else None

            return {
                "timestamp": datetime.now().isoformat(),
                "aggregated_metrics": self.aggregated_metrics.copy(),
                "operation_stats": operation_stats,
                "current_system_health": {
                    "cpu_percent": current_health.cpu_percent,
                    "memory_percent": current_health.memory_percent,
                    "process_memory_mb": current_health.process_memory_mb,
                    "thread_count": current_health.thread_count,
                }
                if current_health
                else None,
                "history_size": {
                    "metrics": len(self.metrics_history),
                    "health_samples": len(self.system_health_history),
                },
            }

    def get_cache_performance(self) -> Dict[str, Any]:
        """Obtener métricas específicas de caché"""
        try:
            from sheily_core.shared.intelligent_cache import get_cache_info

            return get_cache_info()
        except ImportError:
            return {"error": "Cache module not available"}

    def get_memory_performance(self) -> Dict[str, Any]:
        """Obtener métricas específicas de memoria"""
        try:
            from sheily_core.shared.memory_manager import get_memory_stats

            return get_memory_stats()
        except ImportError:
            return {"error": "Memory manager not available"}

    def export_metrics(self, filepath: str) -> bool:
        """Exportar métricas a archivo JSON"""
        try:
            report = self.get_performance_report()
            report["cache_metrics"] = self.get_cache_performance()
            report["memory_metrics"] = self.get_memory_performance()

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Error exportando métricas: {e}")
            return False


# Instancia global del monitor de rendimiento
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Obtener instancia global del monitor de rendimiento"""
    return _performance_monitor


def record_operation(
    operation: str,
    success: bool = True,
    duration: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PerformanceMetrics:
    """Función de conveniencia para registrar operación"""
    return _performance_monitor.record_operation(operation, success, duration, error, metadata)


def get_performance_report() -> Dict[str, Any]:
    """Función de conveniencia para obtener reporte de rendimiento"""
    return _performance_monitor.get_performance_report()


def start_monitoring():
    """Función de conveniencia para iniciar monitoreo"""
    _performance_monitor.start_monitoring()


def stop_monitoring():
    """Función de conveniencia para parar monitoreo"""
    _performance_monitor.stop_monitoring()


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Probando Performance Monitor...")

    # Iniciar monitoreo
    start_monitoring()

    # Registrar algunas operaciones de prueba
    metrics1 = record_operation("test_operation_1", success=True, duration=0.1)
    time.sleep(0.1)
    metrics2 = record_operation("test_operation_2", success=False, error="Test error")

    # Obtener reporte
    report = get_performance_report()

    print(f"✅ Operaciones registradas: {report['aggregated_metrics']['total_operations']}")
    print(f"✅ Tiempo promedio: {report['aggregated_metrics']['avg_response_time']:.3f}s")
    print(
        f"✅ Tasa de éxito: {report['aggregated_metrics']['successful_operations']/max(report['aggregated_metrics']['total_operations'], 1)*100:.1f}%"
    )

    # Parar monitoreo
    stop_monitoring()

    print("✅ Performance Monitor operativo")
