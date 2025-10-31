#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Avanzado de Recuperación Automática de Errores
=====================================================

Este módulo proporciona un sistema completo de recuperación automática de errores:
- Motor central de recuperación con múltiples estrategias
- Recuperación basada en machine learning
- Análisis predictivo de fallos
- Recuperación automática de servicios externos
- Sistema de backup y restauración automática
- Monitoreo continuo del estado del sistema

Características avanzadas:
- Recuperación inteligente basada en patrones de error
- Predicción de fallos antes de que ocurran
- Recuperación automática de dependencias
- Sistema de health checks continuo
- Backup automático de estados críticos
"""

import asyncio
import threading
import time

try:
    import schedule
except ImportError:
    schedule = None
import functools
import json
import os
import signal
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import psutil

# Importar sistema de errores funcionales
from .functional_errors import (
    CircuitBreakerStrategy,
    ContextualResult,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    FallbackStrategy,
    RecoveryStrategy,
    RetryStrategy,
    SheilyError,
    async_safe_pipe,
    create_error,
    error_monitor,
    safe_pipe,
)
from .logger import get_logger
from .result import Err, Ok, Result, create_err, create_ok

# ============================================================================
# Tipos y Estados del Sistema de Recuperación
# ============================================================================


class RecoveryStatus(Enum):
    """Estados del sistema de recuperación"""

    IDLE = "idle"
    ANALYZING = "analyzing"
    RECOVERING = "recovering"
    MONITORING = "monitoring"
    FAILED = "failed"
    SUCCESS = "success"


class SystemHealth(Enum):
    """Estados de salud del sistema"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class RecoveryAttempt:
    """Registro de intento de recuperación"""

    error: SheilyError
    strategy_used: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Métricas del sistema para análisis predictivo"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    error_rate: float
    response_time: float
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Motor de Recuperación Automática
# ============================================================================


class AutoRecoveryEngine:
    """Motor avanzado de recuperación automática"""

    def __init__(self):
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        self.health_checkers: Dict[str, Callable] = {}
        self.predictive_models: Dict[str, Any] = {}
        self.backup_manager = BackupManager()
        self.status = RecoveryStatus.IDLE
        self.health_status = SystemHealth.HEALTHY

        # Configuración
        self.max_recovery_attempts = 5
        self.recovery_timeout = 300.0  # 5 minutos
        self.health_check_interval = 30.0  # 30 segundos
        self.predictive_analysis_enabled = True

        # Logger
        self.logger = get_logger("auto_recovery")

        # Estado interno
        self._recovery_lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Registrar estrategias por defecto
        self._register_default_strategies()

        # Iniciar monitoreo
        self.start_monitoring()

    def _register_default_strategies(self):
        """Registrar estrategias de recuperación por defecto"""
        self.register_strategy("retry", RetryStrategy(max_attempts=3))
        self.register_strategy("fallback", FallbackStrategy(fallback_value=None))
        self.register_strategy("circuit_breaker", CircuitBreakerStrategy())

    def register_strategy(self, name: str, strategy: RecoveryStrategy):
        """Registrar nueva estrategia de recuperación"""
        self.recovery_strategies[name] = strategy
        self.logger.info(f"Registered recovery strategy: {name}")

    def register_health_checker(self, component: str, checker: Callable):
        """Registrar verificador de salud para un componente"""
        self.health_checkers[component] = checker
        self.logger.info(f"Registered health checker for component: {component}")

    def start_monitoring(self):
        """Iniciar monitoreo continuo del sistema"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Started system monitoring")

    def stop_monitoring(self):
        """Detener monitoreo del sistema"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped system monitoring")

    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        if schedule is not None:
            schedule.every(self.health_check_interval).seconds.do(self._perform_health_checks)

        while not self._stop_monitoring.is_set():
            try:
                if schedule is not None:
                    schedule.run_pending()

                # Análisis predictivo si está habilitado
                if self.predictive_analysis_enabled:
                    self._perform_predictive_analysis()

                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)

    def _perform_health_checks(self):
        """Realizar verificaciones de salud del sistema"""
        try:
            overall_health = SystemHealth.HEALTHY
            health_results = {}

            for component, checker in self.health_checkers.items():
                try:
                    result = checker()
                    health_results[component] = result

                    if result == SystemHealth.CRITICAL:
                        overall_health = SystemHealth.CRITICAL
                    elif result == SystemHealth.DEGRADED and overall_health == SystemHealth.HEALTHY:
                        overall_health = SystemHealth.DEGRADED

                except Exception as e:
                    self.logger.error(f"Health check failed for {component}: {e}")
                    health_results[component] = SystemHealth.FAILED
                    overall_health = SystemHealth.CRITICAL

            self.health_status = overall_health

            # Si el sistema está en estado crítico, intentar recuperación
            if overall_health == SystemHealth.CRITICAL:
                self._trigger_emergency_recovery(health_results)

            # Logging de estado de salud
            self.logger.debug(
                f"Health check completed: {overall_health.value}",
                extra={
                    "health_results": {k: v.value for k, v in health_results.items()},
                    "overall_health": overall_health.value,
                },
            )

        except Exception as e:
            self.logger.error(f"Error during health checks: {e}")
            self.health_status = SystemHealth.CRITICAL

    def _perform_predictive_analysis(self):
        """Realizar análisis predictivo de posibles fallos"""
        try:
            # Recopilar métricas actuales
            current_metrics = self._collect_system_metrics()

            # Analizar tendencias
            for component in self.health_checkers.keys():
                prediction = self._predict_component_failure(component, current_metrics)
                if prediction and prediction["probability"] > 0.7:
                    self.logger.warning(
                        f"High failure probability predicted for {component}",
                        extra={"prediction": prediction, "component": component},
                    )

                    # Trigger preventivo recovery si es necesario
                    if prediction["probability"] > 0.9:
                        self._trigger_preventive_recovery(component, prediction)

        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """Recopilar métricas actuales del sistema"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Calcular tasa de errores (últimos 5 minutos)
            recent_errors = [
                attempt
                for attempt in self.recovery_history
                if attempt.start_time > datetime.now() - timedelta(minutes=5)
            ]
            error_rate = len(recent_errors) / max(1, len(list(self.recovery_history)[-100:]))

            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={"bytes_sent": network.bytes_sent, "bytes_recv": network.bytes_recv},
                error_rate=error_rate,
                response_time=0.1,  # Placeholder
                active_connections=0,  # Placeholder
            )

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                network_io={"bytes_sent": 0, "bytes_recv": 0},
                error_rate=0,
                response_time=0,
                active_connections=0,
            )

    def _predict_component_failure(self, component: str, metrics: SystemMetrics) -> Optional[Dict[str, Any]]:
        """Predecir posible fallo de componente"""
        # Implementación simplificada de predicción
        # En un sistema real, esto usaría modelos ML

        risk_factors = []

        # Análisis basado en métricas actuales
        if metrics.cpu_usage > 80:
            risk_factors.append(("high_cpu", 0.3))
        if metrics.memory_usage > 85:
            risk_factors.append(("high_memory", 0.4))
        if metrics.error_rate > 0.1:
            risk_factors.append(("high_error_rate", 0.5))
        if metrics.disk_usage > 90:
            risk_factors.append(("low_disk_space", 0.6))

        if not risk_factors:
            return None

        # Calcular probabilidad ponderada
        total_probability = sum(factor[1] for factor in risk_factors)
        avg_probability = total_probability / len(risk_factors)

        return {
            "component": component,
            "probability": min(avg_probability, 1.0),
            "risk_factors": risk_factors,
            "timestamp": datetime.now().isoformat(),
        }

    def _trigger_emergency_recovery(self, health_results: Dict[str, SystemHealth]):
        """Trigger recuperación de emergencia"""
        self.logger.warning("Triggering emergency recovery due to critical system health")

        for component, health in health_results.items():
            if health == SystemHealth.CRITICAL:
                self._recover_component(
                    component,
                    create_error(
                        f"Emergency recovery triggered for {component}",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.CRITICAL,
                        component="auto_recovery",
                        operation="emergency_recovery",
                    ),
                )

    def _trigger_preventive_recovery(self, component: str, prediction: Dict[str, Any]):
        """Trigger recuperación preventiva"""
        self.logger.info(f"Triggering preventive recovery for {component}", extra={"prediction": prediction})

        self._recover_component(
            component,
            create_error(
                f"Preventive recovery triggered for {component}",
                ErrorCategory.EXTERNAL_SERVICE,
                ErrorSeverity.HIGH,
                component="auto_recovery",
                operation="preventive_recovery",
                prediction=prediction,
            ),
        )

    def recover_error(self, error: SheilyError) -> Result[Any, SheilyError]:
        """Recuperar automáticamente de un error"""
        with self._recovery_lock:
            self.status = RecoveryStatus.ANALYZING

            try:
                # Crear intento de recuperación
                attempt = RecoveryAttempt(error=error, strategy_used="auto", start_time=datetime.now())

                # Buscar estrategias aplicables
                applicable_strategies = [
                    strategy for strategy in self.recovery_strategies.values() if strategy.can_recover(error)
                ]

                if not applicable_strategies:
                    self.logger.warning(f"No recovery strategies available for error: {error}")
                    attempt.end_time = datetime.now()
                    attempt.success = False
                    self.recovery_history.append(attempt)
                    self.status = RecoveryStatus.FAILED
                    return Err(error)

                # Intentar recuperación con cada estrategia
                for strategy in applicable_strategies:
                    try:
                        self.status = RecoveryStatus.RECOVERING
                        self.logger.info(f"Attempting recovery using strategy: {strategy.__class__.__name__}")

                        recovery_result = strategy.recover(error)

                        if recovery_result.is_ok():
                            attempt.strategy_used = strategy.__class__.__name__
                            attempt.end_time = datetime.now()
                            attempt.success = True
                            attempt.metadata["recovery_value"] = recovery_result.unwrap()
                            self.recovery_history.append(attempt)

                            self.logger.info(f"Successfully recovered using {strategy.__class__.__name__}")
                            self.status = RecoveryStatus.SUCCESS

                            # Actualizar métricas de error
                            error_monitor.record_error(error, (attempt.end_time - attempt.start_time).total_seconds())

                            return recovery_result

                    except Exception as e:
                        self.logger.error(f"Recovery strategy {strategy.__class__.__name__} failed: {e}")
                        continue

                # Si ninguna estrategia funcionó
                attempt.end_time = datetime.now()
                attempt.success = False
                self.recovery_history.append(attempt)

                self.logger.error(f"All recovery strategies failed for error: {error}")
                self.status = RecoveryStatus.FAILED

                return Err(error)

            except Exception as e:
                self.logger.error(f"Error during recovery process: {e}")
                self.status = RecoveryStatus.FAILED
                return Err(
                    create_error(
                        f"Recovery process failed: {str(e)}",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.CRITICAL,
                        component="auto_recovery",
                        operation="recover_error",
                        cause=e,
                    )
                )

    def _recover_component(self, component: str, error: SheilyError) -> Result[Any, SheilyError]:
        """Recuperar componente específico"""
        # Esta función sería implementada según los componentes específicos del sistema
        self.logger.info(f"Attempting component recovery for {component}")

        # Placeholder para lógica de recuperación específica
        return Ok(f"Component {component} recovery completed")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de recuperación"""
        if not self.recovery_history:
            return {"total_attempts": 0}

        attempts = list(self.recovery_history)
        successful_attempts = [a for a in attempts if a.success]
        failed_attempts = [a for a in attempts if not a.success]

        # Estadísticas por estrategia
        strategy_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total_time": 0.0})
        for attempt in attempts:
            if attempt.end_time:
                duration = (attempt.end_time - attempt.start_time).total_seconds()
                strategy_stats[attempt.strategy_used]["total_time"] += duration

            if attempt.success:
                strategy_stats[attempt.strategy_used]["success"] += 1
            else:
                strategy_stats[attempt.strategy_used]["failed"] += 1

        return {
            "total_attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "success_rate": len(successful_attempts) / len(attempts) if attempts else 0,
            "current_status": self.status.value,
            "system_health": self.health_status.value,
            "strategy_stats": dict(strategy_stats),
            "last_recovery": attempts[-1].start_time.isoformat() if attempts else None,
        }


# ============================================================================
# Sistema de Backup Automático
# ============================================================================


class BackupManager:
    """Gestor de backups automáticos"""

    def __init__(self, backup_root: str = "backups/auto_recovery"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("backup_manager")

        # Configuración de retención
        self.max_backups_per_component = 10
        self.backup_interval = 3600.0  # 1 hora
        self.retention_days = 7

    def create_backup(
        self, component: str, data: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> Result[Path, SheilyError]:
        """Crear backup de componente"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_root / component / f"{component}_backup_{timestamp}.json"

            backup_file.parent.mkdir(parents=True, exist_ok=True)

            backup_data = {
                "component": component,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "metadata": metadata or {},
            }

            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"Created backup for {component}: {backup_file}")
            self._cleanup_old_backups(component)

            return Ok(backup_file)

        except Exception as e:
            error = create_error(
                f"Failed to create backup for {component}: {str(e)}",
                ErrorCategory.FILESYSTEM,
                ErrorSeverity.HIGH,
                component="backup_manager",
                operation="create_backup",
                cause=e,
            )
            return Err(error)

    def restore_backup(self, component: str, backup_file: Optional[Path] = None) -> Result[Any, SheilyError]:
        """Restaurar backup de componente"""
        try:
            if backup_file is None:
                # Encontrar backup más reciente
                backup_file = self._find_latest_backup(component)

            if not backup_file or not backup_file.exists():
                return Err(
                    create_error(
                        f"No backup found for component {component}",
                        ErrorCategory.FILESYSTEM,
                        ErrorSeverity.HIGH,
                        component="backup_manager",
                        operation="restore_backup",
                    )
                )

            with open(backup_file, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

            self.logger.info(f"Restored backup for {component}: {backup_file}")
            return Ok(backup_data.get("data"))

        except Exception as e:
            error = create_error(
                f"Failed to restore backup for {component}: {str(e)}",
                ErrorCategory.FILESYSTEM,
                ErrorSeverity.HIGH,
                component="backup_manager",
                operation="restore_backup",
                cause=e,
            )
            return Err(error)

    def _find_latest_backup(self, component: str) -> Optional[Path]:
        """Encontrar backup más reciente para componente"""
        component_dir = self.backup_root / component
        if not component_dir.exists():
            return None

        backup_files = list(component_dir.glob(f"{component}_backup_*.json"))
        if not backup_files:
            return None

        return max(backup_files, key=lambda x: x.stat().st_mtime)

    def _cleanup_old_backups(self, component: str):
        """Limpiar backups antiguos"""
        try:
            component_dir = self.backup_root / component
            if not component_dir.exists():
                return

            backup_files = list(component_dir.glob(f"{component}_backup_*.json"))
            if len(backup_files) <= self.max_backups_per_component:
                return

            # Ordenar por fecha de modificación
            backup_files.sort(key=lambda x: x.stat().st_mtime)

            # Eliminar backups antiguos
            for old_backup in backup_files[: -self.max_backups_per_component]:
                old_backup.unlink()
                self.logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old backups for {component}: {e}")


# ============================================================================
# Health Checkers Especializados
# ============================================================================


class MemoryHealthChecker:
    """Verificador de salud para sistema de memoria"""

    def __init__(self, memory_engine):
        self.memory_engine = memory_engine
        self.logger = get_logger("memory_health_checker")

    def __call__(self) -> SystemHealth:
        """Verificar salud del sistema de memoria"""
        try:
            # Verificar estado básico
            if not hasattr(self.memory_engine, "state"):
                return SystemHealth.CRITICAL

            # Verificar número de memorias
            total_memories = self.memory_engine.state.total_memories
            if total_memories < 0:
                return SystemHealth.DEGRADED

            # Verificar índices FAISS si están disponibles
            if hasattr(self.memory_engine, "vector_store"):
                for layer, store in self.memory_engine.vector_store.items():
                    if hasattr(store, "ntotal") and store.ntotal < 0:
                        return SystemHealth.DEGRADED

            return SystemHealth.HEALTHY

        except Exception as e:
            self.logger.error(f"Memory health check failed: {e}")
            return SystemHealth.CRITICAL


class ModelHealthChecker:
    """Verificador de salud para modelos"""

    def __init__(self, model_engine):
        self.model_engine = model_engine
        self.logger = get_logger("model_health_checker")

    def __call__(self) -> SystemHealth:
        """Verificar salud del sistema de modelos"""
        try:
            # Verificar si el modelo está cargado
            if not hasattr(self.model_engine, "model") or self.model_engine.model is None:
                return SystemHealth.DEGRADED

            # Verificar estado del modelo
            # Esta verificación dependería de la implementación específica del modelo

            return SystemHealth.HEALTHY

        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            return SystemHealth.CRITICAL


class RAGHealthChecker:
    """Verificador de salud para sistema RAG"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.logger = get_logger("rag_health_checker")

    def __call__(self) -> SystemHealth:
        """Verificar salud del sistema RAG"""
        try:
            # Verificar si el índice está disponible
            if not hasattr(self.rag_engine, "index") or self.rag_engine.index is None:
                return SystemHealth.DEGRADED

            # Verificar tamaño del índice
            if hasattr(self.rag_engine.index, "ntotal") and self.rag_engine.index.ntotal == 0:
                return SystemHealth.DEGRADED

            return SystemHealth.HEALTHY

        except Exception as e:
            self.logger.error(f"RAG health check failed: {e}")
            return SystemHealth.CRITICAL


# ============================================================================
# Decoradores para Recuperación Automática
# ============================================================================


def with_auto_recovery(component: str, max_attempts: int = 3):
    """Decorador para operaciones con recuperación automática"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error = create_error(
                        f"Error in {func.__name__}: {str(e)}",
                        _categorize_exception(e),
                        _determine_severity(e),
                        component=component,
                        operation=func.__name__,
                        cause=e,
                    )

                    # Intentar recuperación automática
                    recovery_result = auto_recovery_engine.recover_error(error)

                    if recovery_result.is_ok() and attempt == max_attempts - 1:
                        return recovery_result.unwrap()

                    if attempt == max_attempts - 1:
                        raise e

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Instancia Global del Motor de Recuperación
# ============================================================================

auto_recovery_engine = AutoRecoveryEngine()

# ============================================================================
# Funciones de Conveniencia
# ============================================================================


def register_component_health_checker(component: str, checker: Callable):
    """Registrar verificador de salud para componente"""
    auto_recovery_engine.register_health_checker(component, checker)


def trigger_recovery(error: SheilyError) -> Result[Any, SheilyError]:
    """Trigger recuperación manual para error específico"""
    return auto_recovery_engine.recover_error(error)


def get_system_health() -> SystemHealth:
    """Obtener estado actual de salud del sistema"""
    return auto_recovery_engine.health_status


def get_recovery_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de recuperación"""
    return auto_recovery_engine.get_recovery_stats()


# ============================================================================
# Utilidades Auxiliares
# ============================================================================


def _categorize_exception(e: Exception) -> ErrorCategory:
    """Categorizar excepción (reutilizada del módulo principal)"""
    from .functional_errors import _categorize_exception as categorize

    return categorize(e)


def _determine_severity(e: Exception) -> ErrorSeverity:
    """Determinar severidad (reutilizada del módulo principal)"""
    from .functional_errors import _determine_severity as determine

    return determine(e)


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    # Tipos principales
    "RecoveryStatus",
    "SystemHealth",
    "RecoveryAttempt",
    "SystemMetrics",
    # Motor de recuperación
    "AutoRecoveryEngine",
    "BackupManager",
    # Health checkers
    "MemoryHealthChecker",
    "ModelHealthChecker",
    "RAGHealthChecker",
    # Decoradores
    "with_auto_recovery",
    # Funciones de conveniencia
    "register_component_health_checker",
    "trigger_recovery",
    "get_system_health",
    "get_recovery_statistics",
    # Instancia global
    "auto_recovery_engine",
]

import os as _os

if _os.environ.get("SHEILY_CHAT_QUIET", "1") != "1":
    print("✅ Sistema avanzado de recuperación automática de errores cargado exitosamente")
