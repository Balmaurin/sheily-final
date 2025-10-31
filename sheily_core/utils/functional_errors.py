#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema Avanzado de Manejo de Errores Funcionales para Sheily AI
================================================================

Este módulo proporciona un sistema completo de manejo de errores funcionales con:
- Tipos de errores específicos del dominio
- Recuperación automática de errores
- Decoradores funcionales para manejo automático
- Monitoreo y métricas de errores
- Composición segura de operaciones
- Integración perfecta con el sistema de logging existente

Características avanzadas:
- Railway-oriented programming mejorado
- Manejo de errores asíncrono
- Recuperación automática con estrategias configurables
- Métricas detalladas de errores por componente
- Contextos de error enriquecidos
"""

import asyncio
import functools
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .logger import get_logger

# Importar sistema de errores básico existente
from .result import Err, Ok, Result, catch, create_err, create_ok, is_err, is_ok, traverse_results

# Type variables para tipos genéricos
T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")

# ============================================================================
# Tipos de Errores Específicos del Dominio
# ============================================================================


class ErrorSeverity(Enum):
    """Severidad de errores"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categorías de errores"""

    VALIDATION = "validation"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    MODEL = "model"
    RAG = "rag"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorContext:
    """Contexto enriquecido para errores"""

    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "system"
    session_id: str = ""
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0


@dataclass(frozen=True)
class SheilyError:
    """Error específico del dominio Sheily"""

    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    cause: Optional[Exception] = None
    recovery_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message}"


# ============================================================================
# Resultados Mejorados con Contexto
# ============================================================================


@dataclass(frozen=True)
class ContextualResult(Generic[T]):
    """Resultado con contexto enriquecido"""

    result: Result[T, SheilyError]
    context: ErrorContext
    execution_time: float
    warnings: List[str] = field(default_factory=list)

    def is_ok(self) -> bool:
        return is_ok(self.result)

    def is_err(self) -> bool:
        return is_err(self.result)

    def unwrap(self) -> T:
        return self.result.unwrap()

    def unwrap_or(self, default: T) -> T:
        return self.result.unwrap_or(default)

    def map(self, func: Callable[[T], U]) -> "ContextualResult[U]":
        if self.is_ok():
            try:
                new_value = func(self.unwrap())
                return ContextualResult(
                    result=Ok(new_value),
                    context=self.context,
                    execution_time=self.execution_time,
                    warnings=self.warnings,
                )
            except Exception as e:
                error = self._create_error_from_exception(e)
                return ContextualResult(
                    result=Err(error),
                    context=self.context,
                    execution_time=self.execution_time,
                    warnings=self.warnings,
                )
        else:
            return ContextualResult(
                result=Err(self.result.error),
                context=self.context,
                execution_time=self.execution_time,
                warnings=self.warnings,
            )

    def _create_error_from_exception(self, e: Exception) -> SheilyError:
        return SheilyError(
            message=str(e),
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            context=self.context,
            cause=e,
            stack_trace=traceback.format_exc(),
        )


# ============================================================================
# Estrategias de Recuperación
# ============================================================================


class RecoveryStrategy(ABC):
    """Estrategia abstracta de recuperación de errores"""

    @abstractmethod
    def can_recover(self, error: SheilyError) -> bool:
        """Determinar si se puede recuperar del error"""
        pass

    @abstractmethod
    def recover(self, error: SheilyError) -> Result[Any, SheilyError]:
        """Intentar recuperar del error"""
        pass

    @abstractmethod
    def get_max_attempts(self) -> int:
        """Número máximo de intentos de recuperación"""
        pass


class RetryStrategy(RecoveryStrategy):
    """Estrategia de reintento con backoff exponencial"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def can_recover(self, error: SheilyError) -> bool:
        # Solo reintentar errores de red o temporales
        return error.category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]

    def recover(self, error: SheilyError) -> Result[Any, SheilyError]:
        # Esta estrategia se usa en combinación con decoradores
        # El decorador maneja los reintentos reales
        return Err(error)

    def get_max_attempts(self) -> int:
        return self.max_attempts


class FallbackStrategy(RecoveryStrategy):
    """Estrategia de fallback a valores por defecto"""

    def __init__(self, fallback_value: Any = None):
        self.fallback_value = fallback_value

    def can_recover(self, error: SheilyError) -> bool:
        return error.category in [ErrorCategory.MODEL, ErrorCategory.RAG, ErrorCategory.MEMORY]

    def recover(self, error: SheilyError) -> Result[Any, SheilyError]:
        return Ok(self.fallback_value)

    def get_max_attempts(self) -> int:
        return 1


class CircuitBreakerStrategy(RecoveryStrategy):
    """Estrategia de circuit breaker para servicios externos"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "closed"  # closed, open, half-open

    def can_recover(self, error: SheilyError) -> bool:
        if self.state == "open":
            # Verificar si ha pasado suficiente tiempo para intentar recuperación
            if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(
                seconds=self.recovery_timeout
            ):
                self.state = "half-open"
                return True
            return False
        return True

    def recover(self, error: SheilyError) -> Result[Any, SheilyError]:
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            return Ok(None)  # Permitir el intento
        self.record_failure()  # ← CORRECCIÓN: Registrar fallo cuando no se puede recuperar
        return Err(error)

    def get_max_attempts(self) -> int:
        return 1

    def record_failure(self):
        """Registrar un fallo"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# ============================================================================
# Decoradores Funcionales para Manejo de Errores
# ============================================================================


def with_error_handling(
    component: str,
    recovery_strategies: Optional[List[RecoveryStrategy]] = None,
    log_errors: bool = True,
    rethrow_on_failure: bool = False,
):
    """
    Decorador para manejo automático de errores funcionales

    Args:
        component: Nombre del componente para contexto de logging
        recovery_strategies: Lista de estrategias de recuperación a aplicar
        log_errors: Si se deben loguear los errores
        rethrow_on_failure: Si se deben relanzar excepciones después del manejo
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            context = ErrorContext(
                component=component,
                operation=func.__name__,
                metadata={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            )

            try:
                # Ejecutar función
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Si la función ya retorna un Result, devolverlo con contexto
                if isinstance(result, Result):
                    return ContextualResult(
                        result=result, context=context, execution_time=execution_time
                    )

                # Si retorna un valor normal, envolverlo en Ok
                return ContextualResult(
                    result=Ok(result), context=context, execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error = SheilyError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=_categorize_exception(e),
                    severity=_determine_severity(e),
                    context=context,
                    cause=e,
                    stack_trace=traceback.format_exc(),
                )

                # Aplicar estrategias de recuperación
                if recovery_strategies:
                    for strategy in recovery_strategies:
                        if strategy.can_recover(error):
                            recovery_result = strategy.recover(error)
                            if recovery_result.is_ok():
                                if log_errors:
                                    logger = get_logger(component)
                                    logger.warning(
                                        f"Recovered from error using {strategy.__class__.__name__}"
                                    )
                                return ContextualResult(
                                    result=recovery_result,
                                    context=context,
                                    execution_time=execution_time,
                                    warnings=[f"Recovered using {strategy.__class__.__name__}"],
                                )

                # Logging del error
                if log_errors:
                    logger = get_logger(component)
                    logger.error(
                        f"Unhandled error in {func.__name__}",
                        extra={"error": error, "execution_time": execution_time},
                    )

                # Crear resultado de error
                error_result = ContextualResult(
                    result=Err(error), context=context, execution_time=execution_time
                )

                if rethrow_on_failure:
                    raise e

                return error_result

        return wrapper

    return decorator


def async_with_error_handling(
    component: str,
    recovery_strategies: Optional[List[RecoveryStrategy]] = None,
    log_errors: bool = True,
    rethrow_on_failure: bool = False,
):
    """Decorador asíncrono para manejo de errores funcionales"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            context = ErrorContext(
                component=component,
                operation=func.__name__,
                metadata={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            )

            try:
                # Ejecutar función asíncrona
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Si la función ya retorna un Result, devolverlo con contexto
                if isinstance(result, (Ok, Err)):
                    return ContextualResult(
                        result=result, context=context, execution_time=execution_time
                    )

                # Si retorna un valor normal, envolverlo en Ok
                return ContextualResult(
                    result=Ok(result), context=context, execution_time=execution_time
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error = SheilyError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=_categorize_exception(e),
                    severity=_determine_severity(e),
                    context=context,
                    cause=e,
                    stack_trace=traceback.format_exc(),
                )

                # Aplicar estrategias de recuperación
                if recovery_strategies:
                    for strategy in recovery_strategies:
                        if strategy.can_recover(error):
                            recovery_result = strategy.recover(error)
                            if recovery_result.is_ok():
                                if log_errors:
                                    logger = get_logger(component)
                                    logger.warning(
                                        f"Recovered from error using {strategy.__class__.__name__}"
                                    )
                                return ContextualResult(
                                    result=recovery_result,
                                    context=context,
                                    execution_time=execution_time,
                                    warnings=[f"Recovered using {strategy.__class__.__name__}"],
                                )

                # Logging del error
                if log_errors:
                    logger = get_logger(component)
                    logger.error(
                        f"Unhandled error in {func.__name__}",
                        extra={"error": error, "execution_time": execution_time},
                    )

                # Crear resultado de error
                error_result = ContextualResult(
                    result=Err(error), context=context, execution_time=execution_time
                )

                if rethrow_on_failure:
                    raise e

                return error_result

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    component: str = "unknown",
):
    """Decorador específico para reintentos con backoff exponencial"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger = get_logger(component)
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}"
                        )
                        time.sleep(delay)

            # Si llegamos aquí, todos los intentos fallaron
            logger = get_logger(component)
            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_error

        return wrapper

    return decorator


# ============================================================================
# Utilidades Funcionales para Composición Segura
# ============================================================================


def safe_pipe(value: T, *funcs: Callable[[Any], Any]) -> Result[Any, SheilyError]:
    """Ejecutar una serie de funciones en pipeline con manejo de errores"""
    result = Ok(value)

    for func in funcs:
        if result.is_ok():
            try:
                new_value = func(result.unwrap())
                result = Ok(new_value)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in pipeline function {func.__name__}: {str(e)}",
                    category=_categorize_exception(e),
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(component="pipeline", operation=func.__name__),
                    cause=e,
                )
                result = Err(error)
                break

    return result


async def async_safe_pipe(
    value: T, *funcs: Callable[[Any], Awaitable[Any]]
) -> Result[Any, SheilyError]:
    """Ejecutar una serie de funciones asíncronas en pipeline con manejo de errores"""
    result = Ok(value)

    for func in funcs:
        if result.is_ok():
            try:
                new_value = await func(result.unwrap())
                result = Ok(new_value)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in async pipeline function {func.__name__}: {str(e)}",
                    category=_categorize_exception(e),
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(component="async_pipeline", operation=func.__name__),
                    cause=e,
                )
                result = Err(error)
                break

    return result


def bind_results(results: List[Result[T, SheilyError]]) -> Result[List[T], List[SheilyError]]:
    """Combinar múltiples resultados, recolectando todos los errores"""
    values = []
    errors = []

    for result in results:
        if result.is_ok():
            values.append(result.unwrap())
        else:
            errors.append(result.error)

    if errors:
        return Err(errors)
    return Ok(values)


# ============================================================================
# Sistema de Monitoreo de Errores
# ============================================================================


@dataclass
class ErrorMetrics:
    """Métricas de errores por componente"""

    component: str
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    average_recovery_time: float = 0.0
    last_error_time: Optional[datetime] = None
    error_rate_per_minute: float = 0.0

    def record_error(self, error: SheilyError, recovery_time: float = 0.0):
        """Registrar un error"""
        self.total_errors += 1
        self.errors_by_category[error.category] += 1
        self.errors_by_severity[error.severity] += 1
        self.last_error_time = datetime.now()

        # Calcular promedio móvil de tiempo de recuperación
        if recovery_time > 0:
            if self.average_recovery_time == 0:
                self.average_recovery_time = recovery_time
            else:
                # Suavizado exponencial
                alpha = 0.1
                self.average_recovery_time = (
                    alpha * recovery_time + (1 - alpha) * self.average_recovery_time
                )

        # Calcular tasa de errores por minuto
        if self.last_error_time:
            time_window = datetime.now() - self.last_error_time
            minutes = max(time_window.total_seconds() / 60, 1)
            self.error_rate_per_minute = self.total_errors / minutes


class ErrorMonitor:
    """Monitor centralizado de errores"""

    def __init__(self):
        self.metrics: Dict[str, ErrorMetrics] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.logger = get_logger("error_monitor")

    def record_error(self, error: SheilyError, recovery_time: float = 0.0):
        """Registrar un error en el monitor"""
        component = error.context.component

        if component not in self.metrics:
            self.metrics[component] = ErrorMetrics(component=component)

        self.metrics[component].record_error(error, recovery_time)
        self.error_history.append(
            {"error": error, "timestamp": datetime.now(), "recovery_time": recovery_time}
        )

        # Log del error
        self.logger.error(
            f"Error recorded in {component}", extra={"error": error, "recovery_time": recovery_time}
        )

    def get_metrics(self, component: Optional[str] = None) -> Dict[str, ErrorMetrics]:
        """Obtener métricas de errores"""
        if component:
            return {component: self.metrics.get(component)} if component in self.metrics else {}
        return self.metrics.copy()

    def get_error_summary(self) -> Dict[str, Any]:
        """Obtener resumen de errores"""
        total_errors = sum(metrics.total_errors for metrics in self.metrics.values())
        components_with_errors = len([m for m in self.metrics.values() if m.total_errors > 0])

        return {
            "total_errors": total_errors,
            "components_with_errors": components_with_errors,
            "most_problematic_component": max(
                self.metrics.items(), key=lambda x: x[1].total_errors
            )[0]
            if self.metrics
            else None,
            "error_rate_per_minute": sum(m.error_rate_per_minute for m in self.metrics.values()),
            "timestamp": datetime.now().isoformat(),
        }


# Instancia global del monitor de errores
error_monitor = ErrorMonitor()

# ============================================================================
# Utilidades Auxiliares
# ============================================================================


def _categorize_exception(e: Exception) -> ErrorCategory:
    """Categorizar una excepción en una categoría de error"""
    error_msg = str(e).lower()

    if any(word in error_msg for word in ["connection", "network", "timeout", "unreachable"]):
        return ErrorCategory.NETWORK
    elif any(word in error_msg for word in ["file", "path", "directory", "permission"]):
        return ErrorCategory.FILESYSTEM
    elif any(word in error_msg for word in ["memory", "out of memory", "allocation"]):
        return ErrorCategory.MEMORY
    elif any(word in error_msg for word in ["model", "inference", "llama", "tokenizer"]):
        return ErrorCategory.MODEL
    elif any(word in error_msg for word in ["rag", "embedding", "vector", "search"]):
        return ErrorCategory.RAG
    elif any(word in error_msg for word in ["database", "sqlite", "connection"]):
        return ErrorCategory.DATABASE
    elif any(word in error_msg for word in ["config", "configuration", "setting"]):
        return ErrorCategory.CONFIGURATION
    elif any(
        word in error_msg for word in ["auth", "authentication", "permission", "unauthorized"]
    ):
        return ErrorCategory.AUTHENTICATION
    else:
        return ErrorCategory.UNKNOWN


def _determine_severity(e: Exception) -> ErrorSeverity:
    """Determinar la severidad de una excepción"""
    error_msg = str(e).lower()

    if any(word in error_msg for word in ["critical", "fatal", "system", "out of memory"]):
        return ErrorSeverity.CRITICAL
    elif any(word in error_msg for word in ["error", "exception", "failed", "invalid"]):
        return ErrorSeverity.HIGH
    elif any(word in error_msg for word in ["warning", "deprecated", "timeout"]):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW


# ============================================================================
# Context Managers para Manejo de Errores
# ============================================================================


@contextmanager
def error_context(component: str, operation: str, **metadata):
    """Context manager para establecer contexto de errores"""
    old_context = getattr(error_context, "_current_context", None)

    context = ErrorContext(component=component, operation=operation, metadata=metadata)

    error_context._current_context = context

    try:
        yield context
    finally:
        error_context._current_context = old_context


@asynccontextmanager
async def async_error_context(component: str, operation: str, **metadata):
    """Context manager asíncrono para establecer contexto de errores"""
    old_context = getattr(async_error_context, "_current_context", None)

    context = ErrorContext(component=component, operation=operation, metadata=metadata)

    async_error_context._current_context = context

    try:
        yield context
    finally:
        async_error_context._current_context = old_context


# ============================================================================
# Funciones de Conveniencia para Crear Errores
# ============================================================================


def create_error(
    message: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    component: str = "unknown",
    operation: str = "unknown",
    **metadata,
) -> SheilyError:
    """Crear un error estructurado"""
    return SheilyError(
        message=message,
        category=category,
        severity=severity,
        context=ErrorContext(component=component, operation=operation, metadata=metadata),
    )


def create_validation_error(message: str, **metadata) -> SheilyError:
    """Crear error de validación"""
    return create_error(message, ErrorCategory.VALIDATION, ErrorSeverity.HIGH, **metadata)


def create_network_error(message: str, **metadata) -> SheilyError:
    """Crear error de red"""
    return create_error(message, ErrorCategory.NETWORK, ErrorSeverity.HIGH, **metadata)


def create_memory_error(message: str, **metadata) -> SheilyError:
    """Crear error de memoria"""
    return create_error(message, ErrorCategory.MEMORY, ErrorSeverity.CRITICAL, **metadata)


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    # Tipos principales
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "SheilyError",
    "ContextualResult",
    "ErrorMetrics",
    "ErrorMonitor",
    # Estrategias de recuperación
    "RecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "CircuitBreakerStrategy",
    # Decoradores
    "with_error_handling",
    "async_with_error_handling",
    "with_retry",
    # Utilidades funcionales
    "safe_pipe",
    "async_safe_pipe",
    "bind_results",
    # Context managers
    "error_context",
    "async_error_context",
    # Funciones de conveniencia
    "create_error",
    "create_validation_error",
    "create_network_error",
    "create_memory_error",
    # Instancias globales
    "error_monitor",
]

# Información de versión
__version__ = "2.0.0"
__author__ = "Sheily AI Team"

import os as _os

if _os.environ.get("SHEILY_CHAT_QUIET", "1") != "1":
    print("✅ Sistema avanzado de manejo de errores funcionales cargado exitosamente")
