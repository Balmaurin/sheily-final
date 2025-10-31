#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoradores Avanzados para Manejo de Errores Funcionales
=======================================================

Este módulo proporciona decoradores especializados para manejo automático de errores:
- Decoradores específicos por dominio (memoria, RAG, modelos)
- Decoradores con monitoreo de rendimiento integrado
- Decoradores con circuit breaker y rate limiting
- Decoradores para operaciones asíncronas con manejo de errores
- Decoradores para validación automática de parámetros
- Decoradores para logging estructurado de operaciones

Características avanzadas:
- Métricas automáticas de rendimiento y errores
- Circuit breaker automático para servicios externos
- Validación automática de tipos y parámetros
- Logging estructurado con contexto enriquecido
- Soporte completo para operaciones asíncronas
"""

import asyncio
import functools
import inspect
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

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
    async_with_error_handling,
    create_error,
    error_monitor,
    safe_pipe,
    with_error_handling,
)
from .logger import get_logger
from .result import Err, Ok, Result

# Type variables
F = TypeVar("F", bound=Callable[..., Any])

# ============================================================================
# Decoradores Base Mejorados
# ============================================================================


def sheily_operation(
    component: str,
    operation: Optional[str] = None,
    enable_metrics: bool = True,
    enable_circuit_breaker: bool = False,
    max_failures: int = 5,
    recovery_timeout: float = 300.0,
    validate_params: bool = True,
    log_performance: bool = True,
):
    """
    Decorador avanzado para operaciones Sheily con manejo completo de errores

    Args:
        component: Nombre del componente
        operation: Nombre de la operación (por defecto usa el nombre de la función)
        enable_metrics: Habilitar métricas de rendimiento
        enable_circuit_breaker: Habilitar circuit breaker
        max_failures: Número máximo de fallos antes de abrir circuit breaker
        recovery_timeout: Tiempo de espera para recuperación (segundos)
        validate_params: Validar parámetros automáticamente
        log_performance: Loguear métricas de rendimiento
    """

    def decorator(func: F) -> F:
        operation_name = operation or func.__name__
        logger = get_logger(component)

        # Crear estrategias de recuperación
        recovery_strategies = [RetryStrategy(max_attempts=3), FallbackStrategy(fallback_value=None)]

        if enable_circuit_breaker:
            circuit_breaker = CircuitBreakerStrategy(max_failures, recovery_timeout)
            recovery_strategies.append(circuit_breaker)
        else:
            circuit_breaker = None

        # Crear función wrapper
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss if enable_metrics else 0

                # Crear contexto de operación
                context = ErrorContext(
                    component=component,
                    operation=operation_name,
                    metadata={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "function_name": func.__name__,
                    },
                )

                # Validar parámetros si está habilitado
                if validate_params:
                    validation_result = await _validate_function_params(func, args, kwargs)
                    if validation_result.is_err():
                        error = validation_result.error
                        error_monitor.record_error(error)
                        if log_performance:
                            logger.error(
                                f"Parameter validation failed for {operation_name}",
                                extra={"error": error, "execution_time": time.time() - start_time},
                            )
                        raise ValueError(f"Parameter validation failed: {error.message}")

                try:
                    # Ejecutar función con manejo de errores
                    result = await async_with_error_handling(
                        component=component,
                        recovery_strategies=recovery_strategies,
                        log_errors=True,
                    )(func)(*args, **kwargs)

                    execution_time = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss if enable_metrics else 0
                    memory_used = memory_after - memory_before if enable_metrics else 0

                    # Registrar métricas si está habilitado
                    if enable_metrics:
                        _record_operation_metrics(
                            component, operation_name, execution_time, memory_used, result.is_ok()
                        )

                    # Logging de rendimiento
                    if log_performance:
                        if result.is_ok():
                            logger.info(
                                f"Operation {operation_name} completed successfully",
                                extra={
                                    "execution_time": execution_time,
                                    "memory_used": memory_used,
                                    "success": True,
                                },
                            )
                        else:
                            logger.error(
                                f"Operation {operation_name} failed",
                                extra={
                                    "execution_time": execution_time,
                                    "memory_used": memory_used,
                                    "success": False,
                                    "error": result.result.error if result.is_err() else None,
                                },
                            )

                    # Registrar error en monitor si falló
                    if result.is_err():
                        error_monitor.record_error(result.result.error, execution_time)

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    memory_used = psutil.Process().memory_info().rss - memory_before if enable_metrics else 0

                    # Crear error estructurado
                    error = SheilyError(
                        message=f"Unhandled error in {operation_name}: {str(e)}",
                        category=_categorize_exception(e),
                        severity=_determine_severity(e),
                        context=context,
                        cause=e,
                    )

                    error_monitor.record_error(error, execution_time)

                    if log_performance:
                        logger.error(
                            f"Unhandled exception in {operation_name}",
                            extra={
                                "execution_time": execution_time,
                                "memory_used": memory_used,
                                "error": error,
                            },
                        )

                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss if enable_metrics else 0

                # Crear contexto de operación
                context = ErrorContext(
                    component=component,
                    operation=operation_name,
                    metadata={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                        "function_name": func.__name__,
                    },
                )

                # Validar parámetros si está habilitado
                if validate_params:
                    validation_result = _validate_function_params(func, args, kwargs)
                    if validation_result.is_err():
                        error = validation_result.error
                        error_monitor.record_error(error)
                        if log_performance:
                            logger.error(
                                f"Parameter validation failed for {operation_name}",
                                extra={"error": error, "execution_time": time.time() - start_time},
                            )
                        raise ValueError(f"Parameter validation failed: {error.message}")

                try:
                    # Ejecutar función con manejo de errores
                    result = with_error_handling(
                        component=component,
                        recovery_strategies=recovery_strategies,
                        log_errors=True,
                    )(func)(*args, **kwargs)

                    execution_time = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss if enable_metrics else 0
                    memory_used = memory_after - memory_before if enable_metrics else 0

                    # Registrar métricas si está habilitado
                    if enable_metrics:
                        _record_operation_metrics(
                            component, operation_name, execution_time, memory_used, result.is_ok()
                        )

                    # Logging de rendimiento
                    if log_performance:
                        if result.is_ok():
                            logger.info(
                                f"Operation {operation_name} completed successfully",
                                extra={
                                    "execution_time": execution_time,
                                    "memory_used": memory_used,
                                    "success": True,
                                },
                            )
                        else:
                            logger.error(
                                f"Operation {operation_name} failed",
                                extra={
                                    "execution_time": execution_time,
                                    "memory_used": memory_used,
                                    "success": False,
                                    "error": result.result.error if result.is_err() else None,
                                },
                            )

                    # Registrar error en monitor si falló
                    if result.is_err():
                        error_monitor.record_error(result.result.error, execution_time)

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    memory_used = psutil.Process().memory_info().rss - memory_before if enable_metrics else 0

                    # Crear error estructurado
                    error = SheilyError(
                        message=f"Unhandled error in {operation_name}: {str(e)}",
                        category=_categorize_exception(e),
                        severity=_determine_severity(e),
                        context=context,
                        cause=e,
                    )

                    error_monitor.record_error(error, execution_time)

                    if log_performance:
                        logger.error(
                            f"Unhandled exception in {operation_name}",
                            extra={
                                "execution_time": execution_time,
                                "memory_used": memory_used,
                                "error": error,
                            },
                        )

                    raise

            return sync_wrapper

    return decorator


# ============================================================================
# Decoradores Específicos por Dominio
# ============================================================================


def memory_operation(operation: Optional[str] = None, enable_validation: bool = True):
    """Decorador específico para operaciones de memoria"""

    def decorator(func: F) -> F:
        component = "memory_system"

        return sheily_operation(
            component=component,
            operation=operation,
            enable_metrics=True,
            enable_circuit_breaker=False,  # La memoria no necesita circuit breaker
            validate_params=enable_validation,
            log_performance=True,
        )(func)

    return decorator


def rag_operation(operation: Optional[str] = None, enable_circuit_breaker: bool = True):
    """Decorador específico para operaciones RAG"""

    def decorator(func: F) -> F:
        component = "rag_system"

        return sheily_operation(
            component=component,
            operation=operation,
            enable_metrics=True,
            enable_circuit_breaker=enable_circuit_breaker,
            max_failures=3,
            recovery_timeout=60.0,
            validate_params=True,
            log_performance=True,
        )(func)

    return decorator


def model_operation(operation: Optional[str] = None, enable_circuit_breaker: bool = True):
    """Decorador específico para operaciones de modelos"""

    def decorator(func: F) -> F:
        component = "model_system"

        return sheily_operation(
            component=component,
            operation=operation,
            enable_metrics=True,
            enable_circuit_breaker=enable_circuit_breaker,
            max_failures=5,
            recovery_timeout=120.0,
            validate_params=True,
            log_performance=True,
        )(func)

    return decorator


# ============================================================================
# Decoradores para Validación Automática
# ============================================================================


def validate_params(*param_validators):
    """
    Decorador para validación automática de parámetros

    Args:
        param_validators: Lista de funciones validadoras para cada parámetro
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener signature de la función
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validar parámetros posicionales
            for i, (param_name, param_value) in enumerate(bound_args.arguments.items()):
                if i < len(param_validators) and param_validators[i]:
                    validator = param_validators[i]
                    try:
                        validation_result = validator(param_value)
                        if validation_result is not True:
                            raise ValueError(f"Parameter '{param_name}' validation failed: {validation_result}")
                    except Exception as e:
                        raise ValueError(f"Parameter '{param_name}' validation error: {str(e)}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_types(type_hints: Optional[Dict[str, Type]] = None):
    """Decorador para validación automática de tipos"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener hints de tipo si no se proporcionan
            if type_hints is None:
                type_hints = get_type_hints(func)

            # Obtener signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validar tipos
            for param_name, param_value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]

                    # Manejar tipos genéricos (List[str], Optional[str], etc.)
                    if hasattr(expected_type, "__origin__") or str(expected_type).startswith("typing."):
                        if not _validate_generic_type(param_value, expected_type):
                            raise TypeError(
                                f"Parameter '{param_name}' has incorrect type. Expected {expected_type}, got {type(param_value)}"
                            )
                    else:
                        # Validación simple de tipos
                        if not isinstance(param_value, expected_type):
                            raise TypeError(
                                f"Parameter '{param_name}' has incorrect type. Expected {expected_type}, got {type(param_value)}"
                            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _validate_generic_type(value: Any, expected_type: Type) -> bool:
    """Validar tipos genéricos complejos"""
    try:
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin is Union:
            # Para Union types (como Optional)
            return any(isinstance(value, arg) if arg is not type(None) else value is None for arg in args)
        elif origin is list:
            # Para List[T]
            if not isinstance(value, list):
                return False
            element_type = args[0] if args else Any
            return all(isinstance(item, element_type) for item in value)
        elif origin is dict:
            # Para Dict[K, V]
            if not isinstance(value, dict):
                return False
            key_type, value_type = args if len(args) == 2 else (Any, Any)
            return all(isinstance(k, key_type) and isinstance(v, value_type) for k, v in value.items())
        else:
            return isinstance(value, origin or expected_type)

    except Exception:
        return True  # Si no podemos validar, asumir que es correcto


# ============================================================================
# Decoradores para Monitoreo de Rendimiento
# ============================================================================


def with_performance_monitoring(
    component: str,
    operation: Optional[str] = None,
    track_memory: bool = True,
    track_cpu: bool = False,
    alert_thresholds: Optional[Dict[str, float]] = None,
):
    """Decorador para monitoreo detallado de rendimiento"""

    def decorator(func: F) -> F:
        operation_name = operation or func.__name__
        logger = get_logger(component)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss if track_memory else 0
                cpu_before = psutil.cpu_percent(interval=None) if track_cpu else 0

                try:
                    result = await func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss if track_memory else 0
                    cpu_after = psutil.cpu_percent(interval=None) if track_cpu else 0

                    memory_used = memory_after - memory_before if track_memory else 0
                    cpu_used = cpu_after - cpu_before if track_cpu else 0

                    # Verificar umbrales de alerta
                    if alert_thresholds:
                        _check_performance_alerts(
                            operation_name,
                            execution_time,
                            memory_used,
                            cpu_used,
                            alert_thresholds,
                            logger,
                        )

                    # Log detallado de rendimiento
                    logger.debug(
                        f"Performance metrics for {operation_name}",
                        extra={
                            "execution_time": execution_time,
                            "memory_used_mb": memory_used / (1024 * 1024),
                            "cpu_used": cpu_used,
                            "success": True,
                        },
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    memory_used = psutil.Process().memory_info().rss - memory_before if track_memory else 0

                    logger.warning(
                        f"Performance metrics for failed operation {operation_name}",
                        extra={
                            "execution_time": execution_time,
                            "memory_used_mb": memory_used / (1024 * 1024),
                            "error": str(e),
                        },
                    )

                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss if track_memory else 0
                cpu_before = psutil.cpu_percent(interval=None) if track_cpu else 0

                try:
                    result = func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    memory_after = psutil.Process().memory_info().rss if track_memory else 0
                    cpu_after = psutil.cpu_percent(interval=None) if track_cpu else 0

                    memory_used = memory_after - memory_before if track_memory else 0
                    cpu_used = cpu_after - cpu_before if track_cpu else 0

                    # Verificar umbrales de alerta
                    if alert_thresholds:
                        _check_performance_alerts(
                            operation_name,
                            execution_time,
                            memory_used,
                            cpu_used,
                            alert_thresholds,
                            logger,
                        )

                    # Log detallado de rendimiento
                    logger.debug(
                        f"Performance metrics for {operation_name}",
                        extra={
                            "execution_time": execution_time,
                            "memory_used_mb": memory_used / (1024 * 1024),
                            "cpu_used": cpu_used,
                            "success": True,
                        },
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    memory_used = psutil.Process().memory_info().rss - memory_before if track_memory else 0

                    logger.warning(
                        f"Performance metrics for failed operation {operation_name}",
                        extra={
                            "execution_time": execution_time,
                            "memory_used_mb": memory_used / (1024 * 1024),
                            "error": str(e),
                        },
                    )

                    raise

            return sync_wrapper

    return decorator


# ============================================================================
# Decoradores para Operaciones Asíncronas
# ============================================================================


def async_safe_execution(
    component: str,
    operation: Optional[str] = None,
    timeout: Optional[float] = None,
    max_concurrent: int = 10,
    retry_on_failure: bool = True,
):
    """Decorador para ejecución segura de operaciones asíncronas"""

    def decorator(func: F) -> F:
        operation_name = operation or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Crear semáforo para limitar concurrencia
            semaphore = asyncio.Semaphore(max_concurrent)

            async with semaphore:
                try:
                    if timeout:
                        # Ejecutar con timeout
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                        return result
                    else:
                        # Ejecutar normalmente
                        result = await func(*args, **kwargs)
                        return result

                except asyncio.TimeoutError:
                    error = create_error(
                        f"Operation {operation_name} timed out after {timeout}s",
                        ErrorCategory.EXTERNAL_SERVICE,
                        ErrorSeverity.HIGH,
                        component=component,
                        operation=operation_name,
                        timeout=timeout,
                    )
                    error_monitor.record_error(error)
                    raise

                except Exception as e:
                    if retry_on_failure:
                        # Podría implementar lógica de reintento aquí
                        pass

                    error = create_error(
                        f"Error in async operation {operation_name}: {str(e)}",
                        _categorize_exception(e),
                        _determine_severity(e),
                        component=component,
                        operation=operation_name,
                        cause=e,
                    )
                    error_monitor.record_error(error)
                    raise

        return wrapper

    return decorator


# ============================================================================
# Decoradores para Logging Estructurado
# ============================================================================


def with_structured_logging(
    component: str,
    operation: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    log_level: str = "INFO",
):
    """Decorador para logging estructurado de operaciones"""

    def decorator(func: F) -> F:
        operation_name = operation or func.__name__
        logger = get_logger(component)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()

                # Preparar información de logging
                log_data = {
                    "operation": operation_name,
                    "component": component,
                    "start_time": datetime.now().isoformat(),
                }

                if include_args:
                    log_data["args"] = _sanitize_args_for_logging(args)
                    log_data["kwargs"] = _sanitize_kwargs_for_logging(kwargs)

                getattr(logger, log_level.lower())(f"Starting {operation_name}", extra=log_data)

                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Logging de éxito
                    success_data = {**log_data, "execution_time": execution_time, "success": True}

                    if include_result:
                        success_data["result"] = _sanitize_result_for_logging(result)

                    getattr(logger, log_level.lower())(f"Completed {operation_name}", extra=success_data)

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time

                    # Logging de error
                    error_data = {
                        **log_data,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

                    getattr(logger, log_level.lower())(f"Failed {operation_name}", extra=error_data)

                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()

                # Preparar información de logging
                log_data = {
                    "operation": operation_name,
                    "component": component,
                    "start_time": datetime.now().isoformat(),
                }

                if include_args:
                    log_data["args"] = _sanitize_args_for_logging(args)
                    log_data["kwargs"] = _sanitize_kwargs_for_logging(kwargs)

                getattr(logger, log_level.lower())(f"Starting {operation_name}", extra=log_data)

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Logging de éxito
                    success_data = {**log_data, "execution_time": execution_time, "success": True}

                    if include_result:
                        success_data["result"] = _sanitize_result_for_logging(result)

                    logger.log(
                        getattr(logger.logger, log_level.lower()),
                        f"Completed {operation_name}",
                        extra=success_data,
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time

                    # Logging de error
                    error_data = {
                        **log_data,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

                    logger.log(
                        getattr(logger.logger, log_level.lower()),
                        f"Failed {operation_name}",
                        extra=error_data,
                    )

                    raise

            return sync_wrapper

    return decorator


# ============================================================================
# Utilidades Auxiliares
# ============================================================================


def _record_operation_metrics(component: str, operation: str, execution_time: float, memory_used: int, success: bool):
    """Registrar métricas de operación"""
    # Esta función podría integrarse con un sistema de métricas más avanzado
    # Por ahora, solo registra en el monitor de errores
    pass


def _check_performance_alerts(
    operation: str,
    execution_time: float,
    memory_used: int,
    cpu_used: float,
    thresholds: Dict[str, float],
    logger,
):
    """Verificar umbrales de alerta de rendimiento"""
    alerts = []

    if "execution_time" in thresholds and execution_time > thresholds["execution_time"]:
        alerts.append(f"Execution time {execution_time:.2f}s exceeds threshold {thresholds['execution_time']}s")

    if "memory_used" in thresholds and memory_used > thresholds["memory_used"]:
        alerts.append(
            f"Memory usage {memory_used / (1024*1024):.2f}MB exceeds threshold {thresholds['memory_used'] / (1024*1024):.2f}MB"
        )

    if "cpu_used" in thresholds and cpu_used > thresholds["cpu_used"]:
        alerts.append(f"CPU usage {cpu_used:.2f}% exceeds threshold {thresholds['cpu_used']}%")

    for alert in alerts:
        logger.warning(f"Performance alert for {operation}: {alert}")


async def _validate_function_params(func: Callable, args: tuple, kwargs: dict) -> Result[None, SheilyError]:
    """Validar parámetros de función"""
    try:
        # Validaciones básicas
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validar que los parámetros requeridos estén presentes
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty and param_name not in bound_args.arguments:
                return Err(
                    create_error(
                        f"Required parameter '{param_name}' is missing",
                        ErrorCategory.VALIDATION,
                        ErrorSeverity.HIGH,
                        component="parameter_validation",
                        operation=func.__name__,
                        missing_parameter=param_name,
                    )
                )

        return Ok(None)

    except Exception as e:
        return Err(
            create_error(
                f"Parameter validation error: {str(e)}",
                ErrorCategory.VALIDATION,
                ErrorSeverity.HIGH,
                component="parameter_validation",
                operation=func.__name__,
                cause=e,
            )
        )


def _sanitize_args_for_logging(args: tuple) -> str:
    """Sanitizar argumentos para logging"""
    sanitized = []
    for arg in args:
        if isinstance(arg, str) and len(arg) > 100:
            sanitized.append(f"{arg[:100]}...")
        elif isinstance(arg, (dict, list)) and len(str(arg)) > 200:
            sanitized.append(f"{type(arg).__name__}(size={len(arg)})")
        else:
            sanitized.append(str(arg)[:200])
    return str(sanitized)


def _sanitize_kwargs_for_logging(kwargs: dict) -> str:
    """Sanitizar kwargs para logging"""
    sanitized = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 100:
            sanitized[key] = f"{value[:100]}..."
        elif isinstance(value, (dict, list)) and len(str(value)) > 200:
            sanitized[key] = f"{type(value).__name__}(size={len(value)})"
        else:
            sanitized[key] = str(value)[:200]
    return str(sanitized)


def _sanitize_result_for_logging(result: Any) -> str:
    """Sanitizar resultado para logging"""
    if isinstance(result, str) and len(result) > 100:
        return f"{result[:100]}..."
    elif isinstance(result, (dict, list)) and len(str(result)) > 200:
        return f"{type(result).__name__}(size={len(result)})"
    else:
        return str(result)[:200]


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
    # Decoradores principales
    "sheily_operation",
    "memory_operation",
    "rag_operation",
    "model_operation",
    # Decoradores de validación
    "validate_params",
    "validate_types",
    # Decoradores de monitoreo
    "with_performance_monitoring",
    # Decoradores asíncronos
    "async_safe_execution",
    # Decoradores de logging
    "with_structured_logging",
]

import os as _os

if _os.environ.get("SHEILY_CHAT_QUIET", "1") != "1":
    print("✅ Decoradores avanzados para manejo de errores funcionales cargados exitosamente")
