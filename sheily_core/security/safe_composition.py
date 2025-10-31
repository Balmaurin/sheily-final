#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades Funcionales para Composición Segura de Operaciones
============================================================

Este módulo proporciona herramientas avanzadas para composición segura de operaciones:
- Composición funcional con manejo automático de errores
- Railway-oriented programming mejorado
- Operadores seguros para transformación de datos
- Validación y transformación pipeline-based
- Manejo elegante de operaciones asíncronas
- Composición de operaciones con diferentes estrategias de error

Características avanzadas:
- Composición type-safe con validación automática
- Railway-oriented programming con múltiples estrategias
- Operadores funcionales especializados
- Pipelines asíncronos seguros
- Validación composable de datos
- Recuperación automática en pipelines
"""

import asyncio
import functools
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# Importar sistema de errores funcionales
from .functional_errors import (
    ContextualResult,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    SheilyError,
    async_safe_pipe,
    create_error,
    error_monitor,
    safe_pipe,
)
from .logger import get_logger
from .result import Err, Ok, Result, create_err, create_ok

# Type variables
T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

# ============================================================================
# Operadores Funcionales Básicos
# ============================================================================


class SafeOperator(Generic[T, U]):
    """Operador seguro para transformación de datos"""

    def __init__(self, func: Callable[[T], U], error_context: ErrorContext = None):
        self.func = func
        self.error_context = error_context or ErrorContext(
            component="safe_operator", operation=func.__name__
        )
        self.logger = get_logger("safe_operator")

    def __call__(self, value: T) -> Result[U, SheilyError]:
        """Aplicar operador de manera segura"""
        try:
            result = self.func(value)
            return Ok(result)
        except Exception as e:
            error = SheilyError(
                message=f"Error applying operator {self.func.__name__}: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=self.error_context,
                cause=e,
            )
            return Err(error)

    def map(self, value: Result[T, SheilyError]) -> Result[U, SheilyError]:
        """Aplicar operador a un Result"""
        if value.is_err():
            return value

        return self(value.unwrap())


# ============================================================================
# Railway-Oriented Programming Mejorado
# ============================================================================


class Railway(Generic[T]):
    """Implementación avanzada de Railway-oriented programming"""

    def __init__(self, value: Result[T, SheilyError]):
        self._value = value

    @classmethod
    def success(cls, value: T) -> "Railway[T]":
        """Crear railway con valor exitoso"""
        return cls(Ok(value))

    @classmethod
    def failure(cls, error: SheilyError) -> "Railway[T]":
        """Crear railway con error"""
        return cls(Err(error))

    def bind(self, func: Callable[[T], Result[U, SheilyError]]) -> "Railway[U]":
        """Aplicar función que retorna Result"""
        if self._value.is_err():
            return Railway.failure(self._value.error)

        try:
            result = func(self._value.unwrap())
            return Railway(result)
        except Exception as e:
            error = SheilyError(
                message=f"Error in bind operation: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(component="railway", operation=func.__name__),
                cause=e,
            )
            return Railway.failure(error)

    def map(self, func: Callable[[T], U]) -> "Railway[U]":
        """Aplicar función pura"""
        if self._value.is_err():
            return Railway.failure(self._value.error)

        try:
            result = func(self._value.unwrap())
            return Railway.success(result)
        except Exception as e:
            error = SheilyError(
                message=f"Error in map operation: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(component="railway", operation=func.__name__),
                cause=e,
            )
            return Railway.failure(error)

    def tee(self, func: Callable[[T], None]) -> "Railway[T]":
        """Aplicar efecto secundario y continuar"""
        if self._value.is_ok():
            try:
                func(self._value.unwrap())
            except Exception as e:
                # Los efectos secundarios no deberían fallar el railway
                pass
        return self

    def is_success(self) -> bool:
        """Verificar si el railway tiene éxito"""
        return self._value.is_ok()

    def is_failure(self) -> bool:
        """Verificar si el railway tiene error"""
        return self._value.is_err()

    def unwrap(self) -> T:
        """Obtener valor (lanza excepción si hay error)"""
        return self._value.unwrap()

    def unwrap_or(self, default: T) -> T:
        """Obtener valor o default"""
        return self._value.unwrap_or(default)


# ============================================================================
# Pipelines Seguros
# ============================================================================


class SafePipeline(Generic[T]):
    """Pipeline seguro para composición de operaciones"""

    def __init__(self, initial_value: T):
        self._steps: List[Callable] = []
        self._error_handlers: List[Callable[[SheilyError], Result[T, SheilyError]]] = []
        self._current_value: Result[T, SheilyError] = Ok(initial_value)
        self.logger = get_logger("safe_pipeline")

    def pipe(self, func: Callable[[T], U]) -> "SafePipeline[U]":
        """Agregar paso al pipeline"""

        def safe_func(value: T) -> Result[U, SheilyError]:
            try:
                result = func(value)
                return Ok(result)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in pipeline step {func.__name__}: {str(e)}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(component="pipeline", operation=func.__name__),
                    cause=e,
                )
                return Err(error)

        self._steps.append(safe_func)

        # Aplicar inmediatamente si el valor actual es válido
        if self._current_value.is_ok():
            self._current_value = safe_func(self._current_value.unwrap())

        return SafePipeline(self._current_value)  # Type: ignore

    def handle_error(
        self, handler: Callable[[SheilyError], Result[T, SheilyError]]
    ) -> "SafePipeline[T]":
        """Agregar manejador de errores"""
        self._error_handlers.append(handler)

        # Aplicar manejador si hay error actual
        if self._current_value.is_err():
            recovery_result = handler(self._current_value.error)
            if recovery_result.is_ok():
                self._current_value = recovery_result

        return self

    def async_pipe(self, func: Callable[[T], Awaitable[U]]) -> "AsyncSafePipeline[U]":
        """Agregar paso asíncrono al pipeline"""

        async def async_safe_func(value: T) -> Result[U, SheilyError]:
            try:
                result = await func(value)
                return Ok(result)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in async pipeline step {func.__name__}: {str(e)}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(component="async_pipeline", operation=func.__name__),
                    cause=e,
                )
                return Err(error)

        # Crear pipeline asíncrono
        async_pipeline = AsyncSafePipeline(self._current_value)
        async_pipeline._steps.append(async_safe_func)

        return async_pipeline

    def execute(self) -> Result[T, SheilyError]:
        """Ejecutar pipeline completo"""
        return self._current_value

    def get_value_or_default(self, default: T) -> T:
        """Obtener valor o default"""
        return self._current_value.unwrap_or(default)


class AsyncSafePipeline(Generic[T]):
    """Pipeline seguro asíncrono"""

    def __init__(self, initial_value: Result[T, SheilyError]):
        self._steps: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._current_value: Result[T, SheilyError] = initial_value
        self.logger = get_logger("async_safe_pipeline")

    def async_pipe(self, func: Callable[[T], Awaitable[U]]) -> "AsyncSafePipeline[U]":
        """Agregar paso asíncrono"""

        async def async_safe_func(value: T) -> Result[U, SheilyError]:
            try:
                result = await func(value)
                return Ok(result)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in async pipeline step {func.__name__}: {str(e)}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(component="async_pipeline", operation=func.__name__),
                    cause=e,
                )
                return Err(error)

        self._steps.append(async_safe_func)

        # Nota: No aplicamos inmediatamente en pipelines asíncronos
        # para evitar problemas de ejecución prematura

        return AsyncSafePipeline(self._current_value)  # Type: ignore

    def handle_error(
        self, handler: Callable[[SheilyError], Awaitable[Result[T, SheilyError]]]
    ) -> "AsyncSafePipeline[T]":
        """Agregar manejador de errores asíncrono"""
        self._error_handlers.append(handler)
        return self

    async def execute(self) -> Result[T, SheilyError]:
        """Ejecutar pipeline asíncrono completo"""
        current_value = self._current_value

        for step in self._steps:
            if current_value.is_err():
                break

            try:
                current_value = await step(current_value.unwrap())
            except Exception as e:
                error = SheilyError(
                    message=f"Error executing async pipeline step: {str(e)}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(component="async_pipeline", operation="execute"),
                    cause=e,
                )
                current_value = Err(error)
                break

        return current_value


# ============================================================================
# Validadores Composables
# ============================================================================


class ValidationRule(Generic[T]):
    """Regla de validación composable"""

    def __init__(self, name: str, validator: Callable[[T], bool], error_message: str):
        self.name = name
        self.validator = validator
        self.error_message = error_message

    def validate(self, value: T) -> Result[T, SheilyError]:
        """Validar valor"""
        try:
            if self.validator(value):
                return Ok(value)
            else:
                error = SheilyError(
                    message=f"Validation failed for {self.name}: {self.error_message}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(component="validation", operation=self.name),
                )
                return Err(error)
        except Exception as e:
            error = SheilyError(
                message=f"Error during validation {self.name}: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(component="validation", operation=self.name),
                cause=e,
            )
            return Err(error)


class ValidationPipeline(Generic[T]):
    """Pipeline de validación composable"""

    def __init__(self, value: T):
        self.value = value
        self.rules: List[ValidationRule] = []
        self.logger = get_logger("validation_pipeline")

    def add_rule(self, rule: ValidationRule[T]) -> "ValidationPipeline[T]":
        """Agregar regla de validación"""
        self.rules.append(rule)
        return self

    def validate(self) -> Result[T, List[SheilyError]]:
        """Ejecutar todas las validaciones"""
        errors = []

        for rule in self.rules:
            result = rule.validate(self.value)
            if result.is_err():
                errors.append(result.error)

        if errors:
            return Err(errors)

        return Ok(self.value)


# ============================================================================
# Operadores Especializados
# ============================================================================


def safe_map(func: Callable[[T], U]) -> Callable[[Result[T, SheilyError]], Result[U, SheilyError]]:
    """Crear función segura para map"""

    @functools.wraps(func)
    def safe_func(value: Result[T, SheilyError]) -> Result[U, SheilyError]:
        if value.is_err():
            return value

        try:
            result = func(value.unwrap())
            return Ok(result)
        except Exception as e:
            error = SheilyError(
                message=f"Error in safe_map {func.__name__}: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(component="safe_map", operation=func.__name__),
                cause=e,
            )
            return Err(error)

    return safe_func


def safe_bind(
    func: Callable[[T], Result[U, SheilyError]]
) -> Callable[[Result[T, SheilyError]], Result[U, SheilyError]]:
    """Crear función segura para bind"""

    @functools.wraps(func)
    def safe_func(value: Result[T, SheilyError]) -> Result[U, SheilyError]:
        if value.is_err():
            return value

        try:
            return func(value.unwrap())
        except Exception as e:
            error = SheilyError(
                message=f"Error in safe_bind {func.__name__}: {str(e)}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(component="safe_bind", operation=func.__name__),
                cause=e,
            )
            return Err(error)

    return safe_func


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorador para reintentar operaciones en caso de fallo"""

    def decorator(
        func: Callable[[T], Result[U, SheilyError]]
    ) -> Callable[[T], Result[U, SheilyError]]:
        @functools.wraps(func)
        def wrapper(value: T) -> Result[U, SheilyError]:
            last_error = None

            for attempt in range(max_attempts):
                try:
                    result = func(value)
                    if result.is_ok() or attempt == max_attempts - 1:
                        return result

                    last_error = result.error

                except Exception as e:
                    last_error = SheilyError(
                        message=f"Exception in retry attempt {attempt + 1}: {str(e)}",
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.MEDIUM,
                        context=ErrorContext(component="retry", operation=func.__name__),
                        cause=e,
                    )

                # Esperar antes del siguiente intento (excepto en el último)
                if attempt < max_attempts - 1:
                    time.sleep(delay * (2**attempt))  # Backoff exponencial

            return Err(last_error)

        return wrapper

    return decorator


# ============================================================================
# Funciones de Conveniencia para Composición
# ============================================================================


def compose(*funcs: Callable) -> Callable[[T], Result[Any, SheilyError]]:
    """Componer funciones de manera segura"""

    def composed(value: T) -> Result[Any, SheilyError]:
        result = Ok(value)

        for func in funcs:
            if result.is_err():
                break

            try:
                new_result = func(result.unwrap())
                if isinstance(new_result, (Ok, Err)):
                    result = new_result
                else:
                    result = Ok(new_result)
            except Exception as e:
                error = SheilyError(
                    message=f"Error in composed function {func.__name__}: {str(e)}",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(component="composer", operation=func.__name__),
                    cause=e,
                )
                result = Err(error)
                break

        return result

    return composed


def pipeline(*funcs: Callable) -> Callable[[T], "SafePipeline"]:
    """Crear pipeline seguro"""

    def create_pipeline(value: T) -> SafePipeline:
        pipe = SafePipeline(value)

        for func in funcs:
            pipe.pipe(func)

        return pipe

    return create_pipeline


async def async_pipeline(*funcs: Callable) -> Callable[[T], Awaitable["AsyncSafePipeline"]]:
    """Crear pipeline asíncrono seguro"""

    async def create_async_pipeline(value: T) -> AsyncSafePipeline:
        pipe = AsyncSafePipeline(Ok(value))

        for func in funcs:
            pipe.async_pipe(func)

        return pipe

    return create_async_pipeline


# ============================================================================
# Context Managers para Operaciones Seguras
# ============================================================================


@contextmanager
def safe_operation_context(component: str, operation: str, **metadata):
    """Context manager para operaciones seguras"""
    context = ErrorContext(component=component, operation=operation, metadata=metadata)

    try:
        yield context
    except Exception as e:
        error = SheilyError(
            message=f"Error in safe operation context: {str(e)}",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=e,
        )
        error_monitor.record_error(error)
        raise


@asynccontextmanager
async def async_safe_operation_context(component: str, operation: str, **metadata):
    """Context manager asíncrono para operaciones seguras"""
    context = ErrorContext(component=component, operation=operation, metadata=metadata)

    try:
        yield context
    except Exception as e:
        error = SheilyError(
            message=f"Error in async safe operation context: {str(e)}",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            cause=e,
        )
        error_monitor.record_error(error)
        raise


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    # Operadores básicos
    "SafeOperator",
    # Railway-oriented programming
    "Railway",
    # Pipelines seguros
    "SafePipeline",
    "AsyncSafePipeline",
    # Validación composable
    "ValidationRule",
    "ValidationPipeline",
    # Operadores especializados
    "safe_map",
    "safe_bind",
    "retry_on_failure",
    # Funciones de composición
    "compose",
    "pipeline",
    "async_pipeline",
    # Context managers
    "safe_operation_context",
    "async_safe_operation_context",
]

print("✅ Utilidades funcionales para composición segura de operaciones cargadas exitosamente")
