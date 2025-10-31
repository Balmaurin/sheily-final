#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Logging Empresarial para Sheily AI
============================================

Módulo de logging avanzado con:
- Logging estructurado empresarial
- Context managers funcionales
- Configuración avanzada de logs
- Integración con monitoreo empresarial
- Formateo empresarial profesional
"""

import functools
import json
import logging
import logging.handlers
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class LogContext:
    """Contexto estructurado para logging empresarial"""

    component: str
    operation: str
    user_id: str = "system"
    session_id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.session_id is None:
            self.session_id = f"session_{int(datetime.now().timestamp())}"


class EnterpriseFormatter(logging.Formatter):
    """Formateador empresarial avanzado para logs"""

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        # Formato base empresarial
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        log_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Agregar contexto si está disponible
        if hasattr(record, "context") and record.context:
            log_entry["context"] = {
                "component": record.context.component,
                "operation": record.context.operation,
                "user_id": record.context.user_id,
                "session_id": record.context.session_id,
                "metadata": record.context.metadata,
            }

        # Agregar información adicional del record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "context",
                ]:
                    log_entry[f"extra_{key}"] = value

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class EnterpriseLoggerAdapter(logging.LoggerAdapter):
    """Adaptador empresarial para logging con contexto"""

    def __init__(self, logger: logging.Logger, context: LogContext = None):
        super().__init__(logger, {})
        self.context = context

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Procesar mensaje con contexto empresarial"""
        # Agregar contexto al record si existe
        if self.context and "extra" not in kwargs:
            kwargs["extra"] = {}

        if self.context:
            kwargs["extra"]["context"] = self.context

        return msg, kwargs

    # Compatibilidad: permitir uso `with logger.context(**kwargs): ...`
    def context(self, **kwargs):
        """Crear un context manager de logging empresarial.

        Acepta las claves:
        - component (str)
        - operation (str)
        - metadata (dict, opcional)
        - cualquier otra clave se añade al metadata
        """
        from .logger import enterprise_log_context  # import local para evitar ciclos

        component = kwargs.pop("component", "unknown")
        operation = kwargs.pop("operation", "unknown")
        md = kwargs.pop("metadata", {}) or {}
        # incluir cualquier resto (user_id, session_id, etc.) dentro de metadata
        md.update(kwargs)
        return enterprise_log_context(component, operation, **md)


def setup_enterprise_logging(
    name: str = "sheily_ai",
    level: str = "INFO",
    log_file: str = None,
    include_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_json_format: bool = True,
) -> logging.Logger:
    """
    Configurar logging empresarial avanzado

    Args:
        name: Nombre del logger
        level: Nivel de logging
        log_file: Archivo de log (opcional)
        include_console: Incluir salida consola
        max_file_size: Tamaño máximo archivo log
        backup_count: Número de backups
        enable_json_format: Formato JSON estructurado

    Returns:
        Logger empresarial configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Evitar duplicación de handlers
    if logger.handlers:
        return logger

    # Crear directorio de logs si no existe
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Formateador empresarial
    if enable_json_format:
        formatter = EnterpriseFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(component)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Handler de archivo rotativo
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Handler de consola
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str, context: LogContext = None) -> EnterpriseLoggerAdapter:
    """
    Obtener logger empresarial con contexto opcional

    Args:
        name: Nombre del logger
        context: Contexto empresarial opcional

    Returns:
        Logger empresarial adaptado
    """
    logger = setup_enterprise_logging(name)
    return EnterpriseLoggerAdapter(logger, context)


def create_log_context(
    component: str, operation: str, user_id: str = "system", session_id: str = None, **metadata
) -> LogContext:
    """
    Crear contexto de logging empresarial

    Args:
        component: Componente del sistema
        operation: Operación siendo ejecutada
        user_id: ID del usuario
        session_id: ID de sesión opcional
        **metadata: Metadatos adicionales

    Returns:
        Contexto de logging estructurado
    """
    return LogContext(
        component=component,
        operation=operation,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
    )


def log_enterprise_function(
    component: str = "unknown",
    operation: str = "unknown",
    include_args: bool = False,
    include_return: bool = False,
):
    """
    Decorador para logging empresarial de funciones

    Args:
        component: Componente del sistema
        operation: Operación siendo ejecutada
        include_args: Incluir argumentos en logs
        include_return: Incluir valor de retorno en logs

    Returns:
        Decorador de función
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear contexto de logging
            context = create_log_context(
                component=component,
                operation=operation,
                metadata={"function": func.__name__, "module": func.__module__},
            )

            logger = get_logger(f"{component}.{func.__name__}", context)

            # Log de inicio
            start_time = datetime.now()
            logger.info(f"🚀 Iniciando {operation}")

            if include_args:
                logger.debug(f"📥 Argumentos: {args}, {kwargs}")

            try:
                # Ejecutar función
                result = func(*args, **kwargs)

                # Log de éxito
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ {operation} completado en {execution_time:.3f}s")

                if include_return:
                    logger.debug(f"📤 Retorno: {result}")

                return result

            except Exception as e:
                # Log de error
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ {operation} falló en {execution_time:.3f}s: {e}")

                # Agregar información de error al contexto
                if hasattr(logger, "context") and logger.context:
                    logger.context.metadata["error"] = str(e)
                    logger.context.metadata["error_type"] = type(e).__name__

                raise

        return wrapper

    return decorator


def log_enterprise_performance(
    component: str = "unknown",
    operation: str = "unknown",
    warn_threshold: float = 1.0,
    error_threshold: float = 5.0,
):
    """
    Decorador para logging empresarial con monitoreo de performance

    Args:
        component: Componente del sistema
        operation: Operación siendo ejecutada
        warn_threshold: Umbral de warning en segundos
        error_threshold: Umbral de error en segundos

    Returns:
        Decorador de función con monitoreo de performance
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = create_log_context(component=component, operation=operation, metadata={"function": func.__name__})

            logger = get_logger(f"{component}.{func.__name__}", context)
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()

                # Log basado en tiempo de ejecución
                if execution_time > error_threshold:
                    logger.error(f"⏰ {operation} muy lento: {execution_time:.3f}s > {error_threshold}s")
                elif execution_time > warn_threshold:
                    logger.warning(f"⚠️ {operation} lento: {execution_time:.3f}s > {warn_threshold}s")
                else:
                    logger.info(f"⚡ {operation} completado: {execution_time:.3f}s")

                return result

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"💥 {operation} falló: {execution_time:.3f}s - {e}")
                raise

        return wrapper

    return decorator


# Configuración por defecto empresarial
DEFAULT_LOG_CONFIG = {
    "level": "INFO",
    "log_file": "logs/sheily_ai.log",
    "include_console": True,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "enable_json_format": True,
}


def configure_global_logging(config: Dict[str, Any] = None):
    """
    Configurar logging global empresarial

    Args:
        config: Configuración de logging (opcional)
    """
    if config is None:
        config = DEFAULT_LOG_CONFIG

    # Configurar logger raíz
    setup_enterprise_logging(
        name="sheily_ai",
        level=config["level"],
        log_file=config["log_file"],
        include_console=config["include_console"],
        max_file_size=config["max_file_size"],
        backup_count=config["backup_count"],
        enable_json_format=config["enable_json_format"],
    )


# Inicializar logging global si no está configurado
if not logging.getLogger("sheily_ai").handlers:
    configure_global_logging()


# Funciones de conveniencia empresarial
def log_info(message: str, component: str = "system", operation: str = "info", **metadata):
    """Log de información empresarial"""
    context = create_log_context(component, operation, metadata=metadata)
    logger = get_logger(f"sheily_ai.{component}", context)
    logger.info(message)


def log_warning(message: str, component: str = "system", operation: str = "warning", **metadata):
    """Log de warning empresarial"""
    context = create_log_context(component, operation, metadata=metadata)
    logger = get_logger(f"sheily_ai.{component}", context)
    logger.warning(message)


def log_error(message: str, component: str = "system", operation: str = "error", **metadata):
    """Log de error empresarial"""
    context = create_log_context(component, operation, metadata=metadata)
    logger = get_logger(f"sheily_ai.{component}", context)
    logger.error(message)


def log_debug(message: str, component: str = "system", operation: str = "debug", **metadata):
    """Log de debug empresarial"""
    context = create_log_context(component, operation, metadata=metadata)
    logger = get_logger(f"sheily_ai.{component}", context)
    logger.debug(message)


def log_critical(message: str, component: str = "system", operation: str = "critical", **metadata):
    """Log crítico empresarial"""
    context = create_log_context(component, operation, metadata=metadata)
    logger = get_logger(f"sheily_ai.{component}", context)
    logger.critical(message)


# Context manager para logging empresarial
class EnterpriseLogContext:
    """Context manager para logging empresarial con cleanup automático"""

    def __init__(self, component: str, operation: str, **metadata):
        self.component = component
        self.operation = operation
        self.metadata = metadata
        self.context = None
        self.logger = None

    def __enter__(self):
        self.context = create_log_context(self.component, self.operation, metadata=self.metadata)
        self.logger = get_logger(f"sheily_ai.{self.component}", self.context)
        self.logger.info(f"🔄 Iniciando contexto: {self.operation}")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"💥 Error en contexto {self.operation}: {exc_val}")
        else:
            self.logger.info(f"✅ Contexto completado: {self.operation}")


# Función de conveniencia para context manager
def enterprise_log_context(component: str, operation: str, **metadata):
    """Crear context manager de logging empresarial"""
    return EnterpriseLogContext(component, operation, **metadata)


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    "LogContext",
    "EnterpriseFormatter",
    "EnterpriseLoggerAdapter",
    "setup_enterprise_logging",
    "get_logger",
    "create_log_context",
    "log_enterprise_function",
    "log_enterprise_performance",
    "configure_global_logging",
    "log_info",
    "log_warning",
    "log_error",
    "log_debug",
    "log_critical",
    "EnterpriseLogContext",
    "enterprise_log_context",
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "Sheily AI Team - Enterprise Logging System"

# Log de carga del módulo
log_info("Módulo de logging empresarial cargado", "logger", "module_load")
