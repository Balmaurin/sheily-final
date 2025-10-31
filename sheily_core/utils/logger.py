#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Logging System for Sheily AI
========================================

This module provides a centralized logging system with:
- Multiple log levels and outputs
- Structured logging with context
- Performance monitoring
- Error tracking and reporting
- Configurable formatters
"""

import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .config import get_config  # compat
except Exception:
    from ..core.config import get_config


@dataclass
class LogContext:
    """Context information for structured logging"""

    user_id: str = "system"
    session_id: str = ""
    request_id: str = ""
    component: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SheilyFormatter(logging.Formatter):
    """Custom formatter for Sheily logs"""

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

        # Different formats for different log levels
        self.formats = {
            logging.DEBUG: "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            logging.INFO: "%(asctime)s [%(levelname)8s] %(message)s",
            logging.WARNING: "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            logging.ERROR: "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s",
            logging.CRITICAL: "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s",
        }

        # Set default format
        self._style = logging.PercentStyle(self.formats[logging.INFO])

    def format(self, record):
        # Add context information if available
        if self.include_context and hasattr(record, "context"):
            context = record.context
            if context:
                # Add context fields to record
                for key, value in context.__dict__.items():
                    if key != "metadata" and value:
                        setattr(record, f"ctx_{key}", value)

                # Add metadata as a formatted string
                if context.metadata:
                    metadata_str = json.dumps(context.metadata, default=str)
                    setattr(record, "ctx_metadata", metadata_str)

        # Use appropriate format based on log level
        self._style._fmt = self.formats.get(record.levelno, self.formats[logging.INFO])
        self._style = logging.PercentStyle(self._style._fmt)

        return super().format(record)


class SheilyLogger:
    """Centralized logger for Sheily AI system"""

    def __init__(self, name: str = "sheily"):
        self.name = name
        self.config = get_config()
        self.logger = logging.getLogger(name)
        self._context: Optional[LogContext] = None

        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with handlers and formatters"""
        self.logger.setLevel(getattr(logging, self.config.logging.log_level.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        formatter = SheilyFormatter()

        # Console handler (always present)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if enabled)
        if self.config.logging.enable_file_logging:
            try:
                log_file_path = Path(self.config.logging.log_file)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Use rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    self.config.logging.log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            except Exception as e:
                # Fallback to console if file logging fails
                self.logger.warning(f"Could not setup file logging: {e}")

    def set_context(self, context: LogContext):
        """Set logging context for subsequent log messages"""
        self._context = context

    def clear_context(self):
        """Clear logging context"""
        self._context = None

    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary logging context"""
        old_context = self._context
        self._context = LogContext(**kwargs)
        try:
            yield
        finally:
            self._context = old_context

    def _log_with_context(self, level: int, message: str, extra: Dict[str, Any] = None, **kwargs):
        """Log message with context information"""
        if extra is None:
            extra = {}

        if self._context:
            extra["context"] = self._context

        # Add performance information for timing
        if "duration" in kwargs:
            extra["duration"] = kwargs["duration"]

        self.logger.log(level, message, extra=extra, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        self._log_with_context(logging.ERROR, message, **kwargs)

    @contextmanager
    def timer(self, operation: str = "operation"):
        """Context manager for timing operations"""
        start_time = time.time()
        context = LogContext(operation=operation)

        with self.context(component="timer", operation=operation):
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.info(f"Operation '{operation}' completed", duration=duration)


# Global logger instances
_main_logger: Optional[SheilyLogger] = None
_rag_logger: Optional[SheilyLogger] = None
_model_logger: Optional[SheilyLogger] = None
_server_logger: Optional[SheilyLogger] = None


def get_logger(name: str = "sheily") -> SheilyLogger:
    """Get logger instance by name"""
    global _main_logger, _rag_logger, _model_logger, _server_logger

    if name == "sheily" or name == "main":
        if _main_logger is None:
            _main_logger = SheilyLogger("sheily")
        return _main_logger

    elif name == "rag":
        if _rag_logger is None:
            _rag_logger = SheilyLogger("sheily.rag")
        return _rag_logger

    elif name == "model":
        if _model_logger is None:
            _model_logger = SheilyLogger("sheily.model")
        return _model_logger

    elif name == "server":
        if _server_logger is None:
            _server_logger = SheilyLogger("sheily.server")
        return _server_logger

    else:
        return SheilyLogger(f"sheily.{name}")


def init_logging():
    """Initialize logging system"""
    logger = get_logger("main")
    logger.info("Logging system initialized")

    # Log configuration summary
    config = get_config()
    logger.debug(
        "Configuration loaded",
        extra={
            "config_summary": {
                "debug": config.server.debug,
                "log_level": config.logging.log_level,
                "log_file": config.logging.log_file,
                "model_path": config.model.model_path[:50] + "..."
                if config.model.model_path
                else "Not set",
            }
        },
    )


def log_performance(operation: str, duration: float, **metadata):
    """Log performance metrics"""
    logger = get_logger("main")
    logger.info(
        f"Performance: {operation}", duration=duration, extra={"performance_data": metadata}
    )


def log_error(error: Exception, context: str = "", **metadata):
    """Log error with context"""
    logger = get_logger("main")
    logger.exception(f"Error in {context}: {str(error)}", extra={"error_metadata": metadata})


def log_request(
    request_id: str, method: str, endpoint: str, status_code: int, duration: float = None
):
    """Log HTTP request"""
    logger = get_logger("server")

    with logger.context(request_id=request_id, operation=f"{method} {endpoint}"):
        if status_code >= 400:
            logger.warning(f"HTTP request completed with status {status_code}", duration=duration)
        else:
            logger.info(f"HTTP request completed successfully", duration=duration)
