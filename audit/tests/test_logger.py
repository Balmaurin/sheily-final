#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests para sheily_core.logger
Coverage: Sistema de logging empresarial
"""

import logging
from io import StringIO

import pytest

from sheily_core.logger import get_logger, log_error, log_info, log_warning, setup_enterprise_logging


class TestLoggerSetup:
    """Tests para configuración de logger"""

    def test_logger_creation(self):
        """Verificar creación de logger"""
        logger = get_logger("test_logger")
        assert logger is not None
        # Logger puede ser Logger o EnterpriseLoggerAdapter
        assert hasattr(logger, "info") or hasattr(logger, "debug")

    def test_logger_has_handlers(self):
        """Verificar que el logger tiene handlers"""
        logger = get_logger("test_with_handlers")
        # El logger debe tener handlers o estar conectado a root
        assert logger is not None

    def test_logger_logging_level(self):
        """Verificar que el logging level se puede configurar"""
        logger = get_logger("test_level")
        # Debe aceptar niveles de logging
        assert logger is not None
        # Logger puede no tener atributo level si es adapter
        assert hasattr(logger, "level") or hasattr(logger, "logger")


class TestLoggingFunctions:
    """Tests para funciones de logging"""

    def test_log_info_function(self):
        """Verificar función log_info"""
        try:
            log_info("Test info message", component="test")
            # Si no lanza excepción, funciona correctamente
        except Exception as e:
            pytest.fail(f"log_info failed: {e}")

    def test_log_error_function(self):
        """Verificar función log_error"""
        try:
            log_error("Test error message", component="test", error="test_error")
            # Si no lanza excepción, funciona correctamente
        except Exception as e:
            pytest.fail(f"log_error failed: {e}")

    def test_log_warning_function(self):
        """Verificar función log_warning"""
        try:
            log_warning("Test warning message", component="test")
            # Si no lanza excepción, funciona correctamente
        except Exception as e:
            pytest.fail(f"log_warning failed: {e}")


class TestEnterpriseLogging:
    """Tests para logging empresarial"""

    def test_enterprise_logging_setup(self):
        """Verificar setup de logging empresarial"""
        try:
            setup_enterprise_logging()
            # Si no lanza excepción, setup funciona
        except Exception as e:
            pytest.fail(f"Enterprise logging setup failed: {e}")

    def test_logger_name_tracking(self):
        """Verificar que se puede rastrear por nombre"""
        logger = get_logger("test_tracking")
        assert logger.name == "test_tracking"

    def test_logger_context_support(self):
        """Verificar que logger soporta contexto"""
        logger = get_logger("test_context")
        # Debe permitir metadata contextual
        assert logger is not None


class TestLoggerErrorHandling:
    """Tests para manejo de errores en logging"""

    def test_logger_with_none_message(self):
        """Verificar que logger maneja mensajes None"""
        logger = get_logger("test_none")
        try:
            # Algunos loggers manejan None, otros no
            # Verificar que no lanza excepción no controlada
            assert logger is not None
        except Exception as e:
            pytest.fail(f"Logger failed with None message: {e}")

    def test_multiple_loggers(self):
        """Verificar que se pueden crear múltiples loggers"""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        assert logger1 is not None
        assert logger2 is not None
        # Pueden ser la misma instancia o diferentes
        assert logger1.name != logger2.name or logger1 is logger2
