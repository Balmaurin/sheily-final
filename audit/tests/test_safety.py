#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests para sheily_core.safety
Coverage: Seguridad y monitoreo
"""

import pytest

from sheily_core.safety import (
    SecurityConfig,
    SecurityEvent,
    SecurityMonitor,
    create_security_event,
    get_security_monitor,
)


class TestSecurityMonitorSetup:
    """Tests para configuración de monitor de seguridad"""

    def test_security_config_creation(self):
        """Verificar creación de configuración de seguridad"""
        cfg = SecurityConfig()
        assert isinstance(cfg, SecurityConfig)

    def test_security_monitor_creation(self):
        """Verificar creación del monitor de seguridad"""
        monitor = SecurityMonitor()
        assert isinstance(monitor, SecurityMonitor)

    def test_get_security_monitor_function(self):
        """Verificar función get_security_monitor"""
        monitor = get_security_monitor()
        assert isinstance(monitor, SecurityMonitor)


class TestSecurityEventHandling:
    """Tests para manejo de eventos de seguridad"""

    def test_security_event_creation(self):
        """Verificar creación de evento de seguridad"""
        event = create_security_event(event_type="test_event", severity="low", description="Test message")
        assert isinstance(event, SecurityEvent)

    def test_security_event_attributes(self):
        """Verificar atributos del evento de seguridad"""
        event = create_security_event(event_type="test", severity="high", description="Test")
        assert event.event_type == "test"
        assert event.severity == "high"

    def test_record_security_event(self):
        """Verificar grabación de evento"""
        monitor = SecurityMonitor()
        event = create_security_event(event_type="test", severity="low", description="Test")
        try:
            monitor.record_security_event(event)
            # Si no lanza excepción, funciona
        except Exception as e:
            pytest.fail(f"Failed to record event: {e}")


class TestRequestChecking:
    """Tests para validación de requests"""

    def test_check_normal_request(self):
        """Verificar que request normal es válido"""
        monitor = SecurityMonitor()
        is_safe, reason = monitor.check_request("Hello world", client_id="test")
        assert isinstance(is_safe, bool)
        assert isinstance(reason, str)

    def test_check_request_returns_tuple(self):
        """Verificar que check_request retorna tupla"""
        monitor = SecurityMonitor()
        result = monitor.check_request("test query", client_id="client1")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_multiple_client_requests(self):
        """Verificar que se pueden chequear múltiples clientes"""
        monitor = SecurityMonitor()
        result1 = monitor.check_request("query1", client_id="client1")
        result2 = monitor.check_request("query2", client_id="client2")
        assert isinstance(result1, tuple)
        assert isinstance(result2, tuple)


class TestSecuritySummary:
    """Tests para resumen de seguridad"""

    def test_security_summary(self):
        """Verificar obtención de resumen de seguridad"""
        monitor = SecurityMonitor()
        try:
            summary = monitor.get_security_summary()
            assert isinstance(summary, dict)
        except Exception as e:
            pytest.fail(f"Failed to get security summary: {e}")

    def test_security_summary_structure(self):
        """Verificar estructura del resumen"""
        monitor = SecurityMonitor()
        summary = monitor.get_security_summary(hours=24)
        assert isinstance(summary, dict)
