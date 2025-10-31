#!/usr/bin/env python3
"""
Unit Tests: Health Checks System
=================================
Tests para el sistema de health checks (health.py).
"""

import pytest


@pytest.mark.unit
class TestHealthStatus:
    """Tests para HealthStatus enum"""

    def test_health_status_import(self):
        """Verificar que se puede importar HealthStatus"""
        from sheily_core.health import HealthStatus

        assert HealthStatus is not None

    def test_health_status_values(self):
        """Verificar que tiene los valores correctos"""
        from sheily_core.health import HealthStatus

        assert hasattr(HealthStatus, "HEALTHY")
        assert hasattr(HealthStatus, "DEGRADED")
        assert hasattr(HealthStatus, "UNHEALTHY")
        assert hasattr(HealthStatus, "UNKNOWN")


@pytest.mark.unit
class TestComponentHealth:
    """Tests para ComponentHealth dataclass"""

    def test_component_health_creation(self):
        """Verificar creación de ComponentHealth"""
        from sheily_core.health import ComponentHealth, HealthStatus

        component = ComponentHealth(name="test", status=HealthStatus.HEALTHY, message="All good")

        assert component.name == "test"
        assert component.status == HealthStatus.HEALTHY
        assert component.message == "All good"

    def test_component_health_with_metadata(self):
        """Verificar ComponentHealth con metadata"""
        from sheily_core.health import ComponentHealth, HealthStatus

        component = ComponentHealth(
            name="disk", status=HealthStatus.HEALTHY, message="Disk OK", metadata={"free_gb": 100, "used_percent": 50}
        )

        assert component.metadata["free_gb"] == 100
        assert component.metadata["used_percent"] == 50


@pytest.mark.unit
class TestSystemHealth:
    """Tests para SystemHealth dataclass"""

    def test_system_health_creation(self):
        """Verificar creación de SystemHealth"""
        from sheily_core.health import ComponentHealth, HealthStatus, SystemHealth

        components = [ComponentHealth(name="test1", status=HealthStatus.HEALTHY, message="OK")]

        system_health = SystemHealth(
            status=HealthStatus.HEALTHY, components=components, system_info={"version": "1.0.0"}
        )

        assert system_health.status == HealthStatus.HEALTHY
        assert len(system_health.components) == 1

    def test_is_healthy_property(self):
        """Verificar propiedad is_healthy"""
        from sheily_core.health import ComponentHealth, HealthStatus, SystemHealth

        components = [ComponentHealth(name="test", status=HealthStatus.HEALTHY, message="OK")]

        system_health = SystemHealth(status=HealthStatus.HEALTHY, components=components, system_info={})

        assert system_health.is_healthy is True

        system_health.status = HealthStatus.UNHEALTHY
        assert system_health.is_healthy is False

    def test_unhealthy_components_property(self):
        """Verificar propiedad unhealthy_components"""
        from sheily_core.health import ComponentHealth, HealthStatus, SystemHealth

        components = [
            ComponentHealth(name="good", status=HealthStatus.HEALTHY, message="OK"),
            ComponentHealth(name="bad", status=HealthStatus.UNHEALTHY, message="Error"),
        ]

        system_health = SystemHealth(status=HealthStatus.DEGRADED, components=components, system_info={})

        unhealthy = system_health.unhealthy_components
        assert len(unhealthy) == 1
        assert "bad" in unhealthy


@pytest.mark.unit
class TestHealthChecker:
    """Tests para HealthChecker class"""

    def test_health_checker_creation(self):
        """Verificar creación de HealthChecker"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()
        assert checker is not None

    def test_register_check(self):
        """Verificar registro de checks"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()

        def custom_check():
            pass

        checker.register_check("custom", custom_check)
        assert "custom" in checker.checks

    def test_check_all_returns_system_health(self):
        """Verificar que check_all devuelve SystemHealth"""
        from sheily_core.health import HealthChecker, SystemHealth

        checker = HealthChecker()
        result = checker.check_all()

        assert isinstance(result, SystemHealth)
        assert result.status is not None

    def test_to_dict_conversion(self):
        """Verificar conversión a diccionario"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()
        health = checker.check_all()
        health_dict = checker.to_dict(health)

        assert isinstance(health_dict, dict)
        assert "status" in health_dict
        assert "components" in health_dict
        assert "system" in health_dict


@pytest.mark.unit
class TestHealthCheckerChecks:
    """Tests para checks específicos del HealthChecker"""

    def test_system_check(self):
        """Verificar check del sistema"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_system()

        assert result.name == "system"
        assert result.status is not None

    def test_disk_check(self):
        """Verificar check de disco"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_disk()

        assert result.name == "disk"
        assert "percent_used" in result.metadata

    def test_memory_check(self):
        """Verificar check de memoria"""
        from sheily_core.health import HealthChecker

        checker = HealthChecker()
        result = checker._check_memory()

        assert result.name == "memory"
        assert "percent_used" in result.metadata


@pytest.mark.unit
class TestHealthModule:
    """Tests para funciones del módulo health"""

    def test_get_health_checker_singleton(self):
        """Verificar que get_health_checker devuelve singleton"""
        from sheily_core.health import get_health_checker

        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

    def test_check_health_function(self):
        """Verificar función check_health"""
        from sheily_core.health import check_health

        result = check_health()

        assert isinstance(result, dict)
        assert "status" in result
        assert "components" in result

    def test_check_health_has_required_fields(self):
        """Verificar que check_health tiene campos requeridos"""
        from sheily_core.health import check_health

        result = check_health()

        required_fields = ["status", "timestamp", "components", "system"]
        for field in required_fields:
            assert field in result
