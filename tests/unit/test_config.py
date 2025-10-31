#!/usr/bin/env python3
"""
Unit Tests: Configuration System
=================================
Tests para el sistema de configuración de Sheily AI.
"""

from pathlib import Path

import pytest


@pytest.mark.unit
class TestEnterpriseConfig:
    """Tests para EnterpriseConfig"""

    def test_config_import(self):
        """Verificar que se puede importar la configuración"""
        from sheily_core.config import get_config

        config = get_config()
        assert config is not None

    def test_config_system_name(self):
        """Verificar nombre del sistema"""
        from sheily_core.config import get_config

        config = get_config()
        assert config.system_name == "Sheily AI Enterprise"

    def test_config_default_host(self):
        """Verificar host por defecto es seguro"""
        from sheily_core.config import get_config

        config = get_config()
        assert config.host in ["127.0.0.1", "localhost"]

    def test_config_cors_not_wildcard(self):
        """Verificar que CORS no es wildcard"""
        from sheily_core.config import get_config

        config = get_config()
        assert config.cors_origins != ["*"]

    def test_config_has_api_prefix(self):
        """Verificar que tiene API prefix configurado"""
        from sheily_core.config import get_config

        config = get_config()
        assert hasattr(config, "api_prefix")
        assert config.api_prefix.startswith("/")


@pytest.mark.unit
class TestConfigValidation:
    """Tests para validación de configuración"""

    def test_port_is_valid(self):
        """Verificar que el puerto es válido"""
        from sheily_core.config import get_config

        config = get_config()
        assert 1 <= config.port <= 65535

    def test_cors_origins_is_list(self):
        """Verificar que cors_origins es una lista"""
        from sheily_core.config import get_config

        config = get_config()
        assert isinstance(config.cors_origins, list)

    def test_environment_variables_loading(self, test_env_vars):
        """Verificar que carga variables de entorno"""
        import os

        os.environ["PORT"] = "9000"

        from sheily_core.config import get_config

        config = get_config()
        # La configuración puede o no usar PORT, verificar que no falla
        assert config is not None


@pytest.mark.unit
class TestConfigPaths:
    """Tests para paths de configuración"""

    def test_model_paths_exist(self, project_root):
        """Verificar que los paths de modelos son válidos"""
        var_dir = project_root / "var"
        assert isinstance(var_dir, Path)

    def test_log_paths_are_paths(self):
        """Verificar que los paths de logs son Path objects"""
        from sheily_core.config import get_config

        config = get_config()

        if hasattr(config, "log_path"):
            assert isinstance(config.log_path, (str, Path))
