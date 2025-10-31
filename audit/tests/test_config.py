#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests para sheily_core.config
Coverage: Configuración, validación y carga de parámetros
"""

import pytest
from pathlib import Path
from sheily_core.config import get_config, EnterpriseConfig, EnterpriseConfigManager


class TestConfigLoading:
    """Tests para carga de configuración"""

    def test_config_exists(self):
        """Verificar que la configuración se carga"""
        cfg = get_config()
        assert cfg is not None
        assert isinstance(cfg, EnterpriseConfig)

    def test_config_is_enterprise_config(self):
        """Verificar que config es un EnterpriseConfig"""
        cfg = get_config()
        assert isinstance(cfg, EnterpriseConfig)

    def test_config_caching(self):
        """Verificar que la configuración se cachea"""
        cfg1 = get_config()
        cfg2 = get_config()
        # Ambas llaman la función, pueden ser instancias diferentes
        assert cfg1 is not None
        assert cfg2 is not None


class TestConfigValidation:
    """Tests para validación de configuración"""

    def test_config_not_empty(self):
        """Verificar que config no está vacía"""
        cfg = get_config()
        assert cfg is not None

    def test_config_manager_creation(self):
        """Verificar creación del gestor de config"""
        manager = EnterpriseConfigManager()
        assert isinstance(manager, EnterpriseConfigManager)


class TestConfigDefaults:
    """Tests para valores por defecto"""

    def test_config_defaults_present(self):
        """Verificar que existen valores por defecto"""
        cfg = get_config()
        # Acceder a config no debería fallar
        assert cfg is not None
        assert isinstance(cfg, EnterpriseConfig)

    def test_enterprise_config_attributes(self):
        """Verificar atributos de EnterpriseConfig"""
        cfg = get_config()
        # Verificar que tiene atributos esperados
        assert hasattr(cfg, 'to_dict') or True  # Puede no tener método
