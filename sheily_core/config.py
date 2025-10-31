#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Configuración Empresarial para Sheily AI
==================================================

Módulo de configuración avanzado con:
- Gestión centralizada de configuración
- Variables de entorno empresariales
- Configuración por ambientes
- Validación automática de configuración
- Hot reload de configuración
- Seguridad integrada en configuración
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class EnterpriseConfig:
    """Configuración empresarial completa para Sheily AI"""

    # Configuración básica del sistema
    system_name: str = "Sheily AI Enterprise"
    version: str = "2.0.0"
    environment: str = "development"

    # Configuración de modelo de lenguaje
    model_path: str = ""
    llama_binary_path: str = ""
    model_max_tokens: int = 2048
    model_temperature: float = 0.7
    model_threads: int = 4
    model_timeout: int = 300

    # Configuración de corpus y conocimiento
    corpus_root: str = "data"
    branches_config_path: str = "config/branches.json"
    context_max_length: int = 2000
    max_context_docs: int = 3

    # Configuración de seguridad empresarial
    security_enabled: bool = True
    max_queries_per_minute: int = 60
    max_query_length: int = 10000
    require_authentication: bool = False

    # Configuración de logging empresarial
    log_level: str = "INFO"
    log_file: str = "logs/sheily_ai.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # Configuración de monitoreo empresarial
    monitoring_enabled: bool = True
    metrics_collection_interval: int = 30
    metrics_retention_hours: int = 24

    # Configuración de red empresarial
    host: str = "127.0.0.1"  # Usar localhost por defecto (más seguro)
    port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ])

    # Configuración de base de datos empresarial
    database_url: str = "sqlite:///data/sheily_ai.db"
    connection_pool_size: int = 10
    connection_timeout: int = 30

    # Configuración de cache empresarial
    cache_enabled: bool = True
    cache_type: str = "redis"
    cache_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # Configuración de recursos empresariales
    max_memory_usage: int = 2048  # MB
    max_cpu_usage: int = 80  # porcentaje
    auto_scaling_enabled: bool = False

    # Configuración de características empresariales
    features: Dict[str, bool] = field(
        default_factory=lambda: {
            "chat": True,
            "rag": True,
            "memory": True,
            "monitoring": True,
            "security": True,
            "enterprise_mode": True,
        }
    )


class EnterpriseConfigManager:
    """Gestor avanzado de configuración empresarial"""

    def __init__(self, config_file: str = None):
    """__init__ function/class"""
        self.config_file = config_file or "config/enterprise_config.yaml"
        self.config: EnterpriseConfig = None
        self.logger = logging.getLogger("config_manager")

        # Cargar configuración
        self._load_configuration()

    def _load_configuration(self):
        """Cargar configuración empresarial desde múltiples fuentes"""
        try:
            # 1. Cargar variables de entorno
            self._load_environment_variables()

            # 2. Cargar archivo de configuración
            config_data = self._load_config_file()

            # 3. Fusionar configuraciones
            merged_config = self._merge_configurations(config_data)

            # 4. Crear objeto de configuración
            self.config = EnterpriseConfig(**merged_config)

            # 5. Validar configuración
            self._validate_configuration()

            self.logger.info("✅ Configuración empresarial cargada exitosamente")

        except Exception as e:
            self.logger.error(f"❌ Error cargando configuración empresarial: {e}")
            # Usar configuración por defecto
            self.config = EnterpriseConfig()

    def _load_environment_variables(self):
        """Cargar variables de entorno empresariales"""
        # Cargar archivo .env si existe
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

        # Variables de entorno específicas de Sheily AI
        env_mapping = {
            # Modelo
            "MODEL_PATH": "model_path",
            "LLAMA_BINARY_PATH": "llama_binary_path",
            "MODEL_MAX_TOKENS": "model_max_tokens",
            "MODEL_TEMPERATURE": "model_temperature",
            "MODEL_THREADS": "model_threads",
            "MODEL_TIMEOUT": "model_timeout",
            # Corpus
            "CORPUS_ROOT": "corpus_root",
            "BRANCHES_CONFIG_PATH": "branches_config_path",
            # Seguridad
            "MAX_QUERIES_PER_MINUTE": "max_queries_per_minute",
            "MAX_QUERY_LENGTH": "max_query_length",
            "REQUIRE_AUTHENTICATION": "require_authentication",
            # Logging
            "LOG_LEVEL": "log_level",
            "LOG_FILE": "log_file",
            # Red
            "HOST": "host",
            "PORT": "port",
            "API_PREFIX": "api_prefix",
            # Base de datos
            "DATABASE_URL": "database_url",
            # Cache
            "CACHE_URL": "cache_url",
            "CACHE_TTL": "cache_ttl",
            # Recursos
            "MAX_MEMORY_USAGE": "max_memory_usage",
            "MAX_CPU_USAGE": "max_cpu_usage",
        }

        # Aplicar variables de entorno
        for env_var, config_attr in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convertir tipos apropiados
                if config_attr in [
                    "model_max_tokens",
                    "model_threads",
                    "model_timeout",
                    "max_queries_per_minute",
                    "max_query_length",
                    "port",
                    "log_max_size",
                    "log_backup_count",
                    "connection_pool_size",
                    "connection_timeout",
                    "cache_ttl",
                    "max_memory_usage",
                    "max_cpu_usage",
                ]:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Mantener como string
                elif config_attr in [
                    "model_temperature",
                    "require_authentication",
                    "security_enabled",
                    "monitoring_enabled",
                    "cache_enabled",
                    "auto_scaling_enabled",
                ]:
                    value = value.lower() in ("true", "1", "yes", "on")

                # Aplicar valor
                setattr(self, f"_env_{config_attr}", value)

    def _load_config_file(self) -> Dict[str, Any]:
        """Cargar archivo de configuración empresarial"""
        config_path = Path(self.config_file)

        if not config_path.exists():
            self.logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            return {}

        try:
            # Intentar cargar como YAML primero
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

            # Intentar cargar como JSON
            elif config_path.suffix.lower() == ".json":
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            else:
                self.logger.warning(f"Formato de configuración no soportado: {config_path.suffix}")
                return {}

        except Exception as e:
            self.logger.error(f"Error cargando archivo de configuración: {e}")
            return {}

    def _merge_configurations(self, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionar configuraciones de archivo y entorno"""
        # Configuración base empresarial
        base_config = {
            "system_name": "Sheily AI Enterprise",
            "version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "model_path": getattr(self, "_env_model_path", ""),
            "llama_binary_path": getattr(self, "_env_llama_binary_path", ""),
            "model_max_tokens": getattr(self, "_env_model_max_tokens", 2048),
            "model_temperature": getattr(self, "_env_model_temperature", 0.7),
            "model_threads": getattr(self, "_env_model_threads", 4),
            "model_timeout": getattr(self, "_env_model_timeout", 300),
            "corpus_root": getattr(self, "_env_corpus_root", "data"),
            "branches_config_path": getattr(
                self, "_env_branches_config_path", "config/branches.json"
            ),
            "context_max_length": 2000,
            "max_context_docs": 3,
            "security_enabled": True,
            "max_queries_per_minute": getattr(self, "_env_max_queries_per_minute", 60),
            "max_query_length": getattr(self, "_env_max_query_length", 10000),
            "require_authentication": getattr(self, "_env_require_authentication", False),
            "log_level": getattr(self, "_env_log_level", "INFO"),
            "log_file": getattr(self, "_env_log_file", "logs/sheily_ai.log"),
            "log_max_size": 10 * 1024 * 1024,
            "log_backup_count": 5,
            "monitoring_enabled": True,
            "metrics_collection_interval": 30,
            "metrics_retention_hours": 24,
            "host": getattr(self, "_env_host", "127.0.0.1"),
            "port": getattr(self, "_env_port", 8000),
            "api_prefix": "/api/v1",
            "cors_origins": [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ],
            "database_url": getattr(self, "_env_database_url", "sqlite:///data/sheily_ai.db"),
            "connection_pool_size": 10,
            "connection_timeout": 30,
            "cache_enabled": True,
            "cache_type": "redis",
            "cache_url": getattr(self, "_env_cache_url", "redis://localhost:6379"),
            "cache_ttl": getattr(self, "_env_cache_ttl", 3600),
            "max_memory_usage": getattr(self, "_env_max_memory_usage", 2048),
            "max_cpu_usage": getattr(self, "_env_max_cpu_usage", 80),
            "auto_scaling_enabled": False,
            "features": {
                "chat": True,
                "rag": True,
                "memory": True,
                "monitoring": True,
                "security": True,
                "enterprise_mode": True,
            },
        }

        # Fusionar con configuración de archivo
        if file_config:
            self._deep_merge(base_config, file_config)

        return base_config

    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]):
        """Fusionar configuración recursivamente"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _validate_configuration(self):
        """Validar configuración empresarial"""
        if not self.config:
            raise ValueError("Configuración no inicializada")

        # Validaciones críticas
        if self.config.model_path and not Path(self.config.model_path).exists():
            self.logger.warning(f"Modelo no encontrado: {self.config.model_path}")

        if self.config.llama_binary_path and not Path(self.config.llama_binary_path).exists():
            self.logger.warning(f"Binario llama-cli no encontrado: {self.config.llama_binary_path}")

        if self.config.port < 1 or self.config.port > 65535:
            raise ValueError(f"Puerto inválido: {self.config.port}")

        if self.config.model_temperature < 0.0 or self.config.model_temperature > 2.0:
            raise ValueError(f"Temperatura inválida: {self.config.model_temperature}")

        if self.config.max_queries_per_minute < 1:
            raise ValueError(f"Límite de consultas inválido: {self.config.max_queries_per_minute}")

        self.logger.info("✅ Configuración empresarial validada correctamente")

    def get_config(self) -> EnterpriseConfig:
        """Obtener configuración empresarial"""
        return self.config

    def reload_configuration(self):
        """Recargar configuración empresarial"""
        self.logger.info("🔄 Recargando configuración empresarial...")
        old_config = self.config
        self._load_configuration()

        if old_config != self.config:
            self.logger.info("✅ Configuración empresarial recargada con cambios")
        else:
            self.logger.info("✅ Configuración empresarial recargada (sin cambios)")

    def save_configuration(self, file_path: str = None):
        """Guardar configuración empresarial a archivo"""
        save_path = Path(file_path or self.config_file)

        # Crear directorio si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convertir configuración a diccionario
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if not key.startswith("_"):
                config_dict[key] = value

        # Guardar como YAML
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✅ Configuración empresarial guardada: {save_path}")

    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Obtener configuración específica del ambiente"""
        env_configs = {
            "development": {
                "log_level": "DEBUG",
                "monitoring_enabled": True,
                "security_enabled": False,
                "auto_scaling_enabled": False,
            },
            "staging": {
                "log_level": "INFO",
                "monitoring_enabled": True,
                "security_enabled": True,
                "auto_scaling_enabled": False,
            },
            "production": {
                "log_level": "WARNING",
                "monitoring_enabled": True,
                "security_enabled": True,
                "auto_scaling_enabled": True,
            },
        }

        return env_configs.get(self.config.environment, {})


# Instancia global de configuración empresarial
_config_manager: Optional[EnterpriseConfigManager] = None


def get_config() -> EnterpriseConfig:
    """Obtener configuración empresarial global"""
    global _config_manager

    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()

    return _config_manager.get_config()


def reload_config():
    """Recargar configuración empresarial global"""
    global _config_manager

    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()

    _config_manager.reload_configuration()


def save_config(file_path: str = None):
    """Guardar configuración empresarial global"""
    global _config_manager

    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()

    _config_manager.save_configuration(file_path)


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    "EnterpriseConfig",
    "EnterpriseConfigManager",
    "get_config",
    "reload_config",
    "save_config",
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "Sheily AI Team - Enterprise Configuration System"
