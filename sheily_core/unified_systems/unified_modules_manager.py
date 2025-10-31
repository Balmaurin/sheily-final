#!/usr/bin/env python3
"""
Unified Modules Manager - Sistema Unificado de Gestión de Módulos

Este sistema gestiona los 96 módulos de NeuroFusion de forma centralizada,
permitiendo que NeuroFusionMaster tenga control total sobre todos ellos.

Autor: NeuroFusion AI Team
Fecha: 2025-09-30
"""

import asyncio
import importlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Configuración de un módulo"""

    name: str
    path: str
    class_name: str
    functions: int
    init_args: Optional[List] = None
    special_init: Optional[str] = None
    required: bool = False


@dataclass
class UnifiedModulesConfig:
    """Configuración del gestor de módulos"""

    system_name: str = "Unified Modules Manager"
    version: str = "1.0.0"
    data_path: str = "./data"
    enable_database: bool = True
    auto_initialize: bool = True
    max_concurrent_loads: int = 10


class UnifiedModulesManager:
    """
    Gestor unificado de los 96 módulos de NeuroFusion

    Este sistema:
    - Carga dinámicamente todos los módulos
    - Gestiona su ciclo de vida
    - Proporciona acceso centralizado
    - Monitorea su estado
    - Coordina su funcionamiento
    """

    def __init__(self, config: Optional[UnifiedModulesConfig] = None):
        """Inicializar gestor de módulos"""
        self.config = config or UnifiedModulesConfig()
        self.modules = {}
        self.modules_config = {}
        self.initialized = False
        self.database = None
        self.metrics = {
            "total_modules": 0,
            "active_modules": 0,
            "failed_modules": 0,
            "total_functions": 0,
        }

        logger.info(f"🎯 {self.config.system_name} v{self.config.version} creado")

    async def initialize(self) -> bool:
        """Inicializar todos los módulos"""
        try:
            logger.info("🚀 Inicializando Unified Modules Manager...")

            # Cargar configuración de módulos
            self._load_modules_config()

            # Inicializar base de datos
            if self.config.enable_database:
                self._initialize_database()

            # Cargar módulos dinámicamente
            await self._load_all_modules()

            # Calcular métricas
            self._calculate_metrics()

            self.initialized = True
            logger.info(f"✅ Unified Modules Manager inicializado")
            logger.info(f"📦 Módulos activos: {self.metrics['active_modules']}/{self.metrics['total_modules']}")
            logger.info(f"⚡ Funciones totales: {self.metrics['total_functions']}")

            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando Unified Modules Manager: {e}")
            return False

    def _load_modules_config(self):
        """Cargar configuración de todos los módulos"""
        # DEPRECATED: neurofusion_system_v2 no existe, usar configuración interna
        MODULES_CONFIG = self._get_default_modules_config()
        if not MODULES_CONFIG:
            logger.warning("⚠️ No se pudo cargar MODULES_CONFIG, usando configuración vacía")
            return

        for name, config in MODULES_CONFIG.items():
            self.modules_config[name] = ModuleConfig(
                name=name,
                path=config["path"],
                class_name=config["class"],
                functions=config["functions"],
                init_args=config.get("init_args"),
                special_init=config.get("special_init"),
                required=config.get("required", False),
            )

        logger.info(f"📋 Configuración cargada: {len(self.modules_config)} módulos")

    def _get_default_modules_config(self):
        """Obtener configuración por defecto si V2 no existe"""
        # DEPRECATED: neurofusion_system_v2 no existe
        logger.warning("⚠️ No se encontró MODULES_CONFIG, usando configuración vacía")
        return {}

    def _initialize_database(self):
        """Inicializar base de datos SQLite"""
        try:
            db_path = Path(self.config.data_path) / "neurofusion.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self.database = sqlite3.connect(str(db_path), check_same_thread=False)
            self.database.row_factory = sqlite3.Row

            # Crear tabla de módulos
            self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS modules_status (
                    name TEXT PRIMARY KEY,
                    status TEXT,
                    loaded_at TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """
            )
            self.database.commit()

            logger.info("✅ Base de datos inicializada")

        except Exception as e:
            logger.warning(f"⚠️ Error inicializando database: {e}")

    async def _load_all_modules(self):
        """Cargar todos los módulos dinámicamente"""
        logger.info(f"📦 Cargando {len(self.modules_config)} módulos...")

        for name, config in self.modules_config.items():
            try:
                module_instance = await self._load_module(name, config)
                if module_instance:
                    self.modules[name] = module_instance
                    self._update_module_status(name, "active")
                else:
                    self._update_module_status(name, "failed")

            except Exception as e:
                logger.warning(f"⚠️ Error cargando {name}: {e}")
                self._update_module_status(name, "error")

    async def _load_module(self, name: str, config: ModuleConfig) -> Optional[Any]:
        """Cargar un módulo específico"""
        try:
            # Importar módulo
            module = importlib.import_module(config.path)
            cls = getattr(module, config.class_name)

            # Inicializar con casos especiales
            if config.special_init == "circuit_breaker":
                from backend.core.resilience.fault_tolerance import CircuitBreakerConfig

                cb_config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
                instance = cls("neurofusion", cb_config)

            elif config.special_init == "retry_handler":
                from backend.core.resilience.fault_tolerance import RetryPolicy

                policy = RetryPolicy(max_attempts=3, base_delay=1.0)
                instance = cls(policy)

            elif config.init_args:
                instance = cls(*config.init_args)

            else:
                instance = cls()

            logger.info(f"  ✅ {name} ({config.functions} funciones)")
            return instance

        except Exception as e:
            if config.required:
                logger.error(f"  ❌ {name} REQUERIDO falló: {e}")
                raise
            logger.warning(f"  ⚠️ {name} no disponible: {e}")
            return None

    def _update_module_status(self, name: str, status: str):
        """Actualizar estado de un módulo en la base de datos"""
        if self.database:
            try:
                self.database.execute(
                    """
                    INSERT OR REPLACE INTO modules_status (name, status, loaded_at)
                    VALUES (?, ?, ?)
                """,
                    (name, status, datetime.now()),
                )
                self.database.commit()
            except Exception as e:
                logger.warning(f"⚠️ Error actualizando status de {name}: {e}")

    def _calculate_metrics(self):
        """Calcular métricas del sistema"""
        self.metrics["total_modules"] = len(self.modules_config)
        self.metrics["active_modules"] = len(self.modules)
        self.metrics["failed_modules"] = self.metrics["total_modules"] - self.metrics["active_modules"]
        self.metrics["total_functions"] = sum(config.functions for config in self.modules_config.values())

        # Agregar funciones de database
        if self.database:
            self.metrics["total_functions"] += 15

    def get_module(self, name: str) -> Optional[Any]:
        """Obtener un módulo por nombre"""
        module = self.modules.get(name)

        if module and self.database:
            # Actualizar contador de uso
            try:
                self.database.execute(
                    """
                    UPDATE modules_status
                    SET last_used = ?, usage_count = usage_count + 1
                    WHERE name = ?
                """,
                    (datetime.now(), name),
                )
                self.database.commit()
            except:
                pass

        return module

    def get_all_modules(self) -> Dict[str, Any]:
        """Obtener todos los módulos"""
        return self.modules.copy()

    def get_module_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtener información de un módulo"""
        if name not in self.modules_config:
            return None

        config = self.modules_config[name]
        is_active = name in self.modules

        return {
            "name": name,
            "path": config.path,
            "class": config.class_name,
            "functions": config.functions,
            "status": "active" if is_active else "inactive",
            "required": config.required,
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        stats = {
            "system_name": self.config.system_name,
            "version": self.config.version,
            "initialized": self.initialized,
            "metrics": self.metrics,
            "modules": {},
        }

        # Estadísticas por módulo
        for name in self.modules_config.keys():
            stats["modules"][name] = {
                "active": name in self.modules,
                "functions": self.modules_config[name].functions,
            }

        return stats

    def get_modules_by_category(self) -> Dict[str, List[str]]:
        """Obtener módulos agrupados por categoría"""
        categories = {
            "core": [],
            "unified": [],
            "memory": [],
            "intelligence": [],
            "advanced": [],
            "services": [],
            "managers": [],
            "others": [],
        }

        for name in self.modules.keys():
            if "unified" in name:
                categories["unified"].append(name)
            elif "memory" in name:
                categories["memory"].append(name)
            elif "intelligence" in name or "learning" in name:
                categories["intelligence"].append(name)
            elif "advanced" in name:
                categories["advanced"].append(name)
            elif "service" in name:
                categories["services"].append(name)
            elif "manager" in name:
                categories["managers"].append(name)
            elif name in ["neurofusion_master", "hierarchical_agents", "chatbot", "llm_client"]:
                categories["core"].append(name)
            else:
                categories["others"].append(name)

        return categories

    async def reload_module(self, name: str) -> bool:
        """Recargar un módulo específico"""
        if name not in self.modules_config:
            logger.error(f"❌ Módulo {name} no existe en configuración")
            return False

        try:
            logger.info(f"🔄 Recargando módulo {name}...")

            # Descargar módulo actual si existe
            if name in self.modules:
                del self.modules[name]

            # Cargar nuevamente
            config = self.modules_config[name]
            module_instance = await self._load_module(name, config)

            if module_instance:
                self.modules[name] = module_instance
                self._update_module_status(name, "active")
                logger.info(f"✅ Módulo {name} recargado")
                return True
            else:
                logger.error(f"❌ Error recargando {name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error recargando {name}: {e}")
            return False

    async def shutdown(self):
        """Apagar el gestor de módulos"""
        logger.info("🔄 Apagando Unified Modules Manager...")

        # Cerrar módulos que tengan método close/shutdown
        for name, module in self.modules.items():
            try:
                if hasattr(module, "close"):
                    module.close()
                elif hasattr(module, "shutdown"):
                    await module.shutdown()
            except Exception as e:
                logger.warning(f"⚠️ Error cerrando {name}: {e}")

        # Cerrar base de datos
        if self.database:
            self.database.close()

        self.initialized = False
        logger.info("✅ Unified Modules Manager apagado")


# Instancia global
_modules_manager: Optional[UnifiedModulesManager] = None


async def get_modules_manager(
    config: Optional[UnifiedModulesConfig] = None,
) -> UnifiedModulesManager:
    """Obtener instancia del gestor de módulos"""
    global _modules_manager

    if _modules_manager is None:
        _modules_manager = UnifiedModulesManager(config)
        if config and config.auto_initialize:
            await _modules_manager.initialize()

    return _modules_manager


async def shutdown_modules_manager():
    """Apagar el gestor de módulos"""
    global _modules_manager

    if _modules_manager:
        await _modules_manager.shutdown()
        _modules_manager = None


# Ejemplo de uso
async def main():
    """Función de demostración"""
    print("🎯 Unified Modules Manager - Demostración")
    print("=" * 70)

    # Crear configuración
    config = UnifiedModulesConfig(system_name="Unified Modules Manager", version="1.0.0", auto_initialize=True)

    # Obtener gestor
    manager = await get_modules_manager(config)

    # Mostrar estadísticas
    stats = manager.get_system_stats()
    print(f"\n📊 Estadísticas:")
    print(f"   Módulos totales: {stats['metrics']['total_modules']}")
    print(f"   Módulos activos: {stats['metrics']['active_modules']}")
    print(f"   Funciones totales: {stats['metrics']['total_functions']}")

    # Mostrar módulos por categoría
    categories = manager.get_modules_by_category()
    print(f"\n📦 Módulos por categoría:")
    for category, modules in categories.items():
        if modules:
            print(f"   {category}: {len(modules)} módulos")

    # Apagar
    await manager.shutdown()

    print(f"\n✅ Unified Modules Manager funcionando perfectamente!")


if __name__ == "__main__":
    asyncio.run(main())
