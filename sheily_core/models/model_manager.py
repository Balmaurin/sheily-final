#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Manager - Gestión de Modelos para Sheily-AI
=================================================

Gestiona la carga, descarga y administración de múltiples modelos
de lenguaje incluyendo GGUF, transformers, y adaptadores LoRA.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Gestor de modelos de lenguaje con soporte para múltiples formatos
    """

    def __init__(self, config: Dict):
        """
        Inicializar el gestor de modelos

        Args:
            config: Configuración del gestor
        """
        self.config = config
        self.models_path = Path(config.get("model_path", "models/"))
        self.cache_size = config.get("cache_size", 512)  # MB

        # Estado de modelos
        self._loaded_models = {}  # {model_name: model_info}
        self._model_cache = {}  # Cache de modelos recientes
        self._loading_lock = threading.RLock()

        # Allowlist de modelos
        self._load_allowlist()

        # Métricas
        self._model_stats = {
            "models_loaded": 0,
            "models_unloaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_load_time": 0.0,
        }

        logger.info(f"ModelManager inicializado - Path: {self.models_path}")

    def _load_allowlist(self):
        """Cargar lista de modelos permitidos"""
        allowlist_path = self.models_path / "ALLOWLIST.json"
        try:
            if allowlist_path.exists():
                with open(allowlist_path, "r", encoding="utf-8") as f:
                    self._allowlist = json.load(f)
                logger.info(f"Allowlist cargada: {len(self._allowlist)} modelos")
            else:
                self._allowlist = {}
                logger.warning("No se encontró ALLOWLIST.json - todos los modelos permitidos")
        except Exception as e:
            logger.error(f"Error cargando allowlist: {e}")
            self._allowlist = {}

    async def initialize(self) -> bool:
        """
        Inicializar el gestor de modelos

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Verificar directorio de modelos
            if not self.models_path.exists():
                logger.warning(f"Directorio de modelos no existe: {self.models_path}")
                self.models_path.mkdir(parents=True, exist_ok=True)

            # Escanear modelos disponibles
            await self._scan_available_models()

            logger.info("ModelManager inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando ModelManager: {e}")
            return False

    async def _scan_available_models(self):
        """Escanear modelos disponibles en el directorio"""
        available_models = {}

        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir():
                # Buscar archivos de modelo
                model_files = []
                for ext in [".gguf", ".bin", ".safetensors", ".pt"]:
                    model_files.extend(list(model_dir.glob(f"*{ext}")))

                if model_files:
                    available_models[model_dir.name] = {
                        "path": str(model_dir),
                        "files": [str(f) for f in model_files],
                        "type": self._detect_model_type(model_files[0]),
                        "size": sum(f.stat().st_size for f in model_files),
                    }

        self._available_models = available_models
        logger.info(f"Encontrados {len(available_models)} modelos disponibles")

    def _detect_model_type(self, model_path: Path) -> str:
        """Detectar tipo de modelo basado en la extensión"""
        suffix = model_path.suffix.lower()

        type_map = {
            ".gguf": "gguf",
            ".bin": "transformers",
            ".safetensors": "transformers",
            ".pt": "pytorch",
        }

        return type_map.get(suffix, "unknown")

    def _validate_model(self, model_name: str, model_path: str) -> bool:
        """
        Validar modelo contra allowlist y verificar integridad

        Args:
            model_name: Nombre del modelo
            model_path: Ruta del modelo

        Returns:
            bool: True si el modelo es válido
        """
        # Verificar allowlist si existe
        if self._allowlist and model_name not in self._allowlist:
            logger.error(f"Modelo {model_name} no está en allowlist")
            return False

        # Verificar que el archivo existe
        if not Path(model_path).exists():
            logger.error(f"Archivo de modelo no existe: {model_path}")
            return False

        # Verificar hash si está especificado en allowlist
        if self._allowlist and model_name in self._allowlist:
            expected_hash = self._allowlist[model_name].get("sha256")
            if expected_hash:
                actual_hash = self._calculate_file_hash(model_path)
                if actual_hash != expected_hash:
                    logger.error(f"Hash de modelo no coincide: {model_name}")
                    return False

        return True

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcular hash SHA256 de un archivo"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def load_model(self, model_name: str, model_path: Optional[str] = None) -> bool:
        """
        Cargar un modelo en memoria

        Args:
            model_name: Nombre identificador del modelo
            model_path: Ruta opcional del modelo (auto-detectar si no se especifica)

        Returns:
            bool: True si se cargó exitosamente
        """
        start_time = time.time()

        with self._loading_lock:
            # Verificar si ya está cargado
            if model_name in self._loaded_models:
                logger.info(f"Modelo {model_name} ya está cargado")
                self._model_stats["cache_hits"] += 1
                return True

            try:
                # Auto-detectar path si no se especifica
                if not model_path:
                    if model_name in self._available_models:
                        model_info = self._available_models[model_name]
                        model_path = model_info["files"][0]  # Usar primer archivo
                    else:
                        logger.error(f"Modelo {model_name} no encontrado")
                        return False

                # Validar modelo
                if not self._validate_model(model_name, model_path):
                    return False

                # Cargar según el tipo
                model_type = self._detect_model_type(Path(model_path))
                model_instance = await self._load_model_by_type(model_name, model_path, model_type)

                if model_instance:
                    # Registrar modelo cargado
                    self._loaded_models[model_name] = {
                        "instance": model_instance,
                        "path": model_path,
                        "type": model_type,
                        "load_time": time.time(),
                        "access_count": 0,
                        "last_access": time.time(),
                    }

                    # Actualizar métricas
                    load_time = time.time() - start_time
                    self._model_stats["models_loaded"] += 1
                    self._model_stats["total_load_time"] += load_time
                    self._model_stats["cache_misses"] += 1

                    logger.info(f"Modelo {model_name} cargado en {load_time:.2f}s")
                    return True

                return False

            except Exception as e:
                logger.error(f"Error cargando modelo {model_name}: {e}")
                return False

    async def _load_model_by_type(self, model_name: str, model_path: str, model_type: str) -> Optional[Any]:
        """
        Cargar modelo según su tipo

        Args:
            model_name: Nombre del modelo
            model_path: Ruta del modelo
            model_type: Tipo de modelo

        Returns:
            Instancia del modelo cargado o None
        """
        try:
            if model_type == "gguf":
                return await self._load_gguf_model(model_name, model_path)
            elif model_type == "transformers":
                return await self._load_transformers_model(model_name, model_path)
            elif model_type == "pytorch":
                return await self._load_pytorch_model(model_name, model_path)
            else:
                logger.error(f"Tipo de modelo no soportado: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Error cargando modelo {model_type}: {e}")
            return None

    async def _load_gguf_model(self, model_name: str, model_path: str) -> Optional[Any]:
        """Cargar modelo GGUF usando llama.cpp"""
        # Placeholder para integración con llama.cpp
        # En implementación real, usaría llama-cpp-python o wrapper
        logger.info(f"Cargando modelo GGUF: {model_name}")

        # Simulación de carga
        await asyncio.sleep(0.1)  # Simular tiempo de carga

        return {"type": "gguf", "path": model_path, "loaded": True, "backend": "llama_cpp"}

    async def _load_transformers_model(self, model_name: str, model_path: str) -> Optional[Any]:
        """Cargar modelo transformers (bloqueado por DepSwitch en producción)"""
        logger.warning(f"Cargando modelo transformers en modo desarrollo: {model_name}")

        # En producción, esto estaría bloqueado por DepSwitch
        # En desarrollo, se cargaría usando transformers
        await asyncio.sleep(0.1)

        return {
            "type": "transformers",
            "path": model_path,
            "loaded": True,
            "backend": "transformers",
        }

    async def _load_pytorch_model(self, model_name: str, model_path: str) -> Optional[Any]:
        """Cargar modelo PyTorch (bloqueado por DepSwitch en producción)"""
        logger.warning(f"Cargando modelo PyTorch en modo desarrollo: {model_name}")

        await asyncio.sleep(0.1)

        return {"type": "pytorch", "path": model_path, "loaded": True, "backend": "pytorch"}

    async def unload_model(self, model_name: str) -> bool:
        """
        Descargar modelo de la memoria

        Args:
            model_name: Nombre del modelo a descargar

        Returns:
            bool: True si se descargó exitosamente
        """
        with self._loading_lock:
            if model_name not in self._loaded_models:
                logger.warning(f"Modelo {model_name} no está cargado")
                return True

            try:
                model_info = self._loaded_models[model_name]

                # Limpiar instancia del modelo
                if "instance" in model_info:
                    del model_info["instance"]

                # Remover de modelos cargados
                del self._loaded_models[model_name]

                # Actualizar métricas
                self._model_stats["models_unloaded"] += 1

                logger.info(f"Modelo {model_name} descargado")
                return True

            except Exception as e:
                logger.error(f"Error descargando modelo {model_name}: {e}")
                return False

    def get_loaded_models(self) -> List[str]:
        """Obtener lista de modelos cargados"""
        return list(self._loaded_models.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Obtener información de un modelo cargado

        Args:
            model_name: Nombre del modelo

        Returns:
            Información del modelo o None
        """
        if model_name in self._loaded_models:
            info = self._loaded_models[model_name].copy()
            # No incluir la instancia en la respuesta
            info.pop("instance", None)
            return info
        return None

    def get_available_models(self) -> Dict:
        """Obtener modelos disponibles para cargar"""
        return self._available_models.copy()

    def get_model_instance(self, model_name: str) -> Optional[Any]:
        """
        Obtener instancia de modelo cargado

        Args:
            model_name: Nombre del modelo

        Returns:
            Instancia del modelo o None
        """
        if model_name in self._loaded_models:
            model_info = self._loaded_models[model_name]
            # Actualizar estadísticas de acceso
            model_info["access_count"] += 1
            model_info["last_access"] = time.time()

            return model_info["instance"]
        return None

    def get_stats(self) -> Dict:
        """Obtener estadísticas del gestor"""
        return {
            **self._model_stats,
            "loaded_models_count": len(self._loaded_models),
            "available_models_count": len(self._available_models),
            "cache_size_mb": self.cache_size,
        }

    async def health_check(self) -> Dict:
        """Verificar estado de salud del gestor"""
        return {
            "status": "healthy",
            "loaded_models": len(self._loaded_models),
            "available_models": len(self._available_models),
            "models_path_exists": self.models_path.exists(),
            "stats": self.get_stats(),
        }

    async def shutdown(self):
        """Cerrar gestor y limpiar recursos"""
        logger.info("Iniciando shutdown del ModelManager")

        # Descargar todos los modelos
        for model_name in list(self._loaded_models.keys()):
            await self.unload_model(model_name)

        # Limpiar caché
        self._model_cache.clear()

        logger.info("ModelManager shutdown completado")
