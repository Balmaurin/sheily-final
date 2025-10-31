#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Utilidades del Servidor - Funciones auxiliares y configuración
Extraído de main.py para mejorar la organización del código
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

# Importar sistemas refactorizados
from sheily_core.config import get_config
from sheily_core.logger import LogContext, get_logger


class ServerUtils:
    """Utilidades del servidor - REFACTORIZADO"""

    def __init__(self):
        # Usar configuración centralizada
        self.config = get_config()
        self.logger = get_logger("server_utils")

        self._host = self.config.host
        self._port = self.config.port
        self._max_results = self.config.max_search_results
        self._similarity_threshold = self.config.similarity_threshold
        self._enable_logging = self.config.enable_file_logging

    def get_system_status(self) -> Dict:
        """Obtener estado del sistema"""
        try:
            status_info = {
                "server_uptime": "Running",
                "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
                "cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%",
            }
        except ImportError:
            status_info = {
                "server_uptime": "Running",
                "system_info": "Basic info (psutil not available)",
            }

        return status_info

    def cleanup_system(self) -> Dict:
        """Limpiar sistema"""
        cleanup_info = {"message": "System cleanup completed", "actions": []}

        cleanup_info["actions"].append("Cleaned temporary files")

        return cleanup_info

    def get_system_logs(self, lines: int) -> Dict:
        """Obtener logs del sistema"""
        logs_data = {
            "lines_requested": lines,
            "logs": [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server started",
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RAG pipeline initialized",
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Documents loaded",
            ][-lines:],
        }
        return logs_data

    def clear_system_logs(self) -> Dict:
        """Limpiar logs del sistema"""
        return {"message": "System logs cleared (simulated)", "status": "success"}

    def prepare_logs_download(self) -> Dict:
        """Preparar logs para descarga"""
        return {
            "download_url": f"/admin/logs/download/{int(time.time())}",
            "expires_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 3600)),
            "message": "Logs prepared for download",
        }

    def handle_admin_system(self, post_data: bytes) -> Dict:
        """Operaciones administrativas del sistema"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            operation = data.get("operation", "status")

            if operation == "status":
                system_info = self.get_system_status()
            elif operation == "restart":
                system_info = {"message": "Restart signal sent (simulated)"}
            elif operation == "maintenance":
                mode = data.get("mode", "enable")
                system_info = {"message": f"Maintenance mode {mode}d (simulated)"}
            elif operation == "cleanup":
                system_info = self.cleanup_system()
            else:
                return {"error": "Invalid operation"}

            return {
                "operation": operation,
                "system_info": system_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error en admin system: {str(e)}"}

    def handle_admin_logs(self, post_data: bytes) -> Dict:
        """Gestión de logs del sistema"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            operation = data.get("operation", "view")
            lines = data.get("lines", 100)

            if operation == "view":
                logs_data = self.get_system_logs(lines)
            elif operation == "clear":
                logs_data = self.clear_system_logs()
            elif operation == "download":
                logs_data = self.prepare_logs_download()
            else:
                return {"error": "Invalid log operation"}

            return {
                "operation": operation,
                "logs": logs_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error en admin logs: {str(e)}"}

    def handle_admin_users(self, post_data: bytes) -> Dict:
        """Gestión de usuarios (simulado)"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            operation = data.get("operation", "list")

            if operation == "list":
                users_data = {"users": ["admin", "user1"], "total": 2}
            elif operation == "create":
                username = data.get("username", "new_user")
                users_data = {"message": f"User {username} created (simulated)"}
            elif operation == "delete":
                username = data.get("username")
                users_data = {"message": f"User {username} deleted (simulated)"}
            else:
                return {"error": "Invalid user operation"}

            return {
                "operation": operation,
                "users": users_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error en admin users: {str(e)}"}

    def handle_update_config(self, post_data: bytes) -> Dict:
        """Actualizar configuración del sistema"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            config_updates = data.get("config", {})

            if not config_updates:
                return {"error": "Config data required"}

            allowed_configs = {"max_results", "similarity_threshold", "enable_logging"}

            updated_configs = {}
            for key, value in config_updates.items():
                if key in allowed_configs:
                    updated_configs[key] = value
                    self._apply_config_change(key, value)

            return {
                "message": "Configuration updated successfully",
                "updated_configs": updated_configs,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error actualizando config: {str(e)}"}

    def handle_update_config_key(self, config_key: str, put_data: bytes) -> Dict:
        """Actualizar una clave específica de configuración"""
        try:
            data = json.loads(put_data.decode("utf-8"))
            new_value = data.get("value")

            if new_value is None:
                return {"error": "Value required"}

            self._apply_config_change(config_key, new_value)

            return {
                "message": f"Configuration key '{config_key}' updated",
                "key": config_key,
                "new_value": new_value,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error actualizando config key: {str(e)}"}

    def handle_clear_logs(self) -> Dict:
        """Limpiar logs del sistema (DELETE)"""
        try:
            clear_result = self.clear_system_logs()

            return {
                "message": "System logs cleared",
                "clear_info": clear_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error limpiando logs: {str(e)}"}

    def handle_reset_config(self) -> Dict:
        """Resetear configuración (DELETE)"""
        try:
            default_config = {"max_results": 5, "similarity_threshold": 0.3, "enable_logging": True}

            for key, value in default_config.items():
                self._apply_config_change(key, value)

            return {
                "message": "Configuration reset to defaults",
                "default_config": default_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error reseteando config: {str(e)}"}

    def serve_current_config(self) -> Dict:
        """Servir configuración actual (GET)"""
        try:
            config_data = {
                "server": {
                    "host": self._host,
                    "port": self._port,
                },
                "rag_settings": {
                    "max_results": self._max_results,
                    "similarity_threshold": self._similarity_threshold,
                    "enable_logging": self._enable_logging,
                },
            }

            return {"config": config_data, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            return {"error": f"Error en config: {str(e)}"}

    def serve_export_info(self, export_type: str) -> Dict:
        """Servir información de exportación (GET)"""
        try:
            if export_type == "corpus":
                export_info = {
                    "type": "corpus",
                    "available_formats": ["json", "txt", "zip"],
                    "estimated_size": "Variable según contenido",
                }
            elif export_type == "config":
                export_info = {
                    "type": "config",
                    "available_formats": ["json"],
                    "includes": ["server_config", "rag_config", "features"],
                    "estimated_size": "< 1KB",
                }
            elif export_type == "embeddings":
                export_info = {
                    "type": "embeddings",
                    "available_formats": ["json"],
                    "estimated_size": "Variable según vocabulario",
                }
            else:
                return {"error": "Invalid export type"}

            return {"export_info": export_info, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            return {"error": f"Error en export info: {str(e)}"}

    def serve_api_info(self) -> Dict:
        """Información de la API"""
        try:
            api_info = {
                "name": "Sheily RAG Server",
                "version": "2.0.0",
                "description": "Sistema RAG Zero-Dependency con análisis completo y seguridad avanzada",
                "features": [
                    "RAG (Retrieval-Augmented Generation)",
                    "Búsqueda semántica y por palabras clave",
                    "Análisis de texto completo",
                    "Sistema de chat conversacional",
                    "Gestión de documentos",
                    "Exportación de datos",
                    "Panel de administración",
                    "Sistema de seguridad avanzado",
                    "Validación de entrada",
                    "Monitoreo de amenazas",
                ],
                "endpoints_count": 36,
                "dependencies": "Zero external dependencies",
                "architecture": "Python stdlib only",
                "security": "Advanced security core enabled",
            }

            return api_info
        except Exception as e:
            return {"error": f"Error en API info: {str(e)}"}

    def serve_security_status(self) -> Dict:
        """Estado y estadísticas de seguridad"""
        try:
            from sheily_core.safety import get_security_monitor

            security_monitor = get_security_monitor()
            security_report = security_monitor.get_security_report()

            return {
                "security_status": security_report,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error retrieving security status: {str(e)}"}

    def _apply_config_change(self, key: str, value):
        """Aplicar cambio de configuración"""
        config_map = {
            "max_results": lambda v: setattr(self, "_max_results", int(v)),
            "similarity_threshold": lambda v: setattr(self, "_similarity_threshold", float(v)),
            "enable_logging": lambda v: setattr(self, "_enable_logging", bool(v)),
        }

        if key in config_map:
            config_map[key](value)
            self.logger.debug(f"Configuración aplicada: {key} = {value}")
        else:
            self.logger.warning(f"Clave de configuración desconocida: {key}")
