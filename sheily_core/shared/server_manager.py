#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SERVER MANAGER - GESTIÓN UNIFICADA DEL SERVIDOR GGUF
==================================================

Módulo compartido que elimina duplicaciones en la gestión del servidor llama.cpp.
Proporciona una interfaz única y optimizada para:

- Verificación del estado del servidor
- Inicio y parada del servidor
- Configuración optimizada
- Manejo de errores y recuperación
- Monitoreo de salud del servidor
"""

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import requests
from sheily_config import get_config


class ServerManager:
    """Gestor unificado del servidor GGUF"""

    def __init__(self):
        self.config = get_config()
        self.server_process = None
        self.server_status = "stopped"
        self.start_time = None
        self.health_check_thread = None
        self._lock = threading.Lock()

    def check_server_running(self) -> bool:
        """Verificar si el servidor está corriendo - implementación unificada"""
        try:
            response = requests.get(f"http://{self.config.model.host}:{self.config.model.port}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def start_server(self) -> bool:
        """Iniciar servidor GGUF optimizado - implementación unificada"""
        with self._lock:
            # Verificar si ya está corriendo
            if self.check_server_running():
                print("✅ Servidor ya activo, reutilizando...")
                self.server_status = "running"
                return True

            # Validar archivos necesarios
            model_path = Path(self.config.model.path)
            server_path = Path(self.config.model.llama_server_bin)

            if not server_path.exists():
                print(f"❌ No se encontró llama-server en: {server_path}")
                return False

            if not model_path.exists():
                print(f"❌ No se encontró el modelo en: {model_path}")
                return False

            print("🚀 Iniciando servidor GGUF optimizado...")

            # Construir comando optimizado
            cmd = self._build_server_command()

            try:
                # Iniciar proceso en segundo plano
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,  # Crear nuevo grupo de procesos
                )

                # Esperar a que esté listo
                if self._wait_for_server():
                    self.server_status = "running"
                    self.start_time = time.time()
                    self._start_health_monitoring()
                    print(f"✅ Servidor activo en http://{self.config.model.host}:{self.config.model.port}")
                    return True
                else:
                    print("❌ Timeout iniciando servidor")
                    self._cleanup_process()
                    return False

            except Exception as e:
                print(f"❌ Error iniciando servidor: {e}")
                self._cleanup_process()
                return False

    def _build_server_command(self) -> list:
        """Construir comando optimizado del servidor - versión compatible"""
        cmd = [
            str(self.config.model.llama_server_bin),
            "--model",
            self.config.model.path,
            "--threads",
            str(self.config.model.threads),
            "--ctx-size",
            str(self.config.model.ctx_size),
            "--n-predict",
            str(self.config.model.n_predict),
            "--temp",
            str(self.config.model.temperature),
            "--top-p",
            str(self.config.model.top_p),
            "--host",
            self.config.model.host,
            "--port",
            str(self.config.model.port),
            "--timeout",
            str(self.config.model.timeout),
        ]

        # Añadir parámetros compatibles según modo
        if self.config.mode == "ultra-fast":
            cmd.extend(["--repeat-penalty", "1.1", "--top-k", "40"])
        elif self.config.mode == "neuro-advanced":
            cmd.extend(["--repeat-penalty", "1.1", "--top-k", "50"])

        # Solo añadir embedding si está disponible en esta versión
        try:
            # Probar si el servidor soporta embeddings
            test_cmd = [str(self.config.model.llama_server_bin), "--help"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if "--embedding" in result.stdout:
                cmd.append("--embedding")
            if "--pooling" in result.stdout:
                cmd.extend(["--pooling", "mean"])
        except:
            # Si hay error, continuar sin parámetros avanzados
            pass

        return cmd

    def _wait_for_server(self, timeout: int = 120) -> bool:
        """Esperar a que el servidor esté listo"""
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Verificar si el proceso sigue vivo
            if self.server_process.poll() is not None:
                print("❌ Servidor terminó inesperadamente")
                return False

            # Verificar conectividad
            if self.check_server_running():
                return True

            time.sleep(0.5)

        return False

    def _cleanup_process(self):
        """Limpiar proceso del servidor"""
        if self.server_process:
            try:
                # Terminar grupo de procesos completo
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                time.sleep(2)

                # Forzar terminación si es necesario
                if self.server_process.poll() is None:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)

            except (OSError, ProcessLookupError):
                pass  # Proceso ya terminado

            self.server_process = None
            self.server_status = "stopped"

    def stop_server(self) -> bool:
        """Parar servidor GGUF"""
        with self._lock:
            if self.server_status == "stopped":
                return True

            print("🛑 Parando servidor GGUF...")
            self._cleanup_process()
            self._stop_health_monitoring()
            print("✅ Servidor parado")
            return True

    def _start_health_monitoring(self):
        """Iniciar monitoreo de salud del servidor"""
        if self.health_check_thread and self.health_check_thread.is_alive():
            return

        def health_monitor():
            while self.server_status == "running":
                try:
                    if not self.check_server_running():
                        print("⚠️ Servidor no responde, intentando reiniciar...")
                        self._restart_server()
                        break
                    time.sleep(self.config.performance.health_check_interval)
                except Exception as e:
                    print(f"Error en monitoreo de salud: {e}")
                    time.sleep(self.config.performance.health_check_interval)

        self.health_check_thread = threading.Thread(target=health_monitor, daemon=True)
        self.health_check_thread.start()

    def _stop_health_monitoring(self):
        """Parar monitoreo de salud"""
        if self.health_check_thread:
            # El thread daemon se terminará automáticamente
            self.health_check_thread = None

    def _restart_server(self) -> bool:
        """Reiniciar servidor automáticamente"""
        print("🔄 Reiniciando servidor...")
        self.stop_server()
        time.sleep(2)
        return self.start_server()

    def get_server_info(self) -> Dict[str, Any]:
        """Obtener información detallada del servidor"""
        uptime = 0
        if self.start_time:
            uptime = time.time() - self.start_time

        # Información del proceso si está corriendo
        process_info = None
        if self.server_process:
            try:
                process = psutil.Process(self.server_process.pid)
                process_info = {
                    "pid": process.pid,
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "create_time": process.create_time(),
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_info = None

        return {
            "status": self.server_status,
            "uptime_seconds": uptime,
            "model_path": self.config.model.path,
            "host": self.config.model.host,
            "port": self.config.model.port,
            "threads": self.config.model.threads,
            "ctx_size": self.config.model.ctx_size,
            "n_predict": self.config.model.n_predict,
            "process_info": process_info,
            "health_check_enabled": self.health_check_thread is not None,
        }

    def test_embedding_endpoint(self) -> bool:
        """Probar endpoint de embeddings"""
        try:
            response = requests.post(
                f"http://{self.config.model.host}:{self.config.model.port}/v1/embeddings",
                json={"input": "test", "model": "sheily"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                return len(embedding) == self.config.memory.embedding_dim

            return False
        except Exception:
            return False

    def get_gguf_embedding(self, text: str) -> Optional[list]:
        """Obtener embedding GGUF optimizado"""
        try:
            response = requests.post(
                f"http://{self.config.model.host}:{self.config.model.port}/v1/embeddings",
                json={"input": text, "model": "sheily"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["embedding"]

            return None
        except Exception:
            return None


# Instancia global del gestor de servidor
_server_manager = ServerManager()


def get_server_manager() -> ServerManager:
    """Obtener instancia global del gestor de servidor"""
    return _server_manager


def check_server_running() -> bool:
    """Función de conveniencia para verificar servidor"""
    return _server_manager.check_server_running()


def start_server() -> bool:
    """Función de conveniencia para iniciar servidor"""
    return _server_manager.start_server()


def stop_server() -> bool:
    """Función de conveniencia para parar servidor"""
    return _server_manager.stop_server()


def get_server_info() -> Dict[str, Any]:
    """Función de conveniencia para obtener información del servidor"""
    return _server_manager.get_server_info()


def get_gguf_embedding(text: str) -> Optional[list]:
    """Función de conveniencia para obtener embeddings"""
    return _server_manager.get_gguf_embedding(text)


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Probando Server Manager...")

    # Verificar estado inicial
    print(f"Estado inicial: {get_server_info()['status']}")

    # Iniciar servidor
    if start_server():
        print("✅ Servidor iniciado exitosamente")

        # Probar embeddings
        embedding = get_gguf_embedding("test de embeddings")
        if embedding:
            print(f"✅ Embeddings funcionales: {len(embedding)} dimensiones")
        else:
            print("❌ Error en embeddings")

        # Parar servidor
        stop_server()
        print("✅ Servidor parado")
    else:
        print("❌ Error iniciando servidor")
