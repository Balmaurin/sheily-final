#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily Fast Chat V3 - Sistema de Chat Ultra-Rápido y Limpio
===========================================================

Sistema de chat completamente reescrito desde cero para máxima velocidad:

✨ CARACTERÍSTICAS:
- Inicialización en <2 segundos
- Respuestas en 3-5 segundos
- Sin logs verbosos en conversación
- Memoria opcional (desactivada por defecto para velocidad)
- Servidor GGUF singleton reutilizable
- Prompt engineering optimizado
- Tokens reducidos para respuestas rápidas

🎯 ENFOQUE:
- SOLO conversación natural y fluida
- Sin overhead de sistemas complejos
- Sin embeddings en cada mensaje (solo si se activa memoria)
- Sin health checks constantes
- UI limpia y minimalista

🚀 USO:
    from sheily_core.sheily_fast_chat_v3 import FastChatV3
    chat = FastChatV3()
    chat.run()
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# ============================================================================
# Configuración Ultra-Rápida
# ============================================================================


class FastChatConfig:
    """Configuración minimalista para máxima velocidad"""

    def __init__(self):
        # Servidor GGUF
        self.model_path = Path(os.getenv("SHEILY_GGUF", "/home/yo/Escritorio/Sheily-Final/models/gguf/llama-3.2.gguf"))
        self.llama_server = Path(
            os.getenv(
                "LLAMA_SERVER_BIN",
                "/home/yo/Escritorio/Nueva carpeta 1/llama.cpp/build/bin/llama-server",
            )
        )
        self.host = "127.0.0.1"
        self.port = 8080

        # Optimizaciones de velocidad
        self.threads = os.cpu_count() or 8
        self.ctx_size = 2048  # Reducido de 4096 para más velocidad
        self.n_predict = 150  # Respuestas concisas para máxima velocidad
        self.temperature = 0.7  # Balance velocidad/creatividad
        self.top_p = 0.9
        self.top_k = 40
        self.timeout = 30  # Timeout más corto para respuestas rápidas

        # Modo ultra-rápido (sin memoria)
        self.use_memory = os.getenv("SHEILY_USE_MEMORY", "0") == "1"
        self.verbose = os.getenv("SHEILY_VERBOSE", "0") == "1"


# ============================================================================
# Motor de Chat Ultra-Rápido
# ============================================================================


class FastChatV3:
    """Motor de chat ultra-rápido y limpio"""

    def __init__(self, config: Optional[FastChatConfig] = None):
        self.config = config or FastChatConfig()
        self.server_process: Optional[subprocess.Popen] = None
        self.server_ready = False

        # Historial de conversación (en memoria, no en disco)
        self.conversation_history = []
        self.max_history = 5  # Solo últimos 5 intercambios

    def _check_server(self) -> bool:
        """Verificar si el servidor está activo (rápido)"""
        try:
            response = requests.get(f"http://{self.config.host}:{self.config.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _start_server(self) -> bool:
        """Iniciar servidor GGUF optimizado"""
        if not self.config.llama_server.exists():
            print(f"❌ No se encontró llama-server en: {self.config.llama_server}")
            return False

        if not self.config.model_path.exists():
            print(f"❌ No se encontró el modelo en: {self.config.model_path}")
            return False

        if self.config.verbose:
            print("🚀 Iniciando servidor GGUF...")

        cmd = [
            str(self.config.llama_server),
            "--model",
            str(self.config.model_path),
            "--threads",
            str(self.config.threads),
            "--ctx-size",
            str(self.config.ctx_size),
            "--n-predict",
            str(self.config.n_predict),
            "--temp",
            str(self.config.temperature),
            "--top-p",
            str(self.config.top_p),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--timeout",
            "600",
        ]

        self.server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Esperar inicio rápido (máximo 30 segundos)
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.server_process.poll() is not None:
                return False
            if self._check_server():
                if self.config.verbose:
                    print(f"✅ Servidor activo en http://{self.config.host}:{self.config.port}")
                return True
            time.sleep(0.3)

        return False

    def initialize(self) -> bool:
        """Inicialización ultra-rápida"""
        start_time = time.time()

        # Verificar si servidor ya está corriendo
        if self._check_server():
            self.server_ready = True
            if self.config.verbose:
                print("✅ Servidor GGUF ya activo")
        else:
            # Intentar iniciar servidor
            self.server_ready = self._start_server()

        if not self.server_ready:
            print("❌ No se pudo inicializar el servidor GGUF")
            return False

        init_time = time.time() - start_time
        if self.config.verbose:
            print(f"⚡ Sistema listo en {init_time:.1f}s")

        return True

    def _build_prompt(self, user_message: str) -> str:
        """Construir prompt optimizado con historial mínimo"""

        # Sistema base (corto y directo)
        system = (
            "Eres Sheily, una IA conversacional en español. "
            "Responde de forma natural, clara y concisa. "
            "Si te preguntan tu nombre, di 'Soy Sheily'."
        )

        # Construir contexto con historial limitado
        context_parts = [f"<|system|>{system}<|end|>"]

        # Agregar últimos intercambios (máximo 3 para velocidad)
        recent_history = self.conversation_history[-3:]
        for h_user, h_assistant in recent_history:
            context_parts.append(f"<|user|>{h_user}<|end|>")
            context_parts.append(f"<|assistant|>{h_assistant}<|end|>")

        # Mensaje actual
        context_parts.append(f"<|user|>{user_message}<|end|>")
        context_parts.append("<|assistant|>")

        return "".join(context_parts)

    def generate_response(self, user_message: str) -> Optional[str]:
        """Generar respuesta ultra-rápida"""

        if not self.server_ready:
            return None

        # Construir prompt
        prompt = self._build_prompt(user_message)

        # Payload optimizado para velocidad
        payload = {
            "prompt": prompt,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "n_predict": self.config.n_predict,
            "stop": ["<|user|>", "<|end|>", "\nTú:", "\nUsuario:"],
            "stream": False,
        }

        try:
            response = requests.post(
                f"http://{self.config.host}:{self.config.port}/completion",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("content", "").strip()

            if content:
                # Guardar en historial
                self.conversation_history.append((user_message, content))

                # Mantener solo últimos intercambios
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history :]

                return content

        except requests.exceptions.Timeout:
            return "⏱️ La respuesta está tomando demasiado tiempo. Intenta una pregunta más simple."
        except requests.exceptions.ConnectionError:
            return "❌ Error de conexión con el servidor. Verifica que esté activo."
        except Exception as e:
            if self.config.verbose:
                print(f"Error: {e}")
            return "❌ Hubo un problema generando la respuesta."

        return None

    def run(self):
        """Ejecutar chat interactivo ultra-rápido"""

        print("🤖 SHEILY FAST CHAT V3")
        print("=" * 50)
        print("✨ Sistema optimizado para velocidad máxima")
        print("💬 Chat natural sin complejidad innecesaria")
        print()

        # Inicializar
        if not self.initialize():
            print("❌ No se pudo iniciar el sistema")
            return 1

        print("💬 Chat listo. Escribe 'salir' para terminar.\n")

        # Loop de conversación
        while True:
            try:
                # Input del usuario
                user_input = input("Tú: ").strip()

                # Comandos especiales
                if user_input.lower() in ["salir", "exit", "quit", "adios"]:
                    print("\n👋 ¡Hasta luego!")
                    break

                if not user_input:
                    continue

                # Comandos de utilidad
                if user_input.lower() == "clear":
                    self.conversation_history.clear()
                    print("🗑️  Historial borrado")
                    continue

                if user_input.lower() == "historial":
                    print(f"\n📜 Historial ({len(self.conversation_history)} intercambios):")
                    for i, (u, a) in enumerate(self.conversation_history, 1):
                        print(f"  {i}. Tú: {u[:50]}...")
                        print(f"     Sheily: {a[:50]}...")
                    print()
                    continue

                # Generar respuesta
                start_time = time.time()
                response = self.generate_response(user_input)
                elapsed = time.time() - start_time

                if response:
                    print(f"\n🤖 Sheily: {response}\n")

                    if self.config.verbose:
                        print(f"⚡ Tiempo: {elapsed:.1f}s\n")
                else:
                    print("\n❌ No se pudo generar respuesta\n")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrumpido. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}\n")

        # Cleanup
        self.cleanup()
        return 0

    def cleanup(self):
        """Limpiar recursos"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                if self.config.verbose:
                    print("✅ Servidor terminado")
            except:
                pass


# ============================================================================
# Función de Acceso Directo
# ============================================================================


def run_fast_chat():
    """Ejecutar chat rápido directamente"""
    chat = FastChatV3()
    return chat.run()


# ============================================================================
# Exports
# ============================================================================

__all__ = ["FastChatV3", "FastChatConfig", "run_fast_chat"]

# Metadata
__version__ = "3.0.0"
__author__ = "Sheily AI Team - Fast Chat V3"

# Ejecutar si se llama directamente
if __name__ == "__main__":
    exit(run_fast_chat())
