#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily LLM Engine - Real Model Integration
=========================================

Integración real con modelos GGUF usando llama.cpp para reemplazar los fallbacks.
Soporte para Llama 3.2 1B Q4 y otros modelos compatibles.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sheily_core.config import get_config
from sheily_core.logger import get_logger


@dataclass
class LLMConfig:
    """Configuración del motor LLM"""

    model_path: str
    llama_binary_path: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    context_length: int = 2048
    threads: int = 4
    batch_size: int = 512
    verbose: bool = False


class RealLLMEngine:
    """
    Motor LLM real usando llama.cpp para modelos GGUF

    Reemplaza los sistemas fallback con integración real de modelos.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = get_logger("llm_engine")
        self.config = config or self._create_default_config()
        self.model_loaded = False
        self.model_info = {}

        # Validate configuration
        self._validate_config()

        # Try to initialize model
        self._initialize_model()

    def _create_default_config(self) -> LLMConfig:
        """Crear configuración por defecto"""
        # Try to find model and binary
        model_path = self._find_model()
        binary_path = self._find_llama_binary()

        return LLMConfig(
            model_path=model_path,
            llama_binary_path=binary_path,
            max_tokens=256,
            temperature=0.7,
            context_length=2048,
            threads=min(4, os.cpu_count() or 4),
        )

    def _find_model(self) -> str:
        """Buscar modelo GGUF disponible"""
        # Common model locations
        model_locations = [
            "/home/yo/Sheily-Final/models/gguf/llama-3.2.gguf",
            "./models/gguf/llama-3.2.gguf",
            "./models/llama-3.2.gguf",
            os.path.expanduser("~/models/llama-3.2.gguf"),
        ]

        for location in model_locations:
            if Path(location).exists():
                self.logger.info(f"Found model at: {location}")
                return location

        self.logger.warning("No GGUF model found in standard locations")
        return ""

    def _find_llama_binary(self) -> Optional[str]:
        """Buscar binario llama-cli o llama-cpp"""
        # First check our compiled version
        project_binary = "/home/yo/Sheily-Final/tools/llama.cpp/build/bin/llama-cli"
        if Path(project_binary).exists():
            self.logger.info(f"Found compiled llama-cli: {project_binary}")
            return project_binary

        binary_names = ["llama-cli", "llama-cpp", "llama.cpp"]

        # Check in common locations
        search_paths = [
            "/usr/local/bin",
            "/usr/bin",
            "./tools/llama.cpp/build/bin",
            "./tools/llama.cpp",
            os.path.expanduser("~/bin"),
            "./bin",
        ]

        # Also check PATH
        for binary in binary_names:
            try:
                result = subprocess.run(["which", binary], capture_output=True, text=True)
                if result.returncode == 0:
                    binary_path = result.stdout.strip()
                    self.logger.info(f"Found llama binary: {binary_path}")
                    return binary_path
            except Exception:
                pass

        # Check specific locations
        for path in search_paths:
            for binary in binary_names:
                binary_path = Path(path) / binary
                if binary_path.exists() and binary_path.is_file():
                    self.logger.info(f"Found llama binary: {binary_path}")
                    return str(binary_path)

        self.logger.warning("llama-cli binary not found. Will use Python fallback.")
        return None

    def _validate_config(self):
        """Validar configuración"""
        if not self.config.model_path:
            raise ValueError("Model path is required")

        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        # Check model size
        model_size = Path(self.config.model_path).stat().st_size / (1024 * 1024)  # MB
        self.logger.info(f"Model size: {model_size:.1f} MB")

        if model_size < 100:
            self.logger.warning("Model file seems too small, might be corrupted")

        self.logger.info("LLM configuration validated successfully")

    def _initialize_model(self):
        """Inicializar modelo"""
        try:
            self.logger.info(f"Initializing model: {self.config.model_path}")

            # Get model info
            self.model_info = {
                "path": self.config.model_path,
                "size_mb": Path(self.config.model_path).stat().st_size / (1024 * 1024),
                "binary_available": self.config.llama_binary_path is not None,
                "config": {
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "context_length": self.config.context_length,
                },
            }

            # Model is ready if we have binary and model file
            if self.config.llama_binary_path:
                self.model_loaded = True
                self.logger.info("Model initialized successfully (binary available)")

                # Optional: Test basic functionality (can be slow)
                # test_result = self._test_model_basic()
                # if not test_result:
                #     self.logger.warning("Model basic test failed, but model should still work")
            else:
                self.logger.info("Model initialized (binary test skipped)")
                self.model_loaded = True

        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            self.model_loaded = False

    def _test_model_basic(self) -> bool:
        """Test básico del modelo"""
        try:
            if not self.config.llama_binary_path:
                return False

            # Simple test with minimal output
            cmd = [
                self.config.llama_binary_path,
                "-m",
                self.config.model_path,
                "-p",
                "Test:",
                "-n",
                "5",
                "--temp",
                "0.1",
                "--log-disable",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Model test failed: {e}")
            return False

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generar respuesta usando el modelo real
        """
        if not self.model_loaded:
            raise RuntimeError("Modelo no cargado. Initialize primero.")

        start_time = time.time()

        # Merge kwargs with config
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        try:
            if self.config.llama_binary_path:
                response = self._generate_with_binary(prompt, max_tokens, temperature)
            else:
                raise RuntimeError("Binary llama-cli no disponible")

            processing_time = time.time() - start_time

            return {
                "response": response,
                "processing_time": processing_time,
                "model_used": "llama-3.2-gguf",
                "model_info": self.model_info,
                "success": True,
                "method": "llama_binary" if self.config.llama_binary_path else "fallback",
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Generation failed: {e}")

            return {
                "response": self._raise_generation_error(e),
                "processing_time": processing_time,
                "model_used": "fallback",
                "success": False,
                "error": str(e),
            }

    def _generate_with_binary(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generar usando llama-cli binary"""

        # Prepare enhanced prompt with Sheily personality
        enhanced_prompt = f"""Eres Sheily, una asistente de IA inteligente y útil. Respondes de manera clara, precisa y amigable.

Usuario: {prompt}
Sheily:"""

        cmd = [
            self.config.llama_binary_path,
            "-m",
            self.config.model_path,
            "-p",
            enhanced_prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            str(self.config.top_p),
            "--top-k",
            str(self.config.top_k),
            "-c",
            str(self.config.context_length),
            "-t",
            str(self.config.threads),
            "--no-display-prompt",
        ]

        if not self.config.verbose:
            cmd.extend(["--log-disable"])

        self.logger.debug(f"Executing: {' '.join(cmd[:3])} ...")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # 1 minute timeout

        if result.returncode != 0:
            self.logger.error(f"llama-cli failed: {result.stderr}")
            raise RuntimeError(f"Error en binary: {result.stderr}")

        # Clean up response
        response = result.stdout.strip()

        # Remove any remaining prompt artifacts
        if "Usuario:" in response:
            response = response.split("Usuario:")[-1].strip()
        if "Sheily:" in response:
            response = response.replace("Sheily:", "").strip()

        # Basic cleanup
        response = response.replace("\\n", "\n").strip()

        if not response:
            raise RuntimeError("Respuesta vacía del modelo")
        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.config.model_path,
            "binary_path": self.config.llama_binary_path,
            "model_info": self.model_info,
            "config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "context_length": self.config.context_length,
                "threads": self.config.threads,
            },
        }

    def is_available(self) -> bool:
        """Verificar si el motor está disponible"""
        return self.model_loaded


# Factory function
def create_real_llm_engine(config: Optional[LLMConfig] = None) -> RealLLMEngine:
    """Crear instancia del motor LLM real"""
    return RealLLMEngine(config)


# Test function
def test_llm_engine():
    """Test del motor LLM"""
    print("🧪 Testing Real LLM Engine")
    print("=" * 40)

    try:
        engine = create_real_llm_engine()

        print(f"Model loaded: {engine.model_loaded}")
        print(f"Model info: {engine.get_model_info()}")

        if engine.is_available():
            test_prompts = ["Hola, ¿cómo estás?", "¿Qué es Python?", "Explica la ley de Ohm"]

            for prompt in test_prompts:
                print(f"\n📝 Prompt: {prompt}")
                result = engine.generate_response(prompt)
                print(f"   Response: {result['response']}")
                print(f"   Time: {result['processing_time']:.2f}s")
                print(f"   Method: {result.get('method', 'unknown')}")
        else:
            print("❌ Engine not available")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_llm_engine()
