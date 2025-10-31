# Auto-loaded guard - Zero Dependency Version
# Este archivo se ejecuta automáticamente al iniciar Python
# Bloquea acceso a dependencias externas en modo producción

import json
import os
import sys
from pathlib import Path

# Verificar si estamos en modo zero-dependency
ZERO_DEPS_MODE = os.getenv("SHEILY_ZERO_DEPS", "1") == "1"


def _log_security_event(event_type: str, details: str):
    """Log de eventos de seguridad usando solo stdlib"""
    timestamp = __import__("datetime").datetime.now().isoformat()
    event = f"[{timestamp}] SheilyGuard {event_type}: {details}"

    # Escribir a stderr para debugging
    print(event, file=sys.stderr)

    # Opcional: escribir a archivo log
    try:
        log_path = Path.cwd() / "logs" / "security.log"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(event + "\n")
    except Exception:
        pass  # Fallar silenciosamente si no se puede escribir log


def _block_dangerous_imports():
    """Bloquea importaciones peligrosas en modo zero-dependency"""
    if not ZERO_DEPS_MODE:
        return

    dangerous_modules = [
        "torch",
        "tensorflow",
        "transformers",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "gradio",
        "streamlit",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "requests",
        "httpx",
        "aiohttp",
        "pydantic",
        "yaml",
        "toml",
    ]

    # Interceptar __import__ para bloquear módulos peligrosos
    original_import = __builtins__.__import__

    def secure_import(name, *args, **kwargs):
        # Verificar si el módulo está en la lista de bloqueados
        root_module = name.split(".")[0]
        if root_module in dangerous_modules:
            _log_security_event("BLOCKED_IMPORT", f"Bloqueado import de '{name}'")
            raise ImportError(f"Módulo '{name}' bloqueado por SheilyGuard en modo zero-dependency")

        return original_import(name, *args, **kwargs)

    # Reemplazar la función de import global
    __builtins__.__import__ = secure_import
    _log_security_event("GUARD_ACTIVE", "Sistema de bloqueo de imports activado")


# Configurar entorno offline por defecto
if os.getenv("SHEILY_OFFLINE_DEFAULT", "1") == "1":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Deshabilitar GPU por defecto

# Activar protecciones si estamos en modo zero-dependency
if ZERO_DEPS_MODE:
    _block_dangerous_imports()
    _log_security_event("INIT", "SheilyGuard inicializado en modo zero-dependency")
