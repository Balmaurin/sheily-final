"""
Utilidades para importación segura de PyTorch
===============================================

Evita errores de importación cuando PyTorch no está completamente inicializado.
"""

import logging
import sys

logger = logging.getLogger(__name__)

# Flag para verificar si PyTorch está disponible
TORCH_AVAILABLE = False
torch = None
nn = None

try:
    import torch as _torch
    import torch.nn as _nn

    # Verificar que PyTorch está completamente inicializado
    try:
        _ = _torch.tensor([1.0])
        TORCH_AVAILABLE = True
        torch = _torch
        nn = _nn
        logger.info("✅ PyTorch importado correctamente")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch parcialmente disponible: {e}")
        TORCH_AVAILABLE = False
except ImportError as e:
    logger.warning(f"⚠️ PyTorch no disponible: {e}")
    TORCH_AVAILABLE = False


def require_torch(func):
    """Decorador para funciones que requieren PyTorch"""

    def wrapper(*args, **kwargs):
        if not TORCH_AVAILABLE:
            raise RuntimeError(f"PyTorch no está disponible. La función '{func.__name__}' no puede ejecutarse.")
        return func(*args, **kwargs)

    return wrapper


def get_torch():
    """Obtiene el módulo torch de forma segura"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible")
    return torch


def get_nn():
    """Obtiene el módulo torch.nn de forma segura"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible")
    return nn


class TorchStub:
    """Stub para PyTorch cuando no está disponible"""

    def __getattr__(self, name):
        raise RuntimeError(f"PyTorch no está disponible. No se puede acceder a torch.{name}")


class NNStub:
    """Stub para torch.nn cuando no está disponible"""

    def __getattr__(self, name):
        raise RuntimeError(f"PyTorch no está disponible. No se puede acceder a torch.nn.{name}")


# Si PyTorch no está disponible, usar stubs
if not TORCH_AVAILABLE:
    torch = TorchStub()
    nn = NNStub()
