"""
Sistemas Unificados de Sheily
============================

Arquitectura unificada que integra todos los sistemas de IA, memoria,
seguridad y blockchain en una plataforma coherente.
"""

from .unified_master_system import UnifiedMasterSystem
from .unified_system_core import UnifiedSystemCore, get_unified_system

__all__ = [
    "get_unified_system",
    "UnifiedSystemCore",
    "UnifiedMasterSystem",
]
