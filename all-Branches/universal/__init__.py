"""
Sheily Universal System - Sistema de aprendizaje continuo unificado
====================================================================

Un único adaptador LoRA que aprende de TODO el conocimiento,
sin importar el dominio o la fuente de datos.

Arquitectura:
- Corpus unificado global
- Sistema RAG universal
- Adaptador LoRA continuo
- Auto-integración de datasets
- Mejora permanente con cualquier dato
"""

__version__ = "1.0.0"
__author__ = "Sheily Universal Team"

from pathlib import Path

UNIVERSAL_ROOT = Path(__file__).parent
CORPUS_PATH = UNIVERSAL_ROOT / "corpus" / "unified"
INCOMING_PATH = UNIVERSAL_ROOT / "corpus" / "incoming"
ADAPTER_PATH = UNIVERSAL_ROOT / "adapters" / "universal_lora"
RAG_PATH = UNIVERSAL_ROOT / "rag"
SCRIPTS_PATH = UNIVERSAL_ROOT / "scripts"

__all__ = [
    "UNIVERSAL_ROOT",
    "CORPUS_PATH",
    "INCOMING_PATH",
    "ADAPTER_PATH",
    "RAG_PATH",
    "SCRIPTS_PATH"
]
