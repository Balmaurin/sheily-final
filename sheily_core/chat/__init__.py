"""
SISTEMA DE CHAT AVANZADO SHEILY - Módulo Principal

Este módulo contiene el sistema completo de conversación inteligente:

COMPONENTES PRINCIPALES:
- Motor de chat funcional con detección de ramas
- Sistema de conversación unificado
- Chat ultra-rápido optimizado
- Adaptadores de memoria conversacional
- Integración con modelos GGUF

IMPORTS PRINCIPALES:
- ChatEngine: Motor principal de conversación funcional
- UnifiedChatSystem: Sistema unificado de chat
- FastChatV3: Chat ultra-rápido optimizado
- SheilyChatMemoryAdapter: Adaptador de memoria conversacional

EJEMPLO DE USO:
    from sheily_core.chat import ChatEngine, UnifiedChatSystem

    # Crear motor de chat avanzado
    chat_engine = ChatEngine()
    response = chat_engine.process_query("¿Qué es la física cuántica?")

    # Usar sistema unificado
    chat_system = UnifiedChatSystem()
    conversation = chat_system.start_conversation()
"""

# ==============================================================================
# IMPORTS PRINCIPALES DEL MÓDULO CHAT
# ==============================================================================

__all__ = [
    "ChatEngine",
    "UnifiedChatSystem",
    "FastChatV3",
    "SheilyChatMemoryAdapter",
    "ChatContext",
    "ChatMessage",
    "ChatResponse",
]

# ==============================================================================
# CONFIGURACIÓN DEL MÓDULO
# ==============================================================================

__version__ = "2.0.0"
__author__ = "Sheily AI Team"
__description__ = (
    "Sistema completo de conversación inteligente con detección automática de ramas académicas"
)

# ==============================================================================
# IMPORTS CONDICIONALES PARA MEJOR COMPATIBILIDAD
# ==============================================================================

from .chat_engine import ChatContext, ChatEngine, ChatMessage, ChatResponse
from .sheily_chat_memory_adapter import SheilyChatMemoryAdapter
from .sheily_fast_chat_v3 import FastChatV3
from .unified_chat_system import UnifiedChatSystem


# ==============================================================================
# FUNCIONES DE UTILIDAD DEL MÓDULO
# ==============================================================================


def create_chat_engine(config_path: str = None):
    """Crear motor de chat con configuración opcional"""
    return ChatEngine()


def create_unified_chat():
    """Crear sistema unificado de conversación"""
    return UnifiedChatSystem()


def create_fast_chat():
    """Crear chat ultra-rápido optimizado"""
    return FastChatV3()


# ==============================================================================
# INICIALIZACIÓN DEL MÓDULO
# ==============================================================================


def initialize_chat_system():
    """Inicializar sistema completo de chat"""
    print("Inicializando sistema de chat Sheily...")

    try:
        # Inicializar componentes principales
        chat_engine = create_chat_engine()
        unified_chat = create_unified_chat()
        fast_chat = create_fast_chat()

        print("Sistema de chat inicializado correctamente")
        return True

    except Exception as e:
        print(f"Error inicializando sistema de chat: {e}")
        return False


# ==============================================================================
# CONFIGURACIÓN AUTOMÁTICA
# ==============================================================================

# Inicializar sistema automáticamente si se ejecuta directamente
if __name__ == "__main__":
    initialize_chat_system()
