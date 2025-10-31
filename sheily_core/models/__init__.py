"""
SISTEMA DE MODELOS AVANZADO SHEILY - Módulo Principal

Este módulo contiene la gestión completa del ciclo de vida de modelos de IA:

COMPONENTES PRINCIPALES:
- Gestión avanzada de modelos y especialización
- Manejo de ramas académicas especializadas
- Adaptadores LoRA y fine-tuning
- Puente entre sistemas LLM
- Cargador especializado de ramas académicas

IMPORTS PRINCIPALES:
- ModelManager: Gestión completa del ciclo de vida de modelos
- BranchManager: Gestión avanzada de ramas académicas
- BranchSelector: Selección inteligente de ramas por contexto
- SpecializationManager: Especialización avanzada de modelos

EJEMPLO DE USO:
    from sheily_core.models import ModelManager, BranchManager

    # Crear gestor de modelos
    model_mgr = ModelManager()
    model = model_mgr.load_model("matematica")

    # Gestionar ramas académicas
    branch_mgr = BranchManager()
    branch = branch_mgr.get_branch_config("fisica")
"""

# ==============================================================================
# IMPORTS PRINCIPALES DEL MÓDULO MODELS
# ==============================================================================

__all__ = [
    "ModelManager",
    "BranchManager",
    "BranchSelector",
    "SpecializationManager",
    "AdapterManager",
    "SheilyLLMBridge",
]

# ==============================================================================
# CONFIGURACIÓN DEL MÓDULO
# ==============================================================================

__version__ = "2.0.0"
__author__ = "Sheily AI Team"
__description__ = "Sistema completo de gestión de modelos de IA con especialización académica avanzada"

# ==============================================================================
# IMPORTS CONDICIONALES PARA MEJOR COMPATIBILIDAD
# ==============================================================================

try:
    # Imports principales desde componentes internos
    from .adapters import AdapterManager
    from .branch_manager import BranchManager
    from .branch_selector import BranchSelector
    from .model_manager import ModelManager
    from .sheily_llm_bridge import SheilyLLMBridge
    from .specialization import SpecializationManager
    from .specialized_branches_loader import SpecializedBranchesLoader
except ImportError as e:
    # Fallback para desarrollo o imports parciales
    print(f"Advertencia: Algunos componentes de models no disponibles: {e}")

    # Clases básicas como fallback
    class ModelManager:
        """Gestión de modelos (fallback)"""

        pass

    class BranchManager:
        """Gestión de ramas académicas (fallback)"""

        pass

    class BranchSelector:
        """Selección inteligente de ramas (fallback)"""

        pass

    class SpecializationManager:
        """Especialización de modelos (fallback)"""

        pass


# ==============================================================================
# FUNCIONES DE UTILIDAD DEL MÓDULO
# ==============================================================================


def create_model_manager():
    """Crear gestor principal de modelos"""
    return ModelManager()


def create_branch_manager():
    """Crear gestor de ramas académicas"""
    return BranchManager()


def create_adapter_manager():
    """Crear gestor de adaptadores LoRA"""
    return AdapterManager()


# ==============================================================================
# INICIALIZACIÓN DEL MÓDULO
# ==============================================================================


def initialize_models_system():
    """Inicializar sistema completo de modelos"""
    print("🚀 Inicializando sistema de modelos Sheily...")

    try:
        # Inicializar componentes principales
        model_mgr = create_model_manager()
        branch_mgr = create_branch_manager()
        adapter_mgr = create_adapter_manager()

        print("✅ Sistema de modelos inicializado correctamente")
        return True

    except Exception as e:
        print(f"❌ Error inicializando sistema de modelos: {e}")
        return False


# ==============================================================================
# CONFIGURACIÓN AUTOMÁTICA
# ==============================================================================

# Inicializar sistema automáticamente si se ejecuta directamente
if __name__ == "__main__":
    initialize_models_system()
