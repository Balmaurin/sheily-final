"""
Sheily Integration - Integraciones Externas

Este paquete maneja integraciones con servicios externos:
 - Monitoreo y métricas (Langfuse, Trulens)
 - Evaluación de modelos (Giskard, Deepeval)
 - Análisis de calidad y seguridad
 - Gestión central de integración (IntegrationManager)

Además, incluye clientes para interactuar con modelos de lenguaje
utilizando distintos backends:

* ``ollama_client``: Cliente HTTP para interactuar con modelos servidos
  por Ollama (por ejemplo Llama 3.2), siguiendo el endpoint
  ``/api/generate``【485831448119966†L438-L451】.
* ``llama_cpp_client``: Cliente para cargar y ejecutar modelos
  localmente mediante ``llama.cpp``. Estos modelos pueden
  descargarse y ejecutarse con ``ollama run llama3.2``【890854066034281†L13-L33】.
* ``integration_manager``: Gestor central para enrutamiento inteligente,
  validación de seguridad y health monitoring del sistema.
"""

# Exportar IntegrationManager
try:
    from .integration_manager import IntegrationManager  # type: ignore[F401]
except Exception:
    # IntegrationManager puede fallar por dependencias; ignorar gracefully
    pass

# Exportar clientes de generación si están disponibles.  Estos imports
# se realizan en un bloque try/except para no romper en entornos donde
# no se hayan instalado las dependencias opcionales.
try:
    from .ollama_client import generate_completion as generate_with_ollama  # type: ignore[F401]
except Exception:
    # Ollama puede no estar instalado o configurado; ignorar
    pass

try:
    from .llama_cpp_client import LlamaCppClient
    from .llama_cpp_client import generate_completion as generate_with_llama_cpp  # type: ignore[F401]
except Exception:
    pass
