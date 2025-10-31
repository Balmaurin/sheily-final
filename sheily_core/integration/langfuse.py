"""
Adaptador para LangFuse utilizando la API real
"""
from typing import Any, Dict, List, Optional

# Intentamos importar langfuse real
try:
    import langfuse
    from langfuse import Langfuse

    USE_REAL_LANGFUSE = True
    print("✅ Usando LangFuse real")
except ImportError:
    USE_REAL_LANGFUSE = False
    print("⚠️ LangFuse no instalado, usando stub")


def log_model_inference(model: str, prompt: str, completion: str, metadata: Dict = None) -> str:
    """
    Registra una inferencia de modelo en LangFuse

    Args:
        model: Nombre del modelo
        prompt: Prompt utilizado
        completion: Respuesta del modelo
        metadata: Metadatos adicionales

    Returns:
        ID de la traza registrada
    """
    if USE_REAL_LANGFUSE:
        return _log_with_real_langfuse(model, prompt, completion, metadata)
    else:
        return _simulate_langfuse_logging(model, prompt)


def _log_with_real_langfuse(model: str, prompt: str, completion: str, metadata: Dict = None) -> str:
    """Registra en la API real de LangFuse"""
    import os
    import uuid

    # Configurar cliente de LangFuse
    langfuse_client = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "demo"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "demo"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    # Crear ID único
    trace_id = f"trace_{uuid.uuid4().hex[:8]}"

    # Crear traza
    trace = langfuse_client.trace(name="model-inference", id=trace_id, metadata=metadata or {})

    # Registrar generación
    generation = trace.generation(
        name="completion",
        model=model,
        prompt=prompt,
        completion=completion,
        metadata=metadata or {},
    )

    # Finalizar traza
    trace.update(status="success")

    return trace_id


def _simulate_langfuse_logging(model: str, prompt: str) -> str:
    """Simula el registro en LangFuse"""
    import uuid

    return f"trace_{uuid.uuid4().hex[:8]}"
