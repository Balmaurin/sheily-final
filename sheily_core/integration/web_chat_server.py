#!/usr/bin/env python3
"""
Servidor web de chat para Sheily AI con RAG mejorado y modelos Llama.

Este módulo expone un servicio HTTP basado en FastAPI que procesa
consultas de usuario mediante el sistema de recuperación de
información (RAG) mejorado y genera respuestas utilizando un modelo de
lenguaje grande (LLM).

El nuevo sistema RAG utiliza:
- RecursiveCharacterTextSplitter para chunking inteligente
- Embeddings multilingües con sentence-transformers
- FAISS para búsqueda vectorial eficiente
- Microservicio dedicado para escalabilidad

El flujo de procesamiento es el siguiente:

1. Se recibe una consulta via POST /chat con un mensaje del usuario y un contexto opcional.
2. El cliente RAG consulta el servicio dedicado para obtener contexto relevante de los corpus JSONL.
3. Se determina la rama académica basada en los chunks más relevantes encontrados.
4. Se construye un prompt enriquecido con el contexto recuperado y el texto de la consulta.
5. Se invoca el modelo LLM configurado para generar la respuesta.
6. Se devuelve la respuesta, junto con la rama seleccionada, la confianza del sistema RAG y el número de chunks utilizados.

Para usar RAG, el servicio debe estar ejecutándose:
  python start_rag_service.py

El endpoint de generación de Ollama sigue la API oficial según la
documentación: una petición POST a /api/generate con los parámetros
model y prompt devuelve un JSON con la respuesta cuando stream se
establece a false.

Para ejecutar el servidor con llama.cpp se debe establecer la variable
de entorno SHEILY_LLM_PROVIDER=llama_cpp y definir
SHEILY_LLAMA_MODEL_PATH con la ruta al archivo GGUF. Para Ollama, basta
con tener el servidor en ejecución y descargar el modelo Llama 3.2
mediante ollama run llama3.2.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importar cliente del nuevo RAG Service
from .rag_client import initialize_rag_service, get_rag_context, retrieve_relevant_memories

# Seleccionar proveedor de LLM en función de la configuración
LLM_PROVIDER = os.environ.get("SHEILY_LLM_PROVIDER", "ollama").lower()

if LLM_PROVIDER == "llama_cpp":
    try:
        from .llama_cpp_client import generate_completion as llama_generate
    except ImportError as e:
        raise RuntimeError(
            "El proveedor 'llama_cpp' está configurado pero no se pudo importar "
            "la dependencia llama-cpp-python. Instala la librería y define "
            "SHEILY_LLAMA_MODEL_PATH para usar llama.cpp."
        ) from e
else:
    # Proveedor por defecto: Ollama
    from .ollama_client import generate_completion as ollama_generate


class ChatRequest(BaseModel):
    """Esquema de la petición de chat"""

    message: str
    use_rag: bool = True
    use_mcp: bool = False
    mcp_servers: Optional[List[str]] = None
    temperature: float = 0.7
    context: Optional[str] = None


app = FastAPI(title="Sheily AI Chat", version="1.0.0")


@app.on_event("startup")
def _startup_event() -> None:
    """Inicializar componentes al arrancar el servidor."""
    global rag_enabled, mcp_enabled

    # Inicializar RAG Service
    rag_base_url = os.environ.get("SHEILY_RAG_SERVICE_URL", "http://localhost:8002")
    rag_enabled = initialize_rag_service(rag_base_url)

    # Si no está disponible, intentar procesar corpus automáticamente
    if not rag_enabled:
        print(f"⚠️ RAG Service no disponible en {rag_base_url}")
        print("💡 Asegúrate de que el RAG Service esté ejecutándose:")
        print("   python start_rag_service.py")
        print("   O establece SHEILY_RAG_SERVICE_URL si está en otro puerto")

    mcp_enabled = os.environ.get("SHEILY_MCP_ENABLED", "false").lower() == "true"


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Endpoint de salud para comprobar que el servicio está activo."""
    global rag_enabled, mcp_enabled
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "rag_enabled": rag_enabled,
        "mcp_enabled": mcp_enabled,
        "features": {
            "rag": rag_enabled,
            "mcp": mcp_enabled,
            "ollama": LLM_PROVIDER == "ollama",
            "llama_cpp": LLM_PROVIDER == "llama_cpp",
        }
    }


@app.post("/chat")
async def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """Procesar una consulta de usuario utilizando RAG y un modelo LLM.

    Args:
        req: Objeto con la consulta del usuario y un contexto opcional.

    Returns:
        Diccionario con la respuesta generada, la rama seleccionada,
        la confianza de RAG y el número de memorias recuperadas.
    """
    # 1. Usar RAG para obtener contexto y rama si está habilitado
    if req.use_rag and rag_enabled:
        try:
            context, relevant_chunks = get_rag_context(req.message, max_context_length=2000)

            # Determinar rama académica basada en los chunks más relevantes
            if relevant_chunks:
                # Usar la rama del chunk con mayor score
                top_chunk = max(relevant_chunks, key=lambda x: x['score'])
                selected_branch = top_chunk['metadata'].get('domain', 'general')
                rag_confidence = top_chunk['score']
                context_sources = len(relevant_chunks)
            else:
                selected_branch = "general"
                rag_confidence = 0.0
                context_sources = 0

        except Exception as e:
            print(f"Warning: RAG processing failed: {e}")
            context = ""
            selected_branch = "direct"
            rag_confidence = 0.0
            context_sources = 0

    elif req.use_rag and not rag_enabled:
        # RAG solicitado pero no disponible
        raise HTTPException(
            status_code=503,
            detail="RAG solicitado pero el servicio no está disponible. "
                   "Ejecuta: python start_rag_service.py"
        )
    else:
        # RAG deshabilitado
        context = ""
        selected_branch = "direct"
        rag_confidence = 0.0
        context_sources = 0

    # 2. Procesar con MCP si está habilitado
    mcp_tools_used = []
    if req.use_mcp and mcp_enabled:
        try:
            mcp_result = await process_with_mcp(req.message, req.mcp_servers)
            if mcp_result.get("enhanced_context"):
                context += f"\n{mcp_result['enhanced_context']}"
            mcp_tools_used = mcp_result.get("tools_used", [])
        except Exception as e:
            print(f"Warning: MCP processing failed: {e}")

    # 3. Construir prompt combinado
    prompt_parts = []
    if req.context:
        prompt_parts.append(f"Contexto adicional:\n{req.context}\n")
    if context:
        prompt_parts.append(context + "\n")
    prompt_parts.append(f"Pregunta: {req.message}\nRespuesta:")
    full_prompt = "\n".join(prompt_parts)

    # 4. Generar respuesta utilizando el proveedor configurado
    try:
        if LLM_PROVIDER == "llama_cpp":
            model_path = os.environ.get("SHEILY_LLAMA_MODEL_PATH")
            if not model_path:
                raise RuntimeError(
                    "Se requiere la variable de entorno SHEILY_LLAMA_MODEL_PATH con la ruta "
                    "al modelo GGUF para utilizar llama_cpp"
                )
            answer = llama_generate(full_prompt, model_path=model_path)
        else:
            # Ollama con temperatura configurable
            options = {"temperature": req.temperature} if abs(req.temperature - 0.7) > 0.001 else {}
            answer = ollama_generate(full_prompt, options=options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "query": req.message,
        "response": answer.strip(),
        "branch": selected_branch,
        "rag_confidence": rag_confidence,
        "context_sources": context_sources,
    }

    # Incluir información MCP si se usó
    if mcp_tools_used:
        result["mcp_tools_used"] = mcp_tools_used

    return result


async def process_with_mcp(
    message: str, requested_servers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Procesar mensaje utilizando servidores MCP disponibles."""
    try:
        # Importar dinámicamente los servidores MCP disponibles
        mcp_results = []
        enhanced_context = ""

        # Lista de servidores MCP disponibles
        available_servers = {
            "human_memory": "mcp_servers.human_memory_mcp_server",
            "emotional_intelligence": "mcp_servers.emotional_intelligence_mcp_server",
            "continuous_learning": "mcp_servers.continuous_learning_mcp_server",
            "file_processing": "mcp_servers.advanced_file_processing_mcp_server",
        }

        # Determinar qué servidores usar
        servers_to_use = requested_servers or list(available_servers.keys())

        for server_name in servers_to_use:
            if server_name in available_servers:
                try:
                    # Aquí se haría la llamada real al servidor MCP
                    # Por ahora simulamos la respuesta
                    tool_result = {
                        "server": server_name,
                        "tool": f"analyze_{server_name}",
                        "result": f"processed_by_{server_name}",
                    }
                    mcp_results.append(tool_result)
                    enhanced_context += f"[{server_name}] Contexto adicional del servidor MCP.\n"
                except ImportError:
                    print(f"Warning: MCP server {server_name} not available")
                    continue

        return {"enhanced_context": enhanced_context, "tools_used": mcp_results}

    except Exception as e:
        print(f"Error in MCP processing: {e}")
        return {"enhanced_context": "", "tools_used": []}


if __name__ == "__main__":  # pragma: no cover
    # Permitir ejecutar el servidor directamente con python web_chat_server.py
    port = int(os.environ.get("SHEILY_CHAT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
