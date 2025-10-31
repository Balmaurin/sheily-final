#!/usr/bin/env python3
"""
Cliente HTTP para el RAG Service - Integración con Sheily
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class RAGServiceClient:
    """
    Cliente para comunicarse con el RAG Service via HTTP
    """

    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30

    def health_check(self) -> bool:
        """Verificar que el servicio esté funcionando"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except:
            return False

    def process_corpus(self, corpus_path: str, domains: Optional[List[str]] = None) -> Dict:
        """Procesar e indexar un corpus"""
        data = {"corpus_path": corpus_path}
        if domains:
            data["domains"] = domains

        response = self.session.post(f"{self.base_url}/process_corpus", json=data)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Buscar documentos similares"""
        data = {"query": query, "top_k": top_k}
        response = self.session.post(f"{self.base_url}/search", json=data)
        response.raise_for_status()
        return response.json()

    def add_document(self, content: str, metadata: Optional[Dict] = None) -> Dict:
        """Agregar un documento individual"""
        data = {"content": content}
        if metadata:
            data["metadata"] = metadata

        response = self.session.post(f"{self.base_url}/add_document", json=data)
        response.raise_for_status()
        return response.json()

    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> Tuple[str, List[Dict]]:
        """
        Obtener contexto relevante para una consulta
        Returns: (context_string, list_of_chunks)
        """
        try:
            results = self.search(query, top_k=5)

            if not results:
                return "", []

            # Filtrar por score mínimo y construir contexto
            relevant_chunks = [r for r in results if r["score"] > 0.5]

            context_parts = []
            total_length = 0

            for chunk in relevant_chunks:
                chunk_text = chunk["content"]
                if total_length + len(chunk_text) > max_context_length:
                    # Truncar si excede el límite
                    remaining = max_context_length - total_length
                    if remaining > 100:  # Solo agregar si queda espacio significativo
                        chunk_text = chunk_text[:remaining] + "..."
                        context_parts.append(chunk_text)
                    break

                context_parts.append(chunk_text)
                total_length += len(chunk_text)

            context = "\n\n".join(context_parts)
            return context, relevant_chunks

        except Exception as e:
            logger.error(f"Error obteniendo contexto del RAG Service: {e}")
            return "", []


# Instancia global del cliente
rag_client = RAGServiceClient()


def initialize_rag_service(base_url: str = "http://localhost:8002") -> bool:
    """
    Inicializar conexión con el RAG Service
    """
    global rag_client
    rag_client = RAGServiceClient(base_url)

    if rag_client.health_check():
        logger.info(f"✅ RAG Service conectado en {base_url}")
        return True
    else:
        logger.warning(f"⚠️ RAG Service no disponible en {base_url}")
        return False


def get_rag_context(query: str, max_length: int = 2000) -> Tuple[str, List[Dict]]:
    """
    Función de conveniencia para obtener contexto RAG
    """
    return rag_client.get_relevant_context(query, max_length)


# Función de compatibilidad con el sistema existente
def retrieve_relevant_memories(query: str, branch_name: str = None, max_results: int = 5) -> List[Dict]:
    """
    Función de compatibilidad con SheilyLoRARAGIntegrator
    """
    try:
        results = rag_client.search(query, top_k=max_results)

        # Convertir al formato esperado por el sistema existente
        memories = []
        for result in results:
            memories.append(
                {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"],
                    "source": "rag_service",
                }
            )

        return memories

    except Exception as e:
        logger.error(f"Error retrieving memories from RAG Service: {e}")
        return []
