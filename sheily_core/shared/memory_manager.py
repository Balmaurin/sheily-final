#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEMORY MANAGER - GESTIÓN UNIFICADA DE MEMORIA
============================================

Módulo compartido que unifica todas las implementaciones de memoria:
- Memoria humana avanzada V2
- Memoria híbrida legacy
- Memoria local simple
- Integración RAG

Elimina duplicaciones y proporciona una interfaz consistente.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sheily_config import get_config  # compat si existe
except Exception:
    from ..core.config import get_config


class MemoryManager:
    """Gestor unificado de memoria"""

    def __init__(self):
        self.config = get_config()
        self.memory_engine = None
        self.rag_engine = None
        self.initialized = False

    def initialize(self) -> bool:
        """Inicializar sistema de memoria unificado"""
        try:
            print("🚀 Inicializando sistema de memoria unificado...")

            # Inicializar memoria humana avanzada
            if self.config.memory.enable_human_memory:
                self._init_human_memory()

            # Inicializar RAG neurológico
            if self.config.memory.enable_neuro_rag:
                self._init_rag_engine()

            self.initialized = True
            print("✅ Sistema de memoria unificado operativo")
            return True

        except Exception as e:
            print(f"❌ Error inicializando memoria: {e}")
            return False

    def _init_human_memory(self):
        """Inicializar memoria humana avanzada"""
        try:
            from sheily_core.memory.sheily_human_memory_v2 import integrate_human_memory_v2

            self.memory_engine = integrate_human_memory_v2(self.config.memory.user_id)
            print("✅ Memoria humana avanzada V2 operativa")
        except Exception as e:
            print(f"⚠️ Error inicializando memoria humana: {e}")
            self.memory_engine = None

    def _init_rag_engine(self):
        """Inicializar motor RAG neurológico"""
        try:
            from sheily_rag.neuro_rag_engine_v2 import integrate_neuro_rag

            self.rag_engine = integrate_neuro_rag()
            print("✅ Motor RAG neurológico operativo")
        except Exception as e:
            print(f"⚠️ Error inicializando RAG: {e}")
            self.rag_engine = None

    def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda unificada en memoria"""
        if not self.initialized:
            self.initialize()

        results = []

        # Buscar en memoria humana avanzada
        if self.memory_engine:
            try:
                memory_results = self.memory_engine.search_memory(
                    query, top_k=top_k, relevance_threshold=self.config.memory.threshold
                )
                results.extend(memory_results)
            except Exception as e:
                print(f"⚠️ Error en búsqueda de memoria: {e}")

        # Buscar en RAG
        if self.rag_engine:
            try:
                rag_results = self.rag_engine.search(query, top_k=top_k)
                results.extend(rag_results)
            except Exception as e:
                print(f"⚠️ Error en búsqueda RAG: {e}")

        # Ordenar por relevancia
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return results[:top_k]

    def memorize_content(
        self,
        content: str,
        content_type: str = "text",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Memorizar contenido de manera unificada"""
        if not self.initialized:
            self.initialize()

        memory_ids = []

        # Memorizar en memoria humana
        if self.memory_engine:
            try:
                ids = self.memory_engine.memorize_content(
                    content=content,
                    content_type=content_type,
                    importance=importance,
                    metadata=metadata,
                )
                memory_ids.extend(ids)
            except Exception as e:
                print(f"⚠️ Error memorizando en memoria humana: {e}")

        # Indexar en RAG
        if self.rag_engine:
            try:
                document_id = hashlib.sha256((content + str(time.time())).encode()).hexdigest()[:16]
                chunk_ids = self.rag_engine.index_document(
                    content=content,
                    document_id=document_id,
                    content_type=content_type,
                    metadata=metadata,
                )
                memory_ids.extend(chunk_ids)
            except Exception as e:
                print(f"⚠️ Error indexando en RAG: {e}")

        return memory_ids

    def get_memory_context(self, query: str) -> str:
        """Obtener contexto de memoria para enriquecer respuestas"""
        results = self.search_memory(query, top_k=3)

        if not results:
            return ""

        context_parts = []
        for result in results:
            if result.get("relevance_score", 0) >= self.config.memory.threshold:
                # Extraer contenido según el tipo de resultado
                if "memory_context" in result:
                    content = result["memory_context"].content
                elif "chunk" in result:
                    content = result["chunk"].content
                else:
                    continue

                if content:
                    context_parts.append(f"- {content[:200]}...")

        return "\n".join(context_parts) if context_parts else ""

    def consolidate_memory(self) -> Dict[str, Any]:
        """Consolidar memoria de manera unificada"""
        results = {}

        # Consolidar memoria humana
        if self.memory_engine:
            try:
                human_results = self.memory_engine.consolidate_memory()
                results["human_memory"] = human_results
            except Exception as e:
                results["human_memory"] = {"status": "error", "error": str(e)}

        # Consolidar RAG
        if self.rag_engine:
            try:
                rag_results = self.rag_engine.consolidate_memory()
                results["rag_engine"] = rag_results
            except Exception as e:
                results["rag_engine"] = {"status": "error", "error": str(e)}

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas unificadas de memoria"""
        stats = {
            "initialized": self.initialized,
            "human_memory_available": self.memory_engine is not None,
            "rag_engine_available": self.rag_engine is not None,
        }

        # Estadísticas de memoria humana
        if self.memory_engine:
            try:
                human_stats = self.memory_engine.get_memory_stats()
                stats["human_memory"] = human_stats
            except Exception as e:
                stats["human_memory"] = {"error": str(e)}

        # Estadísticas de RAG
        if self.rag_engine:
            try:
                rag_stats = self.rag_engine.get_system_stats()
                stats["rag_engine"] = rag_stats
            except Exception as e:
                stats["rag_engine"] = {"error": str(e)}

        return stats


# Instancia global del gestor de memoria
_memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Obtener instancia global del gestor de memoria"""
    return _memory_manager


def search_memory(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Función de conveniencia para búsqueda de memoria"""
    return _memory_manager.search_memory(query, top_k)


def memorize_content(
    content: str,
    content_type: str = "text",
    importance: float = 0.5,
    metadata: Optional[Dict] = None,
) -> List[str]:
    """Función de conveniencia para memorizar contenido"""
    return _memory_manager.memorize_content(content, content_type, importance, metadata)


def get_memory_context(query: str) -> str:
    """Función de conveniencia para obtener contexto de memoria"""
    return _memory_manager.get_memory_context(query)


def get_memory_stats() -> Dict[str, Any]:
    """Función de conveniencia para obtener estadísticas de memoria"""
    return _memory_manager.get_memory_stats()


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Probando Memory Manager...")

    # Inicializar
    if _memory_manager.initialize():
        print("✅ Memoria inicializada exitosamente")

        # Probar búsqueda
        results = search_memory("hola", top_k=3)
        print(f"✅ Búsqueda realizada: {len(results)} resultados")

        # Probar memorización
        memory_ids = memorize_content("Test de memoria unificada", "test")
        print(f"✅ Contenido memorizado: {len(memory_ids)} IDs")

        # Mostrar estadísticas
        stats = get_memory_stats()
        print(f"✅ Estadísticas: {stats['initialized']}")

    else:
        print("❌ Error inicializando memoria")
