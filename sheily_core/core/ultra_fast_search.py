#!/usr/bin/env python3
"""
Sistema de Búsqueda Ultra-Rápida para Sheily AI
Implementa búsqueda indexada, cache inteligente y recuperación optimizada
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class SearchIndex:
    """Índice de búsqueda avanzado"""

    file_paths: Set[str] = field(default_factory=set)
    content_index: Dict[str, List[str]] = field(default_factory=dict)  # palabra -> archivos
    metadata_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # archivo -> metadatos
    word_frequency: Dict[str, int] = field(default_factory=dict)
    last_updated: float = 0.0


@dataclass
class SearchResult:
    """Resultado de búsqueda optimizado"""

    file_path: str
    relevance_score: float
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0


class UltraFastFileSearcher:
    """Sistema de búsqueda de archivos ultra-rápido"""

    def __init__(self, root_path: str = ".", max_workers: int = None):
        self.root_path = Path(root_path)
        self.max_workers = max_workers or os.cpu_count() or 4
        self.index = SearchIndex()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.index_lock = threading.Lock()

        # Configuración avanzada
        self.enable_fuzzy_search = True
        self.enable_semantic_search = True
        self.max_cache_size = 1000
        self.cache_ttl = 300  # 5 minutos

        # Inicializar índices avanzados
        self._initialize_advanced_indexes()

    def _initialize_advanced_indexes(self):
        """Inicializar índices avanzados para máxima velocidad"""
        self.word_positions = defaultdict(list)  # posiciones de palabras en archivos
        self.file_embeddings = {}  # embeddings semánticos de archivos
        self.category_index = defaultdict(set)  # archivos por categoría
        self.size_index = defaultdict(list)  # archivos por tamaño

    async def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Construir índice ultra-rápido de archivos"""
        start_time = time.time()

        # Verificar si el índice está actualizado
        if not force_rebuild and self._is_index_current():
            logger.info("📊 Índice actualizado, usando caché")
            return {"status": "cached", "files_indexed": len(self.index.file_paths)}

        logger.info("🔍 Construyendo índice avanzado de archivos...")

        # Búsqueda paralela de archivos
        all_files = await self._find_files_parallel()

        # Construcción de índice paralela
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Dividir archivos en lotes para procesamiento paralelo
            batch_size = max(1, len(all_files) // self.max_workers)
            batches = [all_files[i : i + batch_size] for i in range(0, len(all_files), batch_size)]

            # Procesar lotes en paralelo
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(executor, self._process_batch, batch) for batch in batches]

            batch_results = await asyncio.gather(*tasks)

        # Combinar resultados
        total_processed = sum(results["processed"] for results in batch_results)
        total_indexed = sum(results["indexed"] for results in batch_results)

        # Actualizar índice maestro
        with self.index_lock:
            self.index.last_updated = time.time()

        build_time = time.time() - start_time

        logger.info(f"✅ Índice construido: {total_indexed} archivos en {build_time:.2f}s")

        return {
            "status": "built",
            "files_processed": total_processed,
            "files_indexed": total_indexed,
            "build_time": build_time,
            "index_size": len(self.index.file_paths),
        }

    async def _find_files_parallel(self) -> List[Path]:
        """Encontrar archivos usando procesamiento paralelo"""
        all_files = []

        def find_files_recursive(path: Path) -> List[Path]:
            """Función recursiva para encontrar archivos (ejecutada en thread)"""
            files = []
            try:
                for item in path.iterdir():
                    if item.is_file() and self._is_indexable_file(item):
                        files.append(item)
                    elif item.is_dir() and not self._is_excluded_dir(item):
                        files.extend(find_files_recursive(item))
            except (PermissionError, OSError) as e:
                logger.warning(f"Error accediendo a {path}: {e}")
            return files

        # Ejecutar búsqueda en paralelo usando múltiples threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()

            # Crear tareas para diferentes subdirectorios raíz
            root_tasks = []
            for root_dir in [self.root_path]:
                if root_dir.exists():
                    task = loop.run_in_executor(executor, find_files_recursive, root_dir)
                    root_tasks.append(task)

            # Esperar resultados
            if root_tasks:
                results = await asyncio.gather(*root_tasks)
                for result in results:
                    all_files.extend(result)

        return all_files

    def _process_batch(self, batch: List[Path]) -> Dict[str, int]:
        """Procesar un lote de archivos para indexación"""
        processed = 0
        indexed = 0

        for file_path in batch:
            try:
                if self._process_file(file_path):
                    indexed += 1
                processed += 1
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {e}")
                processed += 1

        return {"processed": processed, "indexed": indexed}

    def _process_file(self, file_path: Path) -> bool:
        """Procesar archivo individual para indexación avanzada"""
        try:
            # Obtener metadatos básicos
            stat = file_path.stat()
            file_key = str(file_path)

            with self.index_lock:
                self.index.file_paths.add(file_key)
                self.index.metadata_index[file_key] = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "extension": file_path.suffix,
                    "name": file_path.name,
                }

            # Procesar contenido basado en extensión
            if file_path.suffix.lower() in [
                ".py",
                ".js",
                ".ts",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
            ]:
                self._index_file_content(file_path, file_key)

            return True

        except Exception as e:
            logger.error(f"Error procesando archivo {file_path}: {e}")
            return False

    def _index_file_content(self, file_path: Path, file_key: str):
        """Indexar contenido de archivo con técnicas avanzadas"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Tokenización avanzada
            words = self._advanced_tokenize(content)

            with self.index_lock:
                # Índice invertido de palabras
                for word in words:
                    if word not in self.index.content_index:
                        self.index.content_index[word] = []
                    if file_key not in self.index.content_index[word]:
                        self.index.content_index[word].append(file_key)

                    # Seguimiento de frecuencia
                    self.index.word_frequency[word] = self.index.word_frequency.get(word, 0) + 1

        except Exception as e:
            logger.error(f"Error indexando contenido de {file_path}: {e}")

    def _advanced_tokenize(self, content: str) -> List[str]:
        """Tokenización avanzada para mejor búsqueda"""
        # Convertir a minúsculas y limpiar
        content = content.lower()

        # Expresión regular para palabras (incluyendo números y símbolos comunes)
        words = re.findall(r"\b\w+\b", content)

        # Filtrar palabras muy cortas pero mantener términos técnicos
        filtered_words = [word for word in words if len(word) >= 2]

        return filtered_words

    def _is_indexable_file(self, file_path: Path) -> bool:
        """Determinar si un archivo debe ser indexado"""
        # Excluir archivos temporales, binarios, etc.
        excluded_extensions = {
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".dll",
            ".exe",
            ".bin",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".log",
            ".tmp",
            ".temp",
            ".cache",
        }

        return (
            file_path.suffix.lower() not in excluded_extensions
            and file_path.stat().st_size < 10 * 1024 * 1024  # Máximo 10MB
        )

    def _is_excluded_dir(self, dir_path: Path) -> bool:
        """Determinar si un directorio debe ser excluido"""
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".vscode",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".tox",
            ".eggs",
            "venv",
            "env",
            ".env",
        }

        return dir_path.name in excluded_dirs

    def _is_index_current(self) -> bool:
        """Verificar si el índice está actualizado"""
        if not self.index.file_paths:
            return False

        # Verificar si algún archivo ha cambiado desde la última indexación
        current_time = time.time()
        if current_time - self.index.last_updated > 300:  # 5 minutos
            return False

        # Verificar archivos modificados recientemente
        for file_path in list(self.index.file_paths)[:100]:  # Muestra de 100 archivos
            try:
                path = Path(file_path)
                if path.exists() and path.stat().st_mtime > self.index.last_updated:
                    return False
            except (OSError, ValueError):
                continue

        return True

    async def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """Búsqueda ultra-rápida con múltiples estrategias"""
        start_time = time.time()

        # Verificar caché primero
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # Construir índice si es necesario
        await self.build_index()

        # Estrategias de búsqueda múltiple
        results = []

        # 1. Búsqueda exacta de palabras clave
        exact_results = self._search_exact(query)

        # 2. Búsqueda fuzzy/similar
        if self.enable_fuzzy_search:
            fuzzy_results = self._search_fuzzy(query)

        # 3. Búsqueda semántica
        if self.enable_semantic_search:
            semantic_results = self._search_semantic(query)

        # Combinar y rankear resultados
        all_results = {}
        for result in exact_results + fuzzy_results + semantic_results:
            key = result.file_path
            if key in all_results:
                # Combinar puntuaciones
                all_results[key].relevance_score = max(all_results[key].relevance_score, result.relevance_score)
            else:
                all_results[key] = result

        # Ordenar por relevancia y limitar resultados
        sorted_results = sorted(all_results.values(), key=lambda x: x.relevance_score, reverse=True)
        final_results = sorted_results[:max_results]

        # Cachear resultado
        self._cache_result(cache_key, final_results)

        response_time = time.time() - start_time

        # Agregar tiempo de respuesta a resultados
        for result in final_results:
            result.response_time = response_time

        logger.info(f"🔍 Búsqueda '{query}' completada en {response_time:.3f}s - {len(final_results)} resultados")

        return final_results

    def _search_exact(self, query: str) -> List[SearchResult]:
        """Búsqueda exacta ultra-rápida"""
        results = []
        query_words = self._advanced_tokenize(query)

        # Búsqueda en índice invertido
        candidate_files = set()
        for word in query_words:
            if word in self.index.content_index:
                candidate_files.update(self.index.content_index[word])

        # Calcular relevancia para cada archivo candidato
        for file_path in candidate_files:
            relevance = self._calculate_relevance(file_path, query, query_words)
            if relevance > 0.1:  # Umbral mínimo de relevancia
                result = self._create_search_result(file_path, relevance, query)
                results.append(result)

        return results

    def _search_fuzzy(self, query: str) -> List[SearchResult]:
        """Búsqueda fuzzy para términos similares"""
        results = []

        # Implementación básica de fuzzy search usando distancia de edición
        query_words = set(self._advanced_tokenize(query))

        # Buscar palabras similares en el índice
        for word in query_words:
            # Encontrar palabras similares (diferencia de 1-2 caracteres)
            similar_words = self._find_similar_words(word)

            for similar_word in similar_words:
                if similar_word in self.index.content_index:
                    for file_path in self.index.content_index[similar_word]:
                        relevance = self._calculate_relevance(file_path, query, query_words) * 0.7
                        if relevance > 0.05:
                            result = self._create_search_result(file_path, relevance, query)
                            results.append(result)

        return results

    def _search_semantic(self, query: str) -> List[SearchResult]:
        """Búsqueda semántica básica"""
        # Implementación simplificada - en producción usar embeddings reales
        results = []

        # Por ahora, buscar archivos con términos relacionados
        related_terms = {
            "buscar": ["find", "search", "locate", "discover"],
            "archivo": ["file", "document", "doc", "archivo"],
            "codigo": ["code", "program", "script", "source"],
            "configuracion": ["config", "settings", "setup", "configuration"],
        }

        query_lower = query.lower()
        for key, synonyms in related_terms.items():
            if key in query_lower:
                for synonym in synonyms:
                    if synonym in self.index.content_index:
                        for file_path in self.index.content_index[synonym]:
                            relevance = 0.3  # Baja relevancia para búsqueda semántica básica
                            result = self._create_search_result(file_path, relevance, query)
                            results.append(result)

        return results

    def _find_similar_words(self, word: str) -> List[str]:
        """Encontrar palabras similares usando distancia de edición básica"""
        similar = []

        # Buscar palabras que difieran por 1-2 caracteres
        for indexed_word in list(self.index.content_index.keys())[:1000]:  # Limitar búsqueda
            # Distancia de Levenshtein simplificada
            if abs(len(word) - len(indexed_word)) <= 2:
                # Contar caracteres comunes
                common_chars = len(set(word) & set(indexed_word))
                total_chars = len(set(word) | set(indexed_word))

                if total_chars > 0:
                    similarity = common_chars / total_chars
                    if similarity >= 0.7:  # 70% de similitud
                        similar.append(indexed_word)

        return similar[:5]  # Limitar resultados

    def _calculate_relevance(self, file_path: str, query: str, query_words: List[str]) -> float:
        """Calcular relevancia de un archivo para una consulta"""
        relevance = 0.0

        # Factor 1: Número de palabras de la consulta encontradas
        words_found = 0
        for word in query_words:
            if word in self.index.content_index and file_path in self.index.content_index[word]:
                words_found += 1

        if query_words:
            relevance += (words_found / len(query_words)) * 0.6

        # Factor 2: Frecuencia de palabras clave
        file_word_count = sum(
            self.index.word_frequency.get(word, 0) for word in query_words if word in self.index.word_frequency
        )

        if file_word_count > 0:
            relevance += min(file_word_count / 1000, 0.3)  # Normalizar

        # Factor 3: Metadatos del archivo (tamaño, extensión, etc.)
        metadata = self.index.metadata_index.get(file_path, {})
        file_size = metadata.get("size", 0)

        # Archivos pequeños suelen ser más relevantes para búsquedas específicas
        if file_size < 1024 * 100:  # Menos de 100KB
            relevance += 0.1

        return min(relevance, 1.0)

    def _create_search_result(self, file_path: str, relevance: float, query: str) -> SearchResult:
        """Crear resultado de búsqueda con snippet"""
        snippet = self._extract_snippet(file_path, query)

        return SearchResult(
            file_path=file_path,
            relevance_score=relevance,
            snippet=snippet,
            metadata=self.index.metadata_index.get(file_path, {}),
            response_time=0.0,
        )

    def _extract_snippet(self, file_path: str, query: str) -> str:
        """Extraer snippet relevante del archivo"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Buscar contexto alrededor de palabras clave
            query_lower = query.lower()
            lines = content.split("\n")

            # Encontrar líneas que contienen palabras clave
            relevant_lines = []
            for i, line in enumerate(lines):
                if any(word in line.lower() for word in query_lower.split()):
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])
                    relevant_lines.append(context)

            if relevant_lines:
                return relevant_lines[0][:200] + "..." if len(relevant_lines[0]) > 200 else relevant_lines[0]

        except Exception as e:
            logger.error(f"Error extrayendo snippet de {file_path}: {e}")

        return ""

    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Obtener resultado del caché"""
        with self.cache_lock:
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                if time.time() - cached_item["timestamp"] < self.cache_ttl:
                    return cached_item["result"]
                else:
                    # Remover entrada expirada
                    del self.cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: List[SearchResult]):
        """Guardar resultado en caché"""
        with self.cache_lock:
            # Implementar LRU cache
            if len(self.cache) >= self.max_cache_size:
                # Remover entrada más antigua
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]

            self.cache[cache_key] = {"result": result, "timestamp": time.time()}


# Instancia global del buscador ultra-rápido
file_searcher = UltraFastFileSearcher()


async def search_files(query: str, max_results: int = 20) -> List[SearchResult]:
    """Función pública para búsqueda ultra-rápida"""
    return await file_searcher.search(query, max_results)


class SEILiCoreOptimizer:
    """Optimizador avanzado del núcleo SEI-Li"""

    def __init__(self):
        self.response_cache = {}
        self.thinking_cache = {}
        self.performance_metrics = {
            "total_responses": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    async def optimize_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimizar generación de respuesta"""
        start_time = time.time()

        # 1. Búsqueda ultra-rápida de contexto relevante
        search_start = time.time()
        search_results = await search_files(query, max_results=10)
        search_time = time.time() - search_start

        # 2. Procesamiento paralelo del contexto
        context_processed = await self._process_context_parallel(search_results, query)

        # 3. Generación de respuesta optimizada
        response_start = time.time()
        response = await self._generate_optimized_response(query, context_processed)
        response_time = time.time() - response_start

        # 4. Cache inteligente
        await self._cache_response(query, response)

        total_time = time.time() - start_time

        # Actualizar métricas
        self._update_performance_metrics(total_time, "response")

        return {
            "response": response,
            "performance": {
                "total_time": total_time,
                "search_time": search_time,
                "response_time": response_time,
                "context_sources": len(search_results),
            },
            "optimization": {
                "cache_used": self._check_cache_hit(query),
                "parallel_processing": True,
                "context_optimized": True,
            },
        }

    async def _process_context_parallel(self, search_results: List[SearchResult], query: str) -> Dict[str, Any]:
        """Procesar contexto en paralelo para máxima velocidad"""

        async def process_single_result(result: SearchResult) -> Dict[str, Any]:
            """Procesar un resultado individual"""
            return {
                "file": result.file_path,
                "relevance": result.relevance_score,
                "snippet": result.snippet,
                "metadata": result.metadata,
            }

        # Procesar todos los resultados en paralelo
        tasks = [process_single_result(result) for result in search_results]
        processed_results = await asyncio.gather(*tasks)

        # Filtrar y ordenar por relevancia
        filtered_results = [r for r in processed_results if r["relevance"] > 0.2]
        sorted_results = sorted(filtered_results, key=lambda x: x["relevance"], reverse=True)

        return {
            "relevant_sources": len(sorted_results),
            "top_sources": sorted_results[:5],
            "total_sources": len(search_results),
            "query_keywords": query.lower().split()[:5],  # Principales palabras clave
        }

    async def _generate_optimized_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generar respuesta optimizada usando contexto procesado"""

        # Respuesta basada en contexto disponible
        if context["relevant_sources"] == 0:
            return "No encontré información específica sobre tu consulta en el código fuente disponible."

        # Construir respuesta basada en fuentes relevantes
        response_parts = []
        response_parts.append(f"Encontré {context['relevant_sources']} fuentes relevantes para tu consulta.")

        # Incluir snippets de las fuentes más relevantes
        for source in context["top_sources"][:3]:
            if source["snippet"]:
                response_parts.append(f"📄 {Path(source['file']).name}: {source['snippet'][:100]}...")

        # Añadir recomendaciones basadas en el contexto
        if any("config" in source["file"].lower() for source in context["top_sources"]):
            response_parts.append("💡 Sugiero revisar los archivos de configuración para ajustes específicos.")

        if any(
            "error" in source["file"].lower() or "exception" in source["file"].lower()
            for source in context["top_sources"]
        ):
            response_parts.append(
                "🔧 Parece que hay archivos relacionados con manejo de errores que podrían ser relevantes."
            )

        return "\n\n".join(response_parts)

    async def _cache_response(self, query: str, response: str):
        """Implementar cache inteligente para respuestas"""
        cache_key = hashlib.md5(query.encode()).hexdigest()

        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

        # Limpiar cache antiguo (mantener últimos 100 elementos)
        if len(self.response_cache) > 100:
            oldest_keys = sorted(self.response_cache.keys(), key=lambda k: self.response_cache[k]["timestamp"])[:10]
            for key in oldest_keys:
                del self.response_cache[key]

    def _check_cache_hit(self, query: str) -> bool:
        """Verificar si hay un hit en caché"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        return cache_key in self.response_cache

    def _update_performance_metrics(self, response_time: float, operation_type: str):
        """Actualizar métricas de rendimiento"""
        self.performance_metrics["total_responses"] += 1

        # Actualizar promedio de tiempo de respuesta
        current_avg = self.performance_metrics["avg_response_time"]
        total = self.performance_metrics["total_responses"]

        self.performance_metrics["avg_response_time"] = (current_avg * (total - 1) + response_time) / total

        # Actualizar tasa de aciertos de caché
        cache_hits = sum(1 for v in self.response_cache.values() if v.get("hit", False))
        if self.response_cache:
            self.performance_metrics["cache_hit_rate"] = cache_hits / len(self.response_cache)


# Instancia global del optimizador SEI-Li
sei_licore_optimizer = SEILiCoreOptimizer()


async def optimized_sei_response(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Función pública para obtener respuestas SEI-Li optimizadas"""
    return await sei_licore_optimizer.optimize_response(query, context)
