#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Embeddings de Producción para Sheily-AI
ZERO ABSTRACCIONES - Solo implementaciones reales con sentence-transformers
"""

import asyncio
import hashlib
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# DEPENDENCIAS REALES - Sin fallbacks ni mocks
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------


@dataclass
class EmbeddingConfig:
    """Configuración de producción para embeddings"""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 512


class ProductionEmbeddingManager:
    """
    Gestor de embeddings de producción - SOLO implementaciones reales
    Sin mocks, sin stubs, sin fallbacks
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
    """__init__ function/class"""
        self.config = config or EmbeddingConfig()
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False

        logger.info(f"ProductionEmbeddingManager inicializando en {self.config.device}")

    async def initialize(self) -> bool:
        """Inicializar modelo real de sentence-transformers"""
        try:
            loop = asyncio.get_event_loop()

            # Cargar modelo REAL - Sin fallbacks
            self._model = await loop.run_in_executor(
                self._executor, SentenceTransformer, self.config.model_name, self.config.device
            )

            # Configurar modelo para máximo rendimiento
            self._model.max_seq_length = 512

            self._initialized = True
            logger.info(f"✅ Modelo real cargado: {self.config.model_name} on {self.config.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Error crítico cargando modelo real: {e}")
            # Sin fallback - Si falla, debe fallar completamente
            raise RuntimeError(f"No se pudo cargar el modelo de embeddings: {e}")

    async def encode_text(
        self, text: Union[str, List[str]], normalize: Optional[bool] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generar embeddings usando modelo REAL de sentence-transformers
        Sin fallbacks ni mocks
        """
        if not self._initialized or self._model is None:
            raise RuntimeError("Modelo no inicializado. Llamar a initialize() primero.")

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Verificar caché
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for idx, t in enumerate(texts):
            cache_key = self._generate_cache_key(t)
            cached = self._embedding_cache.get(cache_key)

            if cached and self._is_cache_valid(cached):
                embeddings.append(cached["embedding"])
            else:
                uncached_texts.append(t)
                uncached_indices.append(idx)
                embeddings.append(None)  # Placeholder

        # Generar embeddings REALES para textos no cacheados
        if uncached_texts:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                self._executor,
                lambda: self._model.encode(
                    uncached_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=normalize or self.config.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
            )

            # Insertar nuevos embeddings y cachearlos
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                cache_key = self._generate_cache_key(texts[idx])
                self._cache_embedding(cache_key, embedding)

        return embeddings[0] if is_single else embeddings

    def _generate_cache_key(self, text: str) -> str:
        """Generar clave única de caché"""
        content = f"{text}:{self.config.model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_cache_valid(self, cached_entry: Dict) -> bool:
        """Verificar si entrada de caché es válida (TTL)"""
        if self.config.cache_ttl_seconds <= 0:
            return True

        age = time.time() - cached_entry.get("timestamp", 0)
        return age < self.config.cache_ttl_seconds

    def _cache_embedding(self, key: str, embedding: np.ndarray):
        """Cachear embedding con TTL"""
        self._embedding_cache[key] = {"embedding": embedding, "timestamp": time.time()}

        # Limpiar caché si excede tamaño
        while self._get_cache_size_mb() > self.config.max_cache_size_mb:
            # Eliminar entrada más antigua
            oldest_key = min(
                self._embedding_cache.keys(), key=lambda k: self._embedding_cache[k]["timestamp"]
            )
            del self._embedding_cache[oldest_key]

    def _get_cache_size_mb(self) -> float:
        """Calcular tamaño del caché en MB"""
        total_bytes = sum(entry["embedding"].nbytes for entry in self._embedding_cache.values())
        return total_bytes / (1024 * 1024)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Similitud coseno real usando NumPy"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def find_similar_texts(
        self, query_text: str, candidate_texts: List[str], top_k: int = 5, threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Búsqueda de similitud REAL sin mocks"""
        query_emb = await self.encode_text(query_text)
        candidate_embs = await self.encode_text(candidate_texts)

        similarities = [
            (text, self.calculate_similarity(query_emb, emb))
            for text, emb in zip(candidate_texts, candidate_embs)
            if self.calculate_similarity(query_emb, emb) >= threshold
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def shutdown(self):
        """Cerrar recursos"""
        self._executor.shutdown(wait=True)
        self._embedding_cache.clear()
        logger.info("ProductionEmbeddingManager cerrado")


# Alias de compatibilidad
HighPerformanceEmbeddingManager = ProductionEmbeddingManager
EmbeddingManager = ProductionEmbeddingManager


def create_embedding_config(**kwargs) -> EmbeddingConfig:
    """Crear configuración con valores por defecto"""
    return EmbeddingConfig(**kwargs)


async def initialize_embedding_system(
    config: Optional[EmbeddingConfig] = None,
) -> ProductionEmbeddingManager:
    """Inicializar sistema de embeddings de producción"""
    manager = ProductionEmbeddingManager(config)
    await manager.initialize()
    return manager
