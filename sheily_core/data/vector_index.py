#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Índice vectorial simple en memoria para RAG
------------------------------------------

Proporciona una estructura mínima para añadir y consultar embeddings
usando `EmbeddingManager.calculate_similarity`.

Sin dependencias externas; listo para integrarse con FAISS opcionalmente.
"""
from typing import Any, Dict, List, Tuple


class InMemoryVectorIndex:
    def __init__(self, metric: str = "cosine"):
        self._items: List[Tuple[str, list]] = []
        self.metric = metric

    def add(self, item_id: str, vector: Any):
        # Guardar como lista para compatibilidad con MockNdarray/np.ndarray
        self._items.append((item_id, list(vector)))

    def batch_add(self, items: List[Tuple[str, Any]]):
        for item_id, vec in items:
            self.add(item_id, vec)

    def search(self, query_vector: Any, top_k: int = 5) -> List[Tuple[str, float]]:
        # Cálculo manual de similitud (solo producto punto/ coseno aproximado)
        q = list(query_vector)

        def dot(a: List[float], b: List[float]) -> float:
            return float(sum(x * y for x, y in zip(a, b)))

        def norm(a: List[float]) -> float:
            return sum(x * x for x in a) ** 0.5

        results: List[Tuple[str, float]] = []
        if self.metric == "dot":
            for item_id, vec in self._items:
                results.append((item_id, dot(q, vec)))
        else:
            n_q = norm(q) or 1.0
            for item_id, vec in self._items:
                n_v = norm(vec) or 1.0
                sim = dot(q, vec) / (n_q * n_v)
                results.append((item_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
