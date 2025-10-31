#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio

from sheily_core.data.embeddings import EmbeddingManager


def test_encode_and_similarity_basic():
    async def _run():
        cfg = {
            "embedding_model": "mock",
            "vector_dimension": 64,
            "normalize_embeddings": True,
            "embedding_cache_path": "data/embeddings_cache_test",
        }
        em = EmbeddingManager(cfg)
        ok = await em.initialize()
        assert ok is True

        v1 = await em.encode_text("hola mundo", language="spanish")
        v2 = await em.encode_text("hello world", language="english")

        assert hasattr(v1, "__len__")
        assert hasattr(v2, "__len__")

        sim = em.calculate_similarity(v1, v2, metric="cosine")
        assert 0.0 <= sim <= 1.0

    asyncio.run(_run())
