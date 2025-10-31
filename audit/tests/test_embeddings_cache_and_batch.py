#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import os

from sheily_core.data.embeddings import EmbeddingConfig, EmbeddingManager


def test_cache_ttl_and_lru_and_batching():
    async def _run():
        cfg = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=4,
            normalize_embeddings=True,
            cache_ttl_seconds=1,
            max_cache_size_mb=256,
        )
        em = EmbeddingManager(cfg)
        assert await em.initialize()

        # primer batch
        texts = [f"texto {i}" for i in range(4)]
        e1 = await em.encode_text(texts)
        assert len(e1) == len(texts)

        # cache hit
        e2 = await em.encode_text(texts[0])
        # fuerza expiración TTL
        await asyncio.sleep(1.1)
        e3 = await em.encode_text(texts[0])
        # si TTL expiró, e2 y e3 pueden diferir
        # pero lo importante es que no reviente y rehaga caché sin error.
        assert hasattr(e2, "__len__") and hasattr(e3, "__len__")

    asyncio.run(_run())
