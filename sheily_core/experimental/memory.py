#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibilidad para imports antiguos:
- Expone HumanMemoryEngine delegando al MemoryManager real (no es mock).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..shared.memory_manager import get_memory_manager


class HumanMemoryEngine:
    def __init__(self, user_id: str = "user_persistent", **kwargs: Any):
        self.user_id = user_id
        self._mm = get_memory_manager()
        if not self._mm.initialized:
            self._mm.initialize()

    def memorize_content(
        self,
        content: str,
        content_type: str = "note",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        return self._mm.memorize_content(
            content, content_type=content_type, importance=importance, metadata=metadata or {}
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._mm.search_memory(query, top_k=top_k)

    def get_memory_stats(self) -> Dict[str, Any]:
        return self._mm.get_memory_stats()

    # Métodos de compatibilidad opcionales
    def flush(self) -> None:
        if hasattr(self._mm, "flush"):
            self._mm.flush()

    def close(self) -> None:
        if hasattr(self._mm, "close"):
            self._mm.close()


__all__ = ["HumanMemoryEngine"]
