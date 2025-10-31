# -*- coding: utf-8 -*-
"""
Proveedor 'yaml' minimalista: implementa safe_load/load y safe_dump/dump usando JSON y (opcional) TOML.
No parsea YAML real (evita dependencias pesadas). Útil si tus configs ya están en JSON/TOML.
"""
from __future__ import annotations

import json
from typing import Any, Optional, TextIO

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # opcional; si no está, seguimos sin TOML
    except Exception:
        tomllib = None  # sin TOML


class YamlError(ValueError):
    pass


def _loads(data: str) -> Any:
    data = data.strip()
    if tomllib and ("=" in data or data.startswith("[") or data.startswith("#")):
        try:
            return tomllib.loads(data)
        except Exception:
            pass
    try:
        return json.loads(data)
    except Exception as e:
        raise YamlError(f"No se puede parsear como JSON/TOML: {e}") from e


def _dumps(obj: Any, pretty: bool = True) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None, separators=(",", ": "))


def safe_load(stream: str | TextIO) -> Any:
    s = stream.read() if hasattr(stream, "read") else str(stream)
    return _loads(s)


def load(stream: str | TextIO) -> Any:
    return safe_load(stream)


def safe_dump(data: Any, stream: Optional[TextIO] = None, **kwargs) -> str | None:
    s = _dumps(data, pretty=kwargs.get("default_flow_style", False) is False)
    if stream is None:
        return s
    stream.write(s)
    return None


def dump(data: Any, stream: Optional[TextIO] = None, **kwargs) -> str | None:
    return safe_dump(data, stream, **kwargs)
