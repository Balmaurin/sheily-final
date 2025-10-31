#!/usr/bin/env python3
"""
Cliente de generación para modelos servidos mediante Ollama.

Ollama expone un endpoint HTTP accesible localmente en
``http://localhost:11434/api/generate``. Para generar una respuesta se
debe enviar una petición ``POST`` con un objeto JSON que incluya los
campos ``model`` y ``prompt``. Cuando ``stream`` se establece en
``False`` la API devuelve la respuesta completa en un único objeto
JSON【485831448119966†L438-L451】.

Este módulo encapsula esa llamada en una función utilitaria
``generate_completion`` que puede utilizarse desde el servidor de chat
para obtener respuestas de Llama 3.2.  Meta publicó que el modelo
Llama 3.2 está disponible para ejecutarse localmente con Ollama
mediante ``ollama run llama3.2``【890854066034281†L13-L33】, proporcionando
modelos de 1 y 3 mil millones de parámetros optimizados para uso
local【890854066034281†L34-L40】.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

__all__ = ["generate_completion"]


def generate_completion(
    prompt: str,
    *,
    model: str = "llama3.2:1b-instruct-q4_0",
    api_url: str = "http://localhost:11434/api/generate",
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Enviar un prompt al modelo especificado en Ollama y devolver la respuesta.

    Args:
        prompt: Texto a completar por el LLM.
        model: Nombre del modelo en Ollama (por defecto ``llama3.2:1b-instruct-q4_0``).
        api_url: URL del endpoint Ollama ``/api/generate``.
        stream: Si ``True``, Ollama devolverá una respuesta en
            streaming.  Este cliente utiliza ``False`` por defecto para
            recibir la respuesta completa【485831448119966†L438-L451】.
        options: Diccionario con opciones adicionales de generación
            compatibles con la API de Ollama (temperatura, longitud, etc.).

    Returns:
        La cadena de texto generada por el modelo.

    Raises:
        requests.HTTPError: Si la respuesta HTTP indica un error.
        requests.RequestException: Si ocurre un error de conexión.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if options:
        payload.update(options)

    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")
