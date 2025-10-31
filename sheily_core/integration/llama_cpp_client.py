#!/usr/bin/env python3
"""
Cliente para modelos Llama ejecutados mediante ``llama.cpp``.

Este módulo encapsula la interacción con el binding Python de
``llama.cpp`` (`llama-cpp-python`). Permite cargar un modelo en formato
``gguf`` y generar textos a partir de un prompt. Para usar este módulo
es necesario instalar la dependencia ``llama-cpp-python`` y disponer
del archivo del modelo (por ejemplo un modelo de la familia Llama 3.2
en formato GGUF). La ruta al modelo debe especificarse al crear la
instancia del cliente.

La función pública ``generate_completion`` ofrece una interfaz
compatible con la proporcionada por el cliente de Ollama, lo que
facilita la elección dinámica del proveedor en el servidor de chat.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from llama_cpp import Llama
except ImportError as e:  # pragma: no cover
    # Este error se lanzará cuando el módulo no esté instalado.  Las
    # llamadas al cliente deberían capturar la excepción y avisar al
    # usuario que debe instalar ``llama-cpp-python``.
    Llama = None  # type: ignore


__all__ = ["LlamaCppClient", "generate_completion"]


class LlamaCppClient:
    """Cliente ligero para interactuar con un modelo Llama vía ``llama.cpp``.

    Args:
        model_path: Ruta al archivo del modelo GGUF.  Debe ser un
            modelo compatible con ``llama.cpp``.
        n_ctx: Número máximo de tokens de contexto.  Ajusta este
            parámetro en función de la capacidad del modelo y la
            memoria disponible.
    """

    def __init__(self, model_path: str, *, n_ctx: int = 4096):
        if Llama is None:
            raise ImportError(
                "llama-cpp-python no está instalado. Ejecuta `pip install llama-cpp-python` "
                "para habilitar el soporte de llama.cpp."
            )
        # Cargar modelo
        self._llama = Llama(model_path=model_path, n_ctx=n_ctx)

    def generate_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generar una respuesta a partir de un prompt usando ``llama.cpp``.

        Args:
            prompt: Texto de entrada que incluye el contexto y la
                consulta del usuario.
            max_tokens: Número máximo de tokens a generar.
            temperature: Temperatura de muestreo para controlar la
                aleatoriedad de la respuesta.
            stop: Lista de tokens de parada opcionales.

        Returns:
            Texto generado por el modelo.
        """
        result = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **kwargs,
        )
        # El binding devuelve un diccionario con la clave "choices"
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("text", "").lstrip()
        return ""


def generate_completion(
    prompt: str,
    *,
    model_path: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop: Optional[list[str]] = None,
    **kwargs: Any,
) -> str:
    """Función auxiliar para generar texto con un modelo llama.cpp.

    Esta función crea internamente una instancia de :class:`LlamaCppClient` y
    devuelve el resultado de :meth:`LlamaCppClient.generate_completion`. Se
    proporciona para mantener una API similar a la del cliente de Ollama.

    Args:
        prompt: Texto de entrada para el modelo.
        model_path: Ruta al archivo del modelo GGUF.
        max_tokens: Número máximo de tokens a generar.
        temperature: Temperatura de muestreo.
        stop: Tokens de parada opcionales.
        **kwargs: Parámetros adicionales que se pasan al binding.

    Returns:
        Respuesta generada por el modelo.
    """
    client = LlamaCppClient(model_path)
    return client.generate_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        **kwargs,
    )
