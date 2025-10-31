#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer Manager - Gestión de Tokenizers para Sheily-AI
========================================================

Maneja la carga, caché y uso de tokenizers para diferentes modelos
con soporte para múltiples tipos y optimizaciones de rendimiento.
"""

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Gestor de tokenizers con soporte para múltiples formatos
    """

    def __init__(self, config: Dict):
        """
        Inicializar gestor de tokenizers

        Args:
            config: Configuración del gestor
        """
        self.config = config
        self.tokenizers_path = Path(config.get("tokenizers_path", "models/tokenizers/"))

        # Cache de tokenizers
        self._tokenizer_cache = {}
        self._loading_lock = threading.RLock()

        # Tokenizers por defecto
        self._default_tokenizers = {
            "spanish": "es_tokenizer",
            "english": "en_tokenizer",
            "bilingual": "bilingual_tokenizer",
        }

        # Patrones de pre-procesamiento
        self._preprocessing_patterns = [
            (re.compile(r"\s+"), " "),  # Normalizar espacios
            (re.compile(r"[^\w\s\-\.\,\!\?\:]"), ""),  # Limpiar caracteres especiales
        ]

        # Métricas
        self._tokenizer_stats = {
            "tokenizations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_processed": 0,
            "average_tokens_per_text": 0.0,
        }

        logger.info("TokenizerManager inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el gestor de tokenizers

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Verificar directorio de tokenizers
            if not self.tokenizers_path.exists():
                logger.warning(f"Directorio de tokenizers no existe: {self.tokenizers_path}")
                self.tokenizers_path.mkdir(parents=True, exist_ok=True)

            # Cargar tokenizers por defecto
            await self._load_default_tokenizers()

            logger.info("TokenizerManager inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando TokenizerManager: {e}")
            return False

    async def _load_default_tokenizers(self):
        """Cargar tokenizers por defecto"""
        for lang, tokenizer_name in self._default_tokenizers.items():
            try:
                await self._load_builtin_tokenizer(tokenizer_name, lang)
            except Exception as e:
                logger.warning(f"No se pudo cargar tokenizer {tokenizer_name}: {e}")

    async def _load_builtin_tokenizer(self, tokenizer_name: str, language: str):
        """
        Cargar tokenizer integrado

        Args:
            tokenizer_name: Nombre del tokenizer
            language: Idioma objetivo
        """
        # Implementación de tokenizer básico usando stdlib
        tokenizer = BasicTokenizer(language)

        self._tokenizer_cache[tokenizer_name] = {
            "tokenizer": tokenizer,
            "language": language,
            "type": "basic",
            "loaded_time": time.time(),
            "usage_count": 0,
        }

        logger.info(f"Tokenizer básico cargado: {tokenizer_name} ({language})")

    def tokenize(
        self,
        text: str,
        tokenizer_name: Optional[str] = None,
        language: Optional[str] = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[str], List[int], Any]:
        """
        Tokenizar texto usando el tokenizer especificado

        Args:
            text: Texto a tokenizar
            tokenizer_name: Nombre del tokenizer (auto-detectar si no se especifica)
            language: Idioma del texto
            add_special_tokens: Si añadir tokens especiales
            return_tensors: Formato de retorno ('pt', 'tf', None)

        Returns:
            Tokens como lista de strings, IDs, o tensores
        """
        start_time = time.time()

        try:
            # Auto-detectar tokenizer si no se especifica
            if not tokenizer_name:
                tokenizer_name = self._auto_detect_tokenizer(text, language)

            # Obtener tokenizer del caché
            tokenizer_info = self._get_tokenizer(tokenizer_name)
            if not tokenizer_info:
                raise ValueError(f"Tokenizer no disponible: {tokenizer_name}")

            tokenizer = tokenizer_info["tokenizer"]

            # Pre-procesar texto
            preprocessed_text = self._preprocess_text(text)

            # Tokenizar
            tokens = tokenizer.tokenize(preprocessed_text, add_special_tokens=add_special_tokens)

            # Convertir formato si se solicita
            if return_tensors:
                tokens = self._convert_to_tensors(tokens, return_tensors)

            # Actualizar métricas
            self._update_tokenization_metrics(len(tokens), time.time() - start_time)
            tokenizer_info["usage_count"] += 1

            return tokens

        except Exception as e:
            logger.error(f"Error tokenizando texto: {e}")
            raise

    def encode(
        self,
        text: str,
        tokenizer_name: Optional[str] = None,
        language: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Codificar texto a IDs de tokens

        Args:
            text: Texto a codificar
            tokenizer_name: Nombre del tokenizer
            language: Idioma del texto
            add_special_tokens: Si añadir tokens especiales
            max_length: Longitud máxima
            padding: Si hacer padding
            truncation: Si truncar

        Returns:
            Lista de IDs de tokens
        """
        # Tokenizar primero
        tokens = self.tokenize(text, tokenizer_name, language, add_special_tokens)

        # Obtener tokenizer para conversión a IDs
        if not tokenizer_name:
            tokenizer_name = self._auto_detect_tokenizer(text, language)

        tokenizer_info = self._get_tokenizer(tokenizer_name)
        tokenizer = tokenizer_info["tokenizer"]

        # Convertir tokens a IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Aplicar truncation si es necesario
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Aplicar padding si es necesario
        if padding and max_length and len(token_ids) < max_length:
            pad_token_id = tokenizer.pad_token_id or 0
            token_ids.extend([pad_token_id] * (max_length - len(token_ids)))

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        tokenizer_name: Optional[str] = None,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decodificar IDs de tokens a texto

        Args:
            token_ids: Lista de IDs de tokens
            tokenizer_name: Nombre del tokenizer
            skip_special_tokens: Si omitir tokens especiales
            clean_up_tokenization_spaces: Si limpiar espacios

        Returns:
            Texto decodificado
        """
        if not tokenizer_name:
            tokenizer_name = "bilingual"  # Default

        tokenizer_info = self._get_tokenizer(tokenizer_name)
        if not tokenizer_info:
            raise ValueError(f"Tokenizer no disponible: {tokenizer_name}")

        tokenizer = tokenizer_info["tokenizer"]

        # Convertir IDs a tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Filtrar tokens especiales si se solicita
        if skip_special_tokens:
            tokens = [t for t in tokens if not tokenizer.is_special_token(t)]

        # Unir tokens en texto
        text = tokenizer.convert_tokens_to_string(tokens)

        # Limpiar espacios si se solicita
        if clean_up_tokenization_spaces:
            text = self._cleanup_tokenization_spaces(text)

        return text

    def _auto_detect_tokenizer(self, text: str, language: Optional[str]) -> str:
        """
        Auto-detectar tokenizer apropiado

        Args:
            text: Texto a analizar
            language: Idioma sugerido

        Returns:
            Nombre del tokenizer recomendado
        """
        if language:
            if language.lower() in ["es", "spanish", "español"]:
                return self._default_tokenizers["spanish"]
            elif language.lower() in ["en", "english"]:
                return self._default_tokenizers["english"]

        # Detección automática por contenido
        spanish_indicators = ["ñ", "á", "é", "í", "ó", "ú", "ü", "¿", "¡"]
        spanish_score = sum(1 for char in spanish_indicators if char in text.lower())

        if spanish_score > 0:
            return self._default_tokenizers["spanish"]
        else:
            return self._default_tokenizers["bilingual"]

    def _get_tokenizer(self, tokenizer_name: str) -> Optional[Dict]:
        """
        Obtener tokenizer del caché

        Args:
            tokenizer_name: Nombre del tokenizer

        Returns:
            Información del tokenizer o None
        """
        with self._loading_lock:
            if tokenizer_name in self._tokenizer_cache:
                self._tokenizer_stats["cache_hits"] += 1
                return self._tokenizer_cache[tokenizer_name]
            else:
                self._tokenizer_stats["cache_misses"] += 1
                return None

    def _preprocess_text(self, text: str) -> str:
        """
        Pre-procesar texto antes de tokenización

        Args:
            text: Texto original

        Returns:
            Texto pre-procesado
        """
        processed_text = text

        # Aplicar patrones de pre-procesamiento
        for pattern, replacement in self._preprocessing_patterns:
            processed_text = pattern.sub(replacement, processed_text)

        return processed_text.strip()

    def _convert_to_tensors(self, tokens: List[str], format_type: str) -> Any:
        """
        Convertir tokens a formato tensor

        Args:
            tokens: Lista de tokens
            format_type: Tipo de tensor ('pt', 'tf')

        Returns:
            Tokens en formato tensor (mock en implementación actual)
        """
        # En implementación real, convertiría a PyTorch o TensorFlow tensors
        # Por ahora retornamos la lista original con metadata
        return {
            "input_ids": [hash(token) % 50000 for token in tokens],  # IDs simulados
            "attention_mask": [1] * len(tokens),
            "format": format_type,
            "length": len(tokens),
        }

    def _cleanup_tokenization_spaces(self, text: str) -> str:
        """
        Limpiar espacios de tokenización

        Args:
            text: Texto con espacios de tokenización

        Returns:
            Texto limpio
        """
        # Limpiar espacios extra
        text = re.sub(r"\s+", " ", text)

        # Corregir espacios alrededor de puntuación
        text = re.sub(r"\s+([.,:;!?])", r"\1", text)
        text = re.sub(r"([¿¡])\s+", r"\1", text)

        return text.strip()

    def _update_tokenization_metrics(self, num_tokens: int, processing_time: float):
        """
        Actualizar métricas de tokenización

        Args:
            num_tokens: Número de tokens procesados
            processing_time: Tiempo de procesamiento
        """
        self._tokenizer_stats["tokenizations"] += 1
        self._tokenizer_stats["total_tokens_processed"] += num_tokens

        # Actualizar promedio de tokens por texto
        total_tokenizations = self._tokenizer_stats["tokenizations"]
        current_avg = self._tokenizer_stats["average_tokens_per_text"]

        self._tokenizer_stats["average_tokens_per_text"] = (
            current_avg * (total_tokenizations - 1) + num_tokens
        ) / total_tokenizations

    def get_tokenizer_info(self, tokenizer_name: str) -> Optional[Dict]:
        """
        Obtener información de un tokenizer

        Args:
            tokenizer_name: Nombre del tokenizer

        Returns:
            Información del tokenizer o None
        """
        tokenizer_info = self._get_tokenizer(tokenizer_name)
        if tokenizer_info:
            # Retornar copia sin la instancia del tokenizer
            info = tokenizer_info.copy()
            info.pop("tokenizer", None)
            return info
        return None

    def get_available_tokenizers(self) -> List[str]:
        """Obtener lista de tokenizers disponibles"""
        return list(self._tokenizer_cache.keys())

    def get_stats(self) -> Dict:
        """Obtener estadísticas del gestor"""
        return {**self._tokenizer_stats, "loaded_tokenizers": len(self._tokenizer_cache)}

    async def health_check(self) -> Dict:
        """Verificar estado de salud del gestor"""
        return {
            "status": "healthy",
            "loaded_tokenizers": len(self._tokenizer_cache),
            "default_tokenizers_available": all(
                name in self._tokenizer_cache for name in self._default_tokenizers.values()
            ),
            "stats": self.get_stats(),
        }

    async def shutdown(self):
        """Cerrar gestor y limpiar recursos"""
        logger.info("Iniciando shutdown del TokenizerManager")

        # Limpiar caché
        self._tokenizer_cache.clear()

        logger.info("TokenizerManager shutdown completado")


class BasicTokenizer:
    """
    Tokenizer básico usando solo Python stdlib
    """

    def __init__(self, language: str = "bilingual"):
        """
        Inicializar tokenizer básico

        Args:
            language: Idioma objetivo
        """
        self.language = language
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # Vocabulario básico simulado
        self._vocab = self._build_basic_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        # Patrones de tokenización
        self._token_pattern = re.compile(r"\w+|[^\w\s]")

    def _build_basic_vocab(self) -> Dict[str, int]:
        """Construir vocabulario básico"""
        vocab = {
            "[PAD]": self.pad_token_id,
            "[UNK]": self.unk_token_id,
            "[BOS]": self.bos_token_id,
            "[EOS]": self.eos_token_id,
        }

        # Añadir palabras comunes en español e inglés
        common_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "el",
            "la",
            "un",
            "una",
            "y",
            "o",
            "pero",
            "en",
            "de",
            "para",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "do",
            "does",
            "es",
            "son",
            "era",
            "fueron",
            "tiene",
            "tienen",
            "había",
        ]

        for i, word in enumerate(common_words, start=4):
            vocab[word] = i

        return vocab

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenizar texto en tokens

        Args:
            text: Texto a tokenizar
            add_special_tokens: Si añadir tokens especiales

        Returns:
            Lista de tokens
        """
        # Tokenizar usando patrón regex
        tokens = self._token_pattern.findall(text.lower())

        if add_special_tokens:
            tokens = ["[BOS]"] + tokens + ["[EOS]"]

        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convertir tokens a IDs"""
        return [self._vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convertir IDs a tokens"""
        return [self._id_to_token.get(id_, "[UNK]") for id_ in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convertir tokens a string"""
        return " ".join(tokens)

    def is_special_token(self, token: str) -> bool:
        """Verificar si un token es especial"""
        return token.startswith("[") and token.endswith("]")
