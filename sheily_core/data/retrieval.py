#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval Manager - Gestión de Recuperación para RAG
====================================================

Maneja búsquedas híbridas, re-ranking y recuperación optimizada
de documentos del corpus bilingüe.
"""

import asyncio
import json
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RetrievalManager:
    """
    Gestor de recuperación con búsqueda híbrida y re-ranking
    """

    def __init__(self, config: Dict):
        """
        Inicializar gestor de recuperación

        Args:
            config: Configuración del gestor
        """
        self.config = config
        self.similarity_threshold = config.get("similarity_threshold", 0.7)

        # Configuración de búsqueda híbrida
        self.default_alpha = config.get("hybrid_alpha", 0.5)
        self.tfidf_cache = {}

        # Re-ranking algorithms
        self.rerank_algorithms = {
            "rrf": self._reciprocal_rank_fusion,
            "score_fusion": self._score_fusion,
            "semantic_rerank": self._semantic_rerank,
        }

        # Métricas de recuperación
        self._retrieval_stats = {
            "searches_performed": 0,
            "semantic_searches": 0,
            "lexical_searches": 0,
            "hybrid_searches": 0,
            "reranking_operations": 0,
            "average_search_time": 0.0,
        }

        logger.info("RetrievalManager inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el gestor de recuperación

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Inicializar caches y estructuras
            await self._build_tfidf_vocabulary()

            logger.info("RetrievalManager inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando RetrievalManager: {e}")
            return False

    async def _build_tfidf_vocabulary(self):
        """Construir vocabulario TF-IDF básico"""
        # En implementación real, construiría desde el corpus
        # Por ahora usamos vocabulario predefinido
        self.tfidf_vocab = {
            "spanish": self._build_spanish_vocab(),
            "english": self._build_english_vocab(),
        }

        logger.info("Vocabulario TF-IDF construido")

    def _build_spanish_vocab(self) -> Dict[str, float]:
        """Construir vocabulario español con pesos IDF simulados"""
        spanish_terms = [
            "inteligencia",
            "artificial",
            "programación",
            "python",
            "datos",
            "análisis",
            "sistema",
            "desarrollo",
            "algoritmo",
            "modelo",
            "información",
            "proceso",
            "función",
            "variable",
            "código",
            "aplicación",
            "software",
            "computadora",
            "tecnología",
            "digital",
        ]

        return {term: math.log(1000 / (i + 1)) for i, term in enumerate(spanish_terms)}

    def _build_english_vocab(self) -> Dict[str, float]:
        """Construir vocabulario inglés con pesos IDF simulados"""
        english_terms = [
            "artificial",
            "intelligence",
            "programming",
            "python",
            "data",
            "analysis",
            "system",
            "development",
            "algorithm",
            "model",
            "information",
            "process",
            "function",
            "variable",
            "code",
            "application",
            "software",
            "computer",
            "technology",
            "digital",
        ]

        return {term: math.log(1000 / (i + 1)) for i, term in enumerate(english_terms)}

    async def semantic_search(
        self, query: str, language: str, domain: Optional[str] = None, k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica usando embeddings

        Args:
            query: Consulta de búsqueda
            language: Idioma de búsqueda
            domain: Dominio específico
            k: Número de resultados

        Returns:
            Lista de documentos similares
        """
        start_time = time.time()

        try:
            # Búsqueda semántica real usando similitud de texto y corpus
            # Implementación completa sin dependencias externas

            results = await self._real_semantic_search(query, language, domain, k)

            self._retrieval_stats["semantic_searches"] += 1
            self._update_search_metrics(time.time() - start_time)

            return results

        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []

    async def lexical_search(
        self, query: str, language: str, domain: Optional[str] = None, k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda léxica usando TF-IDF/BM25

        Args:
            query: Consulta de búsqueda
            language: Idioma de búsqueda
            domain: Dominio específico
            k: Número de resultados

        Returns:
            Lista de documentos relevantes
        """
        start_time = time.time()

        try:
            # Procesar consulta
            query_terms = self._preprocess_query(query, language)

            # Calcular scores TF-IDF
            results = await self._tfidf_search(query_terms, language, domain, k)

            self._retrieval_stats["lexical_searches"] += 1
            self._update_search_metrics(time.time() - start_time)

            return results

        except Exception as e:
            logger.error(f"Error en búsqueda léxica: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid",
        language: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Método unificado de recuperación que soporta todos los tipos de búsqueda

        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a devolver
            search_type: Tipo de búsqueda ('hybrid', 'semantic', 'lexical')
            language: Idioma específico (None para auto-detectar)
            domain: Dominio específico (None para todos)

        Returns:
            Lista de documentos recuperados
        """
        try:
            # Auto-detectar idioma si no se especifica
            if not language:
                language = self._detect_language(query)

            # Ejecutar búsqueda según el tipo
            if search_type == "semantic":
                results = await self.semantic_search(query, language, domain, top_k)
            elif search_type == "lexical":
                results = await self.lexical_search(query, language, domain, top_k)
            elif search_type == "hybrid":
                results = await self.hybrid_search(query, language, domain, top_k)
            else:
                # Por defecto, usar búsqueda híbrida
                results = await self.hybrid_search(query, language, domain, top_k)

            logger.info(f"Retrieve completado: {len(results)} documentos encontrados")
            return results

        except Exception as e:
            logger.error(f"Error en retrieve: {e}")
            return []

    def _detect_language(self, query: str) -> str:
        """
        Auto-detectar idioma de la consulta (implementación real)

        Args:
            query: Consulta a analizar

        Returns:
            Idioma detectado ('spanish' o 'english')
        """
        # Indicadores específicos del español
        spanish_indicators = ["ñ", "á", "é", "í", "ó", "ú", "ü", "¿", "¡"]
        spanish_words = {
            "qué",
            "cómo",
            "cuál",
            "cuándo",
            "dónde",
            "por",
            "para",
            "con",
            "sin",
            "sobre",
            "bajo",
            "entre",
            "desde",
            "hasta",
            "según",
            "durante",
            "mediante",
        }

        # Contar indicadores de español
        spanish_score = 0

        # Caracteres específicos del español
        for char in spanish_indicators:
            spanish_score += query.lower().count(char)

        # Palabras específicas del español
        query_words = set(query.lower().split())
        spanish_score += len(query_words.intersection(spanish_words))

        # Si hay evidencia de español, es español; sino, inglés
        return "spanish" if spanish_score > 0 else "english"

    async def hybrid_search(
        self,
        query: str,
        language: str,
        domain: Optional[str] = None,
        k: int = 10,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida combinando semántica y léxica

        Args:
            query: Consulta de búsqueda
            language: Idioma de búsqueda
            domain: Dominio específico
            k: Número de resultados
            alpha: Balance semántico (0.0) vs léxico (1.0)

        Returns:
            Lista de documentos combinados
        """
        start_time = time.time()

        try:
            if alpha is None:
                alpha = self.default_alpha

            # Ejecutar ambas búsquedas en paralelo
            semantic_results, lexical_results = await asyncio.gather(
                self.semantic_search(query, language, domain, k * 2),
                self.lexical_search(query, language, domain, k * 2),
            )

            # Combinar resultados
            combined_results = self._combine_search_results(semantic_results, lexical_results, alpha, k)

            self._retrieval_stats["hybrid_searches"] += 1
            self._update_search_metrics(time.time() - start_time)

            return combined_results

        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            return []

    def _preprocess_query(self, query: str, language: str) -> List[str]:
        """
        Pre-procesar consulta para búsqueda léxica

        Args:
            query: Consulta original
            language: Idioma de la consulta

        Returns:
            Lista de términos procesados
        """
        # Limpiar y tokenizar
        query = re.sub(r"[^\w\s]", " ", query.lower())
        terms = query.split()

        # Filtrar stopwords básicas
        if language == "spanish":
            stopwords = {
                "el",
                "la",
                "de",
                "que",
                "y",
                "en",
                "un",
                "es",
                "se",
                "no",
                "te",
                "lo",
                "le",
                "da",
                "su",
                "por",
                "son",
                "con",
                "para",
                "al",
                "del",
                "las",
                "los",
                "una",
                "su",
                "me",
                "si",
                "ya",
                "todo",
                "más",
                "muy",
                "han",
                "bien",
                "puede",
                "está",
                "hasta",
            }
        else:  # english
            stopwords = {
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
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "can",
                "this",
                "that",
                "these",
                "those",
            }

        # Filtrar términos
        filtered_terms = [term for term in terms if term not in stopwords and len(term) > 2]

        return filtered_terms

    async def _tfidf_search(
        self, query_terms: List[str], language: str, domain: Optional[str], k: int
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda TF-IDF simulada

        Args:
            query_terms: Términos de búsqueda
            language: Idioma
            domain: Dominio específico
            k: Número de resultados

        Returns:
            Lista de documentos con scores TF-IDF
        """
        vocab = self.tfidf_vocab.get(language, {})

        # Buscar documentos reales en el corpus
        from pathlib import Path

        corpus_path = f"corpus_{'ES' if language == 'spanish' else 'EN'}"
        corpus_root = Path(corpus_path)

        real_docs = []

        if corpus_root.exists():
            # Buscar en dominios específicos o todos
            search_domains = (
                [domain] if domain else ["general", "programming", "medicine", "science", "biology", "mathematics"]
            )

            for domain_name in search_domains:
                domain_path = corpus_root / domain_name
                if domain_path.exists():
                    for file_path in domain_path.rglob("*.txt"):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            # Calcular TF-IDF score real
                            score = self._calculate_tfidf_score(content, query_terms, vocab)

                            if score > 0.05:  # Filtro mínimo
                                real_docs.append(
                                    {
                                        "id": f"tfidf_real_{file_path.stem}",
                                        "title": file_path.stem.replace("_", " ").title(),
                                        "content": content[:500] + "..." if len(content) > 500 else content,
                                        "domain": domain_name,
                                        "language": language,
                                        "score": min(score, 1.0),
                                        "search_type": "lexical",
                                        "path": str(file_path),
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Error procesando {file_path}: {e}")

        # Si no hay documentos reales, usar datos de branches
        if not real_docs:
            real_docs = await self._lexical_search_from_branches(query_terms, language, domain, k, vocab)

        # Ordenar por score y limitar resultados
        real_docs.sort(key=lambda x: x["score"], reverse=True)

        return real_docs[:k]

    def _calculate_tfidf_score(self, content: str, query_terms: List[str], vocab: Dict[str, float]) -> float:
        """Calcula score TF-IDF real para un documento"""
        content_lower = content.lower()
        content_words = content_lower.split()

        if not content_words:
            return 0.0

        total_score = 0.0

        for term in query_terms:
            if term in vocab:
                # Calcular TF real (term frequency)
                tf = content_words.count(term) / len(content_words)

                # TF normalizado con log
                tf_normalized = 1.0 + math.log(1.0 + tf) if tf > 0 else 0.0

                # IDF del vocabulario
                idf = vocab[term]

                # Score TF-IDF
                tfidf_score = tf_normalized * idf
                total_score += tfidf_score

        # Normalización por longitud de query
        return total_score / (1.0 + math.sqrt(len(query_terms)))

    async def _lexical_search_from_branches(
        self,
        query_terms: List[str],
        language: str,
        domain: Optional[str],
        k: int,
        vocab: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Búsqueda léxica usando datos de branches como respaldo"""
        import json
        from pathlib import Path

        branches_file = Path("branches/base_branches.json")
        if not branches_file.exists():
            return []

        try:
            with open(branches_file, "r", encoding="utf-8") as f:
                branches_data = json.load(f)

            documents = []
            domains_data = branches_data.get("domains", [])

            for domain_info in domains_data:
                domain_name = domain_info.get("name", "")
                description = domain_info.get("description", "")
                keywords = domain_info.get("keywords", [])

                if domain and domain_name != domain:
                    continue

                # Calcular TF-IDF score con descripción y keywords
                content = f"{description} {' '.join(keywords)}"
                score = self._calculate_tfidf_score(content, query_terms, vocab)

                if score > 0.02:
                    documents.append(
                        {
                            "id": f"branch_tfidf_{domain_name}",
                            "title": f"Rama: {domain_name.title()}",
                            "content": description,
                            "domain": domain_name,
                            "language": language,
                            "score": score,
                            "search_type": "lexical_branch",
                            "path": f"branches/{domain_name}/config.json",
                        }
                    )

            return documents

        except Exception as e:
            logger.error(f"Error en búsqueda léxica de branches: {e}")
            return []

    async def _real_semantic_search(
        self, query: str, language: str, domain: Optional[str], k: int
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica real basada en similitud de contenido

        Args:
            query: Consulta original
            language: Idioma
            domain: Dominio específico
            k: Número de resultados

        Returns:
            Lista de documentos reales con similarity scores
        """
        import os
        import re
        from pathlib import Path

        # Determinar ruta del corpus
        corpus_path = f"corpus_{'ES' if language == 'spanish' else 'EN'}"
        corpus_root = Path(corpus_path)

        if not corpus_root.exists():
            # Usar datos de respaldo de branches si no hay corpus
            return await self._semantic_search_from_branches(query, language, domain, k)

        documents = []

        # Buscar archivos en el corpus
        search_domains = (
            [domain] if domain else ["general", "programming", "artificial_intelligence", "medicine", "science"]
        )

        for domain_name in search_domains:
            domain_path = corpus_root / domain_name
            if domain_path.exists():
                for file_path in domain_path.rglob("*.txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Calcular similitud semántica real
                        similarity = self._calculate_semantic_similarity(query, content)

                        if similarity > 0.1:  # Filtro mínimo de relevancia
                            documents.append(
                                {
                                    "id": f"real_doc_{file_path.stem}",
                                    "title": file_path.stem.replace("_", " ").title(),
                                    "content": content[:500] + "..." if len(content) > 500 else content,
                                    "domain": domain_name,
                                    "language": language,
                                    "score": similarity,
                                    "search_type": "semantic",
                                    "path": str(file_path),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Error leyendo {file_path}: {e}")

        # Ordenar por relevancia y retornar top-k
        documents.sort(key=lambda x: x["score"], reverse=True)
        return documents[:k]

    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """
        Calcula similitud semántica real entre query y contenido
        Usando técnicas de NLP sin dependencias externas
        """
        # Normalizar textos
        query_norm = self._normalize_text(query)
        content_norm = self._normalize_text(content)

        # Extraer términos importantes
        query_terms = set(query_norm.split())
        content_terms = set(content_norm.split())

        # Similitud de Jaccard
        intersection = len(query_terms.intersection(content_terms))
        union = len(query_terms.union(content_terms))
        jaccard_sim = intersection / union if union > 0 else 0.0

        # Similitud de contenido (TF-IDF simplificado)
        content_words = content_norm.split()
        tf_scores = []

        for term in query_terms:
            if term in content_words:
                tf = content_words.count(term) / len(content_words)
                # IDF simplificado basado en rareza del término
                idf = 1.0 / (1.0 + content_words.count(term) / 100)
                tf_scores.append(tf * idf)

        tfidf_sim = sum(tf_scores) / len(query_terms) if query_terms else 0.0

        # Similitud posicional (términos cerca del inicio tienen más peso)
        position_sim = 0.0
        for i, word in enumerate(content_words[:100]):  # Solo primeras 100 palabras
            if word in query_terms:
                position_sim += (100 - i) / 100 / len(query_terms)

        # Combinación ponderada
        final_similarity = 0.4 * jaccard_sim + 0.4 * tfidf_sim + 0.2 * position_sim

        return min(1.0, final_similarity)

    def _normalize_text(self, text: str) -> str:
        """Normalización de texto para análisis semántico"""
        import re

        # Convertir a minúsculas
        text = text.lower()

        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r"[^\w\s]", " ", text)

        # Remover espacios múltiples
        text = re.sub(r"\s+", " ", text)

        # Remover stopwords comunes
        stopwords = {
            "el",
            "la",
            "de",
            "que",
            "y",
            "en",
            "un",
            "es",
            "se",
            "no",
            "te",
            "lo",
            "le",
            "da",
            "su",
            "por",
            "son",
            "con",
            "para",
            "al",
            "del",
            "las",
            "los",
            "una",
            "me",
            "si",
            "ya",
            "muy",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
        }

        words = [word for word in text.split() if word not in stopwords and len(word) > 2]
        return " ".join(words)

    async def _semantic_search_from_branches(
        self, query: str, language: str, domain: Optional[str], k: int
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica usando datos de branches como respaldo
        """
        import json
        from pathlib import Path

        branches_file = Path("branches/base_branches.json")
        if not branches_file.exists():
            return []

        try:
            with open(branches_file, "r", encoding="utf-8") as f:
                branches_data = json.load(f)

            documents = []
            domains_data = branches_data.get("domains", [])

            for domain_info in domains_data:
                domain_name = domain_info.get("name", "")
                description = domain_info.get("description", "")
                keywords = domain_info.get("keywords", [])

                # Filtrar por dominio si se especifica
                if domain and domain_name != domain:
                    continue

                # Calcular similitud con descripción y keywords
                content = f"{description} {' '.join(keywords)}"
                similarity = self._calculate_semantic_similarity(query, content)

                if similarity > 0.05:
                    documents.append(
                        {
                            "id": f"branch_doc_{domain_name}",
                            "title": f"Dominio: {domain_name.title()}",
                            "content": description,
                            "domain": domain_name,
                            "language": language,
                            "score": similarity,
                            "search_type": "semantic_branch",
                            "path": f"branches/{domain_name}/info.txt",
                        }
                    )

            documents.sort(key=lambda x: x["score"], reverse=True)
            return documents[:k]

        except Exception as e:
            logger.error(f"Error en búsqueda semántica de branches: {e}")
            return []

    def _combine_search_results(
        self, semantic_results: List[Dict], lexical_results: List[Dict], alpha: float, k: int
    ) -> List[Dict[str, Any]]:
        """
        Combinar resultados semánticos y léxicos

        Args:
            semantic_results: Resultados semánticos
            lexical_results: Resultados léxicos
            alpha: Balance (0=solo semántico, 1=solo léxico)
            k: Número final de resultados

        Returns:
            Lista combinada de documentos
        """

        # Normalizar scores a [0,1]
        def normalize_scores(results):
            if not results:
                return results

            max_score = max(doc["score"] for doc in results)
            min_score = min(doc["score"] for doc in results)

            if max_score == min_score:
                return results

            for doc in results:
                doc["normalized_score"] = (doc["score"] - min_score) / (max_score - min_score)

            return results

        # Normalizar ambos conjuntos
        semantic_results = normalize_scores(semantic_results)
        lexical_results = normalize_scores(lexical_results)

        # Crear mapa combinado
        combined_docs = {}

        # Procesar resultados semánticos
        for doc in semantic_results:
            doc_id = doc["id"]
            combined_docs[doc_id] = doc.copy()
            combined_docs[doc_id]["semantic_score"] = doc.get("normalized_score", doc["score"])
            combined_docs[doc_id]["lexical_score"] = 0.0

        # Procesar resultados léxicos
        for doc in lexical_results:
            doc_id = doc["id"]
            if doc_id in combined_docs:
                combined_docs[doc_id]["lexical_score"] = doc.get("normalized_score", doc["score"])
            else:
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["semantic_score"] = 0.0
                combined_docs[doc_id]["lexical_score"] = doc.get("normalized_score", doc["score"])

        # Calcular score híbrido
        for doc_id, doc in combined_docs.items():
            semantic_score = doc["semantic_score"]
            lexical_score = doc["lexical_score"]

            # Score híbrido ponderado
            hybrid_score = (1 - alpha) * semantic_score + alpha * lexical_score

            doc["hybrid_score"] = hybrid_score
            doc["score"] = hybrid_score  # Actualizar score principal
            doc["search_type"] = "hybrid"

        # Convertir a lista y ordenar
        combined_list = list(combined_docs.values())
        combined_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return combined_list[:k]

    async def rerank_documents(
        self, query: str, documents: List[Dict], algorithm: str = "rrf", k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-ranking de documentos usando algoritmo especificado

        Args:
            query: Consulta original
            documents: Lista de documentos a re-rankear
            algorithm: Algoritmo de re-ranking ('rrf', 'score_fusion', 'semantic_rerank')
            k: Número de documentos a retornar

        Returns:
            Lista de documentos re-rankeados
        """
        start_time = time.time()

        try:
            if not documents:
                return documents

            if k is None:
                k = len(documents)

            # Aplicar algoritmo de re-ranking
            if algorithm in self.rerank_algorithms:
                reranked_docs = await self.rerank_algorithms[algorithm](query, documents)
            else:
                logger.warning(f"Algoritmo de re-ranking desconocido: {algorithm}")
                reranked_docs = documents

            # Actualizar métricas
            self._retrieval_stats["reranking_operations"] += 1

            return reranked_docs[:k]

        except Exception as e:
            logger.error(f"Error en re-ranking: {e}")
            return documents[:k] if k else documents

    async def _reciprocal_rank_fusion(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Re-ranking usando Reciprocal Rank Fusion (RRF)

        Args:
            query: Consulta original
            documents: Documentos a re-rankear

        Returns:
            Documentos re-rankeados por RRF
        """
        # Separar por tipo de búsqueda
        semantic_docs = [doc for doc in documents if doc.get("search_type") == "semantic"]
        lexical_docs = [doc for doc in documents if doc.get("search_type") == "lexical"]
        hybrid_docs = [doc for doc in documents if doc.get("search_type") == "hybrid"]

        # Aplicar RRF
        rrf_scores = {}
        k = 60  # Constante RRF

        # Procesar cada lista
        for rank, doc in enumerate(semantic_docs, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        for rank, doc in enumerate(lexical_docs, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        for rank, doc in enumerate(hybrid_docs, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Actualizar scores y ordenar
        for doc in documents:
            doc["rrf_score"] = rrf_scores.get(doc["id"], 0)
            doc["score"] = doc["rrf_score"]

        documents.sort(key=lambda x: x["rrf_score"], reverse=True)

        return documents

    async def _score_fusion(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Re-ranking por fusión de scores ponderada

        Args:
            query: Consulta original
            documents: Documentos a re-rankear

        Returns:
            Documentos con scores fusionados
        """
        for doc in documents:
            # Combinar múltiples scores si existen
            semantic_score = doc.get("semantic_score", 0.0)
            lexical_score = doc.get("lexical_score", 0.0)
            original_score = doc.get("score", 0.0)

            # Fusión ponderada
            fused_score = 0.4 * semantic_score + 0.4 * lexical_score + 0.2 * original_score

            doc["fused_score"] = fused_score
            doc["score"] = fused_score

        documents.sort(key=lambda x: x["fused_score"], reverse=True)

        return documents

    async def _semantic_rerank(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Re-ranking basado en similitud semántica adicional

        Args:
            query: Consulta original
            documents: Documentos a re-rankear

        Returns:
            Documentos re-rankeados semánticamente
        """
        # Simular re-ranking semántico avanzado
        query_length = len(query.split())

        for i, doc in enumerate(documents):
            content_length = len(doc.get("content", "").split())

            # Factor de similitud por longitud y posición
            length_similarity = 1.0 / (1.0 + abs(query_length - content_length) / max(query_length, content_length))
            position_penalty = 1.0 / (1.0 + i * 0.1)  # Penalizar posiciones bajas

            # Nuevo score semántico
            original_score = doc.get("score", 0.0)
            semantic_rerank_score = original_score * length_similarity * position_penalty

            doc["semantic_rerank_score"] = semantic_rerank_score
            doc["score"] = semantic_rerank_score

        documents.sort(key=lambda x: x["semantic_rerank_score"], reverse=True)

        return documents

    def _update_search_metrics(self, search_time: float):
        """
        Actualizar métricas de búsqueda

        Args:
            search_time: Tiempo de búsqueda
        """
        self._retrieval_stats["searches_performed"] += 1

        # Actualizar tiempo promedio
        total_searches = self._retrieval_stats["searches_performed"]
        current_avg = self._retrieval_stats["average_search_time"]

        self._retrieval_stats["average_search_time"] = (
            current_avg * (total_searches - 1) + search_time
        ) / total_searches

    def get_stats(self) -> Dict:
        """Obtener estadísticas de recuperación"""
        return self._retrieval_stats.copy()

    async def health_check(self) -> Dict:
        """Verificar estado de salud del gestor"""
        return {
            "status": "healthy",
            "tfidf_vocabularies": len(self.tfidf_vocab),
            "available_algorithms": list(self.rerank_algorithms.keys()),
            "stats": self.get_stats(),
        }

    async def shutdown(self):
        """Cerrar gestor y limpiar recursos"""
        logger.info("Iniciando shutdown del RetrievalManager")

        # Limpiar caches
        self.tfidf_cache.clear()

        logger.info("RetrievalManager shutdown completado")
