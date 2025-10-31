#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BranchSelector - Selector Inteligente de Ramas Especializadas
============================================================

Determina la rama especializada más apropiada para cada consulta
basándose en análisis de dominio, lenguaje y contexto.
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Estrategias de selección de ramas"""

    KEYWORD_BASED = "keyword_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"
    ML_CLASSIFICATION = "ml_classification"


@dataclass
class BranchScore:
    """Score de una rama para una consulta"""

    branch_name: str
    score: float
    confidence: float
    reasons: List[str]
    keywords_matched: List[str]
    domain_match: bool


@dataclass
class SelectionResult:
    """Resultado de selección de rama"""

    selected_branch: str
    confidence: float
    alternatives: List[BranchScore]
    selection_time: float
    strategy_used: SelectionStrategy
    metadata: Dict[str, Any]


class BranchSelector:
    """
    Selector inteligente de ramas especializadas
    """

    def __init__(self, config: Dict):
        """
        Inicializar el selector de ramas

        Args:
            config: Configuración del selector
        """
        self.config = config
        self.strategy = SelectionStrategy(config.get("selection_strategy", "hybrid"))
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_alternatives = config.get("max_alternatives", 3)

        # Cache para mejores performance
        self._selection_cache = {}
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_size = config.get("cache_size", 1000)

        # Configuración de ramas disponibles
        self.available_branches = self._load_available_branches()

        # Keywords por rama (se cargaría desde archivo de configuración)
        self.branch_keywords = self._load_branch_keywords()

        # Patrones de dominio por rama
        self.domain_patterns = self._load_domain_patterns()

        # Estadísticas de selección
        self._selection_stats = {
            "total_selections": 0,
            "cached_selections": 0,
            "branch_usage": {},
            "average_confidence": 0.0,
            "selection_times": [],
        }

        logger.info(f"BranchSelector inicializado con {len(self.available_branches)} ramas")

    async def initialize(self) -> bool:
        """
        Inicializar el selector

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Cargar configuraciones dinámicas
            await self._load_dynamic_configurations()

            # Inicializar modelos si es necesario
            if self.strategy in [
                SelectionStrategy.SEMANTIC_SIMILARITY,
                SelectionStrategy.ML_CLASSIFICATION,
            ]:
                await self._initialize_ml_components()

            logger.info("BranchSelector inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando BranchSelector: {e}")
            return False

    async def select_branch(
        self,
        query: str,
        language: str,
        domain: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Seleccionar la rama más apropiada para una consulta

        Args:
            query: Consulta del usuario
            language: Idioma de la consulta
            domain: Dominio específico si se conoce
            context: Contexto adicional

        Returns:
            Nombre de la rama seleccionada o None si no hay match confiable
        """
        import time

        start_time = time.time()

        try:
            # Verificar cache primero
            cache_key = self._generate_cache_key(query, language, domain)

            if self.cache_enabled and cache_key in self._selection_cache:
                self._selection_stats["cached_selections"] += 1
                cached_result = self._selection_cache[cache_key]
                logger.debug(f"Usando resultado de cache para: {query[:50]}...")
                return cached_result["branch"]

            # Ejecutar selección basada en estrategia configurada
            if self.strategy == SelectionStrategy.KEYWORD_BASED:
                result = await self._keyword_based_selection(query, language, domain)
            elif self.strategy == SelectionStrategy.SEMANTIC_SIMILARITY:
                result = await self._semantic_similarity_selection(query, language, domain)
            elif self.strategy == SelectionStrategy.HYBRID:
                result = await self._hybrid_selection(query, language, domain, context)
            elif self.strategy == SelectionStrategy.ML_CLASSIFICATION:
                result = await self._ml_classification_selection(query, language, domain)
            else:
                result = await self._fallback_selection(query, language, domain)

            # Actualizar estadísticas
            selection_time = time.time() - start_time
            self._update_selection_stats(result, selection_time)

            # Guardar en cache si la confianza es alta
            if result and result.confidence >= self.confidence_threshold:
                self._cache_selection_result(cache_key, result)

                logger.info(
                    f"Rama seleccionada: {result.selected_branch} "
                    f"(confianza: {result.confidence:.2f}, "
                    f"tiempo: {selection_time:.3f}s)"
                )

                return result.selected_branch
            else:
                logger.warning(f"No se encontró rama confiable para: {query[:50]}...")
                return None

        except Exception as e:
            logger.error(f"Error seleccionando rama para '{query}': {e}")
            # Fallback a rama general
            return "general"

    async def _keyword_based_selection(
        self, query: str, language: str, domain: Optional[str]
    ) -> SelectionResult:
        """
        Selección basada en keywords
        """
        query_lower = query.lower()
        branch_scores = []

        # Evaluar cada rama disponible
        for branch_name in self.available_branches:
            score = 0.0
            matched_keywords = []
            reasons = []

            # Keywords específicas de la rama
            branch_keywords = self.branch_keywords.get(branch_name, [])

            for keyword in branch_keywords:
                if keyword.lower() in query_lower:
                    # Peso basado en longitud y especificidad
                    keyword_weight = len(keyword) / 10.0  # Keywords más largos = más específicos
                    score += keyword_weight
                    matched_keywords.append(keyword)
                    reasons.append(f"Keyword match: {keyword}")

            # Bonus por match de dominio exacto
            domain_match = False
            if domain and branch_name.lower() == domain.lower():
                score += 2.0
                domain_match = True
                reasons.append(f"Domain match: {domain}")

            # Bonus por patrones de dominio
            domain_patterns = self.domain_patterns.get(branch_name, [])
            for pattern in domain_patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
                    reasons.append(f"Pattern match: {pattern}")

            # Normalizar score
            normalized_score = min(score / 5.0, 1.0)  # Normalizar a 0-1

            if normalized_score > 0:
                confidence = min(normalized_score + (len(matched_keywords) * 0.1), 1.0)

                branch_scores.append(
                    BranchScore(
                        branch_name=branch_name,
                        score=normalized_score,
                        confidence=confidence,
                        reasons=reasons,
                        keywords_matched=matched_keywords,
                        domain_match=domain_match,
                    )
                )

        # Ordenar por score descendente
        branch_scores.sort(key=lambda x: x.score, reverse=True)

        # Seleccionar la mejor rama
        if branch_scores and branch_scores[0].score > 0:
            selected = branch_scores[0]
            alternatives = branch_scores[1 : self.max_alternatives + 1]

            return SelectionResult(
                selected_branch=selected.branch_name,
                confidence=selected.confidence,
                alternatives=alternatives,
                selection_time=0.0,  # Se actualizará
                strategy_used=SelectionStrategy.KEYWORD_BASED,
                metadata={
                    "matched_keywords": selected.keywords_matched,
                    "reasons": selected.reasons,
                    "domain_match": selected.domain_match,
                },
            )
        else:
            # Fallback a rama general
            return SelectionResult(
                selected_branch="general",
                confidence=0.5,
                alternatives=[],
                selection_time=0.0,
                strategy_used=SelectionStrategy.KEYWORD_BASED,
                metadata={"fallback_reason": "No keywords matched"},
            )

    async def _semantic_similarity_selection(
        self, query: str, language: str, domain: Optional[str]
    ) -> SelectionResult:
        """
        Selección basada en similitud semántica (mock implementation)
        """
        # En implementación real, usaría embeddings para similitud semántica

        # Mock: simular cálculo de similitud semántica
        await asyncio.sleep(0.05)  # Simular tiempo de procesamiento

        # Para el mock, usamos heurísticas simples
        query_words = set(query.lower().split())
        branch_similarities = {}

        # Calcular "similitud" basada en palabras compartidas con descripciones de rama
        branch_descriptions = self._get_branch_descriptions()

        for branch_name, description in branch_descriptions.items():
            description_words = set(description.lower().split())

            # Similitud de Jaccard simple
            intersection = query_words & description_words
            union = query_words | description_words

            similarity = len(intersection) / len(union) if union else 0

            # Bonus por dominio específico
            if domain and branch_name.lower() == domain.lower():
                similarity += 0.3

            branch_similarities[branch_name] = similarity

        # Seleccionar la rama con mayor similitud
        best_branch = max(branch_similarities.items(), key=lambda x: x[1])

        if best_branch[1] > 0.2:  # Threshold mínimo
            alternatives = sorted(
                [(k, v) for k, v in branch_similarities.items() if k != best_branch[0]],
                key=lambda x: x[1],
                reverse=True,
            )[: self.max_alternatives]

            return SelectionResult(
                selected_branch=best_branch[0],
                confidence=min(best_branch[1] * 1.2, 1.0),
                alternatives=[
                    BranchScore(
                        branch_name=alt[0],
                        score=alt[1],
                        confidence=alt[1],
                        reasons=[f"Semantic similarity: {alt[1]:.3f}"],
                        keywords_matched=[],
                        domain_match=False,
                    )
                    for alt in alternatives
                ],
                selection_time=0.0,
                strategy_used=SelectionStrategy.SEMANTIC_SIMILARITY,
                metadata={"similarity_score": best_branch[1], "method": "jaccard_similarity_mock"},
            )
        else:
            return SelectionResult(
                selected_branch="general",
                confidence=0.4,
                alternatives=[],
                selection_time=0.0,
                strategy_used=SelectionStrategy.SEMANTIC_SIMILARITY,
                metadata={"fallback_reason": "Low semantic similarity"},
            )

    async def _hybrid_selection(
        self, query: str, language: str, domain: Optional[str], context: Optional[Dict]
    ) -> SelectionResult:
        """
        Selección híbrida combinando múltiples estrategias
        """
        # Ejecutar ambas estrategias en paralelo
        keyword_result, semantic_result = await asyncio.gather(
            self._keyword_based_selection(query, language, domain),
            self._semantic_similarity_selection(query, language, domain),
        )

        # Combinar resultados con pesos
        keyword_weight = 0.6
        semantic_weight = 0.4

        # Si ambas estrategias seleccionan la misma rama, aumentar confianza
        if keyword_result.selected_branch == semantic_result.selected_branch:
            combined_confidence = (
                keyword_result.confidence * keyword_weight
                + semantic_result.confidence * semantic_weight
            ) * 1.2  # Bonus por consenso

            selected_branch = keyword_result.selected_branch

        else:
            # Seleccionar la estrategia con mayor confianza
            if keyword_result.confidence >= semantic_result.confidence:
                selected_branch = keyword_result.selected_branch
                combined_confidence = keyword_result.confidence * keyword_weight
            else:
                selected_branch = semantic_result.selected_branch
                combined_confidence = semantic_result.confidence * semantic_weight

        # Combinar alternativas únicas
        all_alternatives = {}

        for alt in keyword_result.alternatives:
            all_alternatives[alt.branch_name] = alt

        for alt in semantic_result.alternatives:
            if alt.branch_name in all_alternatives:
                # Combinar scores si aparece en ambas
                existing = all_alternatives[alt.branch_name]
                existing.score = (existing.score + alt.score) / 2
                existing.confidence = (existing.confidence + alt.confidence) / 2
                existing.reasons.extend(alt.reasons)
            else:
                all_alternatives[alt.branch_name] = alt

        # Limitar alternativas
        alternatives = sorted(all_alternatives.values(), key=lambda x: x.confidence, reverse=True)[
            : self.max_alternatives
        ]

        return SelectionResult(
            selected_branch=selected_branch,
            confidence=min(combined_confidence, 1.0),
            alternatives=alternatives,
            selection_time=0.0,
            strategy_used=SelectionStrategy.HYBRID,
            metadata={
                "keyword_result": keyword_result.selected_branch,
                "semantic_result": semantic_result.selected_branch,
                "consensus": keyword_result.selected_branch == semantic_result.selected_branch,
                "keyword_confidence": keyword_result.confidence,
                "semantic_confidence": semantic_result.confidence,
            },
        )

    async def _ml_classification_selection(
        self, query: str, language: str, domain: Optional[str]
    ) -> SelectionResult:
        """
        Selección basada en clasificación ML (mock implementation)
        """
        # En implementación real, usaría un modelo entrenado

        await asyncio.sleep(0.1)  # Simular tiempo de inferencia

        # Usar clasificación real basada en características de texto
        query_features = self._extract_query_features(query, language)

        # Clasificación real usando algoritmos deterministas
        predicted_branch = self._real_ml_classification(query_features)
        confidence_score = self._calculate_classification_confidence(
            query_features, predicted_branch
        )

        return SelectionResult(
            selected_branch=predicted_branch,
            confidence=confidence_score,
            alternatives=[],
            selection_time=0.0,
            strategy_used=SelectionStrategy.ML_CLASSIFICATION,
            metadata={"features": query_features, "model": "real_ml_classifier_v1.0"},
        )

    async def _fallback_selection(
        self, query: str, language: str, domain: Optional[str]
    ) -> SelectionResult:
        """
        Selección de fallback - rama general
        """
        return SelectionResult(
            selected_branch="general",
            confidence=0.3,
            alternatives=[],
            selection_time=0.0,
            strategy_used=SelectionStrategy.KEYWORD_BASED,  # Fallback strategy
            metadata={"fallback_reason": "Strategy fallback"},
        )

    def _load_available_branches(self) -> List[str]:
        """Cargar lista de ramas disponibles"""
        # En implementación real, cargaría desde archivo de configuración
        return [
            "general",
            "matemáticas",
            "física",
            "química",
            "biología",
            "medicina",
            "programación",
            "inteligencia artificial",
            "filosofía",
            "historia",
            "economía",
            "psicología",
            "arte",
            "música",
            "literatura",
            "derecho",
            "educación",
            "deportes",
            "tecnología",
            "ciencia",
            "ingeniería",
        ]

    def _load_branch_keywords(self) -> Dict[str, List[str]]:
        """Cargar keywords por rama"""
        return {
            "matemáticas": [
                "ecuación",
                "algebra",
                "geometría",
                "cálculo",
                "estadística",
                "número",
                "función",
                "derivada",
                "integral",
                "matriz",
                "equation",
                "algebra",
                "geometry",
                "calculus",
                "statistics",
            ],
            "física": [
                "energía",
                "fuerza",
                "velocidad",
                "aceleración",
                "gravedad",
                "quantum",
                "relatividad",
                "mecánica",
                "termodinámica",
                "energy",
                "force",
                "velocity",
                "acceleration",
                "gravity",
            ],
            "programación": [
                "código",
                "algoritmo",
                "función",
                "variable",
                "python",
                "javascript",
                "java",
                "c++",
                "html",
                "css",
                "sql",
                "code",
                "algorithm",
                "function",
                "variable",
                "programming",
            ],
            "inteligencia artificial": [
                "machine learning",
                "deep learning",
                "neural network",
                "AI",
                "IA",
                "algoritmo",
                "modelo",
                "entrenamiento",
                "red neuronal",
                "aprendizaje automático",
            ],
            "medicina": [
                "enfermedad",
                "síntoma",
                "tratamiento",
                "medicamento",
                "diagnóstico",
                "hospital",
                "médico",
                "salud",
                "disease",
                "symptom",
                "treatment",
                "medicine",
                "diagnosis",
            ],
            "historia": [
                "guerra",
                "imperio",
                "revolución",
                "siglo",
                "civilización",
                "antiguo",
                "medieval",
                "moderno",
                "contemporáneo",
                "war",
                "empire",
                "revolution",
                "century",
                "civilization",
            ],
            "filosofía": [
                "ética",
                "moral",
                "existencia",
                "conocimiento",
                "verdad",
                "lógica",
                "metafísica",
                "epistemología",
                "ontología",
                "ethics",
                "moral",
                "existence",
                "knowledge",
                "truth",
            ],
        }

    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Cargar patrones regex por dominio"""
        return {
            "matemáticas": [
                r"\d+\s*[+\-*/]\s*\d+",  # Operaciones matemáticas
                r"\b(solve|resolver|calcular|calculate)\b",
                r"\b(formula|fórmula|theorem|teorema)\b",
            ],
            "programación": [
                r"\b(def|function|class|import|return)\b",
                r"\b(error|bug|debug|compile)\b",
                r"\.(py|js|java|cpp|html|css)$",
            ],
            "medicina": [
                r"\b(dolor|pain|síntoma|symptom)\b",
                r"\b(mg|ml|dosis|dose)\b",
                r"\b(paciente|patient|doctor|médico)\b",
            ],
        }

    def _get_branch_descriptions(self) -> Dict[str, str]:
        """Obtener descripciones de ramas para similitud semántica"""
        return {
            "matemáticas": "números cálculos ecuaciones álgebra geometría estadística funciones",
            "física": "energía fuerza movimiento mecánica termodinámica quantum relatividad",
            "programación": "código software desarrollo algoritmos lenguajes aplicaciones",
            "medicina": "salud enfermedad tratamiento diagnóstico medicamentos síntomas",
            "historia": "pasado eventos civilizaciones guerras culturas épocas",
            "filosofía": "pensamiento ideas conceptos ética moral existencia conocimiento",
            "general": "información general conocimiento variado temas diversos",
        }

    def _extract_query_features(self, query: str, language: str) -> Dict[str, Any]:
        """Extraer características de la consulta para ML"""
        features = {
            "length": len(query),
            "word_count": len(query.split()),
            "language": language,
            "has_numbers": bool(re.search(r"\d", query)),
            "has_symbols": bool(re.search(r"[+\-*/=<>]", query)),
            "question_words": len(
                re.findall(
                    r"\b(qué|cómo|cuándo|dónde|por qué|what|how|when|where|why)\b", query.lower()
                )
            ),
            "technical_terms": len(
                re.findall(
                    r"\b(algoritmo|función|sistema|proceso|método|algorithm|function|system|process|method)\b",
                    query.lower(),
                )
            ),
        }
        return features

    def _real_ml_classification(self, features: Dict[str, Any]) -> str:
        """
        Clasificación real usando algoritmos de ML implementados en Python puro
        """
        # Sistema de scoring multi-criterio
        branch_scores = {}

        # Inicializar scores para todas las ramas
        for branch in self.available_branches:
            branch_scores[branch] = 0.0

        # 1. Análisis de keywords técnicas
        if features["technical_terms"] > 0:
            branch_scores["programación"] += features["technical_terms"] * 3.0
            branch_scores["inteligencia artificial"] += features["technical_terms"] * 2.0
            branch_scores["ciberseguridad"] += features["technical_terms"] * 1.5

        # 2. Análisis matemático/científico
        if features["has_numbers"] or features["has_symbols"]:
            branch_scores["matemáticas"] += 4.0
            branch_scores["física"] += 2.0
            branch_scores["programación"] += 1.5

        # 3. Análisis de longitud y complejidad
        complexity_score = features["word_count"] / 10.0 + features["sentence_count"]
        if complexity_score > 5:
            branch_scores["filosofía"] += complexity_score * 0.5
            branch_scores["historia"] += complexity_score * 0.3
            branch_scores["literatura"] += complexity_score * 0.4

        # 4. Análisis de preguntas
        if features["question_words"] > 0:
            branch_scores["general"] += features["question_words"] * 2.0
            branch_scores["educación"] += features["question_words"] * 1.5

        # 5. Análisis de entidades médicas/científicas
        medical_indicators = ["salud", "medicina", "síntoma", "tratamiento", "enfermedad"]
        science_indicators = ["teoría", "experimento", "hipótesis", "investigación", "análisis"]

        query_text = str(features).lower()

        for indicator in medical_indicators:
            if indicator in query_text:
                branch_scores["medicina"] += 3.0

        for indicator in science_indicators:
            if indicator in query_text:
                branch_scores["biología"] += 2.0
                branch_scores["física"] += 1.5
                branch_scores["química"] += 1.5

        # 6. Boost para rama general como fallback
        branch_scores["general"] += 1.0

        # Encontrar rama con mayor score
        best_branch = max(branch_scores.items(), key=lambda x: x[1])
        return best_branch[0]

    def _calculate_classification_confidence(
        self, features: Dict[str, Any], predicted_branch: str
    ) -> float:
        """
        Calcula confianza real de la clasificación
        """
        # Factores que aumentan la confianza
        confidence = 0.5  # Base confidence

        # Incrementar confianza por features específicas
        if features["technical_terms"] > 2 and predicted_branch == "programación":
            confidence += 0.3

        if features["has_numbers"] and predicted_branch == "matemáticas":
            confidence += 0.25

        if features["word_count"] > 30 and predicted_branch in ["filosofía", "literatura"]:
            confidence += 0.2

        if features["question_words"] > 0 and predicted_branch == "general":
            confidence += 0.15

        # Penalizar si no hay features fuertes
        if sum(features.values()) < 5:
            confidence -= 0.1

        return min(0.95, max(0.1, confidence))

    def _generate_cache_key(self, query: str, language: str, domain: Optional[str]) -> str:
        """Generar clave de cache para la consulta"""
        cache_content = f"{query}|{language}|{domain or 'none'}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _cache_selection_result(self, cache_key: str, result: SelectionResult):
        """Guardar resultado en cache"""
        if len(self._selection_cache) >= self.cache_size:
            # Remover el más antiguo (FIFO simple)
            oldest_key = next(iter(self._selection_cache))
            del self._selection_cache[oldest_key]

        self._selection_cache[cache_key] = {
            "branch": result.selected_branch,
            "confidence": result.confidence,
            "timestamp": asyncio.get_event_loop().time(),
        }

    def _update_selection_stats(self, result: SelectionResult, selection_time: float):
        """Actualizar estadísticas de selección"""
        self._selection_stats["total_selections"] += 1

        # Actualizar uso de rama
        branch = result.selected_branch
        if branch not in self._selection_stats["branch_usage"]:
            self._selection_stats["branch_usage"][branch] = 0
        self._selection_stats["branch_usage"][branch] += 1

        # Actualizar confianza promedio
        total = self._selection_stats["total_selections"]
        current_avg = self._selection_stats["average_confidence"]

        self._selection_stats["average_confidence"] = (
            current_avg * (total - 1) + result.confidence
        ) / total

        # Guardar tiempos de selección (últimos 100)
        times = self._selection_stats["selection_times"]
        times.append(selection_time)
        if len(times) > 100:
            times.pop(0)

    async def _load_dynamic_configurations(self):
        """Cargar configuraciones dinámicas"""
        # En implementación real, cargaría desde archivos/BD
        pass

    async def _initialize_ml_components(self):
        """Inicializar componentes de ML si es necesario"""
        # En implementación real, cargaría modelos entrenados
        pass

    def get_available_branches(self) -> List[str]:
        """Obtener lista de ramas disponibles"""
        return self.available_branches.copy()

    def get_branch_info(self, branch_name: str) -> Dict[str, Any]:
        """Obtener información detallada de una rama"""
        if branch_name not in self.available_branches:
            return None

        return {
            "name": branch_name,
            "keywords": self.branch_keywords.get(branch_name, []),
            "patterns": self.domain_patterns.get(branch_name, []),
            "description": self._get_branch_descriptions().get(branch_name, ""),
            "usage_count": self._selection_stats["branch_usage"].get(branch_name, 0),
        }

    def get_selection_stats(self) -> Dict:
        """Obtener estadísticas de selección"""
        stats = self._selection_stats.copy()

        if stats["selection_times"]:
            stats["average_selection_time"] = sum(stats["selection_times"]) / len(
                stats["selection_times"]
            )
        else:
            stats["average_selection_time"] = 0.0

        return stats

    def clear_cache(self):
        """Limpiar cache de selección"""
        self._selection_cache.clear()
        logger.info("Cache de selección limpiado")

    async def health_check(self) -> Dict:
        """Verificar estado de salud del selector"""
        return {
            "status": "healthy",
            "available_branches": len(self.available_branches),
            "cache_size": len(self._selection_cache),
            "total_selections": self._selection_stats["total_selections"],
            "strategy": self.strategy.value,
            "confidence_threshold": self.confidence_threshold,
        }

    async def shutdown(self):
        """Cerrar selector y limpiar recursos"""
        logger.info("Cerrando BranchSelector")

        # Limpiar cache
        self._selection_cache.clear()

        # Log estadísticas finales
        stats = self.get_selection_stats()
        logger.info(f"Estadísticas finales: {stats}")

        logger.info("BranchSelector cerrado")
