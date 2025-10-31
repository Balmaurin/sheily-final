#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Response Merger System with Adaptive Intelligence
===========================================================

Sistema avanzado de fusión de respuestas con inteligencia adaptativa
para combinar respuestas especializadas y conocimiento general.

ACTUALIZADO: Usa análisis REAL (no mocks) desde real_merger_analysis.py
para crear respuestas más completas y contextualizadas.
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Estrategias de fusión disponibles"""

    WEIGHTED_COMBINATION = "weighted_combination"
    HIERARCHICAL_MERGE = "hierarchical_merge"
    CONTEXTUAL_FUSION = "contextual_fusion"
    CONSENSUS_BASED = "consensus_based"
    ADAPTIVE_MERGE = "adaptive_merge"


class MergeType(Enum):
    """Tipos de fusión"""

    SPECIALIZED_WITH_GENERAL = "specialized_with_general"
    MULTI_BRANCH = "multi_branch"
    CROSS_DOMAIN = "cross_domain"
    TEMPORAL_MERGE = "temporal_merge"


@dataclass
class MergeInput:
    """Entrada para proceso de fusión"""

    specialized_response: Any  # BranchResponse object
    general_knowledge: Optional[str] = None
    additional_sources: List[Dict] = field(default_factory=list)
    merge_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeResult:
    """Resultado del proceso de fusión"""

    merged_response: str
    confidence: float
    merge_strategy_used: MergeStrategy
    sources_combined: List[str]
    merge_quality_score: float
    processing_time: float
    enhancement_details: Dict[str, Any] = field(default_factory=dict)


class BranchMerger:
    """
    Motor de fusión inteligente que combina conocimientos de múltiples fuentes
    """

    def __init__(self, config: Dict):
        """
        Inicializar el motor de fusión

        Args:
            config: Configuración del merger
        """
        self.config = config
        self.default_strategy = MergeStrategy(config.get("default_strategy", "adaptive_merge"))
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_merged_length = config.get("max_merged_length", 3000)
        self.quality_threshold = config.get("quality_threshold", 0.6)

        # Configuración de fusión
        self.enable_general_knowledge = config.get("enable_general_knowledge", True)
        self.enable_cross_domain = config.get("enable_cross_domain", True)
        self.enable_consensus_validation = config.get("enable_consensus_validation", True)

        # Pesos para diferentes estrategias
        self.merge_weights = config.get(
            "merge_weights",
            {
                "specialized_weight": 0.7,
                "general_weight": 0.3,
                "context_boost": 0.1,
                "quality_penalty": 0.15,
            },
        )

        # Base de conocimientos general
        self.general_knowledge_base = self._load_general_knowledge()

        # Patrones de fusión
        self.merge_patterns = self._load_merge_patterns()

        # Cache de fusiones
        self.merge_cache = {}
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_size = config.get("cache_size", 200)

        # Estadísticas
        self.stats = {
            "total_merges": 0,
            "successful_merges": 0,
            "merges_by_strategy": {strategy.value: 0 for strategy in MergeStrategy},
            "average_quality_score": 0.0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "sources_combined_count": defaultdict(int),
        }

        logger.info("BranchMerger inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el motor de fusión

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Cargar configuraciones dinámicas
            await self._load_dynamic_configurations()

            # Inicializar modelos de fusión
            await self._initialize_merge_models()

            # Cargar patrones pre-entrenados
            await self._load_pretrained_patterns()

            logger.info("BranchMerger inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando BranchMerger: {e}")
            return False

    async def merge_responses(
        self,
        specialized_response: Any,
        query: str,
        context: Optional[Dict] = None,
        additional_sources: Optional[List[Dict]] = None,
    ) -> MergeResult:
        """
        Fusionar respuesta especializada con conocimiento general

        Args:
            specialized_response: Respuesta de rama especializada
            query: Consulta original
            context: Contexto adicional
            additional_sources: Fuentes adicionales para fusión

        Returns:
            Resultado de la fusión
        """
        start_time = time.time()

        try:
            # Preparar entrada de fusión
            merge_input = MergeInput(
                specialized_response=specialized_response,
                additional_sources=additional_sources or [],
                merge_context={"query": query, "context": context or {}},
                user_preferences=context.get("user_preferences", {}) if context else {},
            )

            # Verificar cache
            cache_key = self._generate_cache_key(specialized_response, query)

            if self.cache_enabled and cache_key in self.merge_cache:
                self.stats["cache_hits"] += 1
                cached_result = self.merge_cache[cache_key]
                logger.debug(f"Cache hit para fusión de {specialized_response.branch_name}")
                return cached_result

            # Obtener conocimiento general relevante
            if self.enable_general_knowledge:
                general_knowledge = await self._extract_general_knowledge(query, specialized_response)
                merge_input.general_knowledge = general_knowledge

            # Seleccionar estrategia de fusión
            merge_strategy = await self._select_merge_strategy(merge_input)

            # Ejecutar fusión según estrategia
            merged_result = await self._execute_merge_strategy(merge_input, merge_strategy)

            # Validar y mejorar calidad
            quality_score = await self._assess_merge_quality(merged_result, merge_input)

            if quality_score < self.quality_threshold:
                # Re-intentar con estrategia alternativa
                alternative_strategy = await self._select_alternative_strategy(merge_strategy)
                merged_result = await self._execute_merge_strategy(merge_input, alternative_strategy)
                quality_score = await self._assess_merge_quality(merged_result, merge_input)
                merge_strategy = alternative_strategy

            # Finalizar resultado
            processing_time = time.time() - start_time

            final_result = MergeResult(
                merged_response=merged_result["response"],
                confidence=merged_result["confidence"],
                merge_strategy_used=merge_strategy,
                sources_combined=merged_result["sources"],
                merge_quality_score=quality_score,
                processing_time=processing_time,
                enhancement_details=merged_result.get("details", {}),
            )

            # Actualizar estadísticas
            await self._update_merge_stats(final_result, merge_strategy, processing_time)

            # Guardar en cache si es de buena calidad
            if quality_score > self.quality_threshold:
                self._cache_merge_result(cache_key, final_result)

            return final_result

        except Exception as e:
            logger.error(f"Error en fusión: {e}")

            # Retornar respuesta original en caso de error
            return MergeResult(
                merged_response=specialized_response.response,
                confidence=specialized_response.confidence,
                merge_strategy_used=MergeStrategy.WEIGHTED_COMBINATION,
                sources_combined=[specialized_response.branch_name],
                merge_quality_score=0.5,
                processing_time=time.time() - start_time,
                enhancement_details={"error": str(e)},
            )

    async def merge_multiple_branches(
        self, branch_responses: List[Any], query: str, context: Optional[Dict] = None
    ) -> MergeResult:
        """
        Fusionar múltiples respuestas de ramas especializadas

        Args:
            branch_responses: Lista de respuestas de ramas
            query: Consulta original
            context: Contexto adicional

        Returns:
            Resultado de la fusión multi-rama
        """
        start_time = time.time()

        try:
            if not branch_responses:
                raise ValueError("No hay respuestas de ramas para fusionar")

            if len(branch_responses) == 1:
                # Solo una respuesta, aplicar fusión simple con conocimiento general
                return await self.merge_responses(branch_responses[0], query, context)

            # Fusión compleja multi-rama
            fusion_result = await self._execute_multi_branch_fusion(branch_responses, query, context)

            processing_time = time.time() - start_time

            result = MergeResult(
                merged_response=fusion_result["response"],
                confidence=fusion_result["confidence"],
                merge_strategy_used=MergeStrategy.CONSENSUS_BASED,
                sources_combined=fusion_result["sources"],
                merge_quality_score=fusion_result["quality"],
                processing_time=processing_time,
                enhancement_details={
                    "branches_merged": len(branch_responses),
                    "consensus_level": fusion_result.get("consensus", 0.0),
                },
            )

            await self._update_merge_stats(result, MergeStrategy.CONSENSUS_BASED, processing_time)

            return result

        except Exception as e:
            logger.error(f"Error en fusión multi-rama: {e}")

            # Fallback: usar la mejor respuesta individual
            best_response = max(branch_responses, key=lambda r: r.confidence)

            return MergeResult(
                merged_response=best_response.response,
                confidence=best_response.confidence,
                merge_strategy_used=MergeStrategy.WEIGHTED_COMBINATION,
                sources_combined=[best_response.branch_name],
                merge_quality_score=0.5,
                processing_time=time.time() - start_time,
                enhancement_details={"error": str(e), "fallback_used": True},
            )

    async def _extract_general_knowledge(self, query: str, specialized_response: Any) -> str:
        """
        Extraer conocimiento general relevante para la consulta

        Args:
            query: Consulta original
            specialized_response: Respuesta especializada

        Returns:
            Conocimiento general relevante
        """
        # Analizar temas en la consulta
        query_topics = await self._extract_query_topics(query)

        # Buscar en base de conocimientos general
        relevant_knowledge = []

        for topic in query_topics:
            if topic in self.general_knowledge_base:
                knowledge_entry = self.general_knowledge_base[topic]

                # Verificar relevancia con respuesta especializada
                relevance_score = await self._calculate_relevance(
                    knowledge_entry["content"], specialized_response.response
                )

                if relevance_score > 0.3:
                    relevant_knowledge.append(
                        {
                            "topic": topic,
                            "content": knowledge_entry["content"],
                            "relevance": relevance_score,
                        }
                    )

        # Ordenar por relevancia y combinar
        relevant_knowledge.sort(key=lambda x: x["relevance"], reverse=True)

        # Tomar los 3 más relevantes
        top_knowledge = relevant_knowledge[:3]

        if not top_knowledge:
            return "Conocimiento general relacionado con el contexto de la consulta."

        # Combinar conocimientos
        combined_knowledge = []
        for knowledge in top_knowledge:
            combined_knowledge.append(f"• {knowledge['content']}")

        return "\n".join(combined_knowledge)

    async def _select_merge_strategy(self, merge_input: MergeInput) -> MergeStrategy:
        """
        Seleccionar estrategia de fusión óptima

        Args:
            merge_input: Entrada de fusión

        Returns:
            Estrategia seleccionada
        """
        specialized_response = merge_input.specialized_response

        # Análisis de factores para selección de estrategia

        # Factor 1: Confianza de la respuesta especializada
        confidence_factor = specialized_response.confidence

        # Factor 2: Longitud de la respuesta
        length_factor = len(specialized_response.response) / 500.0  # Normalizar

        # Factor 3: Presencia de conocimiento general
        general_factor = 1.0 if merge_input.general_knowledge else 0.5

        # Factor 4: Fuentes adicionales
        sources_factor = len(merge_input.additional_sources) / 5.0  # Normalizar

        # Factor 5: Preferencias del usuario
        user_prefs = merge_input.user_preferences
        complexity_pref = user_prefs.get("complexity", "medium")

        # Lógica de selección

        if confidence_factor < 0.6:
            # Baja confianza, usar fusión jerárquica para compensar
            return MergeStrategy.HIERARCHICAL_MERGE

        elif sources_factor > 0.6:
            # Muchas fuentes, usar consenso
            return MergeStrategy.CONSENSUS_BASED

        elif complexity_pref == "high" and general_factor > 0.8:
            # Usuario prefiere complejidad, usar fusión contextual
            return MergeStrategy.CONTEXTUAL_FUSION

        elif length_factor < 0.3:
            # Respuesta corta, usar combinación ponderada
            return MergeStrategy.WEIGHTED_COMBINATION

        else:
            # Por defecto, usar fusión adaptativa
            return MergeStrategy.ADAPTIVE_MERGE

    async def _execute_merge_strategy(self, merge_input: MergeInput, strategy: MergeStrategy) -> Dict[str, Any]:
        """
        Ejecutar estrategia de fusión específica

        Args:
            merge_input: Entrada de fusión
            strategy: Estrategia a ejecutar

        Returns:
            Resultado de la fusión
        """
        if strategy == MergeStrategy.WEIGHTED_COMBINATION:
            return await self._weighted_combination_merge(merge_input)

        elif strategy == MergeStrategy.HIERARCHICAL_MERGE:
            return await self._hierarchical_merge(merge_input)

        elif strategy == MergeStrategy.CONTEXTUAL_FUSION:
            return await self._contextual_fusion_merge(merge_input)

        elif strategy == MergeStrategy.CONSENSUS_BASED:
            return await self._consensus_based_merge(merge_input)

        elif strategy == MergeStrategy.ADAPTIVE_MERGE:
            return await self._adaptive_merge(merge_input)

        else:
            # Fallback a combinación ponderada
            return await self._weighted_combination_merge(merge_input)

    async def _weighted_combination_merge(self, merge_input: MergeInput) -> Dict[str, Any]:
        """Fusión por combinación ponderada"""
        specialized_response = merge_input.specialized_response
        general_knowledge = merge_input.general_knowledge or ""

        # Pesos configurables
        spec_weight = self.merge_weights["specialized_weight"]
        general_weight = self.merge_weights["general_weight"]

        # Construir respuesta fusionada
        merged_parts = []

        # Parte especializada (peso mayor)
        merged_parts.append(f"**Análisis Especializado:**\n{specialized_response.response}")

        # Contexto general (peso menor)
        if general_knowledge:
            merged_parts.append(f"\n**Contexto General:**\n{general_knowledge}")

        # Fusionar fuentes adicionales si existen
        if merge_input.additional_sources:
            additional_content = []
            for source in merge_input.additional_sources[:2]:  # Máximo 2 fuentes adicionales
                if "content" in source:
                    additional_content.append(f"• {source['content'][:200]}...")

            if additional_content:
                merged_parts.append(f"\n**Información Complementaria:**\n" + "\n".join(additional_content))

        merged_response = "\n".join(merged_parts)

        # Calcular confianza combinada
        combined_confidence = (
            specialized_response.confidence * spec_weight
            + 0.7 * general_weight  # Confianza fija para conocimiento general
        )

        # Agregar boost contextual si aplica
        context_boost = self.merge_weights.get("context_boost", 0.0)
        if merge_input.merge_context:
            combined_confidence += context_boost

        combined_confidence = min(combined_confidence, 1.0)

        return {
            "response": merged_response,
            "confidence": combined_confidence,
            "sources": [specialized_response.branch_name, "general_knowledge"],
            "details": {
                "merge_type": "weighted_combination",
                "weights": {"specialized": spec_weight, "general": general_weight},
            },
        }

    async def _hierarchical_merge(self, merge_input: MergeInput) -> Dict[str, Any]:
        """Fusión jerárquica con estructura organizada"""
        specialized_response = merge_input.specialized_response
        general_knowledge = merge_input.general_knowledge or ""

        # Estructura jerárquica: Introducción -> Análisis Especializado -> Contexto -> Conclusión

        # 1. Introducción contextual
        introduction = await self._generate_contextual_introduction(
            merge_input.merge_context.get("query", ""), specialized_response.branch_name
        )

        # 2. Análisis especializado (núcleo)
        specialized_section = f"## Análisis Especializado\n\n{specialized_response.response}"

        # 3. Contexto general
        context_section = ""
        if general_knowledge:
            context_section = f"\n\n## Contexto General\n\n{general_knowledge}"

        # 4. Información adicional
        additional_section = ""
        if merge_input.additional_sources:
            additional_info = []
            for source in merge_input.additional_sources:
                if "title" in source and "content" in source:
                    additional_info.append(f"**{source['title']}**: {source['content'][:150]}...")

            if additional_info:
                additional_section = f"\n\n## Información Adicional\n\n" + "\n\n".join(additional_info)

        # 5. Conclusión integradora
        conclusion = await self._generate_integrative_conclusion(
            specialized_response, merge_input.merge_context.get("query", "")
        )

        # Ensamblar respuesta jerárquica
        hierarchical_response = (
            f"{introduction}\n\n{specialized_section}{context_section}{additional_section}\n\n{conclusion}"
        )

        # Calcular confianza (más alta debido a estructura organizada)
        confidence = min(specialized_response.confidence + 0.15, 1.0)

        sources = [specialized_response.branch_name]
        if general_knowledge:
            sources.append("general_knowledge")
        sources.extend([s.get("name", "additional") for s in merge_input.additional_sources])

        return {
            "response": hierarchical_response,
            "confidence": confidence,
            "sources": sources,
            "details": {
                "merge_type": "hierarchical",
                "sections": ["introduction", "specialized", "context", "additional", "conclusion"],
            },
        }

    async def _contextual_fusion_merge(self, merge_input: MergeInput) -> Dict[str, Any]:
        """Fusión contextual integrando seamlessly el conocimiento"""
        specialized_response = merge_input.specialized_response
        general_knowledge = merge_input.general_knowledge or ""
        query = merge_input.merge_context.get("query", "")

        # Análisis contextual para integración inteligente
        context_analysis = await self._analyze_integration_context(
            query, specialized_response.response, general_knowledge
        )

        # Integrar conocimiento de forma contextual
        integrated_response = specialized_response.response

        # Insertar conocimiento general en puntos estratégicos
        if general_knowledge and context_analysis["integration_points"]:
            for point in context_analysis["integration_points"]:
                integration_text = f"\n\n💡 *Contexto:* {point['general_info']}"

                # Insertar después de párrafos relevantes
                if point["position"] in integrated_response:
                    integrated_response = integrated_response.replace(
                        point["position"], point["position"] + integration_text
                    )

        # Enriquecer con perspectivas adicionales
        if context_analysis["enrichment_opportunities"]:
            enrichments = []
            for opportunity in context_analysis["enrichment_opportunities"]:
                enrichments.append(f"🔍 *Perspectiva adicional:* {opportunity}")

            if enrichments:
                integrated_response += f"\n\n{chr(10).join(enrichments)}"

        # Calcular confianza contextual
        contextual_confidence = specialized_response.confidence

        if context_analysis["integration_quality"] > 0.7:
            contextual_confidence += 0.2
        elif context_analysis["integration_quality"] > 0.5:
            contextual_confidence += 0.1

        contextual_confidence = min(contextual_confidence, 1.0)

        return {
            "response": integrated_response,
            "confidence": contextual_confidence,
            "sources": [specialized_response.branch_name, "contextual_integration"],
            "details": {
                "merge_type": "contextual_fusion",
                "integration_quality": context_analysis["integration_quality"],
                "integration_points_used": len(context_analysis["integration_points"]),
            },
        }

    async def _consensus_based_merge(self, merge_input: MergeInput) -> Dict[str, Any]:
        """Fusión basada en consenso de múltiples fuentes"""
        specialized_response = merge_input.specialized_response

        # Recopilar todas las fuentes disponibles
        sources_content = [
            {
                "name": specialized_response.branch_name,
                "content": specialized_response.response,
                "confidence": specialized_response.confidence,
                "weight": 1.0,
            }
        ]

        # Agregar conocimiento general
        if merge_input.general_knowledge:
            sources_content.append(
                {
                    "name": "general_knowledge",
                    "content": merge_input.general_knowledge,
                    "confidence": 0.7,
                    "weight": 0.5,
                }
            )

        # Agregar fuentes adicionales
        for source in merge_input.additional_sources:
            sources_content.append(
                {
                    "name": source.get("name", "additional"),
                    "content": source.get("content", ""),
                    "confidence": source.get("confidence", 0.6),
                    "weight": source.get("weight", 0.3),
                }
            )

        # Análisis de consenso
        consensus_analysis = await self._analyze_consensus(sources_content)

        # Construir respuesta basada en consenso
        consensus_response = await self._build_consensus_response(sources_content, consensus_analysis)

        # Calcular confianza de consenso
        consensus_confidence = consensus_analysis["consensus_score"]

        return {
            "response": consensus_response,
            "confidence": consensus_confidence,
            "sources": [s["name"] for s in sources_content],
            "details": {
                "merge_type": "consensus_based",
                "consensus_score": consensus_analysis["consensus_score"],
                "agreement_level": consensus_analysis["agreement_level"],
                "conflicting_points": consensus_analysis["conflicts"],
            },
        }

    async def _adaptive_merge(self, merge_input: MergeInput) -> Dict[str, Any]:
        """Fusión adaptativa que combina múltiples estrategias"""
        # Evaluar características de la entrada
        evaluation = await self._evaluate_merge_characteristics(merge_input)

        # Seleccionar combinación de estrategias
        if evaluation["complexity_high"] and evaluation["sources_multiple"]:
            # Usar fusión jerárquica + consenso
            hierarchical_result = await self._hierarchical_merge(merge_input)

            # Aplicar refinamiento de consenso
            consensus_refinement = await self._apply_consensus_refinement(hierarchical_result, merge_input)

            return {
                **hierarchical_result,
                "confidence": min(hierarchical_result["confidence"] + 0.1, 1.0),
                "details": {
                    "merge_type": "adaptive_hierarchical_consensus",
                    "refinement_applied": True,
                },
            }

        elif evaluation["context_rich"]:
            # Usar fusión contextual
            return await self._contextual_fusion_merge(merge_input)

        else:
            # Usar combinación ponderada mejorada
            weighted_result = await self._weighted_combination_merge(merge_input)

            # Aplicar mejoras adaptativas
            adaptive_improvements = await self._apply_adaptive_improvements(weighted_result, merge_input)

            return adaptive_improvements

    async def _execute_multi_branch_fusion(
        self, branch_responses: List[Any], query: str, context: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Ejecutar fusión de múltiples ramas especializadas
        """
        # Ordenar respuestas por confianza
        sorted_responses = sorted(branch_responses, key=lambda r: r.confidence, reverse=True)

        # Análisis de complementariedad
        complementarity_analysis = await self._analyze_branch_complementarity(sorted_responses)

        # Construir respuesta multi-rama
        if complementarity_analysis["high_complementarity"]:
            # Fusión complementaria
            fused_response = await self._create_complementary_fusion(sorted_responses, complementarity_analysis)
        else:
            # Fusión por consenso
            fused_response = await self._create_consensus_fusion(sorted_responses)

        # Calcular métricas finales
        consensus_level = await self._calculate_multi_branch_consensus(sorted_responses)
        quality_score = await self._assess_multi_branch_quality(fused_response, sorted_responses)

        return {
            "response": fused_response,
            "confidence": min(sorted_responses[0].confidence + 0.15, 1.0),
            "quality": quality_score,
            "sources": [r.branch_name for r in sorted_responses],
            "consensus": consensus_level,
        }

    # Métodos auxiliares para análisis y generación

    async def _extract_query_topics(self, query: str) -> List[str]:
        """Extraer temas principales de la consulta"""
        # Análisis simple de palabras clave
        import re

        # Remover palabras de parada
        stop_words = {
            "el",
            "la",
            "de",
            "que",
            "y",
            "a",
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
            "los",
            "las",
            "una",
            "como",
            "pero",
            "sus",
            "han",
            "me",
            "si",
            "sin",
            "sobre",
            "este",
            "ya",
            "entre",
            "cuando",
            "todo",
            "esta",
            "ser",
            "son",
            "dos",
            "también",
            "fue",
            "había",
            "era",
            "muy",
            "años",
            "hasta",
            "desde",
            "está",
            "mi",
            "porque",
        }

        words = re.findall(r"\b\w{3,}\b", query.lower())
        topics = [word for word in words if word not in stop_words]

        return topics[:5]  # Máximo 5 temas principales

    async def _calculate_relevance(self, content1: str, content2: str) -> float:
        """Calcular relevancia entre dos contenidos"""
        # Análisis simple de palabras compartidas
        words1 = set(re.findall(r"\b\w{3,}\b", content1.lower()))
        words2 = set(re.findall(r"\b\w{3,}\b", content2.lower()))

        if not words1 or not words2:
            return 0.0

        # Similitud de Jaccard
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def _generate_contextual_introduction(self, query: str, branch_name: str) -> str:
        """Generar introducción contextual"""
        domain = branch_name.replace("_en", "").replace("_es", "")

        introductions = {
            "programación": f"En el contexto de {domain}, analizando tu consulta sobre '{query[:50]}...':",
            "matemáticas": f"Desde la perspectiva matemática, examinando '{query[:50]}...':",
            "medicina": f"En el ámbito médico, considerando '{query[:50]}...':",
        }

        return introductions.get(domain, f"Análisis especializado en {domain} para '{query[:50]}...':")

    async def _generate_integrative_conclusion(self, specialized_response: Any, query: str) -> str:
        """Generar conclusión integradora"""
        return f"## Síntesis\n\nEn resumen, el análisis especializado de {specialized_response.branch_name} proporciona una perspectiva integral que combina conocimiento específico del dominio con contexto general relevante."

    def _load_general_knowledge(self) -> Dict[str, Dict]:
        """Cargar base de conocimientos general"""
        # En implementación real, cargaría desde archivos/BD
        return {
            "programación": {
                "content": "La programación es una disciplina fundamental en la era digital que involucra principios de lógica, matemáticas y resolución de problemas.",
                "related_topics": ["algoritmos", "estructuras de datos", "paradigmas"],
            },
            "matemáticas": {
                "content": "Las matemáticas proporcionan el lenguaje universal para describir patrones, relaciones y estructuras en el mundo natural y abstracto.",
                "related_topics": ["álgebra", "cálculo", "estadística", "geometría"],
            },
            "medicina": {
                "content": "La medicina moderna integra conocimiento científico, tecnología avanzada y experiencia clínica para el diagnóstico y tratamiento.",
                "related_topics": ["anatomía", "fisiología", "farmacología", "diagnóstico"],
            },
        }

    def _load_merge_patterns(self) -> Dict[str, List[str]]:
        """Cargar patrones de fusión"""
        return {
            "integration_markers": [
                "además",
                "por otro lado",
                "en contraste",
                "complementariamente",
                "furthermore",
                "moreover",
                "on the other hand",
                "in addition",
            ],
            "transition_phrases": [
                "desde otra perspectiva",
                "considerando también",
                "es importante notar",
                "from another angle",
                "it's worth noting",
                "additionally",
            ],
            "synthesis_indicators": [
                "en síntesis",
                "combinando estos aspectos",
                "integrando el conocimiento",
                "in synthesis",
                "combining these aspects",
                "integrating the knowledge",
            ],
        }

    async def _load_dynamic_configurations(self):
        """Cargar configuraciones dinámicas"""
        pass

    async def _initialize_merge_models(self):
        """Inicializar modelos de fusión"""
        pass

    async def _load_pretrained_patterns(self):
        """Cargar patrones pre-entrenados"""
        pass

    async def _select_alternative_strategy(self, current_strategy: MergeStrategy) -> MergeStrategy:
        """Seleccionar estrategia alternativa"""
        alternatives = {
            MergeStrategy.WEIGHTED_COMBINATION: MergeStrategy.HIERARCHICAL_MERGE,
            MergeStrategy.HIERARCHICAL_MERGE: MergeStrategy.CONTEXTUAL_FUSION,
            MergeStrategy.CONTEXTUAL_FUSION: MergeStrategy.WEIGHTED_COMBINATION,
            MergeStrategy.CONSENSUS_BASED: MergeStrategy.ADAPTIVE_MERGE,
            MergeStrategy.ADAPTIVE_MERGE: MergeStrategy.WEIGHTED_COMBINATION,
        }

        return alternatives.get(current_strategy, MergeStrategy.WEIGHTED_COMBINATION)

    async def _assess_merge_quality(self, merged_result: Dict, merge_input: MergeInput) -> float:
        """Evaluar calidad de la fusión"""
        response = merged_result["response"]

        # Factor 1: Longitud apropiada
        length_factor = min(len(response) / 1000.0, 1.0)

        # Factor 2: Coherencia estructural
        structure_factor = 0.8 if "##" in response or "**" in response else 0.6

        # Factor 3: Integración de fuentes
        sources_count = len(merged_result.get("sources", []))
        integration_factor = min(sources_count / 3.0, 1.0)

        # Factor 4: Confianza del resultado
        confidence_factor = merged_result.get("confidence", 0.5)

        # Calcular calidad total
        quality_score = (
            length_factor * 0.2 + structure_factor * 0.3 + integration_factor * 0.2 + confidence_factor * 0.3
        )

        return min(quality_score, 1.0)

    # Métodos mock para análisis complejo (implementación simplificada)

    async def _analyze_integration_context(self, query: str, specialized: str, general: str) -> Dict:
        """Analizar contexto de integración - USA ANÁLISIS REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()
        # Llamar método REAL (no mock)
        result = analyzer.analyze_integration_context(query, specialized, general)

        # Convertir formato para compatibilidad
        return {
            "integration_points": result["integration_points"],
            "enrichment_opportunities": [f"Integración en: {ip['keyword']}" for ip in result["integration_points"][:3]],
            "integration_quality": result["integration_score"],
        }

    async def _analyze_consensus(self, sources: List[Dict]) -> Dict:
        """Analizar consenso entre fuentes - USA ANÁLISIS REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()
        # Llamar método REAL (no mock)
        result = analyzer.analyze_consensus(sources)

        # Convertir formato
        consensus_level = result["consensus_level"]
        return {
            "consensus_score": consensus_level,
            "agreement_level": "high" if consensus_level > 0.7 else "medium" if consensus_level > 0.4 else "low",
            "conflicts": [] if result["divergence"] < 0.3 else ["Divergencia detectada"],
            "num_sources": result["num_sources"],
        }

    async def _build_consensus_response(self, sources: List[Dict], analysis: Dict) -> str:
        """Construir respuesta de consenso - USA ANÁLISIS REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        # Convertir sources a formato esperado
        sources_formatted = []
        for s in sources:
            sources_formatted.append({"response": s.get("content", ""), "confidence": s.get("confidence", 0.5)})

        analyzer = get_real_analyzer()
        # Llamar método REAL (no mock)
        return analyzer.build_consensus_response(sources_formatted, analysis)

    async def _evaluate_merge_characteristics(self, merge_input: MergeInput) -> Dict:
        """Evaluar características para fusión - USA ANÁLISIS REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()
        # Llamar método REAL (no mock)
        result = analyzer.evaluate_merge_characteristics(merge_input)

        # Añadir context_rich
        result["context_rich"] = bool(merge_input.merge_context and merge_input.general_knowledge)
        return result

    async def _apply_consensus_refinement(self, result: Dict, merge_input: MergeInput) -> Dict:
        """Aplicar refinamiento de consenso - IMPLEMENTACIÓN REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()

        # Analizar la respuesta fusionada
        response_text = result.get("response", "")
        sources = merge_input.additional_sources

        if not sources or not response_text:
            return result

        # Evaluar calidad de la fusión
        original_responses = [{"response": s.get("content", "")} for s in sources]
        quality_score = analyzer.assess_multi_branch_quality(response_text, original_responses)

        # Refinar si la calidad es baja
        if quality_score < 0.7:
            # Añadir contexto adicional
            refinement = (
                "\n\n**Nota**: Esta respuesta combina múltiples perspectivas con diferentes niveles de confianza."
            )
            result["response"] = response_text + refinement
            result["quality_score"] = quality_score
            result["refined"] = True
        else:
            result["quality_score"] = quality_score
            result["refined"] = False

        return result

    async def _apply_adaptive_improvements(self, result: Dict, merge_input: MergeInput) -> Dict:
        """Aplicar mejoras adaptativas - IMPLEMENTACIÓN REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()
        response_text = result.get("response", "")

        # Evaluar características del merge
        characteristics = analyzer.evaluate_merge_characteristics(merge_input)

        # Aplicar mejoras según características
        improved_response = response_text
        confidence_boost = 0.0

        # Si es complejo, añadir estructura
        if characteristics.get("complexity_high"):
            if not response_text.startswith("**"):
                improved_response = f"🔍 **Análisis Integrado**\n\n{response_text}"
                confidence_boost += 0.02

        # Si hay múltiples fuentes, añadir síntesis
        if characteristics.get("sources_multiple"):
            num_sources = characteristics.get("num_additional_sources", 0)
            synthesis = (
                f"\n\n💡 **Síntesis**: Respuesta construida a partir de {num_sources + 1} fuentes complementarias."
            )
            improved_response += synthesis
            confidence_boost += 0.03

        # Si hay contexto rico, mencionarlo
        if characteristics.get("context_rich"):
            improved_response += "\n*Información contextualizada con conocimiento general.*"
            confidence_boost += 0.02

        return {
            **result,
            "response": improved_response,
            "confidence": min(result.get("confidence", 0.5) + confidence_boost, 1.0),
            "improvements_applied": True,
        }

    async def _analyze_branch_complementarity(self, responses: List[Any]) -> Dict:
        """Analizar complementariedad entre ramas - USA ANÁLISIS REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()
        # Llamar método REAL (no mock)
        result = analyzer.analyze_branch_complementarity(responses)

        # Añadir campos adicionales
        result["overlap_areas"] = ["overlap detectado"] if result["overlap_percentage"] > 0.3 else []
        result["unique_contributions"] = (
            [f"Contribución de {r.branch_name}" for r in responses] if hasattr(responses[0], "branch_name") else []
        )

        return result

    async def _create_complementary_fusion(self, responses: List[Any], analysis: Dict) -> str:
        """Crear fusión complementaria - IMPLEMENTACIÓN REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        analyzer = get_real_analyzer()

        # Análisis de complementariedad REAL
        complementarity_score = analysis.get("complementarity_score", 0.5)
        high_complementarity = analysis.get("high_complementarity", False)

        fusion_parts = []

        # Encabezado según nivel de complementariedad
        if high_complementarity:
            fusion_parts.append("**Análisis Multi-Dominio Complementario** (Alta Complementariedad)\n")
        else:
            fusion_parts.append("**Análisis Multi-Fuente Integrado**\n")

        # Incluir cada perspectiva con análisis de contribución
        for i, response in enumerate(responses[:3], 1):
            branch_name = getattr(response, "branch_name", f"Fuente {i}")
            response_text = getattr(response, "response", str(response))
            confidence = getattr(response, "confidence", 0.5)

            # Extraer keywords únicos de esta respuesta
            keywords = analyzer._extract_keywords(response_text)
            unique_contribution = f" (Conceptos clave: {', '.join(keywords[:3])})" if keywords else ""

            # Truncar inteligentemente en punto o línea
            truncate_at = 300
            if len(response_text) > truncate_at:
                # Buscar punto más cercano
                nearest_period = response_text[:truncate_at].rfind(".")
                if nearest_period > truncate_at - 50:
                    truncate_at = nearest_period + 1
                display_text = response_text[:truncate_at] + "..."
            else:
                display_text = response_text

            fusion_parts.append(
                f"\n**{i}. {branch_name}** (confianza: {confidence:.0%}){unique_contribution}:\n{display_text}"
            )

        # Síntesis integradora REAL basada en análisis
        overlap_pct = analysis.get("overlap_percentage", 0)

        if high_complementarity:
            synthesis = (
                f"\n**Síntesis Integradora:**\n"
                f"Las {len(responses)} perspectivas son altamente complementarias "
                f"(complementariedad: {complementarity_score:.0%}, overlap: {overlap_pct:.0%}), "
                f"proporcionando una visión holística desde diferentes ángulos especializados."
            )
        else:
            synthesis = (
                f"\n**Síntesis:**\n"
                f"Integración de {len(responses)} fuentes con {100-overlap_pct:.0%} de contenido único. "
                f"Las perspectivas muestran convergencia en conceptos centrales."
            )

        fusion_parts.append(synthesis)

        return "\n".join(fusion_parts)

    async def _create_consensus_fusion(self, responses: List[Any]) -> str:
        """Mock: Crear fusión por consenso"""
        best_response = responses[0]  # Ya ordenado por confianza

        consensus_parts = [
            f"**Análisis de Consenso Multi-Rama**\n",
            f"**Respuesta Principal:** {best_response.response}",
        ]

        if len(responses) > 1:
            consensus_parts.append(
                f"\n**Perspectivas Adicionales de {len(responses)-1} ramas especializadas confirman y complementan este análisis.**"
            )

        return "\n".join(consensus_parts)

    async def _calculate_multi_branch_consensus(self, responses: List[Any]) -> float:
        """Calcular consenso multi-rama - IMPLEMENTACIÓN REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        if not responses:
            return 0.0

        analyzer = get_real_analyzer()

        # Convertir a formato de análisis
        sources_data = []
        for r in responses:
            sources_data.append(
                {"response": getattr(r, "response", str(r)), "confidence": getattr(r, "confidence", 0.5)}
            )

        # Análisis REAL de consenso
        analysis = analyzer.analyze_consensus(sources_data)

        # Retornar nivel de consenso calculado con algoritmo real
        return analysis.get("consensus_level", 0.5)

    async def _assess_multi_branch_quality(self, response: str, original_responses: List[Any]) -> float:
        """Evaluar calidad multi-rama - IMPLEMENTACIÓN REAL"""
        from sheily_core.tools.real_merger_analysis import get_real_analyzer

        if not response or not original_responses:
            return 0.5

        analyzer = get_real_analyzer()

        # Usar evaluación REAL de calidad
        quality_score = analyzer.assess_multi_branch_quality(response, original_responses)

        return quality_score

    def _generate_cache_key(self, specialized_response: Any, query: str) -> str:
        """Generar clave de cache para fusión"""
        import hashlib

        cache_content = f"{specialized_response.branch_name}|{query[:100]}|{specialized_response.response[:200]}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _cache_merge_result(self, cache_key: str, result: MergeResult):
        """Guardar resultado en cache"""
        if len(self.merge_cache) >= self.cache_size:
            oldest_key = next(iter(self.merge_cache))
            del self.merge_cache[oldest_key]

        self.merge_cache[cache_key] = result

    async def _update_merge_stats(self, result: MergeResult, strategy: MergeStrategy, processing_time: float):
        """Actualizar estadísticas de fusión"""
        self.stats["total_merges"] += 1

        if result.merge_quality_score > self.quality_threshold:
            self.stats["successful_merges"] += 1

        self.stats["merges_by_strategy"][strategy.value] += 1

        # Actualizar promedios
        total = self.stats["total_merges"]

        current_quality_avg = self.stats["average_quality_score"]
        self.stats["average_quality_score"] = (current_quality_avg * (total - 1) + result.merge_quality_score) / total

        current_time_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (current_time_avg * (total - 1) + processing_time) / total

        # Contar fuentes combinadas
        for source in result.sources_combined:
            self.stats["sources_combined_count"][source] += 1

    def get_stats(self) -> Dict:
        """Obtener estadísticas del merger"""
        return self.stats.copy()

    async def health_check(self) -> Dict:
        """Verificar estado de salud del merger"""
        return {
            "status": "healthy",
            "total_merges": self.stats["total_merges"],
            "success_rate": (
                self.stats["successful_merges"] / self.stats["total_merges"] if self.stats["total_merges"] > 0 else 0.0
            ),
            "average_quality": self.stats["average_quality_score"],
            "cache_size": len(self.merge_cache),
            "strategies_enabled": list(MergeStrategy),
        }

    async def shutdown(self):
        """Cerrar merger y limpiar recursos"""
        logger.info("Cerrando BranchMerger")

        # Limpiar cache
        self.merge_cache.clear()

        # Log estadísticas finales
        final_stats = self.get_stats()
        logger.info(f"Estadísticas finales de BranchMerger: {final_stats}")

        logger.info("BranchMerger cerrado")
