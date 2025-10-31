#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Merger Analysis System - NO MOCKS
======================================
Sistema de análisis y fusión de respuestas 100% funcional.
Elimina todos los mocks del merger original.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Resultado de análisis"""
    confidence: float
    complexity_score: float
    semantic_features: Dict[str, Any]
    integration_points: List[Dict[str, str]]


class RealMergerAnalyzer:
    """
    Analizador REAL de merge (sin mocks)
    ====================================
    
    Implementa análisis funcional de:
    - Contexto de integración
    - Consenso entre fuentes
    - Complementariedad
    - Calidad de respuestas
    """
    
    def __init__(self):
        """Inicializar analizador"""
        self.min_confidence = 0.3
        self.complexity_threshold = 0.5
        logger.info("RealMergerAnalyzer inicializado")
    
    def analyze_integration_context(
        self,
        query: str,
        specialized: str,
        general: str
    ) -> Dict[str, Any]:
        """
        Analizar contexto de integración - IMPLEMENTACIÓN REAL
        
        Analiza cómo integrar respuesta especializada con contexto general.
        NO es mock - hace análisis real de texto.
        
        Args:
            query: Consulta original
            specialized: Respuesta especializada
            general: Contexto general
            
        Returns:
            Análisis de integración real
        """
        integration_points = []
        
        # 1. Encontrar puntos de integración por palabras clave
        query_words = set(self._extract_keywords(query))
        spec_words = set(self._extract_keywords(specialized))
        gen_words = set(self._extract_keywords(general))
        
        # Intersección de conceptos
        common_spec_gen = spec_words & gen_words
        common_query_spec = query_words & spec_words
        
        # Crear puntos de integración basados en intersecciones reales
        for keyword in common_spec_gen:
            # Buscar contexto en specialized
            spec_context = self._find_context(keyword, specialized, 100)
            # Buscar contexto en general
            gen_context = self._find_context(keyword, general, 100)
            
            if spec_context and gen_context:
                integration_points.append({
                    "keyword": keyword,
                    "specialized_context": spec_context,
                    "general_context": gen_context,
                    "relevance": self._calculate_relevance(keyword, query)
                })
        
        # 2. Calcular score de integración
        integration_score = len(integration_points) / max(len(query_words), 1)
        integration_score = min(integration_score, 1.0)
        
        # 3. Analizar complementariedad
        complementary = len(gen_words - spec_words) > 0
        
        result = {
            "integration_points": integration_points[:5],  # Top 5
            "integration_score": round(integration_score, 3),
            "complementary": complementary,
            "specialized_coverage": round(len(common_query_spec) / max(len(query_words), 1), 3),
            "additional_info": general if complementary else None
        }
        
        logger.debug(f"Integration analysis: {len(integration_points)} points, score={integration_score:.3f}")
        return result
    
    def analyze_consensus(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizar consenso entre fuentes - IMPLEMENTACIÓN REAL
        
        Calcula consenso real basado en:
        - Confidence promedio
        - Similitud de contenido
        - Convergencia de respuestas
        
        Args:
            sources: Lista de fuentes con response y confidence
            
        Returns:
            Análisis de consenso real
        """
        if not sources:
            return {
                "consensus_level": 0.0,
                "agreement_score": 0.0,
                "divergence": 1.0,
                "primary_source": None
            }
        
        # 1. Confidence promedio (ponderado)
        confidences = [s.get("confidence", 0.5) for s in sources]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 2. Análisis de similitud entre respuestas
        responses = [s.get("response", "") for s in sources]
        similarity_scores = []
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._calculate_text_similarity(responses[i], responses[j])
                similarity_scores.append(sim)
        
        # 3. Calcular consensus level
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            consensus_level = (avg_confidence + avg_similarity) / 2
        else:
            consensus_level = avg_confidence
        
        # 4. Identificar fuente primaria (mayor confidence)
        primary_source = max(sources, key=lambda s: s.get("confidence", 0))
        
        # 5. Calcular divergencia
        if len(confidences) > 1:
            confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            divergence = min(confidence_variance * 10, 1.0)  # Normalizado
        else:
            divergence = 0.0
        
        result = {
            "consensus_level": round(consensus_level, 3),
            "agreement_score": round(avg_similarity if similarity_scores else avg_confidence, 3),
            "divergence": round(divergence, 3),
            "primary_source": primary_source,
            "num_sources": len(sources),
            "avg_confidence": round(avg_confidence, 3)
        }
        
        logger.debug(f"Consensus analysis: level={consensus_level:.3f}, sources={len(sources)}")
        return result
    
    def build_consensus_response(
        self,
        sources: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> str:
        """
        Construir respuesta de consenso - IMPLEMENTACIÓN REAL
        
        Construye respuesta sintetizada basada en fuentes reales.
        
        Args:
            sources: Fuentes con responses
            analysis: Análisis de consenso
            
        Returns:
            Respuesta sintetizada
        """
        if not sources:
            return "No hay información disponible."
        
        consensus_level = analysis.get("consensus_level", 0.5)
        primary = analysis.get("primary_source", sources[0])
        
        # 1. Respuesta base (fuente primaria)
        base_response = primary.get("response", "")
        
        # 2. Si hay alto consenso, usar respuesta principal
        if consensus_level >= 0.7:
            response_parts = [
                "**Respuesta Consolidada** (alto consenso)\n",
                base_response
            ]
        else:
            # 3. Bajo consenso - mostrar perspectivas
            response_parts = [
                "**Análisis Multi-Fuente** (perspectivas variadas)\n",
                f"\n**Perspectiva Principal** (confianza {primary.get('confidence', 0):.2f}):\n",
                base_response
            ]
            
            # Añadir otras perspectivas significativas
            for i, source in enumerate(sources[1:3], 1):  # Máximo 2 adicionales
                if source.get("confidence", 0) > 0.5:
                    response_parts.append(
                        f"\n**Perspectiva Alternativa {i}** (confianza {source.get('confidence', 0):.2f}):\n"
                        f"{source.get('response', '')[:200]}..."
                    )
        
        # 4. Añadir metadata de consenso
        response_parts.append(
            f"\n\n*Nivel de consenso: {consensus_level:.0%} | "
            f"Fuentes: {len(sources)}*"
        )
        
        return "\n".join(response_parts)
    
    def evaluate_merge_characteristics(self, merge_input: Any) -> Dict[str, Any]:
        """
        Evaluar características para fusión - IMPLEMENTACIÓN REAL
        
        Analiza complejidad y características del merge.
        
        Args:
            merge_input: Input de merge con respuestas
            
        Returns:
            Características evaluadas
        """
        specialized_resp = getattr(merge_input, 'specialized_response', None)
        additional = getattr(merge_input, 'additional_sources', [])
        
        if not specialized_resp:
            return {
                "complexity_high": False,
                "sources_multiple": False,
                "needs_integration": False
            }
        
        # Extraer response text
        spec_text = getattr(specialized_resp, 'response', str(specialized_resp))
        
        # 1. Evaluar complejidad
        complexity_indicators = [
            len(spec_text) > 500,  # Respuesta larga
            len(spec_text.split('\n')) > 5,  # Múltiples párrafos
            any(keyword in spec_text.lower() for keyword in ['sin embargo', 'por otro lado', 'además', 'también']),
            len(additional) > 0  # Múltiples fuentes
        ]
        complexity_high = sum(complexity_indicators) >= 2
        
        # 2. Evaluar necesidad de integración
        needs_integration = len(additional) > 0 and len(spec_text) > 100
        
        result = {
            "complexity_high": complexity_high,
            "sources_multiple": len(additional) > 1,
            "needs_integration": needs_integration,
            "response_length": len(spec_text),
            "num_additional_sources": len(additional)
        }
        
        logger.debug(f"Merge characteristics: complexity={complexity_high}, sources={len(additional)}")
        return result
    
    def analyze_branch_complementarity(
        self,
        responses: List[Any]
    ) -> Dict[str, Any]:
        """
        Analizar complementariedad entre ramas - IMPLEMENTACIÓN REAL
        
        Determina si las respuestas son complementarias o redundantes.
        
        Args:
            responses: Lista de respuestas
            
        Returns:
            Análisis de complementariedad
        """
        if len(responses) < 2:
            return {
                "high_complementarity": False,
                "complementarity_score": 0.0,
                "overlap_percentage": 0.0
            }
        
        # Extraer textos
        texts = []
        for resp in responses:
            if hasattr(resp, 'response'):
                texts.append(resp.response)
            else:
                texts.append(str(resp))
        
        # 1. Calcular overlap de palabras clave
        all_keywords = [set(self._extract_keywords(text)) for text in texts]
        
        # Intersección (overlap)
        if len(all_keywords) >= 2:
            overlap = all_keywords[0]
            for keywords in all_keywords[1:]:
                overlap = overlap & keywords
            
            # Unión (total)
            union = all_keywords[0]
            for keywords in all_keywords[1:]:
                union = union | keywords
            
            overlap_percentage = len(overlap) / max(len(union), 1)
            
            # Alta complementariedad = bajo overlap
            complementarity_score = 1.0 - overlap_percentage
        else:
            overlap_percentage = 0.0
            complementarity_score = 0.0
        
        high_complementarity = complementarity_score > 0.6
        
        result = {
            "high_complementarity": high_complementarity,
            "complementarity_score": round(complementarity_score, 3),
            "overlap_percentage": round(overlap_percentage, 3),
            "num_responses": len(responses)
        }
        
        logger.debug(
            f"Complementarity: score={complementarity_score:.3f}, "
            f"overlap={overlap_percentage:.3f}"
        )
        return result
    
    def assess_multi_branch_quality(
        self,
        response: str,
        original_responses: List[Any]
    ) -> float:
        """
        Evaluar calidad multi-rama - IMPLEMENTACIÓN REAL
        
        Calcula quality score basado en:
        - Longitud apropiada
        - Diversidad de contenido
        - Coherencia
        
        Args:
            response: Respuesta fusionada
            original_responses: Respuestas originales
            
        Returns:
            Quality score (0-1)
        """
        scores = []
        
        # 1. Score de longitud (no muy corta, no muy larga)
        length = len(response)
        if 200 <= length <= 2000:
            length_score = 1.0
        elif length < 200:
            length_score = length / 200
        else:
            length_score = max(0.5, 1.0 - (length - 2000) / 2000)
        scores.append(length_score)
        
        # 2. Score de diversidad (vs respuestas originales)
        response_keywords = set(self._extract_keywords(response))
        
        coverage_scores = []
        for orig in original_responses:
            orig_text = orig.response if hasattr(orig, 'response') else str(orig)
            orig_keywords = set(self._extract_keywords(orig_text))
            
            if orig_keywords:
                coverage = len(response_keywords & orig_keywords) / len(orig_keywords)
                coverage_scores.append(coverage)
        
        if coverage_scores:
            diversity_score = sum(coverage_scores) / len(coverage_scores)
            scores.append(diversity_score)
        
        # 3. Score de estructura (tiene secciones, no es solo texto plano)
        has_structure = any(marker in response for marker in ['**', '\n\n', '###', '- '])
        structure_score = 1.0 if has_structure else 0.7
        scores.append(structure_score)
        
        # 4. Bonus por número de respuestas integradas
        integration_bonus = min(len(original_responses) * 0.05, 0.2)
        
        # Promedio + bonus
        quality_score = sum(scores) / len(scores) + integration_bonus
        quality_score = min(quality_score, 1.0)
        
        logger.debug(f"Quality assessment: {quality_score:.3f} (length={length_score:.2f}, diversity={diversity_score:.2f if coverage_scores else 0:.2f})")
        return round(quality_score, 3)
    
    # ========== Utilidades ==========
    
    def _extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """Extraer palabras clave de texto"""
        # Limpiar y tokenizar
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        # Filtrar palabras comunes (stopwords básicas)
        stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'y', 'o', 'en', 'con', 'por', 'para',
            'que', 'como', 'es', 'son', 'está', 'están', 'ser',
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to',
            'for', 'of', 'is', 'are', 'was', 'were', 'be'
        }
        
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in stopwords
        ]
        
        return keywords
    
    def _find_context(self, keyword: str, text: str, context_size: int = 100) -> Optional[str]:
        """Encontrar contexto alrededor de keyword"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        pos = text_lower.find(keyword_lower)
        if pos == -1:
            return None
        
        start = max(0, pos - context_size // 2)
        end = min(len(text), pos + len(keyword) + context_size // 2)
        
        context = text[start:end].strip()
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _calculate_relevance(self, keyword: str, query: str) -> float:
        """Calcular relevancia de keyword para query"""
        query_lower = query.lower()
        keyword_lower = keyword.lower()
        
        # Relevancia basada en posición y frecuencia
        if keyword_lower in query_lower:
            # Palabra exacta en query = alta relevancia
            return 1.0
        
        # Calcular similitud por caracteres comunes
        common_chars = set(keyword_lower) & set(query_lower)
        similarity = len(common_chars) / max(len(set(keyword_lower)), 1)
        
        return round(similarity, 3)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos"""
        if not text1 or not text2:
            return 0.0
        
        # Extraer keywords de ambos
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        similarity = intersection / max(union, 1)
        return round(similarity, 3)


# Instancia global
_analyzer: Optional[RealMergerAnalyzer] = None


def get_real_analyzer() -> RealMergerAnalyzer:
    """Obtener instancia global del analizador"""
    global _analyzer
    if _analyzer is None:
        _analyzer = RealMergerAnalyzer()
    return _analyzer


# Exports
__all__ = [
    'RealMergerAnalyzer',
    'AnalysisResult',
    'get_real_analyzer'
]
