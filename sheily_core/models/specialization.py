#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpecializationEngine - Motor de Especialización Avanzada
=======================================================

Aplica especialización dinámica y contextual a las respuestas
basándose en el dominio, experiencia previa y patrones de uso.
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SpecializationType(Enum):
    """Tipos de especialización disponibles"""

    DOMAIN_EXPERTISE = "domain_expertise"
    CONTEXTUAL_ADAPTATION = "contextual_adaptation"
    HISTORICAL_LEARNING = "historical_learning"
    USER_PREFERENCE = "user_preference"
    TECHNICAL_DEPTH = "technical_depth"


class SpecializationLevel(Enum):
    """Niveles de especialización"""

    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    RESEARCH = 5


@dataclass
class SpecializationContext:
    """Contexto para aplicar especialización"""

    domain: str
    user_level: SpecializationLevel
    query_type: str
    historical_interactions: List[Dict] = field(default_factory=list)
    domain_preferences: Dict[str, Any] = field(default_factory=dict)
    technical_requirements: List[str] = field(default_factory=list)


@dataclass
class SpecializationResult:
    """Resultado de la especialización"""

    enhanced_response: str
    confidence: float
    specialization_applied: List[SpecializationType]
    technical_level: SpecializationLevel
    domain_accuracy: float
    enhancement_details: Dict[str, Any] = field(default_factory=dict)


class SpecializationEngine:
    """
    Motor de especialización que mejora respuestas con conocimiento específico
    """

    def __init__(self, config: Dict):
        """
        Inicializar el motor de especialización

        Args:
            config: Configuración del motor
        """
        self.config = config
        self.enable_domain_expertise = config.get("enable_domain_expertise", True)
        self.enable_contextual_adaptation = config.get("enable_contextual_adaptation", True)
        self.enable_historical_learning = config.get("enable_historical_learning", True)
        self.enable_technical_depth = config.get("enable_technical_depth", True)

        # Configuración de especialización
        self.max_enhancement_length = config.get("max_enhancement_length", 2000)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.specialization_timeout = config.get("specialization_timeout", 10.0)

        # Base de conocimientos por dominio
        self.domain_knowledge = self._load_domain_knowledge()

        # Patrones de especialización
        self.specialization_patterns = self._load_specialization_patterns()

        # Histórico de interacciones por usuario/sesión
        self.interaction_history = defaultdict(lambda: deque(maxlen=50))

        # Cache de especializaciones
        self.specialization_cache = {}
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_size = config.get("cache_size", 500)

        # Estadísticas de especialización
        self.stats = {
            "total_specializations": 0,
            "successful_specializations": 0,
            "specializations_by_type": {spec_type.value: 0 for spec_type in SpecializationType},
            "specializations_by_domain": defaultdict(int),
            "average_confidence_improvement": 0.0,
            "cache_hits": 0,
            "processing_times": deque(maxlen=100),
        }

        logger.info("SpecializationEngine inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el motor de especialización

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Cargar configuraciones dinámicas
            await self._load_dynamic_configurations()

            # Inicializar modelos de especialización
            await self._initialize_specialization_models()

            # Cargar patrones pre-entrenados
            await self._load_pretrained_patterns()

            logger.info("SpecializationEngine inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando SpecializationEngine: {e}")
            return False

    async def specialize_response(
        self, query: str, base_response: str, branch_name: str, context: Optional[Dict] = None
    ) -> SpecializationResult:
        """
        Aplicar especialización a una respuesta base

        Args:
            query: Consulta original
            base_response: Respuesta base sin especializar
            branch_name: Nombre de la rama especializada
            context: Contexto adicional

        Returns:
            Resultado de la especialización
        """
        start_time = time.time()

        try:
            # Verificar cache primero
            cache_key = self._generate_cache_key(query, base_response, branch_name)

            if self.cache_enabled and cache_key in self.specialization_cache:
                self.stats["cache_hits"] += 1
                cached_result = self.specialization_cache[cache_key]
                logger.debug(f"Cache hit para especialización de {branch_name}")
                return cached_result

            # Extraer contexto de especialización
            spec_context = await self._extract_specialization_context(query, branch_name, context)

            # Aplicar diferentes tipos de especialización
            enhanced_response = base_response
            applied_specializations = []
            confidence_improvements = []

            # 1. Especialización por dominio
            if self.enable_domain_expertise:
                domain_result = await self._apply_domain_expertise(enhanced_response, spec_context)
                if domain_result["enhanced"]:
                    enhanced_response = domain_result["response"]
                    applied_specializations.append(SpecializationType.DOMAIN_EXPERTISE)
                    confidence_improvements.append(domain_result["confidence_boost"])

            # 2. Adaptación contextual
            if self.enable_contextual_adaptation:
                contextual_result = await self._apply_contextual_adaptation(enhanced_response, query, spec_context)
                if contextual_result["enhanced"]:
                    enhanced_response = contextual_result["response"]
                    applied_specializations.append(SpecializationType.CONTEXTUAL_ADAPTATION)
                    confidence_improvements.append(contextual_result["confidence_boost"])

            # 3. Aprendizaje histórico
            if self.enable_historical_learning:
                historical_result = await self._apply_historical_learning(enhanced_response, spec_context)
                if historical_result["enhanced"]:
                    enhanced_response = historical_result["response"]
                    applied_specializations.append(SpecializationType.HISTORICAL_LEARNING)
                    confidence_improvements.append(historical_result["confidence_boost"])

            # 4. Profundidad técnica
            if self.enable_technical_depth:
                technical_result = await self._apply_technical_depth(enhanced_response, query, spec_context)
                if technical_result["enhanced"]:
                    enhanced_response = technical_result["response"]
                    applied_specializations.append(SpecializationType.TECHNICAL_DEPTH)
                    confidence_improvements.append(technical_result["confidence_boost"])

            # Calcular métricas finales
            processing_time = time.time() - start_time
            final_confidence = self._calculate_final_confidence(
                base_response, enhanced_response, confidence_improvements
            )

            technical_level = await self._assess_technical_level(enhanced_response, spec_context)
            domain_accuracy = await self._assess_domain_accuracy(enhanced_response, spec_context)

            # Crear resultado
            result = SpecializationResult(
                enhanced_response=enhanced_response,
                confidence=final_confidence,
                specialization_applied=applied_specializations,
                technical_level=technical_level,
                domain_accuracy=domain_accuracy,
                enhancement_details={
                    "processing_time": processing_time,
                    "confidence_improvements": confidence_improvements,
                    "original_length": len(base_response),
                    "enhanced_length": len(enhanced_response),
                    "branch_name": branch_name,
                },
            )

            # Actualizar estadísticas
            await self._update_specialization_stats(result, processing_time)

            # Guardar en cache si es de buena calidad
            if final_confidence > self.confidence_threshold:
                self._cache_specialization(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Error en especialización: {e}")

            # Retornar respuesta sin especializar en caso de error
            return SpecializationResult(
                enhanced_response=base_response,
                confidence=0.5,
                specialization_applied=[],
                technical_level=SpecializationLevel.BASIC,
                domain_accuracy=0.5,
                enhancement_details={"error": str(e)},
            )

    async def _extract_specialization_context(
        self, query: str, branch_name: str, context: Optional[Dict]
    ) -> SpecializationContext:
        """
        Extraer contexto necesario para la especialización
        """
        # Determinar dominio base
        domain = branch_name.replace("_en", "").replace("_es", "")

        # Análizar nivel técnico requerido
        user_level = await self._analyze_query_level(query)

        # Determinar tipo de consulta
        query_type = await self._classify_query_type(query)

        # Obtener histórico si está disponible
        session_id = context.get("session_id", "default") if context else "default"
        historical_interactions = list(self.interaction_history.get(session_id, []))

        # Extraer preferencias de dominio
        domain_preferences = await self._extract_domain_preferences(historical_interactions, domain)

        # Identificar requerimientos técnicos
        technical_requirements = await self._identify_technical_requirements(query)

        return SpecializationContext(
            domain=domain,
            user_level=user_level,
            query_type=query_type,
            historical_interactions=historical_interactions,
            domain_preferences=domain_preferences,
            technical_requirements=technical_requirements,
        )

    async def _apply_domain_expertise(self, response: str, context: SpecializationContext) -> Dict[str, Any]:
        """
        Aplicar conocimiento especializado del dominio
        """
        domain_knowledge = self.domain_knowledge.get(context.domain, {})

        if not domain_knowledge:
            return {"enhanced": False, "response": response, "confidence_boost": 0.0}

        enhanced_response = response
        enhancement_applied = False
        confidence_boost = 0.0

        # Agregar terminología específica del dominio
        terminology = domain_knowledge.get("terminology", {})
        for term, definition in terminology.items():
            if term.lower() in response.lower() and definition not in response:
                # Agregar definición contextual
                enhanced_response = enhanced_response.replace(term, f"{term} ({definition})")
                enhancement_applied = True
                confidence_boost += 0.05

        # Agregar conceptos relacionados relevantes
        related_concepts = domain_knowledge.get("related_concepts", [])
        query_concepts = set(re.findall(r"\b\w+\b", response.lower()))

        for concept in related_concepts:
            if any(word in query_concepts for word in concept["keywords"]):
                if concept["name"] not in response:
                    enhanced_response += f"\n\n[Concepto relacionado: {concept['name']} - {concept['description']}]"
                    enhancement_applied = True
                    confidence_boost += 0.1

        # Agregar mejores prácticas si es relevante
        best_practices = domain_knowledge.get("best_practices", [])
        for practice in best_practices:
            if any(keyword in response.lower() for keyword in practice.get("triggers", [])):
                enhanced_response += f"\n\n💡 Mejor práctica: {practice['description']}"
                enhancement_applied = True
                confidence_boost += 0.08

        return {
            "enhanced": enhancement_applied,
            "response": enhanced_response,
            "confidence_boost": min(confidence_boost, 0.3),
        }

    async def _apply_contextual_adaptation(
        self, response: str, query: str, context: SpecializationContext
    ) -> Dict[str, Any]:
        """
        Adaptar respuesta al contexto específico de la consulta
        """
        enhanced_response = response
        enhancement_applied = False
        confidence_boost = 0.0

        # Ajustar nivel de detalle según el nivel del usuario
        if context.user_level == SpecializationLevel.BASIC:
            # Simplificar terminología técnica
            technical_terms = re.findall(r"\b[A-Z]{2,}\b|\b\w*[Tt]ech\w*\b", response)
            for term in technical_terms[:3]:  # Limitar a 3 términos
                if term not in enhanced_response:
                    continue
                simplified = self._simplify_technical_term(term, context.domain)
                if simplified:
                    enhanced_response = enhanced_response.replace(term, f"{term} ({simplified})")
                    enhancement_applied = True
                    confidence_boost += 0.05

        elif context.user_level in [SpecializationLevel.EXPERT, SpecializationLevel.RESEARCH]:
            # Agregar detalles técnicos avanzados
            advanced_details = await self._generate_advanced_details(query, context.domain)
            if advanced_details:
                enhanced_response += f"\n\n🔬 Detalles avanzados:\n{advanced_details}"
                enhancement_applied = True
                confidence_boost += 0.15

        # Adaptar formato según el tipo de consulta
        if context.query_type == "how_to":
            # Estructurar como pasos
            if "1." not in response and "paso" not in response.lower():
                steps = self._extract_action_items(response)
                if len(steps) > 1:
                    steps_formatted = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
                    enhanced_response = f"Pasos a seguir:\n{steps_formatted}\n\n{response}"
                    enhancement_applied = True
                    confidence_boost += 0.1

        elif context.query_type == "comparison":
            # Estructurar como comparación
            if "vs." not in response and "comparación" not in response.lower():
                enhanced_response = f"📊 Análisis comparativo:\n{response}"
                enhancement_applied = True
                confidence_boost += 0.08

        return {
            "enhanced": enhancement_applied,
            "response": enhanced_response,
            "confidence_boost": min(confidence_boost, 0.25),
        }

    async def _apply_historical_learning(self, response: str, context: SpecializationContext) -> Dict[str, Any]:
        """
        Aplicar aprendizaje basado en interacciones históricas
        """
        if not context.historical_interactions:
            return {"enhanced": False, "response": response, "confidence_boost": 0.0}

        enhanced_response = response
        enhancement_applied = False
        confidence_boost = 0.0

        # Análizar patrones en interacciones previas
        recent_interactions = context.historical_interactions[-10:]  # Últimas 10

        # Identificar temas recurrentes
        recurring_topics = self._identify_recurring_topics(recent_interactions)

        for topic in recurring_topics:
            if topic["frequency"] >= 3 and topic["name"] in response.lower():
                # Agregar contexto basado en interacciones previas
                historical_insight = self._generate_historical_insight(topic, context.domain)
                if historical_insight:
                    enhanced_response += f"\n\n📚 Basado en consultas anteriores: {historical_insight}"
                    enhancement_applied = True
                    confidence_boost += 0.12

        # Aplicar preferencias detectadas
        for preference_key, preference_value in context.domain_preferences.items():
            if preference_key == "detail_level" and preference_value == "high":
                if len(response) < 200:  # Respuesta corta, el usuario prefiere detalle
                    detailed_addition = await self._generate_additional_details(response, context.domain)
                    if detailed_addition:
                        enhanced_response += f"\n\n{detailed_addition}"
                        enhancement_applied = True
                        confidence_boost += 0.1

            elif preference_key == "format_preference" and preference_value == "examples":
                if "ejemplo" not in response.lower() and "example" not in response.lower():
                    example = await self._generate_domain_example(response, context.domain)
                    if example:
                        enhanced_response += f"\n\n💡 Ejemplo: {example}"
                        enhancement_applied = True
                        confidence_boost += 0.1

        return {
            "enhanced": enhancement_applied,
            "response": enhanced_response,
            "confidence_boost": min(confidence_boost, 0.2),
        }

    async def _apply_technical_depth(self, response: str, query: str, context: SpecializationContext) -> Dict[str, Any]:
        """
        Agregar profundidad técnica según requerimientos
        """
        enhanced_response = response
        enhancement_applied = False
        confidence_boost = 0.0

        # Verificar si se necesitan detalles de implementación
        if any(req in context.technical_requirements for req in ["implementation", "code", "algorithm"]):
            # Agregar detalles de implementación
            impl_details = await self._generate_implementation_details(query, context.domain)
            if impl_details:
                enhanced_response += f"\n\n⚙️ Detalles de implementación:\n{impl_details}"
                enhancement_applied = True
                confidence_boost += 0.15

        # Verificar si se necesitan fundamentos teóricos
        if "theory" in context.technical_requirements or "fundamentos" in query.lower():
            theoretical_foundation = await self._generate_theoretical_foundation(query, context.domain)
            if theoretical_foundation:
                enhanced_response += f"\n\n📖 Fundamentos teóricos:\n{theoretical_foundation}"
                enhancement_applied = True
                confidence_boost += 0.12

        # Verificar si se necesitan consideraciones de performance
        if "performance" in context.technical_requirements or "rendimiento" in query.lower():
            performance_notes = await self._generate_performance_considerations(query, context.domain)
            if performance_notes:
                enhanced_response += f"\n\n⚡ Consideraciones de rendimiento:\n{performance_notes}"
                enhancement_applied = True
                confidence_boost += 0.1

        # Agregar referencias técnicas si es relevante
        if context.user_level in [SpecializationLevel.EXPERT, SpecializationLevel.RESEARCH]:
            technical_refs = await self._generate_technical_references(query, context.domain)
            if technical_refs:
                enhanced_response += f"\n\n📚 Referencias técnicas:\n{technical_refs}"
                enhancement_applied = True
                confidence_boost += 0.08

        return {
            "enhanced": enhancement_applied,
            "response": enhanced_response,
            "confidence_boost": min(confidence_boost, 0.3),
        }

    async def _analyze_query_level(self, query: str) -> SpecializationLevel:
        """Analizar el nivel técnico requerido por la consulta"""
        query_lower = query.lower()

        # Indicadores de nivel avanzado
        advanced_indicators = [
            "implementación",
            "algoritmo",
            "optimización",
            "arquitectura",
            "implementation",
            "algorithm",
            "optimization",
            "architecture",
            "performance",
            "benchmark",
            "análisis",
            "research",
        ]

        # Indicadores de nivel básico
        basic_indicators = [
            "qué es",
            "what is",
            "cómo funciona",
            "how does",
            "explicar",
            "explain",
            "definir",
            "define",
            "introducción",
            "introduction",
        ]

        # Indicadores de nivel intermedio
        intermediate_indicators = [
            "configurar",
            "setup",
            "usar",
            "use",
            "aplicar",
            "apply",
            "comparar",
            "compare",
            "diferencia",
            "difference",
        ]

        advanced_count = sum(1 for indicator in advanced_indicators if indicator in query_lower)
        basic_count = sum(1 for indicator in basic_indicators if indicator in query_lower)
        intermediate_count = sum(1 for indicator in intermediate_indicators if indicator in query_lower)

        if advanced_count >= 2:
            return SpecializationLevel.EXPERT
        elif advanced_count >= 1:
            return SpecializationLevel.ADVANCED
        elif intermediate_count >= 1:
            return SpecializationLevel.INTERMEDIATE
        elif basic_count >= 1:
            return SpecializationLevel.BASIC
        else:
            # Por defecto, nivel intermedio
            return SpecializationLevel.INTERMEDIATE

    async def _classify_query_type(self, query: str) -> str:
        """Clasificar el tipo de consulta"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["cómo", "how", "pasos", "steps"]):
            return "how_to"
        elif any(word in query_lower for word in ["vs", "versus", "diferencia", "difference", "comparar", "compare"]):
            return "comparison"
        elif any(word in query_lower for word in ["qué es", "what is", "definir", "define"]):
            return "definition"
        elif any(word in query_lower for word in ["por qué", "why", "razón", "reason"]):
            return "explanation"
        elif "?" in query:
            return "question"
        else:
            return "general"

    def _load_domain_knowledge(self) -> Dict[str, Dict]:
        """Cargar base de conocimientos por dominio"""
        # En implementación real, cargaría desde archivos/BD
        return {
            "programación": {
                "terminology": {
                    "API": "Interfaz de Programación de Aplicaciones",
                    "Framework": "Marco de trabajo de desarrollo",
                    "Refactoring": "Reestructuración del código sin cambiar funcionalidad",
                },
                "related_concepts": [
                    {
                        "name": "Principios SOLID",
                        "keywords": ["código", "diseño", "arquitectura"],
                        "description": "Cinco principios fundamentales para diseño de software mantenible",
                    },
                    {
                        "name": "Patrones de Diseño",
                        "keywords": ["patrón", "estructura", "solución"],
                        "description": "Soluciones reutilizables a problemas comunes en diseño de software",
                    },
                ],
                "best_practices": [
                    {
                        "triggers": ["código", "función", "variable"],
                        "description": "Usar nombres descriptivos para variables y funciones",
                    },
                    {
                        "triggers": ["test", "testing", "prueba"],
                        "description": "Implementar pruebas unitarias para garantizar calidad",
                    },
                ],
            },
            "matemáticas": {
                "terminology": {
                    "Derivada": "Razón de cambio instantánea de una función",
                    "Integral": "Área bajo la curva de una función",
                    "Límite": "Valor al que se aproxima una función",
                },
                "related_concepts": [
                    {
                        "name": "Teorema Fundamental del Cálculo",
                        "keywords": ["derivada", "integral", "cálculo"],
                        "description": "Conecta los conceptos de derivación e integración",
                    }
                ],
                "best_practices": [
                    {
                        "triggers": ["ecuación", "problema", "resolver"],
                        "description": "Verificar la solución sustituyendo en la ecuación original",
                    }
                ],
            },
        }

    def _load_specialization_patterns(self) -> Dict[str, List[str]]:
        """Cargar patrones de especialización"""
        return {
            "technical_indicators": [
                r"\b(implementar|desarrollar|configurar|optimizar)\b",
                r"\b(algoritmo|estructura|arquitectura|framework)\b",
                r"\b(performance|benchmark|escalabilidad)\b",
            ],
            "educational_indicators": [
                r"\b(aprender|entender|explicar|enseñar)\b",
                r"\b(concepto|principio|fundamento|teoría)\b",
                r"\b(ejemplo|caso|práctica)\b",
            ],
            "problem_solving_indicators": [
                r"\b(resolver|solucionar|debuggear|arreglar)\b",
                r"\b(error|problema|bug|issue)\b",
                r"\b(ayuda|consejo|recomendación)\b",
            ],
        }

    async def _load_dynamic_configurations(self):
        """Cargar configuraciones dinámicas"""
        # En implementación real, cargaría configuraciones actualizables
        pass

    async def _initialize_specialization_models(self):
        """Inicializar modelos de especialización"""
        # En implementación real, cargaría modelos ML para especialización
        pass

    async def _load_pretrained_patterns(self):
        """Cargar patrones pre-entrenados"""
        # En implementación real, cargaría patrones aprendidos
        pass

    # Métodos auxiliares para generación de contenido especializado

    def _simplify_technical_term(self, term: str, domain: str) -> Optional[str]:
        """Simplificar término técnico para nivel básico"""
        simplifications = {
            "API": "forma de comunicación entre programas",
            "Framework": "conjunto de herramientas de desarrollo",
            "Algorithm": "serie de pasos para resolver un problema",
            "Database": "sistema para guardar información de forma organizada",
        }
        return simplifications.get(term)

    async def _generate_advanced_details(self, query: str, domain: str) -> Optional[str]:
        """Generar detalles avanzados para usuarios expertos"""
        if domain == "programación":
            return "Consideraciones de complejidad temporal O(n), patrones de arquitectura recomendados, y optimizaciones específicas del lenguaje."
        elif domain == "matemáticas":
            return "Demostración formal, casos especiales, y conexiones con otros teoremas relacionados."
        return None

    def _extract_action_items(self, text: str) -> List[str]:
        """Extraer elementos de acción del texto"""
        # Buscar verbos de acción y oraciones imperativas
        action_patterns = [
            r"[A-Z][^.!?]*(?:instalar|configurar|ejecutar|crear|definir)[^.!?]*[.!?]",
            r"[A-Z][^.!?]*(?:asegúrate|verifica|comprueba|revisa)[^.!?]*[.!?]",
        ]

        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            actions.extend(matches)

        return actions[:5]  # Máximo 5 acciones

    def _identify_recurring_topics(self, interactions: List[Dict]) -> List[Dict]:
        """Identificar temas recurrentes en interacciones"""
        topic_counts = defaultdict(int)

        for interaction in interactions:
            query = interaction.get("query", "").lower()

            # Extraer palabras clave potenciales
            words = re.findall(r"\b\w{4,}\b", query)  # Palabras de 4+ caracteres
            for word in words:
                if word not in ["cómo", "qué", "cuál", "dónde", "cuándo", "por", "para"]:
                    topic_counts[word] += 1

        # Convertir a lista de temas con frecuencia
        recurring_topics = [{"name": topic, "frequency": count} for topic, count in topic_counts.items() if count >= 2]

        return sorted(recurring_topics, key=lambda x: x["frequency"], reverse=True)[:3]

    def _generate_historical_insight(self, topic: Dict, domain: str) -> Optional[str]:
        """Generar insight basado en tema recurrente"""
        topic_name = topic["name"]
        frequency = topic["frequency"]

        insights = {
            "programación": f"Has consultado sobre {topic_name} {frequency} veces. Esto sugiere interés en profundizar en este aspecto.",
            "matemáticas": f"El tema {topic_name} ha aparecido {frequency} veces en tus consultas. Te recomiendo explorar las aplicaciones prácticas.",
        }

        return insights.get(domain, f"Has mostrado interés recurrente en {topic_name}.")

    async def _extract_domain_preferences(self, interactions: List[Dict], domain: str) -> Dict[str, str]:
        """Extraer preferencias del usuario para el dominio"""
        preferences = {}

        if not interactions:
            return preferences

        # Analizar longitud promedio de consultas para determinar nivel de detalle preferido
        query_lengths = [len(i.get("query", "")) for i in interactions]
        avg_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0

        if avg_length > 100:
            preferences["detail_level"] = "high"
        elif avg_length < 30:
            preferences["detail_level"] = "low"
        else:
            preferences["detail_level"] = "medium"

        # Detectar preferencia por ejemplos
        example_requests = sum(1 for i in interactions if "ejemplo" in i.get("query", "").lower())
        if example_requests >= 2:
            preferences["format_preference"] = "examples"

        return preferences

    async def _identify_technical_requirements(self, query: str) -> List[str]:
        """Identificar requerimientos técnicos de la consulta"""
        requirements = []
        query_lower = query.lower()

        if any(word in query_lower for word in ["implementar", "implement", "código", "code"]):
            requirements.append("implementation")

        if any(word in query_lower for word in ["algoritmo", "algorithm"]):
            requirements.append("algorithm")

        if any(word in query_lower for word in ["performance", "rendimiento", "optimizar"]):
            requirements.append("performance")

        if any(word in query_lower for word in ["teoría", "theory", "fundamentos"]):
            requirements.append("theory")

        return requirements

    async def _generate_additional_details(self, response: str, domain: str) -> Optional[str]:
        """Generar detalles adicionales para respuesta"""
        if domain == "programación":
            return "Detalles adicionales: Considera también las implicaciones de seguridad, mantenibilidad del código y documentación apropiada."
        elif domain == "matemáticas":
            return "Contexto adicional: Explora las aplicaciones en otros campos y las variaciones del concepto."
        return None

    async def _generate_domain_example(self, response: str, domain: str) -> Optional[str]:
        """Generar ejemplo específico del dominio"""
        if domain == "programación":
            return "En Python: `if __name__ == '__main__':` es un patrón común para ejecutar código solo cuando el script se ejecuta directamente."
        elif domain == "matemáticas":
            return (
                "Por ejemplo, la derivada de f(x) = x² es f'(x) = 2x, lo que significa que la pendiente en x=3 sería 6."
            )
        return None

    async def _generate_implementation_details(self, query: str, domain: str) -> Optional[str]:
        """Generar detalles de implementación"""
        if domain == "programación":
            return "• Estructura de datos recomendada\n• Manejo de excepciones\n• Pruebas unitarias\n• Documentación del código"
        return None

    async def _generate_theoretical_foundation(self, query: str, domain: str) -> Optional[str]:
        """Generar fundamentos teóricos"""
        if domain == "matemáticas":
            return "Base teórica: Axiomas fundamentales, definiciones formales y propiedades matemáticas subyacentes."
        return None

    async def _generate_performance_considerations(self, query: str, domain: str) -> Optional[str]:
        """Generar consideraciones de rendimiento"""
        if domain == "programación":
            return "Complejidad temporal, uso de memoria, escalabilidad y optimizaciones específicas del sistema."
        return None

    async def _generate_technical_references(self, query: str, domain: str) -> Optional[str]:
        """Generar referencias técnicas"""
        return "Consulta documentación oficial, papers de investigación y estándares de la industria para información más detallada."

    def _calculate_final_confidence(
        self, base_response: str, enhanced_response: str, confidence_improvements: List[float]
    ) -> float:
        """Calcular confianza final de la especialización"""
        base_confidence = 0.6  # Confianza base

        # Sumar mejoras de confianza
        total_improvement = sum(confidence_improvements)

        # Factor basado en cantidad de texto agregado
        length_factor = len(enhanced_response) / len(base_response) if base_response else 1.0
        length_bonus = min((length_factor - 1.0) * 0.1, 0.2)

        final_confidence = base_confidence + total_improvement + length_bonus

        return min(final_confidence, 1.0)

    async def _assess_technical_level(self, response: str, context: SpecializationContext) -> SpecializationLevel:
        """Evaluar el nivel técnico de la respuesta"""
        # Contar indicadores técnicos
        technical_indicators = len(
            re.findall(r"\b(?:implementación|algoritmo|arquitectura|optimización)\b", response.lower())
        )

        if technical_indicators >= 5:
            return SpecializationLevel.RESEARCH
        elif technical_indicators >= 3:
            return SpecializationLevel.EXPERT
        elif technical_indicators >= 2:
            return SpecializationLevel.ADVANCED
        elif technical_indicators >= 1:
            return SpecializationLevel.INTERMEDIATE
        else:
            return SpecializationLevel.BASIC

    async def _assess_domain_accuracy(self, response: str, context: SpecializationContext) -> float:
        """Evaluar la precisión del dominio en la respuesta"""
        domain_keywords = self.domain_knowledge.get(context.domain, {}).get("terminology", {})

        if not domain_keywords:
            return 0.7  # Valor por defecto

        # Calcular proporción de keywords del dominio presentes
        keywords_found = sum(1 for keyword in domain_keywords.keys() if keyword.lower() in response.lower())

        if not domain_keywords:
            return 0.7

        accuracy = keywords_found / len(domain_keywords)
        return min(accuracy + 0.3, 1.0)  # Mínimo 0.3, máximo 1.0

    def _generate_cache_key(self, query: str, response: str, branch_name: str) -> str:
        """Generar clave de cache para especialización"""
        import hashlib

        cache_content = f"{query[:100]}|{response[:200]}|{branch_name}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _cache_specialization(self, cache_key: str, result: SpecializationResult):
        """Guardar especialización en cache"""
        if len(self.specialization_cache) >= self.cache_size:
            # Remover el más antiguo
            oldest_key = next(iter(self.specialization_cache))
            del self.specialization_cache[oldest_key]

        self.specialization_cache[cache_key] = result

    async def _update_specialization_stats(self, result: SpecializationResult, processing_time: float):
        """Actualizar estadísticas de especialización"""
        self.stats["total_specializations"] += 1

        if result.confidence > self.confidence_threshold:
            self.stats["successful_specializations"] += 1

        # Contar por tipo de especialización
        for spec_type in result.specialization_applied:
            self.stats["specializations_by_type"][spec_type.value] += 1

        # Actualizar tiempo de procesamiento
        self.stats["processing_times"].append(processing_time)

        # Calcular mejora promedio de confianza
        confidence_improvement = result.confidence - 0.6  # Confianza base
        if self.stats["total_specializations"] == 1:
            self.stats["average_confidence_improvement"] = confidence_improvement
        else:
            current_avg = self.stats["average_confidence_improvement"]
            total = self.stats["total_specializations"]
            self.stats["average_confidence_improvement"] = (current_avg * (total - 1) + confidence_improvement) / total

    def get_stats(self) -> Dict:
        """Obtener estadísticas del motor de especialización"""
        stats = self.stats.copy()

        if self.stats["processing_times"]:
            stats["average_processing_time"] = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        else:
            stats["average_processing_time"] = 0.0

        return stats

    async def health_check(self) -> Dict:
        """Verificar estado de salud del motor"""
        return {
            "status": "healthy",
            "total_specializations": self.stats["total_specializations"],
            "success_rate": (
                self.stats["successful_specializations"] / self.stats["total_specializations"]
                if self.stats["total_specializations"] > 0
                else 0.0
            ),
            "cache_size": len(self.specialization_cache),
            "domain_expertise_enabled": self.enable_domain_expertise,
            "contextual_adaptation_enabled": self.enable_contextual_adaptation,
        }

    async def shutdown(self):
        """Cerrar motor de especialización"""
        logger.info("Cerrando SpecializationEngine")

        # Limpiar cache
        self.specialization_cache.clear()

        # Limpiar histórico
        self.interaction_history.clear()

        # Log estadísticas finales
        final_stats = self.get_stats()
        logger.info(f"Estadísticas finales de SpecializationEngine: {final_stats}")

        logger.info("SpecializationEngine cerrado")
