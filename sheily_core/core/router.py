#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperRouter - Router Principal del Sistema Sheily-AI
====================================================

Coordina el routing inteligente de consultas hacia los componentes
más apropiados del sistema basado en análisis de contenido y contexto.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .branch_selector import BranchSelector
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Tipos de rutas disponibles"""

    RAG_SEARCH = "rag_search"
    LLM_GENERATION = "llm_generation"
    HYBRID_RAG_LLM = "hybrid_rag_llm"
    SPECIALIZED_BRANCH = "specialized_branch"
    FALLBACK = "fallback"


class Priority(Enum):
    """Niveles de prioridad de procesamiento"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RouteRequest:
    """Estructura de solicitud de routing"""

    id: str
    query: str
    language: str
    domain: Optional[str] = None
    route_type: Optional[RouteType] = None
    priority: Priority = Priority.MEDIUM
    context: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RouteResponse:
    """Estructura de respuesta de routing"""

    request_id: str
    success: bool
    route_taken: RouteType
    response: Optional[str] = None
    components_used: Optional[List[str]] = None
    selected_components: Optional[List[str]] = None
    execution_time: float = 0.0
    processing_time: Optional[float] = None  # Alias para compatibilidad
    result: Optional[Any] = None  # Resultado completo de la operación
    error: Optional[str] = None  # Mensaje de error si falló
    metadata: Optional[Dict[str, Any]] = None


class HyperRouter:
    """
    Router principal del sistema con capacidades inteligentes
    """

    def __init__(self, config: Dict):
        """
        Inicializar el HyperRouter

        Args:
            config: Configuración del router
        """
        self.config = config
        self.routing_strategy = config.get("routing_strategy", "intelligent")
        self.max_concurrent = config.get("max_concurrent_routes", 50)
        self.route_timeout = config.get("route_timeout", 30.0)
        self.fallback_enabled = config.get("fallback_enabled", True)

        # Inicializar componentes
        self.branch_selector = BranchSelector(config)
        self.load_balancer = LoadBalancer(config)

        # Estado del router
        self._active_routes = {}  # {request_id: RouteRequest}
        self._route_history = []
        self._component_health = {}

        # Estrategias de routing disponibles
        self._routing_strategies = {
            "intelligent": self._execute_intelligent_routing,
            "round_robin": self._round_robin_routing,
            "weighted": self._weighted_routing,
            "priority": self._priority_routing,
        }

        # Patrones de consulta para routing
        self._query_patterns = {
            "search_patterns": [
                r"\b(buscar|encontrar|search|find)\b",
                r"\b(qué es|what is|define|definition)\b",
                r"\b(información|information|about)\b",
            ],
            "generation_patterns": [
                r"\b(generar|generate|crear|create|escribir|write)\b",
                r"\b(explícame|explain|cuéntame|tell me)\b",
                r"\b(cómo|how to|tutorial)\b",
            ],
            "analytical_patterns": [
                r"\b(analizar|analyze|comparar|compare)\b",
                r"\b(diferencia|difference|similitud|similarity)\b",
                r"\b(pros y contras|pros and cons)\b",
            ],
        }

        # Métricas de routing
        self._routing_stats = {
            "total_routes": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_routing_time": 0.0,
            "routes_by_type": {route_type.value: 0 for route_type in RouteType},
            "component_usage": {},
        }

        logger.info("HyperRouter inicializado")

    async def initialize(self) -> bool:
        """
        Inicializar el router y sus componentes

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Inicializar componentes
            await asyncio.gather(self.branch_selector.initialize(), self.load_balancer.initialize())

            # Iniciar monitoreo de salud
            asyncio.create_task(self._health_monitoring_loop())

            logger.info("HyperRouter inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando HyperRouter: {e}")
            return False

    async def route(self, request: RouteRequest) -> RouteResponse:
        """
        Procesar solicitud de routing

        Args:
            request: Solicitud de routing

        Returns:
            Respuesta con resultado del routing
        """
        start_time = time.time()

        try:
            # Verificar capacidad
            if len(self._active_routes) >= self.max_concurrent:
                return RouteResponse(
                    request_id=request.id,
                    route_taken=RouteType.FALLBACK,
                    selected_components=[],
                    processing_time=0.0,
                    success=False,
                    error="Sistema saturado - máxima capacidad alcanzada",
                )

            # Registrar ruta activa
            self._active_routes[request.id] = request

            try:
                # Determinar estrategia de routing
                if request.route_type:
                    # Ruta específica solicitada
                    route_response = await self._execute_specific_route(request)
                else:
                    # Routing inteligente automático
                    route_response = await self._execute_intelligent_routing(request)

                # Actualizar métricas
                processing_time = time.time() - start_time
                route_response.processing_time = processing_time

                self._update_routing_metrics(route_response, processing_time)

                return route_response

            finally:
                # Limpiar ruta activa
                if request.id in self._active_routes:
                    del self._active_routes[request.id]

        except Exception as e:
            logger.error(f"Error en routing para {request.id}: {e}")

            return RouteResponse(
                request_id=request.id,
                route_taken=RouteType.FALLBACK,
                selected_components=[],
                processing_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    async def _execute_intelligent_routing(self, request: RouteRequest) -> RouteResponse:
        """
        Ejecutar routing inteligente basado en análisis de contenido

        Args:
            request: Solicitud de routing

        Returns:
            Respuesta del routing inteligente
        """
        # Analizar consulta para determinar tipo óptimo
        analyzed_type = await self._analyze_query_type(request.query)

        # Seleccionar rama especializada si aplica
        selected_branch = await self.branch_selector.select_branch(
            request.query, request.language, request.domain
        )

        # Determinar componentes necesarios
        components = await self._select_components(analyzed_type, selected_branch)

        # Balancear carga entre componentes disponibles
        balanced_components = await self.load_balancer.balance_components(
            components, request.priority
        )

        # Ejecutar routing
        try:
            if analyzed_type == RouteType.RAG_SEARCH:
                result = await self._execute_rag_search(request, balanced_components)
            elif analyzed_type == RouteType.LLM_GENERATION:
                result = await self._execute_llm_generation(request, balanced_components)
            elif analyzed_type == RouteType.HYBRID_RAG_LLM:
                result = await self._execute_hybrid_route(request, balanced_components)
            elif analyzed_type == RouteType.SPECIALIZED_BRANCH:
                result = await self._execute_specialized_branch(
                    request, balanced_components, selected_branch
                )
            else:
                result = await self._execute_fallback_route(request)
                analyzed_type = RouteType.FALLBACK

            return RouteResponse(
                request_id=request.id,
                route_taken=analyzed_type,
                selected_components=balanced_components,
                processing_time=0.0,  # Se actualizará después
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error(f"Error ejecutando route {analyzed_type}: {e}")

            if self.fallback_enabled:
                fallback_result = await self._execute_fallback_route(request)
                return RouteResponse(
                    request_id=request.id,
                    route_taken=RouteType.FALLBACK,
                    selected_components=["fallback"],
                    processing_time=0.0,
                    success=True,
                    result=fallback_result,
                    error=f"Fallback usado debido a: {str(e)}",
                )
            else:
                raise e

    async def _analyze_query_type(self, query: str) -> RouteType:
        """
        Analizar consulta para determinar el tipo de routing óptimo

        Args:
            query: Consulta a analizar

        Returns:
            Tipo de routing recomendado
        """
        import re

        query_lower = query.lower()

        # Contadores para cada tipo
        search_score = 0
        generation_score = 0
        analytical_score = 0

        # Evaluar patrones de búsqueda
        for pattern in self._query_patterns["search_patterns"]:
            if re.search(pattern, query_lower):
                search_score += 1

        # Evaluar patrones de generación
        for pattern in self._query_patterns["generation_patterns"]:
            if re.search(pattern, query_lower):
                generation_score += 1

        # Evaluar patrones analíticos
        for pattern in self._query_patterns["analytical_patterns"]:
            if re.search(pattern, query_lower):
                analytical_score += 1

        # Factores adicionales
        if "?" in query and len(query.split()) < 10:
            search_score += 1

        if len(query.split()) > 20:
            generation_score += 1

        if any(word in query_lower for word in ["vs", "versus", "compare", "diferencia"]):
            analytical_score += 2

        # Determinar tipo basado en scores
        if analytical_score > 0 and (generation_score > 0 or search_score > 0):
            return RouteType.HYBRID_RAG_LLM
        elif generation_score > search_score:
            return RouteType.LLM_GENERATION
        elif search_score > 0:
            return RouteType.RAG_SEARCH
        elif generation_score > 0:
            return RouteType.LLM_GENERATION
        else:
            # Por defecto, usar híbrido para máxima flexibilidad
            return RouteType.HYBRID_RAG_LLM

    async def _select_components(self, route_type: RouteType, branch: Optional[str]) -> List[str]:
        """
        Seleccionar componentes necesarios para un tipo de routing

        Args:
            route_type: Tipo de routing
            branch: Rama especializada seleccionada

        Returns:
            Lista de componentes necesarios
        """
        components = []

        if route_type == RouteType.RAG_SEARCH:
            components = ["rag_engine", "retrieval_manager", "embedding_manager"]

        elif route_type == RouteType.LLM_GENERATION:
            components = ["llm_engine", "model_manager", "tokenizer_manager"]

        elif route_type == RouteType.HYBRID_RAG_LLM:
            components = [
                "rag_engine",
                "llm_engine",
                "retrieval_manager",
                "model_manager",
                "embedding_manager",
            ]

        elif route_type == RouteType.SPECIALIZED_BRANCH:
            components = [
                "branch_manager",
                "llm_engine",
                "rag_engine",
                f"branch_{branch}" if branch else "branch_general",
            ]

        # Siempre incluir componentes de seguridad
        components.append("depswitch")

        return components

    async def _execute_rag_search(
        self, request: RouteRequest, components: List[str]
    ) -> Dict[str, Any]:
        """Ejecutar routing de búsqueda RAG"""
        # En implementación real, llamaría al RAG Engine
        result = {
            "type": "rag_search",
            "query": request.query,
            "language": request.language,
            "domain": request.domain,
            "documents": [
                {
                    "title": "Documento RAG 1",
                    "content": f"Información relevante para: {request.query}",
                    "score": 0.9,
                },
                {
                    "title": "Documento RAG 2",
                    "content": f"Contexto adicional sobre: {request.query}",
                    "score": 0.8,
                },
            ],
            "components_used": components,
            "routing_method": "rag_search",
        }

        # Simular tiempo de procesamiento
        await asyncio.sleep(0.1)

        return result

    async def _execute_llm_generation(
        self, request: RouteRequest, components: List[str]
    ) -> Dict[str, Any]:
        """Ejecutar routing de generación LLM"""
        # En implementación real, llamaría al LLM Engine
        result = {
            "type": "llm_generation",
            "query": request.query,
            "generated_text": f"Respuesta generada para: {request.query}",
            "language": request.language,
            "model_used": "sheily_specialized",
            "components_used": components,
            "routing_method": "llm_generation",
        }

        # Simular tiempo de procesamiento
        await asyncio.sleep(0.2)

        return result

    async def _execute_hybrid_route(
        self, request: RouteRequest, components: List[str]
    ) -> Dict[str, Any]:
        """Ejecutar routing híbrido RAG+LLM"""
        # Combinar búsqueda RAG con generación LLM

        # Paso 1: Búsqueda RAG
        rag_result = await self._execute_rag_search(request, components)

        # Paso 2: Generación LLM basada en contexto RAG
        enhanced_request = RouteRequest(
            id=f"{request.id}_gen",
            query=f"Basado en esta información: {rag_result['documents'][0]['content']}, responde: {request.query}",
            language=request.language,
            domain=request.domain,
            context={"rag_context": rag_result},
        )

        llm_result = await self._execute_llm_generation(enhanced_request, components)

        # Combinar resultados
        result = {
            "type": "hybrid_rag_llm",
            "query": request.query,
            "rag_results": rag_result["documents"],
            "generated_response": llm_result["generated_text"],
            "language": request.language,
            "components_used": components,
            "routing_method": "hybrid_rag_llm",
        }

        return result

    async def _execute_specialized_branch(
        self, request: RouteRequest, components: List[str], branch: str
    ) -> Dict[str, Any]:
        """Ejecutar routing de rama especializada"""
        result = {
            "type": "specialized_branch",
            "branch": branch,
            "query": request.query,
            "specialized_response": f"Respuesta especializada de {branch} para: {request.query}",
            "language": request.language,
            "domain": request.domain,
            "components_used": components,
            "routing_method": "specialized_branch",
        }

        # Simular procesamiento especializado
        await asyncio.sleep(0.15)

        return result

    async def _execute_fallback_route(self, request: RouteRequest) -> Dict[str, Any]:
        """Ejecutar routing de fallback"""
        result = {
            "type": "fallback",
            "query": request.query,
            "fallback_response": f"Respuesta básica para: {request.query}",
            "language": request.language,
            "components_used": ["fallback"],
            "routing_method": "fallback",
            "note": "Respuesta generada por sistema de fallback",
        }

        return result

    async def _execute_specific_route(self, request: RouteRequest) -> RouteResponse:
        """Ejecutar routing específico solicitado"""
        components = await self._select_components(request.route_type, None)
        balanced_components = await self.load_balancer.balance_components(
            components, request.priority
        )

        if request.route_type == RouteType.RAG_SEARCH:
            result = await self._execute_rag_search(request, balanced_components)
        elif request.route_type == RouteType.LLM_GENERATION:
            result = await self._execute_llm_generation(request, balanced_components)
        elif request.route_type == RouteType.HYBRID_RAG_LLM:
            result = await self._execute_hybrid_route(request, balanced_components)
        else:
            result = await self._execute_fallback_route(request)

        return RouteResponse(
            request_id=request.id,
            route_taken=request.route_type,
            selected_components=balanced_components,
            processing_time=0.0,
            success=True,
            result=result,
        )

    async def _health_monitoring_loop(self):
        """Loop de monitoreo de salud de componentes"""
        health_check_interval = self.config.get("health_check_interval", 60.0)

        while True:
            try:
                # Verificar salud de componentes
                component_health = await self._check_component_health()
                self._component_health.update(component_health)

                # Log estado de salud
                healthy_components = sum(1 for status in component_health.values() if status)
                total_components = len(component_health)

                logger.info(
                    f"Health check: {healthy_components}/{total_components} componentes healthy"
                )

                await asyncio.sleep(health_check_interval)

            except Exception as e:
                logger.error(f"Error en health monitoring: {e}")
                await asyncio.sleep(health_check_interval)

    async def _check_component_health(self) -> Dict[str, bool]:
        """Verificar salud de todos los componentes"""
        # En implementación real, verificaría cada componente
        components = ["rag_engine", "llm_engine", "branch_selector", "load_balancer", "depswitch"]

        health_status = {}

        for component in components:
            try:
                # Simular health check
                await asyncio.sleep(0.01)
                health_status[component] = True
            except:
                health_status[component] = False

        return health_status

    def _update_routing_metrics(self, response: RouteResponse, processing_time: float):
        """Actualizar métricas de routing"""
        self._routing_stats["total_routes"] += 1

        if response.success:
            self._routing_stats["successful_routes"] += 1
        else:
            self._routing_stats["failed_routes"] += 1

        # Actualizar tiempo promedio
        total_routes = self._routing_stats["total_routes"]
        current_avg = self._routing_stats["average_routing_time"]

        self._routing_stats["average_routing_time"] = (
            current_avg * (total_routes - 1) + processing_time
        ) / total_routes

        # Contar por tipo de ruta
        if response.route_taken:
            route_type_key = response.route_taken.value
            self._routing_stats["routes_by_type"][route_type_key] += 1

        # Contar uso de componentes
        for component in response.selected_components:
            if component not in self._routing_stats["component_usage"]:
                self._routing_stats["component_usage"][component] = 0
            self._routing_stats["component_usage"][component] += 1

    def get_active_routes(self) -> Dict[str, RouteRequest]:
        """Obtener rutas actualmente activas"""
        return self._active_routes.copy()

    def get_component_health(self) -> Dict[str, bool]:
        """Obtener estado de salud de componentes"""
        return self._component_health.copy()

    async def _round_robin_routing(self, request: RouteRequest) -> RouteResponse:
        """Routing round-robin simple"""
        # Implementación básica para compatibilidad
        return await self._execute_intelligent_routing(request)

    async def _weighted_routing(self, request: RouteRequest) -> RouteResponse:
        """Routing basado en pesos de componentes"""
        # Implementación básica para compatibilidad
        return await self._execute_intelligent_routing(request)

    async def _priority_routing(self, request: RouteRequest) -> RouteResponse:
        """Routing basado en prioridad de requests"""
        # Implementación básica para compatibilidad
        return await self._execute_intelligent_routing(request)

    def get_stats(self) -> Dict:
        """Obtener estadísticas del router"""
        return {
            **self._routing_stats,
            "active_routes": len(self._active_routes),
            "component_health": self._component_health,
            "routing_strategy": self.routing_strategy,
        }

    async def health_check(self) -> Dict:
        """Verificar estado de salud del router"""
        component_health = await self._check_component_health()
        healthy_count = sum(1 for status in component_health.values() if status)
        total_count = len(component_health)

        return {
            "status": "healthy" if healthy_count == total_count else "degraded",
            "healthy_components": healthy_count,
            "total_components": total_count,
            "active_routes": len(self._active_routes),
            "routing_strategy": self.routing_strategy,
            "stats": self.get_stats(),
        }

    async def shutdown(self):
        """Cerrar router y limpiar recursos"""
        logger.info("Iniciando shutdown del HyperRouter")

        try:
            # Esperar que terminen las rutas activas
            if self._active_routes:
                logger.info(f"Esperando {len(self._active_routes)} rutas activas...")
                # En implementación real, esperaría o cancelaría rutas activas
                await asyncio.sleep(1.0)

            # Shutdown de componentes
            await asyncio.gather(self.branch_selector.shutdown(), self.load_balancer.shutdown())

            logger.info("HyperRouter shutdown completado")

        except Exception as e:
            logger.error(f"Error durante shutdown: {e}")
            raise
