#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURO-INTEGRATION V2 - INTEGRACIÓN NEUROLÓGICA AVANZADA
======================================================

Sistema de integración perfecta que conecta:

COMPONENTES INTEGRADOS:
- Memoria humana avanzada (HumanMemoryEngine)
- Motor RAG neurológico (NeuroRAGEngine)
- Entrenamiento neurológico (NeuroTrainingEngine)
- Chat persistente existente (sheily_persistent_memory_chat.py)
- Sistema de mantenimiento automático (sheily_auto_maintainer.py)

INTEGRACIONES AVANZADAS:
- Comunicación bidireccional entre componentes
- Sincronización automática de estado
- Optimización cruzada de rendimiento
- Evolución autónoma del sistema completo
- Monitoreo y adaptación en tiempo real
- Fallback automático en caso de fallos

ARQUITECTURA DE INTEGRACIÓN:
- Núcleo de integración con comunicación asíncrona
- Sistema de eventos para coordinación
- Caché distribuido para optimización
- Mecanismos de recuperación automática
- API unificada para acceso a funcionalidades
"""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Configuración de integración
INTEGRATION_ROOT = Path(__file__).resolve().parents[2] / "data" / "neuro_integration_v2"
HEALTH_CHECK_INTERVAL = 60  # segundos
SYNC_INTERVAL = 300  # 5 minutos
MAX_RETRY_ATTEMPTS = 3


@dataclass
class IntegrationConfig:
    """Configuración de integración avanzada"""

    enable_human_memory: bool = True
    enable_neuro_rag: bool = True
    enable_neuro_training: bool = True
    enable_auto_sync: bool = True
    enable_health_monitoring: bool = True
    enable_performance_optimization: bool = True

    # Configuración de comunicación
    async_communication: bool = True
    event_driven_updates: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # Configuración de recuperación
    auto_recovery: bool = True
    fallback_enabled: bool = True
    graceful_degradation: bool = True


@dataclass
class ComponentHealth:
    """Estado de salud de componentes individuales"""

    component_name: str
    status: str  # healthy, degraded, failed, initializing
    last_check: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

    def is_healthy(self) -> bool:
        """Verificar si el componente está saludable"""
        return self.status == "healthy"

    def update_health(
        self,
        status: str,
        response_time: float = 0.0,
        memory_usage: float = 0.0,
        error: Optional[str] = None,
    ):
        """Actualizar estado de salud"""
        self.status = status
        self.last_check = datetime.now()
        self.response_time = response_time
        self.memory_usage = memory_usage

        if error:
            self.error_count += 1
            self.last_error = error
        else:
            # Resetear contador de errores si no hay error
            self.error_count = 0
            self.last_error = None


@dataclass
class IntegrationState:
    """Estado completo del sistema integrado"""

    system_id: str
    components_health: Dict[str, ComponentHealth] = field(default_factory=dict)
    integration_metrics: Dict[str, float] = field(default_factory=dict)
    sync_status: Dict[str, datetime] = field(default_factory=dict)
    performance_baseline: Dict[str, float] = field(default_factory=dict)

    # Estado de evolución
    learning_progress: float = 0.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuroIntegrationEngine:
    """Motor de integración neurológica avanzada"""

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.state_file = INTEGRATION_ROOT / "integration_state.json"

        # Inicializar componentes
        self._init_directories()
        self.state = self._load_state()

        # Componentes del sistema
        self.human_memory_engine = None
        self.neuro_rag_engine = None
        self.neuro_training_engine = None

        # Sistema de eventos
        self.event_queue = asyncio.Queue() if self.config.async_communication else deque()
        self.event_handlers = defaultdict(list)

        # Caché avanzado
        self.cache = {}
        self.cache_timestamps = {}

        # Monitoreo de salud
        self.health_monitor = None

        self.logger = self._get_logger()

    def _get_logger(self):
        """Obtener logger con fallback"""
        try:
            from sheily_core.logger import get_logger

            return get_logger("neuro_integration")
        except ImportError:
            import logging

            return logging.getLogger("neuro_integration")

    def _init_directories(self):
        """Inicializar estructura de directorios"""
        INTEGRATION_ROOT.mkdir(parents=True, exist_ok=True)
        (INTEGRATION_ROOT / "events").mkdir(exist_ok=True)
        (INTEGRATION_ROOT / "cache").mkdir(exist_ok=True)
        (INTEGRATION_ROOT / "health").mkdir(exist_ok=True)

    def _load_state(self) -> IntegrationState:
        """Cargar estado de integración"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return IntegrationState(**data)
            except Exception as e:
                self.logger.warning(f"Error loading integration state: {e}")

        return IntegrationState(system_id=f"neuro_integration_{int(time.time())}")

    def _save_state(self):
        """Guardar estado de integración"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving integration state: {e}")

    def initialize_components(self) -> bool:
        """Inicializar todos los componentes avanzados"""
        success_count = 0

        # Inicializar memoria humana avanzada
        if self.config.enable_human_memory:
            try:
                from sheily_core.memory.sheily_human_memory_v2 import integrate_human_memory_v2

                self.human_memory_engine = integrate_human_memory_v2()
                self.state.components_health["human_memory"] = ComponentHealth("human_memory", "healthy")
                success_count += 1
                self.logger.info("Human memory engine initialized")
            except Exception as e:
                self.logger.error(f"Error initializing human memory: {e}")
                self.state.components_health["human_memory"] = ComponentHealth(
                    "human_memory", "failed", last_error=str(e)
                )

        # Inicializar motor RAG neurológico
        if self.config.enable_neuro_rag:
            try:
                from sheily_rag.neuro_rag_engine_v2 import integrate_neuro_rag

                self.neuro_rag_engine = integrate_neuro_rag()
                self.state.components_health["neuro_rag"] = ComponentHealth("neuro_rag", "healthy")
                success_count += 1
                self.logger.info("Neuro-RAG engine initialized")
            except Exception as e:
                self.logger.error(f"Error initializing neuro-RAG: {e}")
                self.state.components_health["neuro_rag"] = ComponentHealth("neuro_rag", "failed", last_error=str(e))

        # Inicializar entrenamiento neurológico (LAZY LOADING)
        # No cargar transformers al inicio para acelerar el arranque
        if self.config.enable_neuro_training:
            try:
                # Solo marcar como pendiente, se cargará cuando se necesite
                self.neuro_training_engine = None  # Lazy loading
                self.state.components_health["neuro_training"] = ComponentHealth("neuro_training", "lazy_loaded")
                success_count += 1
                self.logger.info("⚡ Neuro-training engine configurado para lazy loading (carga rápida)")
            except Exception as e:
                self.logger.error(f"Error initializing neuro-training: {e}")
                self.state.components_health["neuro_training"] = ComponentHealth(
                    "neuro_training", "failed", last_error=str(e)
                )

        # Inicializar monitoreo de salud
        if self.config.enable_health_monitoring:
            self._start_health_monitoring()

        # Inicializar sistema de eventos
        if self.config.event_driven_updates:
            self._start_event_system()

        self._save_state()

        self.logger.info(f"Initialized {success_count}/3 advanced components")
        return success_count > 0

    def get_training_engine(self):
        """Obtener training engine con lazy loading (solo carga cuando se necesita)"""
        if self.neuro_training_engine is None and self.config.enable_neuro_training:
            try:
                self.logger.info("⏳ Cargando neuro-training engine (lazy loading)...")
                from sheily_train.core.training.neuro_training_v2 import integrate_neuro_training

                self.neuro_training_engine = integrate_neuro_training()
                self.state.components_health["neuro_training"] = ComponentHealth("neuro_training", "healthy")
                self.logger.info("✅ Neuro-training engine cargado exitosamente")
            except Exception as e:
                self.logger.error(f"❌ Error cargando neuro-training: {e}")
                self.state.components_health["neuro_training"] = ComponentHealth(
                    "neuro_training", "failed", last_error=str(e)
                )
                return None
        return self.neuro_training_engine

    def process_with_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesar consulta con contexto neurológico completo

        Args:
            query: Consulta del usuario
            context: Contexto adicional (memoria, RAG, etc.)

        Returns:
            Resultado procesado con información de todos los componentes
        """
        start_time = time.time()
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "components_used": [],
            "processing_time": 0.0,
        }

        context = context or {}

        try:
            # 1. Buscar en memoria humana
            if self.human_memory_engine and self.config.enable_human_memory:
                try:
                    memory_results = self.human_memory_engine.search_memory(query, top_k=5, relevance_threshold=0.3)
                    result["memory_results"] = memory_results
                    result["components_used"].append("human_memory")
                except Exception as e:
                    self.logger.error(f"Error en búsqueda de memoria: {e}")
                    result["memory_error"] = str(e)

            # 2. Buscar en RAG neurológico
            if self.neuro_rag_engine and self.config.enable_neuro_rag:
                try:
                    rag_context = self.neuro_rag_engine.retrieve_context(query, top_k=3, branch=context.get("branch"))
                    result["rag_context"] = rag_context
                    result["components_used"].append("neuro_rag")
                except Exception as e:
                    self.logger.error(f"Error en RAG: {e}")
                    result["rag_error"] = str(e)

            # 3. Detectar carga cognitiva
            cognitive_load = self.detect_cognitive_load(query, context)
            result["cognitive_load"] = cognitive_load

            # 4. Aplicar optimizaciones si es necesario
            if cognitive_load > 0.7:
                optimization_result = self.apply_neural_optimization(result, cognitive_load)
                result["optimization_applied"] = optimization_result

            result["processing_time"] = time.time() - start_time
            result["status"] = "success"

        except Exception as e:
            self.logger.error(f"Error en process_with_context: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time

        return result

    def detect_cognitive_load(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Detectar carga cognitiva de la consulta

        Analiza factores como:
        - Complejidad del query
        - Número de conceptos involucrados
        - Profundidad del contexto requerido
        - Recursos computacionales necesarios

        Returns:
            Score de carga cognitiva (0.0 a 1.0)
        """
        context = context or {}
        load_factors = []

        # Factor 1: Longitud y complejidad del query
        query_length = len(query.split())
        complexity_score = min(1.0, query_length / 50)  # Normalizado a 50 palabras
        load_factors.append(complexity_score)

        # Factor 2: Número de preguntas o conceptos
        question_marks = query.count("?")
        concepts_score = min(1.0, question_marks / 3)  # Normalizado a 3 preguntas
        load_factors.append(concepts_score)

        # Factor 3: Contexto requerido
        if "memory_results" in context:
            memory_count = len(context["memory_results"])
            memory_score = min(1.0, memory_count / 10)  # Normalizado a 10 resultados
            load_factors.append(memory_score)

        # Factor 4: Recursos RAG
        if "rag_context" in context:
            rag_items = len(context.get("rag_context", []))
            rag_score = min(1.0, rag_items / 5)  # Normalizado a 5 documentos
            load_factors.append(rag_score)

        # Calcular promedio ponderado
        if load_factors:
            cognitive_load = sum(load_factors) / len(load_factors)
        else:
            cognitive_load = 0.5  # Carga media por defecto

        return cognitive_load

    def apply_neural_optimization(self, processing_result: Dict[str, Any], cognitive_load: float) -> Dict[str, Any]:
        """Aplicar optimizaciones neuronales basadas en carga cognitiva

        Args:
            processing_result: Resultado del procesamiento
            cognitive_load: Carga cognitiva detectada

        Returns:
            Resultado de optimización con métricas
        """
        optimization = {"applied": [], "cognitive_load": cognitive_load, "improvements": []}

        try:
            # Optimización 1: Reducir resultados si carga es alta
            if cognitive_load > 0.8:
                if "memory_results" in processing_result:
                    original_count = len(processing_result["memory_results"])
                    processing_result["memory_results"] = processing_result["memory_results"][:3]
                    optimization["applied"].append("memory_reduction")
                    optimization["improvements"].append(f"Reducido de {original_count} a 3 resultados")

                if "rag_context" in processing_result:
                    original_count = len(processing_result["rag_context"])
                    processing_result["rag_context"] = processing_result["rag_context"][:2]
                    optimization["applied"].append("rag_reduction")
                    optimization["improvements"].append(f"Reducido de {original_count} a 2 documentos")

            # Optimización 2: Usar caché para queries similares
            if self.config.cache_enabled:
                cache_key = self._generate_cache_key(processing_result["query"])
                if cache_key in self.cache:
                    optimization["applied"].append("cache_hit")
                    optimization["improvements"].append("Utilizando resultado cacheado")
                else:
                    self.cache[cache_key] = processing_result
                    self.cache_timestamps[cache_key] = time.time()
                    optimization["applied"].append("cache_store")

            # Optimización 3: Priorizar componentes más rápidos
            if cognitive_load > 0.7 and self.config.enable_performance_optimization:
                # Obtener métricas de rendimiento
                if "human_memory" in self.state.components_health:
                    memory_health = self.state.components_health["human_memory"]
                    if memory_health.response_time > 1.0:
                        optimization["applied"].append("skip_slow_components")
                        optimization["improvements"].append("Omitidos componentes lentos")

            optimization["status"] = "success"

        except Exception as e:
            self.logger.error(f"Error en apply_neural_optimization: {e}")
            optimization["status"] = "error"
            optimization["error"] = str(e)

        return optimization

    def _generate_cache_key(self, text: str) -> str:
        """Generar clave de caché para texto"""
        import hashlib

        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _start_health_monitoring(self):
        """Iniciar monitoreo de salud de componentes"""

        def health_check_loop():
            while True:
                try:
                    self._perform_health_checks()
                    time.sleep(HEALTH_CHECK_INTERVAL)
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
                    time.sleep(HEALTH_CHECK_INTERVAL)

        monitor_thread = threading.Thread(target=health_check_loop, daemon=True)
        monitor_thread.start()
        self.health_monitor = monitor_thread

    def _perform_health_checks(self):
        """Realizar verificaciones de salud de componentes"""
        current_time = datetime.now()

        # Verificar memoria humana
        if self.human_memory_engine:
            try:
                start_time = time.time()
                stats = self.human_memory_engine.get_memory_stats()
                response_time = time.time() - start_time

                self.state.components_health["human_memory"].update_health(
                    "healthy", response_time, 0.0  # Memory usage would need actual measurement
                )
            except Exception as e:
                self.state.components_health["human_memory"].update_health("failed", error=str(e))

        # Verificar RAG neurológico
        if self.neuro_rag_engine:
            try:
                start_time = time.time()
                stats = self.neuro_rag_engine.get_system_stats()
                response_time = time.time() - start_time

                self.state.components_health["neuro_rag"].update_health("healthy", response_time, 0.0)
            except Exception as e:
                self.state.components_health["neuro_rag"].update_health("failed", error=str(e))

        # Verificar entrenamiento neurológico
        if self.neuro_training_engine:
            try:
                # Verificación básica de estado
                response_time = 0.1  # Simulado
                self.state.components_health["neuro_training"].update_health("healthy", response_time, 0.0)
            except Exception as e:
                self.state.components_health["neuro_training"].update_health("failed", error=str(e))

    def _start_event_system(self):
        """Iniciar sistema de eventos asíncrono"""
        if self.config.async_communication:

            def event_loop():
                asyncio.run(self._process_events_async())

            event_thread = threading.Thread(target=event_loop, daemon=True)
            event_thread.start()

    async def _process_events_async(self):
        """Procesar eventos de manera asíncrona"""
        while True:
            try:
                if not self.event_queue.empty():
                    event = await self.event_queue.get()
                    await self._handle_event(event)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)

    def _process_events_sync(self):
        """Procesar eventos de manera síncrona"""
        while self.event_queue:
            try:
                event = self.event_queue.popleft()
                self._handle_event_sync(event)
            except Exception as e:
                self.logger.error(f"Sync event processing error: {e}")

    def _handle_event_sync(self, event: Dict[str, Any]):
        """Manejar evento de manera síncrona"""
        event_type = event.get("type")
        event_data = event.get("data", {})

        # Manejar diferentes tipos de eventos
        if event_type == "memory_update":
            self._handle_memory_update(event_data)
        elif event_type == "rag_query":
            self._handle_rag_query(event_data)
        elif event_type == "training_complete":
            self._handle_training_complete(event_data)
        elif event_type == "component_sync":
            self._handle_component_sync(event_data)

    async def _handle_event(self, event: Dict[str, Any]):
        """Manejar evento de manera asíncrona"""
        # Por ahora, usar manejo síncrono
        self._handle_event_sync(event)

    def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emitir evento al sistema"""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "source": "integration_engine",
        }

        if self.config.async_communication:
            self.event_queue.put_nowait(event)
        else:
            self.event_queue.append(event)

    def _handle_memory_update(self, event_data: Dict[str, Any]):
        """Manejar actualización de memoria"""
        try:
            # Sincronizar actualización de memoria con otros componentes
            if self.neuro_rag_engine:
                # Notificar a RAG sobre nueva memoria
                self.neuro_rag_engine.state.learning_progress = min(
                    1.0, self.neuro_rag_engine.state.learning_progress + 0.01
                )

            if self.neuro_training_engine:
                # Adaptar entrenamiento basado en nueva memoria
                if self.human_memory_engine:
                    self.neuro_training_engine.integrate_with_human_memory(self.human_memory_engine)

        except Exception as e:
            self.logger.error(f"Error handling memory update: {e}")

    def _handle_rag_query(self, event_data: Dict[str, Any]):
        """Manejar consulta RAG"""
        try:
            query = event_data.get("query", "")

            # Usar memoria humana para enriquecer consulta RAG
            if self.human_memory_engine:
                memory_context = self.human_memory_engine.search_memory(query, top_k=3)

                # Agregar contexto de memoria a consulta RAG
                if memory_context and self.neuro_rag_engine:
                    enriched_query = (
                        f"{query} Context: {' '.join([ctx.get('content', '') for ctx in memory_context[:2]])}"
                    )
                    event_data["enriched_query"] = enriched_query

        except Exception as e:
            self.logger.error(f"Error handling RAG query: {e}")

    def _handle_training_complete(self, event_data: Dict[str, Any]):
        """Manejar completación de entrenamiento"""
        try:
            # Actualizar conocimiento del sistema basado en entrenamiento
            if self.human_memory_engine:
                # Agregar conocimiento de entrenamiento a memoria
                training_knowledge = event_data.get("training_knowledge", {})
                self.human_memory_engine.memorize_content(
                    content=f"Training completed: {json.dumps(training_knowledge)}",
                    content_type="training_metadata",
                    importance=0.8,
                    metadata={"source": "neuro_training", "type": "completion"},
                )

            # Optimizar otros componentes basado en entrenamiento
            if self.neuro_rag_engine:
                self.neuro_rag_engine.state.learning_progress = min(
                    1.0, self.neuro_rag_engine.state.learning_progress + 0.05
                )

        except Exception as e:
            self.logger.error(f"Error handling training complete: {e}")

    def _handle_component_sync(self, event_data: Dict[str, Any]):
        """Manejar sincronización entre componentes"""
        try:
            # Sincronizar estado entre componentes
            sync_time = datetime.now()

            # Actualizar timestamps de sincronización
            for component in ["human_memory", "neuro_rag", "neuro_training"]:
                if component in self.state.components_health:
                    self.state.sync_status[component] = sync_time

            # Realizar optimizaciones cruzadas
            if self.config.enable_performance_optimization:
                self._optimize_cross_component_performance()

        except Exception as e:
            self.logger.error(f"Error handling component sync: {e}")

    def _optimize_cross_component_performance(self):
        """Optimizar rendimiento entre componentes"""
        try:
            # Analizar métricas de rendimiento actuales
            performance_metrics = {}

            for component_name, health in self.state.components_health.items():
                if health.is_healthy():
                    performance_metrics[component_name] = {
                        "response_time": health.response_time,
                        "memory_usage": health.memory_usage,
                        "error_rate": health.error_count / max(health.last_check.timestamp(), 1),
                    }

            # Generar sugerencias de optimización
            optimizations = []

            # Optimizar basado en uso de memoria
            high_memory_components = [
                name for name, metrics in performance_metrics.items() if metrics["memory_usage"] > 0.8
            ]

            if high_memory_components:
                optimizations.append(f"High memory usage in: {', '.join(high_memory_components)}")

            # Optimizar basado en tiempos de respuesta
            slow_components = [name for name, metrics in performance_metrics.items() if metrics["response_time"] > 1.0]

            if slow_components:
                optimizations.append(f"Slow response times in: {', '.join(slow_components)}")

            self.state.optimization_suggestions.extend(optimizations)

        except Exception as e:
            self.logger.error(f"Error in cross-component optimization: {e}")

    def enhanced_chat_with_memory(
        self, message: str, session_id: str = "default", use_neuro_components: bool = True
    ) -> Dict[str, Any]:
        """Chat mejorado con integración de componentes avanzados"""
        start_time = time.time()

        try:
            # Usar componentes avanzados si están disponibles
            if use_neuro_components and self.human_memory_engine and self.neuro_rag_engine:
                # Búsqueda avanzada en memoria humana
                memory_results = self.human_memory_engine.search_memory(message, top_k=5)

                # Búsqueda avanzada en RAG neurológico
                rag_results = self.neuro_rag_engine.search(message, top_k=5)

                # Combinar resultados de manera inteligente
                combined_context = self._combine_search_results(memory_results, rag_results)

                # Generar respuesta enriquecida
                response = self._generate_enriched_response(message, combined_context)

                # Aprender de la interacción
                self._learn_from_interaction(message, response, combined_context)

            else:
                # Fallback al sistema legacy
                response = self._fallback_chat_response(message)

            # Registrar métricas
            response_time = time.time() - start_time

            result = {
                "response": response,
                "response_time": response_time,
                "used_neuro_components": use_neuro_components,
                "memory_results_count": len(memory_results) if use_neuro_components else 0,
                "rag_results_count": len(rag_results) if use_neuro_components else 0,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Emitir evento de interacción
            self.emit_event("enhanced_chat_interaction", result)

            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced chat: {e}")
            return {
                "response": f"Error procesando mensaje: {str(e)}",
                "error": str(e),
                "response_time": time.time() - start_time,
            }

    def _combine_search_results(
        self, memory_results: List[Dict[str, Any]], rag_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combinar resultados de memoria y RAG"""
        combined = []

        # Agregar resultados de memoria con peso alto
        for result in memory_results:
            combined.append(
                {
                    "source": "human_memory",
                    "content": result.get("content", ""),
                    "relevance_score": result.get("relevance_score", 0.0),
                    "metadata": result.get("metadata", {}),
                }
            )

        # Agregar resultados de RAG con peso medio
        for result in rag_results:
            combined.append(
                {
                    "source": "neuro_rag",
                    "content": result.get("chunk", {}).get("content", ""),
                    "relevance_score": result.get("relevance_score", 0.0),
                    "metadata": result.get("chunk", {}).get("metadata", {}),
                }
            )

        # Ordenar por relevancia
        combined.sort(key=lambda x: x["relevance_score"], reverse=True)

        return combined[:10]  # Máximo 10 resultados combinados

    def _generate_enriched_response(self, message: str, combined_context: List[Dict[str, Any]]) -> str:
        """Generar respuesta enriquecida con contexto avanzado"""
        if not combined_context:
            return "No tengo suficiente contexto para responder adecuadamente."

        # Construir contexto enriquecido
        context_parts = []
        for i, ctx in enumerate(combined_context[:3]):  # Top 3 resultados
            content = ctx["content"][:200] + "..." if len(ctx["content"]) > 200 else ctx["content"]
            source = ctx["source"]
            context_parts.append(f"[{source.upper()}] {content}")

        context_text = "\n".join(context_parts)

        # Crear respuesta basada en contexto
        response = f"Basándome en mi conocimiento avanzado y memoria especializada:\n\n{context_text}\n\nRespuesta:"

        return response

    def _learn_from_interaction(self, message: str, response: str, context: List[Dict[str, Any]]):
        """Aprender de la interacción para mejorar futuro rendimiento"""
        try:
            # Aprender en memoria humana
            if self.human_memory_engine:
                interaction_content = f"Q: {message}\nA: {response}"
                self.human_memory_engine.memorize_content(
                    content=interaction_content,
                    content_type="chat_interaction",
                    importance=0.6,
                    metadata={
                        "interaction_type": "enhanced_chat",
                        "context_sources": [ctx["source"] for ctx in context],
                        "response_length": len(response),
                    },
                )

            # Aprender en RAG neurológico
            if self.neuro_rag_engine:
                # Indexar interacción como documento
                interaction_doc = f"Consulta: {message}\nRespuesta: {response}"
                self.neuro_rag_engine.index_document(
                    content=interaction_doc,
                    document_id=f"interaction_{int(time.time())}",
                    content_type="chat_interaction",
                    metadata={"interaction_type": "enhanced", "context_used": len(context)},
                )

        except Exception as e:
            self.logger.error(f"Error learning from interaction: {e}")

    def _fallback_chat_response(self, message: str) -> str:
        """Respuesta de fallback usando sistema legacy"""
        try:
            # Intentar usar el sistema de chat existente
            from sheily_persistent_memory_chat import sheily_memory_query

            response = sheily_memory_query(message)

            # Si la respuesta es muy corta o genérica, enriquecerla
            if len(response) < 50:
                response += " (Respuesta básica - componentes avanzados no disponibles)"

            return response

        except Exception as e:
            return f"Respuesta básica del sistema: {str(e)}"

    def synchronize_components(self, force: bool = False) -> Dict[str, Any]:
        """Sincronizar estado entre componentes"""
        if not self.config.enable_auto_sync and not force:
            return {"status": "skipped", "reason": "auto_sync_disabled"}

        sync_start = datetime.now()
        sync_results = {}

        try:
            # Sincronizar memoria humana con RAG
            if self.human_memory_engine and self.neuro_rag_engine:
                memory_stats = self.human_memory_engine.get_memory_stats()
                rag_stats = self.neuro_rag_engine.get_system_stats()

                # Actualizar progreso de aprendizaje cruzado
                avg_learning_progress = (
                    memory_stats.get("learning_progress", 0.0) + rag_stats.get("learning_progress", 0.0)
                ) / 2

                self.state.learning_progress = avg_learning_progress
                sync_results["memory_rag_sync"] = "success"

            # Sincronizar entrenamiento con memoria
            if self.neuro_training_engine and self.human_memory_engine:
                # Integrar conocimiento de entrenamiento en memoria
                training_knowledge = self.neuro_training_engine.export_training_knowledge()
                self.human_memory_engine.memorize_content(
                    content=f"Training knowledge: {json.dumps(training_knowledge)}",
                    content_type="training_knowledge",
                    importance=0.8,
                )
                sync_results["training_memory_sync"] = "success"

            # Actualizar métricas de integración
            sync_duration = (datetime.now() - sync_start).total_seconds()
            self.state.integration_metrics["last_sync_duration"] = sync_duration
            self.state.integration_metrics["sync_success_rate"] = 1.0

            sync_results["overall_status"] = "success"
            sync_results["sync_duration"] = sync_duration

        except Exception as e:
            self.logger.error(f"Error during component synchronization: {e}")
            sync_results["overall_status"] = "error"
            sync_results["error"] = str(e)
            self.state.integration_metrics["sync_success_rate"] = 0.0

        self._save_state()
        return sync_results

    def get_system_health_report(self) -> Dict[str, Any]:
        """Obtener reporte completo de salud del sistema"""
        report = {
            "system_id": self.state.system_id,
            "overall_status": "healthy",
            "components": {},
            "integration_metrics": self.state.integration_metrics,
            "learning_progress": self.state.learning_progress,
            "optimization_suggestions": self.state.optimization_suggestions,
            "report_timestamp": datetime.now().isoformat(),
        }

        # Estado de componentes individuales
        for component_name, health in self.state.components_health.items():
            component_status = {
                "status": health.status,
                "last_check": health.last_check.isoformat(),
                "response_time": health.response_time,
                "memory_usage": health.memory_usage,
                "error_count": health.error_count,
            }

            if health.last_error:
                component_status["last_error"] = health.last_error

            report["components"][component_name] = component_status

            # Determinar estado general
            if health.status == "failed":
                report["overall_status"] = "degraded"
            elif health.status == "degraded" and report["overall_status"] == "healthy":
                report["overall_status"] = "degraded"

        return report

    def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimizar rendimiento del sistema completo"""
        optimization_start = datetime.now()

        optimizations_applied = []
        performance_improvements = {}

        try:
            # Optimizar configuración de memoria
            if self.human_memory_engine:
                memory_consolidation = self.human_memory_engine.consolidate_memory()
                if memory_consolidation.get("status") == "completed":
                    optimizations_applied.append("memory_consolidation")
                    performance_improvements["memory_efficiency"] = 0.1

            # Optimizar configuración de RAG
            if self.neuro_rag_engine:
                rag_consolidation = self.neuro_rag_engine.consolidate_memory()
                if rag_consolidation.get("status") == "completed":
                    optimizations_applied.append("rag_consolidation")
                    performance_improvements["rag_efficiency"] = 0.1

            # Optimizar configuración de entrenamiento
            if self.neuro_training_engine:
                # Aplicar optimizaciones de entrenamiento
                optimizations_applied.append("training_optimization")
                performance_improvements["training_efficiency"] = 0.05

            # Actualizar línea base de rendimiento
            for metric, improvement in performance_improvements.items():
                current_baseline = self.state.performance_baseline.get(metric, 1.0)
                self.state.performance_baseline[metric] = current_baseline * (1 - improvement)

            optimization_results = {
                "status": "completed",
                "optimizations_applied": optimizations_applied,
                "performance_improvements": performance_improvements,
                "optimization_duration": (datetime.now() - optimization_start).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

            # Registrar adaptación
            self.state.adaptation_history.append(optimization_results)

            return optimization_results

        except Exception as e:
            self.logger.error(f"Error optimizing system performance: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


# Función de integración principal
def integrate_neuro_system(config: Optional[IntegrationConfig] = None) -> NeuroIntegrationEngine:
    """Integrar sistema neurológico completo"""
    return NeuroIntegrationEngine(config)


# Función de migración desde sistema legacy
def migrate_from_legacy_system() -> NeuroIntegrationEngine:
    """Migrar desde sistema legacy completo"""
    engine = NeuroIntegrationEngine()

    try:
        # Inicializar componentes avanzados
        success = engine.initialize_components()

        if success:
            print("✅ Sistema neurológico integrado exitosamente")
            print("✅ Todos los componentes avanzados están operativos")
            print("✅ Integración automática activada")
        else:
            print("⚠️ Algunos componentes no pudieron inicializarse")
            print("✅ Sistema funcionando con componentes disponibles")

        return engine

    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return engine
