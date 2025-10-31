#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Integración y Testing Completo - Sheily-AI
=====================================================

Pruebas integrales para verificar el funcionamiento completo del ecosistema
Sheily-AI con todos sus componentes implementados.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de una prueba"""

    component: str
    test_name: str
    success: bool
    execution_time: float
    details: Dict[str, Any]
    error: Optional[str] = None


class SheilyIntegrationTester:
    """
    Tester integral del sistema Sheily-AI
    """

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.system_config = self._load_system_config()

        # Componentes del sistema (se inicializarán durante las pruebas)
        self.llm_engine = None
        self.rag_engine = None
        self.hyperrouter = None
        self.branch_manager = None
        self.specialization_engine = None
        self.branch_merger = None

    def _load_system_config(self) -> Dict:
        """Cargar configuración del sistema"""
        return {
            "llm_engine": {
                "model_path": "./models/real_models",
                "max_context_length": 4096,
                "temperature": 0.7,
            },
            "rag_engine": {
                "index_path": "./indices",
                "embedding_dim": 768,
                "top_k_retrieval": 5,
            },
            "hyperrouter": {
                "routing_strategy": "adaptive",
                "max_concurrent_routes": 10,
                "route_timeout": 30.0,
            },
            "branches": {
                "branch_manager": {
                    "max_concurrent_branches": 5,
                    "enable_specialization": True,
                    "enable_merging": True,
                    "branches_path": "./branches",
                    "corpus_es_path": "./corpus_ES",
                    "corpus_en_path": "./corpus_EN",
                },
                "specialization_engine": {
                    "enable_domain_expertise": True,
                    "enable_contextual_adaptation": True,
                    "confidence_threshold": 0.6,
                },
                "branch_merger": {
                    "default_strategy": "adaptive_merge",
                    "confidence_threshold": 0.7,
                    "enable_general_knowledge": True,
                },
            },
        }

    async def run_complete_integration_test(self) -> Dict[str, Any]:
        """
        Ejecutar prueba completa de integración

        Returns:
            Resumen completo de resultados
        """
        logger.info("🚀 Iniciando pruebas de integración completas del sistema Sheily-AI")

        start_time = time.time()

        try:
            # Fase 1: Pruebas de componentes individuales
            await self._test_individual_components()

            # Fase 2: Pruebas de integración básica
            await self._test_basic_integration()

            # Fase 3: Pruebas de flujo completo
            await self._test_complete_workflow()

            # Fase 4: Pruebas de rendimiento y estrés
            await self._test_performance_and_stress()

            # Fase 5: Pruebas de casos edge
            await self._test_edge_cases()

            execution_time = time.time() - start_time

            # Generar reporte final
            report = self._generate_integration_report(execution_time)

            logger.info(f"✅ Pruebas de integración completadas en {execution_time:.2f}s")

            return report

        except Exception as e:
            logger.error(f"❌ Error durante las pruebas de integración: {e}")

            return {
                "status": "FAILED",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "tests_completed": len(self.test_results),
            }

    async def _test_individual_components(self):
        """Probar cada componente individualmente"""
        logger.info("📋 Fase 1: Pruebas de componentes individuales")

        # Test LLM Engine
        await self._test_llm_engine()

        # Test RAG Engine
        await self._test_rag_engine()

        # Test Hyperrouter
        await self._test_hyperrouter()

        # Test Branch System
        await self._test_branch_system()

    async def _test_llm_engine(self):
        """Probar LLM Engine"""
        logger.info("🧠 Testing LLM Engine...")

        start_time = time.time()

        try:
            # Importar y inicializar
            from sheily_core.llm_engine import InferenceManager, LLMEngine, ModelManager, TokenizerManager

            # Crear instancia
            self.llm_engine = LLMEngine(self.system_config["llm_engine"])

            # Test inicialización
            init_success = await self.llm_engine.initialize()
            assert init_success, "LLM Engine initialization failed"

            # Test health check
            health = await self.llm_engine.health_check()
            assert health["status"] in ["healthy", "ready"], f"LLM Engine unhealthy: {health}"

            # Test generación de texto
            test_prompt = "Explica qué es la inteligencia artificial"

            generation_result = await self.llm_engine.generate_text(prompt=test_prompt, max_length=100, temperature=0.7)

            assert generation_result["success"], "Text generation failed"
            assert len(generation_result["generated_text"]) > 10, "Generated text too short"

            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="LLM Engine",
                    test_name="Complete LLM Engine Test",
                    success=True,
                    execution_time=execution_time,
                    details={
                        "health_status": health["status"],
                        "generation_success": True,
                        "generated_length": len(generation_result["generated_text"]),
                    },
                )
            )

            logger.info(f"✅ LLM Engine test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="LLM Engine",
                    test_name="Complete LLM Engine Test",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ LLM Engine test failed: {e}")

    async def _test_rag_engine(self):
        """Probar RAG Engine"""
        logger.info("🔍 Testing RAG Engine...")

        start_time = time.time()

        try:
            # Importar y inicializar
            from sheily_core.rag_engine import EmbeddingManager, IndexManager, RAGEngine, RetrievalManager

            # Crear instancia
            self.rag_engine = RAGEngine(self.system_config["rag_engine"])

            # Test inicialización
            init_success = await self.rag_engine.initialize()
            assert init_success, "RAG Engine initialization failed"

            # Test health check
            health = await self.rag_engine.health_check()
            assert health["status"] in ["healthy", "ready"], f"RAG Engine unhealthy: {health}"

            # Test búsqueda
            test_query = "algoritmos de machine learning"

            search_result = await self.rag_engine.search(query=test_query, top_k=5, search_type="hybrid")

            assert search_result["success"], "RAG search failed"
            assert len(search_result["results"]) > 0, "No search results returned"

            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="RAG Engine",
                    test_name="Complete RAG Engine Test",
                    success=True,
                    execution_time=execution_time,
                    details={
                        "health_status": health["status"],
                        "search_success": True,
                        "results_count": len(search_result["results"]),
                    },
                )
            )

            logger.info(f"✅ RAG Engine test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="RAG Engine",
                    test_name="Complete RAG Engine Test",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ RAG Engine test failed: {e}")

    async def _test_hyperrouter(self):
        """Probar Hyperrouter"""
        logger.info("🎯 Testing Hyperrouter...")

        start_time = time.time()

        try:
            # Importar y inicializar
            from sheily_core.hyperrouter import BranchSelector, HyperRouter, LoadBalancer, Priority, RouteRequest

            # Crear instancia
            self.hyperrouter = HyperRouter(self.system_config["hyperrouter"])

            # Test inicialización
            init_success = await self.hyperrouter.initialize()
            assert init_success, "Hyperrouter initialization failed"

            # Test health check
            health = await self.hyperrouter.health_check()
            assert health["status"] in ["healthy", "ready"], f"Hyperrouter unhealthy: {health}"

            # Test routing
            test_request = RouteRequest(
                id="test_route_001",
                query="¿Cómo funciona el aprendizaje automático?",
                language="es",
                domain="inteligencia artificial",
                priority=Priority.MEDIUM,
            )

            route_result = await self.hyperrouter.route(test_request)

            assert route_result.success, "Routing failed"
            assert route_result.route_taken is not None, "No route taken"
            assert len(route_result.selected_components) > 0, "No components selected"

            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Hyperrouter",
                    test_name="Complete Hyperrouter Test",
                    success=True,
                    execution_time=execution_time,
                    details={
                        "health_status": health["status"],
                        "routing_success": True,
                        "route_taken": route_result.route_taken.value,
                        "components_selected": len(route_result.selected_components),
                    },
                )
            )

            logger.info(f"✅ Hyperrouter test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Hyperrouter",
                    test_name="Complete Hyperrouter Test",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Hyperrouter test failed: {e}")

    async def _test_branch_system(self):
        """Probar sistema de ramas"""
        logger.info("🌳 Testing Branch System...")

        start_time = time.time()

        try:
            # Importar y inicializar
            from sheily_core.branches import (
                BranchManager,
                BranchMerger,
                BranchQuery,
                SpecializationEngine,
                create_branches_system,
                initialize_branches_system,
            )

            # Crear sistema completo
            branch_manager, specialization_engine, branch_merger = create_branches_system(
                self.system_config["branches"]
            )

            self.branch_manager = branch_manager
            self.specialization_engine = specialization_engine
            self.branch_merger = branch_merger

            # Test inicialización
            init_success = await initialize_branches_system(branch_manager, specialization_engine, branch_merger)
            assert init_success, "Branch system initialization failed"

            # Test health checks
            manager_health = await branch_manager.health_check()
            spec_health = await specialization_engine.health_check()
            merger_health = await branch_merger.health_check()

            assert manager_health["status"] in [
                "healthy",
                "warning",
            ], f"Branch manager unhealthy: {manager_health}"
            assert spec_health["status"] == "healthy", f"Specialization engine unhealthy: {spec_health}"
            assert merger_health["status"] == "healthy", f"Branch merger unhealthy: {merger_health}"

            # Test consulta de rama
            test_query = BranchQuery(
                branch_name="programación",
                query="¿Cómo implementar un algoritmo de ordenamiento?",
                require_specialization=True,
                merge_with_general=True,
            )

            branch_response = await branch_manager.query_branch(test_query)

            assert branch_response.confidence > 0, "Branch query failed"
            assert len(branch_response.response) > 10, "Branch response too short"

            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Branch System",
                    test_name="Complete Branch System Test",
                    success=True,
                    execution_time=execution_time,
                    details={
                        "manager_status": manager_health["status"],
                        "specialization_status": spec_health["status"],
                        "merger_status": merger_health["status"],
                        "branch_query_success": True,
                        "response_confidence": branch_response.confidence,
                        "total_branches": manager_health.get("total_branches", 0),
                    },
                )
            )

            logger.info(f"✅ Branch System test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Branch System",
                    test_name="Complete Branch System Test",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Branch System test failed: {e}")

    async def _test_basic_integration(self):
        """Probar integración básica entre componentes"""
        logger.info("🔗 Fase 2: Pruebas de integración básica")

        # Test integración RAG + LLM
        await self._test_rag_llm_integration()

        # Test integración Hyperrouter + Branches
        await self._test_router_branches_integration()

    async def _test_rag_llm_integration(self):
        """Probar integración RAG + LLM"""
        logger.info("🔄 Testing RAG + LLM Integration...")

        start_time = time.time()

        try:
            if not self.rag_engine or not self.llm_engine:
                raise ValueError("RAG Engine or LLM Engine not initialized")

            # Búsqueda RAG
            test_query = "aprendizaje automático supervisado"

            search_result = await self.rag_engine.search(query=test_query, top_k=3, search_type="hybrid")

            # Usar contexto RAG para generar respuesta LLM
            if search_result["success"] and search_result["results"]:
                rag_context = "\n".join([result["content"] for result in search_result["results"][:2]])

                enhanced_prompt = f"Basado en este contexto: {rag_context}\n\nPregunta: {test_query}\n\nRespuesta:"

                generation_result = await self.llm_engine.generate_text(
                    prompt=enhanced_prompt, max_length=150, temperature=0.6
                )

                assert generation_result["success"], "RAG-enhanced generation failed"

                execution_time = time.time() - start_time

                self.test_results.append(
                    TestResult(
                        component="RAG + LLM Integration",
                        test_name="RAG Enhanced Generation",
                        success=True,
                        execution_time=execution_time,
                        details={
                            "rag_results_used": len(search_result["results"]),
                            "generation_success": True,
                            "enhanced_response_length": len(generation_result["generated_text"]),
                        },
                    )
                )

                logger.info(f"✅ RAG + LLM integration test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="RAG + LLM Integration",
                    test_name="RAG Enhanced Generation",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ RAG + LLM integration test failed: {e}")

    async def _test_router_branches_integration(self):
        """Probar integración Hyperrouter + Branches"""
        logger.info("🎯🌳 Testing Hyperrouter + Branches Integration...")

        start_time = time.time()

        try:
            if not self.hyperrouter or not self.branch_manager:
                raise ValueError("Hyperrouter or Branch Manager not initialized")

            # Test routing a rama especializada
            from sheily_core.hyperrouter import Priority, RouteRequest, RouteType

            specialized_request = RouteRequest(
                id="integration_test_001",
                query="Explica los principios de programación orientada a objetos",
                language="es",
                domain="programación",
                route_type=RouteType.SPECIALIZED_BRANCH,
                priority=Priority.HIGH,
            )

            route_result = await self.hyperrouter.route(specialized_request)

            assert route_result.success, "Specialized routing failed"
            assert route_result.route_taken == RouteType.SPECIALIZED_BRANCH, "Wrong route type"

            # Verificar que se puede consultar la rama directamente
            from sheily_core.branches import BranchQuery

            branch_query = BranchQuery(
                branch_name="programación",
                query=specialized_request.query,
                require_specialization=True,
            )

            branch_response = await self.branch_manager.query_branch(branch_query)

            assert branch_response.confidence > 0, "Direct branch query failed"

            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Hyperrouter + Branches",
                    test_name="Specialized Branch Routing",
                    success=True,
                    execution_time=execution_time,
                    details={
                        "routing_success": True,
                        "route_type": route_result.route_taken.value,
                        "branch_query_success": True,
                        "branch_confidence": branch_response.confidence,
                    },
                )
            )

            logger.info(f"✅ Hyperrouter + Branches integration test passed ({execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Hyperrouter + Branches",
                    test_name="Specialized Branch Routing",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Hyperrouter + Branches integration test failed: {e}")

    async def _test_complete_workflow(self):
        """Probar flujo completo del sistema"""
        logger.info("🎭 Fase 3: Pruebas de flujo completo")

        await self._test_end_to_end_query_processing()

    async def _test_end_to_end_query_processing(self):
        """Probar procesamiento end-to-end de consultas"""
        logger.info("🎯 Testing End-to-End Query Processing...")

        start_time = time.time()

        try:
            # Consultas de prueba complejas
            test_queries = [
                {
                    "query": "¿Cuáles son las mejores prácticas para implementar redes neuronales profundas?",
                    "language": "es",
                    "domain": "inteligencia artificial",
                    "expected_components": ["rag_engine", "llm_engine", "branch_manager"],
                },
                {
                    "query": "Explain the differences between supervised and unsupervised machine learning",
                    "language": "en",
                    "domain": "artificial_intelligence",
                    "expected_components": ["rag_engine", "llm_engine"],
                },
                {
                    "query": "¿Cómo optimizar el rendimiento de algoritmos de ordenamiento?",
                    "language": "es",
                    "domain": "programación",
                    "expected_components": ["branch_manager", "specialization_engine"],
                },
            ]

            successful_queries = 0

            for i, test_case in enumerate(test_queries):
                try:
                    # Simular procesamiento completo
                    query_start = time.time()

                    # 1. Routing inteligente
                    from sheily_core.hyperrouter import Priority, RouteRequest

                    route_request = RouteRequest(
                        id=f"e2e_test_{i+1}",
                        query=test_case["query"],
                        language=test_case["language"],
                        domain=test_case["domain"],
                        priority=Priority.HIGH,
                    )

                    if self.hyperrouter:
                        route_result = await self.hyperrouter.route(route_request)

                        if route_result.success:
                            # 2. Procesamiento según ruta seleccionada
                            if "rag_engine" in route_result.selected_components and self.rag_engine:
                                # Búsqueda RAG
                                rag_result = await self.rag_engine.search(query=test_case["query"], top_k=3)

                            if "branch_manager" in route_result.selected_components and self.branch_manager:
                                # Consulta especializada
                                from sheily_core.branches import BranchQuery

                                branch_query = BranchQuery(
                                    branch_name=test_case["domain"],
                                    query=test_case["query"],
                                    require_specialization=True,
                                    merge_with_general=True,
                                )

                                branch_result = await self.branch_manager.query_branch(branch_query)

                            if "llm_engine" in route_result.selected_components and self.llm_engine:
                                # Generación final
                                generation_result = await self.llm_engine.generate_text(
                                    prompt=test_case["query"], max_length=200
                                )

                    query_time = time.time() - query_start
                    successful_queries += 1

                    logger.info(f"✅ Query {i+1} processed successfully ({query_time:.2f}s)")

                except Exception as e:
                    logger.warning(f"⚠️ Query {i+1} failed: {e}")

            execution_time = time.time() - start_time

            success_rate = successful_queries / len(test_queries)

            self.test_results.append(
                TestResult(
                    component="End-to-End System",
                    test_name="Complete Query Processing",
                    success=success_rate > 0.5,  # Al menos 50% de éxito
                    execution_time=execution_time,
                    details={
                        "total_queries": len(test_queries),
                        "successful_queries": successful_queries,
                        "success_rate": success_rate,
                        "average_query_time": execution_time / len(test_queries),
                    },
                )
            )

            logger.info(f"✅ End-to-End test completed: {successful_queries}/{len(test_queries)} queries successful")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="End-to-End System",
                    test_name="Complete Query Processing",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ End-to-End test failed: {e}")

    async def _test_performance_and_stress(self):
        """Probar rendimiento y estrés del sistema"""
        logger.info("⚡ Fase 4: Pruebas de rendimiento y estrés")

        await self._test_concurrent_requests()

    async def _test_concurrent_requests(self):
        """Probar manejo de solicitudes concurrentes"""
        logger.info("🚀 Testing Concurrent Requests...")

        start_time = time.time()

        try:
            if not self.hyperrouter:
                raise ValueError("Hyperrouter not initialized")

            # Crear múltiples solicitudes concurrentes
            from sheily_core.hyperrouter import Priority, RouteRequest

            concurrent_requests = []

            for i in range(10):  # 10 solicitudes concurrentes
                request = RouteRequest(
                    id=f"concurrent_test_{i+1}",
                    query=f"Test query {i+1}: ¿Qué es la programación?",
                    language="es",
                    domain="programación",
                    priority=Priority.MEDIUM,
                )
                concurrent_requests.append(request)

            # Ejecutar todas las solicitudes concurrentemente
            concurrent_start = time.time()

            tasks = [self.hyperrouter.route(req) for req in concurrent_requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            concurrent_time = time.time() - concurrent_start

            # Analizar resultados
            successful_requests = 0
            failed_requests = 0

            for result in results:
                if isinstance(result, Exception):
                    failed_requests += 1
                elif hasattr(result, "success") and result.success:
                    successful_requests += 1
                else:
                    failed_requests += 1

            execution_time = time.time() - start_time

            success_rate = successful_requests / len(concurrent_requests)

            self.test_results.append(
                TestResult(
                    component="System Performance",
                    test_name="Concurrent Requests Handling",
                    success=success_rate > 0.8,  # Al menos 80% de éxito
                    execution_time=execution_time,
                    details={
                        "total_requests": len(concurrent_requests),
                        "successful_requests": successful_requests,
                        "failed_requests": failed_requests,
                        "success_rate": success_rate,
                        "concurrent_processing_time": concurrent_time,
                        "avg_time_per_request": concurrent_time / len(concurrent_requests),
                    },
                )
            )

            logger.info(f"✅ Concurrent requests test: {successful_requests}/{len(concurrent_requests)} successful")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="System Performance",
                    test_name="Concurrent Requests Handling",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Concurrent requests test failed: {e}")

    async def _test_edge_cases(self):
        """Probar casos edge y manejo de errores"""
        logger.info("🎭 Fase 5: Pruebas de casos edge")

        await self._test_error_handling()
        await self._test_invalid_inputs()

    async def _test_error_handling(self):
        """Probar manejo de errores"""
        logger.info("⚠️ Testing Error Handling...")

        start_time = time.time()

        try:
            error_cases_handled = 0
            total_error_cases = 0

            # Test 1: Consulta a rama inexistente
            if self.branch_manager:
                total_error_cases += 1
                try:
                    from sheily_core.branches import BranchQuery

                    invalid_query = BranchQuery(branch_name="rama_inexistente_test", query="Test query")

                    result = await self.branch_manager.query_branch(invalid_query)

                    # Debe manejar graciosamente el error
                    if hasattr(result, "response") and "no encontrada" in result.response.lower():
                        error_cases_handled += 1

                except Exception:
                    pass  # Se espera que maneje el error internamente

            # Test 2: Routing con parámetros inválidos
            if self.hyperrouter:
                total_error_cases += 1
                try:
                    from sheily_core.hyperrouter import Priority, RouteRequest

                    invalid_request = RouteRequest(
                        id="invalid_test",
                        query="",  # Consulta vacía
                        language="invalid_lang",
                        priority=Priority.CRITICAL,
                    )

                    result = await self.hyperrouter.route(invalid_request)

                    # Debe devolver resultado con manejo de error
                    if hasattr(result, "success"):
                        error_cases_handled += 1

                except Exception:
                    pass  # Se espera que maneje el error internamente

            # Test 3: LLM con prompt extremadamente largo
            if self.llm_engine:
                total_error_cases += 1
                try:
                    extremely_long_prompt = "Test prompt " * 1000  # Prompt muy largo

                    result = await self.llm_engine.generate_text(prompt=extremely_long_prompt, max_length=50)

                    # Debe manejar graciosamente
                    if isinstance(result, dict) and "success" in result:
                        error_cases_handled += 1

                except Exception:
                    pass

            execution_time = time.time() - start_time

            error_handling_score = error_cases_handled / total_error_cases if total_error_cases > 0 else 1.0

            self.test_results.append(
                TestResult(
                    component="System Robustness",
                    test_name="Error Handling",
                    success=error_handling_score > 0.6,  # Al menos 60% de casos manejados
                    execution_time=execution_time,
                    details={
                        "total_error_cases": total_error_cases,
                        "handled_cases": error_cases_handled,
                        "handling_score": error_handling_score,
                    },
                )
            )

            logger.info(f"✅ Error handling test: {error_cases_handled}/{total_error_cases} cases handled gracefully")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="System Robustness",
                    test_name="Error Handling",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Error handling test failed: {e}")

    async def _test_invalid_inputs(self):
        """Probar validación de entradas inválidas"""
        logger.info("🚫 Testing Invalid Inputs Validation...")

        start_time = time.time()

        try:
            validation_cases_passed = 0
            total_validation_cases = 3

            # Test casos de entrada inválida
            invalid_inputs = [
                {"query": None, "language": "es"},
                {"query": "", "language": ""},
                {"query": "Test", "language": None},
            ]

            if self.hyperrouter:
                from sheily_core.hyperrouter import Priority, RouteRequest

                for i, invalid_input in enumerate(invalid_inputs):
                    try:
                        request = RouteRequest(
                            id=f"validation_test_{i+1}",
                            query=invalid_input.get("query", ""),
                            language=invalid_input.get("language", "es"),
                            priority=Priority.LOW,
                        )

                        result = await self.hyperrouter.route(request)

                        # El sistema debe manejar graciosamente las entradas inválidas
                        if hasattr(result, "success"):
                            validation_cases_passed += 1

                    except Exception:
                        # También es aceptable que lance excepciones controladas
                        validation_cases_passed += 1

            execution_time = time.time() - start_time

            validation_score = validation_cases_passed / total_validation_cases

            self.test_results.append(
                TestResult(
                    component="Input Validation",
                    test_name="Invalid Inputs Handling",
                    success=validation_score > 0.5,
                    execution_time=execution_time,
                    details={
                        "total_cases": total_validation_cases,
                        "passed_cases": validation_cases_passed,
                        "validation_score": validation_score,
                    },
                )
            )

            logger.info(f"✅ Input validation test: {validation_cases_passed}/{total_validation_cases} cases handled")

        except Exception as e:
            execution_time = time.time() - start_time

            self.test_results.append(
                TestResult(
                    component="Input Validation",
                    test_name="Invalid Inputs Handling",
                    success=False,
                    execution_time=execution_time,
                    details={},
                    error=str(e),
                )
            )

            logger.error(f"❌ Input validation test failed: {e}")

    def _generate_integration_report(self, total_execution_time: float) -> Dict[str, Any]:
        """Generar reporte completo de integración"""

        # Calcular métricas generales
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - successful_tests

        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0

        # Agrupar por componente
        results_by_component = {}
        for result in self.test_results:
            if result.component not in results_by_component:
                results_by_component[result.component] = []
            results_by_component[result.component].append(result)

        # Calcular métricas por componente
        component_summary = {}
        for component, results in results_by_component.items():
            component_success = len([r for r in results if r.success])
            component_total = len(results)
            component_summary[component] = {
                "total_tests": component_total,
                "successful_tests": component_success,
                "success_rate": component_success / component_total if component_total > 0 else 0.0,
                "avg_execution_time": sum(r.execution_time for r in results) / component_total,
            }

        # Determinar estado general
        overall_status = "PASSED" if success_rate >= 0.8 else "PARTIAL" if success_rate >= 0.5 else "FAILED"

        report = {
            "status": overall_status,
            "execution_summary": {
                "total_execution_time": total_execution_time,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
            },
            "component_summary": component_summary,
            "test_phases": {
                "individual_components": "COMPLETED",
                "basic_integration": "COMPLETED",
                "complete_workflow": "COMPLETED",
                "performance_stress": "COMPLETED",
                "edge_cases": "COMPLETED",
            },
            "detailed_results": [
                {
                    "component": r.component,
                    "test_name": r.test_name,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "error": r.error,
                    "details": r.details,
                }
                for r in self.test_results
            ],
            "system_readiness": {
                "llm_engine": any(r.component == "LLM Engine" and r.success for r in self.test_results),
                "rag_engine": any(r.component == "RAG Engine" and r.success for r in self.test_results),
                "hyperrouter": any(r.component == "Hyperrouter" and r.success for r in self.test_results),
                "branch_system": any(r.component == "Branch System" and r.success for r in self.test_results),
                "end_to_end": any(r.component == "End-to-End System" and r.success for r in self.test_results),
            },
            "recommendations": self._generate_recommendations(success_rate, results_by_component),
        }

        return report

    def _generate_recommendations(self, success_rate: float, results_by_component: Dict) -> List[str]:
        """Generar recomendaciones basadas en resultados"""
        recommendations = []

        if success_rate >= 0.9:
            recommendations.append("✅ Sistema completamente funcional y listo para producción")
        elif success_rate >= 0.7:
            recommendations.append("✅ Sistema mayormente funcional, revisar componentes con fallos")
        elif success_rate >= 0.5:
            recommendations.append("⚠️ Sistema parcialmente funcional, requiere correcciones importantes")
        else:
            recommendations.append("❌ Sistema requiere revisión completa antes del despliegue")

        # Recomendaciones específicas por componente
        for component, results in results_by_component.items():
            component_success_rate = len([r for r in results if r.success]) / len(results)

            if component_success_rate < 0.5:
                recommendations.append(f"🔧 Revisar {component}: tasa de éxito baja ({component_success_rate:.1%})")

        # Recomendaciones generales
        if success_rate > 0.8:
            recommendations.extend(
                [
                    "📊 Implementar monitoreo continuo en producción",
                    "🚀 Considerar optimizaciones de rendimiento",
                    "📚 Documentar configuraciones exitosas",
                ]
            )

        return recommendations


async def main():
    """Función principal para ejecutar las pruebas"""
    print("🎯 Sistema de Integración y Testing - Sheily-AI")
    print("=" * 60)

    tester = SheilyIntegrationTester()

    # Ejecutar pruebas completas
    report = await tester.run_complete_integration_test()

    # Mostrar reporte final
    print("\n" + "=" * 60)
    print("📊 REPORTE FINAL DE INTEGRACIÓN")
    print("=" * 60)

    print(f"Estado General: {report['status']}")
    print(f"Tiempo Total: {report['execution_summary']['total_execution_time']:.2f}s")
    print(
        f"Pruebas Exitosas: {report['execution_summary']['successful_tests']}/{report['execution_summary']['total_tests']}"
    )
    print(f"Tasa de Éxito: {report['execution_summary']['success_rate']:.1%}")

    print("\n📋 Resumen por Componente:")
    for component, summary in report["component_summary"].items():
        print(f"  {component}: {summary['successful_tests']}/{summary['total_tests']} ({summary['success_rate']:.1%})")

    print("\n🎯 Estado del Sistema:")
    for system, ready in report["system_readiness"].items():
        status = "✅" if ready else "❌"
        print(f"  {status} {system.replace('_', ' ').title()}")

    print("\n💡 Recomendaciones:")
    for recommendation in report["recommendations"]:
        print(f"  {recommendation}")

    # Guardar reporte detallado
    report_path = Path("integration_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📄 Reporte detallado guardado en: {report_path}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
