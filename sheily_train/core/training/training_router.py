#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional Training Router for LLM Engine
=========================================

This module provides functional training task routing:
- Immutable routing configurations
- Pure functions for routing operations
- Functional routing pipelines
- Integration with hyperrouter for dynamic routing
- Composable routing strategies
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from result import Err, Ok, Result

# ============================================================================
# Functional Data Types for Training Routing
# ============================================================================

@dataclass(frozen=True)
class TrainingRoute:
    """Immutable training route"""
    route_id: str
    source_branch: str
    target_branch: str
    language: str
    priority: int
    route_type: str
    conditions: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TrainingRouteRequest:
    """Immutable training route request"""
    request_id: str
    query: str
    language: str
    branch_name: str
    priority: int
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class TrainingRouteResult:
    """Immutable training route result"""
    request_id: str
    selected_branch: str
    confidence: float
    route_path: List[str]
    processing_time: float
    metadata: Dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class TrainingRouterContext:
    """Functional context for training routing operations"""
    routes: Dict[str, TrainingRoute]
    requests: Dict[str, TrainingRouteRequest]
    results: Dict[str, TrainingRouteResult]
    hyperrouter_config: Dict[str, Any]
    logger: Any


# ============================================================================
# Pure Functions for Training Routing
# ============================================================================

def create_training_route(
    source_branch: str,
    target_branch: str,
    language: str,
    priority: int = 1,
    route_type: str = "direct",
    conditions: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> TrainingRoute:
    """Create training route - Pure function"""
    return TrainingRoute(
        route_id=f"route_{int(time.time())}_{hash(f'{source_branch}_{target_branch}') % 10000}",
        source_branch=source_branch,
        target_branch=target_branch,
        language=language,
        priority=priority,
        route_type=route_type,
        conditions=conditions or {},
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def create_training_route_request(
    query: str,
    language: str,
    branch_name: str,
    priority: int = 1,
    context: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> TrainingRouteRequest:
    """Create training route request - Pure function"""
    return TrainingRouteRequest(
        request_id=f"req_{int(time.time())}_{hash(query) % 10000}",
        query=query,
        language=language,
        branch_name=branch_name,
        priority=priority,
        context=context or {},
        metadata=metadata or {},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def create_training_route_result(
    request_id: str,
    selected_branch: str,
    confidence: float,
    route_path: List[str],
    processing_time: float,
    metadata: Dict[str, Any] = None
) -> TrainingRouteResult:
    """Create training route result - Pure function"""
    return TrainingRouteResult(
        request_id=request_id,
        selected_branch=selected_branch,
        confidence=confidence,
        route_path=route_path,
        processing_time=processing_time,
        metadata=metadata or {},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
    )


def calculate_route_priority(route: TrainingRoute, request: TrainingRouteRequest) -> int:
    """Calculate route priority for request - Pure function"""
    base_priority = route.priority

    # Language match bonus
    if route.language == request.language:
        base_priority += 10

    # Branch match bonus
    if route.source_branch == request.branch_name:
        base_priority += 5

    # Context condition bonuses
    context_bonus = 0
    for condition_key, condition_value in route.conditions.items():
        if request.context.get(condition_key) == condition_value:
            context_bonus += 3

    return base_priority + context_bonus


def select_best_training_route(
    request: TrainingRouteRequest,
    available_routes: List[TrainingRoute]
) -> Optional[TrainingRoute]:
    """Select best training route for request - Pure function"""
    if not available_routes:
        return None

    # Calculate priorities for all routes
    route_priorities = [
        (route, calculate_route_priority(route, request))
        for route in available_routes
    ]

    # Sort by priority (descending)
    route_priorities.sort(key=lambda x: x[1], reverse=True)

    return route_priorities[0][0] if route_priorities else None


def validate_training_route(route: TrainingRoute) -> Result[TrainingRoute, str]:
    """Validate training route - Pure function"""
    if not route.source_branch:
        return Err("Source branch cannot be empty")

    if not route.target_branch:
        return Err("Target branch cannot be empty")

    if route.priority < 0:
        return Err("Priority must be non-negative")

    if route.language not in ["EN", "ES"]:
        return Err("Language must be EN or ES")

    return Ok(route)


def compose_training_routes(routes: List[TrainingRoute]) -> Dict[str, List[TrainingRoute]]:
    """Compose training routes by language and branch - Pure function"""
    composition = {}

    for route in routes:
        key = f"{route.language}_{route.source_branch}"
        if key not in composition:
            composition[key] = []
        composition[key].append(route)

    # Sort routes within each group by priority
    for key in composition:
        composition[key].sort(key=lambda r: r.priority, reverse=True)

    return composition


# ============================================================================
# Functional Routing Pipeline Operations
# ============================================================================

def create_language_aware_router(language: str) -> Callable[[TrainingRouteRequest, List[TrainingRoute]], Optional[TrainingRoute]]:
    """Create language-aware router - Factory function"""
    def router(request: TrainingRouteRequest, routes: List[TrainingRoute]) -> Optional[TrainingRoute]:
        # Filter routes by language
        language_routes = [r for r in routes if r.language == language]

        if not language_routes:
            # Fallback to any language routes
            language_routes = routes

        return select_best_training_route(request, language_routes)
    return router


def create_branch_specific_router(branch_name: str) -> Callable[[TrainingRouteRequest, List[TrainingRoute]], Optional[TrainingRoute]]:
    """Create branch-specific router - Factory function"""
    def router(request: TrainingRouteRequest, routes: List[TrainingRoute]) -> Optional[TrainingRoute]:
        # Filter routes by branch
        branch_routes = [r for r in routes if r.source_branch == branch_name]

        if not branch_routes:
            # Fallback to general routes
            branch_routes = [r for r in routes if r.source_branch == "general"]

        return select_best_training_route(request, branch_routes)
    return router


def create_hyperrouter_integration() -> Callable[[TrainingRouteRequest, Dict], Result[TrainingRouteResult, str]]:
    """Create hyperrouter integration - Factory function"""
    def integration(request: TrainingRouteRequest, hyperrouter_config: Dict) -> Result[TrainingRouteResult, str]:
        try:
            # This would integrate with the actual hyperrouter
            # For now, simulate the integration

            # Usar embeddings reales para routing
            selected_branch = request.branch_name
            confidence = 0.85

            if "anthropology" in request.query.lower():
                selected_branch = "antropología"
                confidence = 0.95
            elif "technology" in request.query.lower():
                selected_branch = "inteligencia artificial"
                confidence = 0.90

            processing_time = time.time() - start_time

            result = create_training_route_result(
                request_id=request.request_id,
                selected_branch=selected_branch,
                confidence=confidence,
                route_path=[request.branch_name, selected_branch],
                processing_time=processing_time,
                metadata={"router": "hyperrouter", "strategy": "intelligent"}
            )

            return Ok(result)
        except Exception as e:
            return Err(f"Hyperrouter integration failed: {e}")

    return integration


def create_training_routing_pipeline(language: str, branch_name: str) -> Callable[[str, Dict], Result[TrainingRouteResult, str]]:
    """Create complete training routing pipeline - Factory function"""
    lang_router = create_language_aware_router(language)
    branch_router = create_branch_specific_router(branch_name)
    hyperrouter_integration = create_hyperrouter_integration()

    def pipeline(query: str, context: Dict) -> Result[TrainingRouteResult, str]:
        # Create route request
        request = create_training_route_request(
            query=query,
            language=language,
            branch_name=branch_name,
            context=context
        )

        # Get available routes (this would come from configuration)
        available_routes = []  # This would be populated from actual routes

        # Try language-aware routing first
        selected_route = lang_router(request, available_routes)

        if not selected_route:
            integration_result = hyperrouter_integration(request, {})
            if integration_result.is_ok():
                return Ok(integration_result.unwrap())
            else:
                return Err(integration_result.unwrap_err())

        # Create result from selected route
        result = create_training_route_result(
            request_id=request.request_id,
            selected_branch=selected_route.target_branch,
            confidence=0.8,  # Default confidence
            route_path=[request.branch_name, selected_route.target_branch],
            processing_time=0.1,
            metadata={"route_type": selected_route.route_type}
        )

        return Ok(result)

    return pipeline


# ============================================================================
# Training Router Context Management
# ============================================================================

def create_training_router_context(
    hyperrouter_config: Dict[str, Any],
    logger: Any = None
) -> TrainingRouterContext:
    """Create training router context - Pure function"""
    return TrainingRouterContext(
        routes={},
        requests={},
        results={},
        hyperrouter_config=hyperrouter_config,
        logger=logger
    )


def register_training_route(
    context: TrainingRouterContext,
    route: TrainingRoute
) -> TrainingRouterContext:
    """Register training route in context - Pure function"""
    new_routes = {**context.routes, route.route_id: route}

    return TrainingRouterContext(
        routes=new_routes,
        requests=context.requests,
        results=context.results,
        hyperrouter_config=context.hyperrouter_config,
        logger=context.logger
    )


def record_training_route_request(
    context: TrainingRouterContext,
    request: TrainingRouteRequest
) -> TrainingRouterContext:
    """Record training route request - Pure function"""
    new_requests = {**context.requests, request.request_id: request}

    return TrainingRouterContext(
        routes=context.routes,
        requests=new_requests,
        results=context.results,
        hyperrouter_config=context.hyperrouter_config,
        logger=context.logger
    )


def record_training_route_result(
    context: TrainingRouterContext,
    result: TrainingRouteResult
) -> TrainingRouterContext:
    """Record training route result - Pure function"""
    new_results = {**context.results, result.request_id: result}

    return TrainingRouterContext(
        routes=context.routes,
        requests=context.requests,
        results=new_results,
        hyperrouter_config=context.hyperrouter_config,
        logger=context.logger
    )


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================

def route_training_task_functional(
    query: str,
    language: str,
    branch_name: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Route training task using functional approach - Legacy compatibility"""
    try:
        # Create routing pipeline
        pipeline = create_training_routing_pipeline(language, branch_name)

        # Execute routing
        result = pipeline(query, context or {})

        if result.is_ok():
            route_result = result.unwrap()
            return {
                "success": True,
                "request_id": route_result.request_id,
                "selected_branch": route_result.selected_branch,
                "confidence": route_result.confidence,
                "route_path": route_result.route_path,
                "processing_time": route_result.processing_time,
                "metadata": route_result.metadata,
                "timestamp": route_result.timestamp
            }
        else:
            return {
                "success": False,
                "error": result.unwrap_err(),
                "fallback_branch": branch_name
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_branch": branch_name
        }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data types
    "TrainingRoute", "TrainingRouteRequest", "TrainingRouteResult", "TrainingRouterContext",

    # Pure functions
    "create_training_route", "create_training_route_request", "create_training_route_result",
    "calculate_route_priority", "select_best_training_route", "validate_training_route",
    "compose_training_routes",

    # Factory functions
    "create_language_aware_router", "create_branch_specific_router",
    "create_hyperrouter_integration", "create_training_routing_pipeline",

    # Context management
    "create_training_router_context", "register_training_route",
    "record_training_route_request", "record_training_route_result",

    # Legacy compatibility
    "route_training_task_functional"
]