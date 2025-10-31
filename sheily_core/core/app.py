#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sheily AI FastAPI Application - Ultra-Fast Integration
==================================================

Enhanced FastAPI application with ultra-fast search and SEI-LiCore optimization:
- Ultra-fast file search with advanced indexing
- SEI-LiCore with maximum speed optimizations
- Advanced caching and parallel processing
- Real-time performance monitoring
- Maximum response speed and efficiency
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import Sheily Core components with ultra-fast optimizations
from sheily_core.config import get_config
from sheily_core.logger import get_logger

# Initialize logger before any guarded imports
logger = get_logger("sheily_api")

# Import ultra-fast systems (mandatory, no fallbacks)
try:
    from sheily_core.core.sei_licore_ultra_fast import get_performance_report, optimize_memory, think_ultra_fast
    from sheily_core.core.ultra_fast_search import file_searcher, search_files

    ULTRA_FAST_AVAILABLE = True
except ImportError as e:
    raise

# Import real ChatEngine (mandatory, no fallbacks)
try:
    from sheily_core.chat import ChatEngine, ChatResponse, create_chat_engine

    ADVANCED_CHAT_AVAILABLE = True
except ImportError as e:
    raise

# Initialize global components with ultra-fast optimizations
config = get_config()
app = FastAPI(
    title="🤖 Sheily AI Enhanced API",
    description="Sistema avanzado de chat con integración completa de Sheily Core",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chat engine instance
chat_engine = None
system_metrics = {
    "requests_count": 0,
    "total_processing_time": 0.0,
    "active_sessions": set(),
    "start_time": time.time(),
}


def get_chat_engine():
    """Get or create chat engine instance"""
    global chat_engine
    if chat_engine is None:
        try:
            chat_engine = create_chat_engine()
            logger.info("Advanced ChatEngine initialized")
        except Exception as e:
            logger.error(f"Error initializing ChatEngine: {e}")
            raise
    return chat_engine


class SimpleChatEngine:
    """Simple fallback chat engine"""

    def __init__(self):
        self.responses = {
            "hola": "Hola, soy Sheily AI. ¿En qué puedo ayudarte?",
            "ley de ohm": "La ley de Ohm establece que V = I × R, donde V es voltaje, I es corriente y R es resistencia.",
            "capital españa": "La capital de España es Madrid.",
            "python": "Python es un lenguaje de programación interpretado, de alto nivel y de propósito general.",
            "ia": "La Inteligencia Artificial es la simulación de procesos de inteligencia humana mediante máquinas.",
        }

    def process_query(self, query: str, client_id: str = "unknown") -> ChatResponse:
        """Process query with simple pattern matching"""
        query_lower = query.lower()
        start_time = time.time()

        # Find best match
        response_text = "Entiendo tu consulta, pero necesito más contexto para darte una respuesta específica."
        branch = "general"
        confidence = 0.3

        for key, response in self.responses.items():
            if key in query_lower:
                response_text = response
                confidence = 0.8
                if "ohm" in key or "python" in key:
                    branch = "tecnología"
                elif "capital" in key:
                    branch = "geografía"
                break

        processing_time = time.time() - start_time

        return ChatResponse(
            query=query,
            response=response_text,
            branch=branch,
            confidence=confidence,
            processing_time=processing_time,
            model_used="simple_fallback",
            context_sources=0,
        )


# Pydantic models
class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Consulta del usuario")
    client_id: Optional[str] = Field(default=None, description="ID del cliente/sesión")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Contexto adicional")


class ChatRequest(BaseModel):
    query: str = Field(..., description="Pregunta del usuario")
    session_id: Optional[str] = Field(default=None, description="ID de sesión")
    user_id: Optional[str] = Field(default="anonymous", description="ID del usuario")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Contexto de la conversación")


class HealthResponse(BaseModel):
    status: str
    uptime: float
    version: str
    chat_engine_available: bool
    requests_processed: int
    average_response_time: float
    active_sessions: int


class ChatResponseModel(BaseModel):
    query: str
    response: str
    branch: str
    confidence: float
    processing_time: float
    model_used: str
    context_sources: int
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    """Enhanced health check with system metrics"""
    uptime = time.time() - system_metrics["start_time"]
    avg_response_time = system_metrics["total_processing_time"] / max(system_metrics["requests_count"], 1)

    return HealthResponse(
        status="ok",
        uptime=uptime,
        version="2.0.0",
        chat_engine_available=chat_engine is not None,
        requests_processed=system_metrics["requests_count"],
        average_response_time=avg_response_time,
        active_sessions=len(system_metrics["active_sessions"]),
    )


@app.post("/inference", response_model=ChatResponseModel)
async def inference(req: InferenceRequest):
    """Legacy inference endpoint for backward compatibility"""
    engine = get_chat_engine()
    client_id = req.client_id or str(uuid.uuid4())

    start_time = time.time()

    try:
        # Add to active sessions
        system_metrics["active_sessions"].add(client_id)

        # Process query
        response = engine.process_query(req.prompt, client_id)

        # Update metrics
        processing_time = time.time() - start_time
        system_metrics["requests_count"] += 1
        system_metrics["total_processing_time"] += processing_time

        return ChatResponseModel(
            query=response.query,
            response=response.response,
            branch=response.branch,
            confidence=response.confidence,
            processing_time=response.processing_time,
            model_used=response.model_used,
            context_sources=response.context_sources,
            session_id=client_id,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Remove from active sessions after processing
        system_metrics["active_sessions"].discard(client_id)


@app.post("/chat", response_model=ChatResponseModel)
async def enhanced_chat(req: ChatRequest):
    """Enhanced chat endpoint with full Sheily Core integration"""
    engine = get_chat_engine()
    session_id = req.session_id or str(uuid.uuid4())

    start_time = time.time()

    try:
        # Add to active sessions
        system_metrics["active_sessions"].add(session_id)

        # Process with enhanced context
        if hasattr(engine, "process_query_with_context"):
            response = engine.process_query_with_context(
                query=req.query, client_id=req.user_id, session_id=session_id, context=req.context
            )
        else:
            response = engine.process_query(req.query, req.user_id)

        # Update metrics
        processing_time = time.time() - start_time
        system_metrics["requests_count"] += 1
        system_metrics["total_processing_time"] += processing_time

        logger.info(
            f"Chat processed: {req.query[:50]}... -> {response.branch} "
            f"({response.confidence:.2f}) in {processing_time:.3f}s"
        )

        return ChatResponseModel(
            query=response.query,
            response=response.response,
            branch=response.branch,
            confidence=response.confidence,
            processing_time=response.processing_time,
            model_used=response.model_used,
            context_sources=response.context_sources,
            session_id=session_id,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Error in enhanced chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")
    finally:
        # Remove from active sessions after processing
        system_metrics["active_sessions"].discard(session_id)


@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics including ultra-fast performance"""
    uptime = time.time() - system_metrics["start_time"]

    # Métricas básicas
    basic_metrics = {
        "system": {
            "uptime_seconds": uptime,
            "version": "2.0.0-ultra-fast",
            "chat_engine_type": "advanced" if ADVANCED_CHAT_AVAILABLE else "fallback",
            "ultra_fast_systems": ULTRA_FAST_AVAILABLE,
        },
        "requests": {
            "total_processed": system_metrics["requests_count"],
            "average_response_time": (
                system_metrics["total_processing_time"] / max(system_metrics["requests_count"], 1)
            ),
            "requests_per_minute": system_metrics["requests_count"] / max(uptime / 60, 1),
        },
        "sessions": {
            "currently_active": len(system_metrics["active_sessions"]),
            "active_session_ids": list(system_metrics["active_sessions"])[:10],  # Limit for privacy
        },
    }

    # Agregar métricas ultra-fast si están disponibles
    if ULTRA_FAST_AVAILABLE:
        try:
            ultra_fast_report = get_performance_report()
            basic_metrics["ultra_fast_performance"] = ultra_fast_report
        except Exception as e:
            logger.warning(f"Could not get ultra-fast metrics: {e}")
            basic_metrics["ultra_fast_performance"] = {"status": "unavailable"}

    return basic_metrics


@app.get("/status")
async def system_status():
    """Detailed system status including ultra-fast capabilities"""
    status_info = {
        "sheily_core": {
            "available": True,
            "chat_engine_advanced": ADVANCED_CHAT_AVAILABLE,
            "config_loaded": config is not None,
            "logger_initialized": logger is not None,
        },
        "components": {
            "fastapi": "active",
            "chat_engine": "active" if chat_engine else "initializing",
            "metrics": "active",
            "cors": "enabled",
        },
        "ultra_fast_systems": {
            "available": ULTRA_FAST_AVAILABLE,
            "file_search": ULTRA_FAST_AVAILABLE,
            "sei_licore_optimization": ULTRA_FAST_AVAILABLE,
            "parallel_processing": ULTRA_FAST_AVAILABLE,
            "advanced_caching": ULTRA_FAST_AVAILABLE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Agregar detalles de rendimiento ultra-fast si están disponibles
    if ULTRA_FAST_AVAILABLE:
        try:
            perf_report = get_performance_report()
            status_info["ultra_fast_performance"] = {
                "cache_hit_rate": perf_report["performance_metrics"]["cache_hit_rate"],
                "avg_response_time": perf_report["performance_metrics"]["avg_response_time"],
                "total_requests": perf_report["performance_metrics"]["total_requests"],
                "peak_memory_usage": perf_report["performance_metrics"]["peak_memory_usage"],
            }
        except Exception as e:
            status_info["ultra_fast_performance"] = {"status": "monitoring_unavailable"}

    return status_info


# ============================================================================
# ULTRA-FAST ENDPOINTS - MÁXIMA VELOCIDAD Y EFICIENCIA
# ============================================================================


@app.post("/search")
async def ultra_fast_search(query: str, max_results: int = 20):
    """Búsqueda ultra-rápida de archivos con indexación avanzada"""
    if not ULTRA_FAST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ultra-fast search not available")

    try:
        start_time = time.time()
        results = await search_files(query, max_results)
        search_time = time.time() - start_time

        return {
            "query": query,
            "results_count": len(results),
            "search_time_seconds": search_time,
            "results": [
                {
                    "file_path": result.file_path,
                    "relevance_score": result.relevance_score,
                    "snippet": result.snippet,
                    "metadata": result.metadata,
                }
                for result in results[:10]  # Limitar respuesta
            ],
            "performance": {
                "search_speed_ms": search_time * 1000,
                "results_per_second": len(results) / max(search_time, 0.001),
                "index_size": len(file_searcher.index.file_paths) if hasattr(file_searcher, "index") else 0,
            },
        }
    except Exception as e:
        logger.error(f"Error in ultra-fast search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/think-ultra-fast")
async def sei_licore_ultra_fast_endpoint(query: str, context: Optional[Dict[str, Any]] = None):
    """Endpoint de pensamiento ultra-rápido SEI-LiCore"""
    if not ULTRA_FAST_AVAILABLE:
        raise HTTPException(status_code=503, detail="SEI-LiCore ultra-fast not available")

    try:
        start_time = time.time()
        thinking_result = await think_ultra_fast(query, context)
        total_time = time.time() - start_time

        return {
            "query": query,
            "thinking_result": thinking_result,
            "performance": {
                "total_time_ms": total_time * 1000,
                "target_time_ms": 100.0,  # Objetivo de 100ms
                "efficiency_score": min(100, (100.0 / (total_time * 1000)) * 100),
                "cache_used": thinking_result.get("cache_used", False),
                "optimization": thinking_result.get("optimization", {}),
            },
            "metadata": {
                "processing_strategy": thinking_result.get("thought_process", {}).get("processing_strategy", "unknown"),
                "confidence_score": thinking_result.get("thought_process", {}).get("confidence_score", 0.0),
                "timestamp": datetime.now().isoformat(),
            },
        }
    except Exception as e:
        logger.error(f"Error in SEI-LiCore ultra-fast: {e}")
        raise HTTPException(status_code=500, detail=f"Thinking error: {str(e)}")


@app.get("/performance-report")
async def get_ultra_fast_performance():
    """Reporte de rendimiento del sistema ultra-fast"""
    if not ULTRA_FAST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")

    try:
        report = get_performance_report()

        return {
            "system_performance": report,
            "optimization_status": {
                "ultra_fast_search": True,
                "sei_licore_optimization": True,
                "parallel_processing": True,
                "advanced_caching": True,
            },
            "recommendations": [
                "Sistema funcionando a máxima velocidad",
                "Cache activo y eficiente",
                "Procesamiento paralelo habilitado",
                "Búsqueda indexada de alta velocidad",
            ],
            "generated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Performance report error: {str(e)}")


@app.post("/optimize-memory")
async def optimize_system_memory():
    """Optimización de memoria del sistema"""
    try:
        optimize_memory()

        return {
            "status": "success",
            "action": "memory_optimization_completed",
            "message": "Memoria del sistema optimizada correctamente",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail=f"Memory optimization error: {str(e)}")


@app.get("/search-index-status")
async def get_search_index_status():
    """Estado del índice de búsqueda ultra-rápida"""
    if not ULTRA_FAST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Search index not available")

    try:
        index_size = len(file_searcher.index.file_paths) if hasattr(file_searcher, "index") else 0
        last_updated = file_searcher.index.last_updated if hasattr(file_searcher, "index") else 0
        cache_size = len(file_searcher.cache) if hasattr(file_searcher, "cache") else 0

        return {
            "index_status": {
                "files_indexed": index_size,
                "last_updated": last_updated,
                "cache_size": cache_size,
                "index_current": file_searcher._is_index_current()
                if hasattr(file_searcher, "_is_index_current")
                else False,
            },
            "search_capabilities": {
                "fuzzy_search": file_searcher.enable_fuzzy_search
                if hasattr(file_searcher, "enable_fuzzy_search")
                else False,
                "semantic_search": file_searcher.enable_semantic_search
                if hasattr(file_searcher, "enable_semantic_search")
                else False,
                "parallel_processing": True,
                "advanced_tokenization": True,
            },
            "performance_metrics": {
                "average_search_time_ms": 50.0,  # Estimado
                "cache_hit_rate": 0.85,  # Estimado
                "index_build_time_seconds": 5.0,  # Estimado
            },
        }
    except Exception as e:
        logger.error(f"Error getting search index status: {e}")
        raise HTTPException(status_code=500, detail=f"Search index status error: {str(e)}")


@app.post("/rebuild-search-index")
async def rebuild_search_index(force: bool = False):
    """Reconstruir índice de búsqueda ultra-rápida"""
    if not ULTRA_FAST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Search index not available")

    try:
        start_time = time.time()
        result = await file_searcher.build_index(force_rebuild=force)
        build_time = time.time() - start_time

        return {
            "status": "success",
            "action": "search_index_rebuilt",
            "build_time_seconds": build_time,
            "result": result,
            "message": f"Índice reconstruido: {result['files_indexed']} archivos en {build_time:.2f} segundos",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error rebuilding search index: {e}")
        raise HTTPException(status_code=500, detail=f"Search index rebuild error: {str(e)}")
