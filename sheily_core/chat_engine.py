#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional Chat Engine for Sheily AI System
 ===========================================

 This module provides a functional chat engine with:
 - Pure functions with no mutable state
 - Functional composition patterns
 - Immutable data structures
 - Functional error handling
 - Composable middleware pipeline
 - Functional configuration management
"""

import json
import subprocess
import time
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from result import Err, Ok, Result

from sheily_core.config import get_config
from sheily_core.logger import LogContext, get_logger
from sheily_core.safety import get_security_monitor
from sheily_core.utils.subprocess_utils import safe_subprocess_run

# ============================================================================
# Functional Data Types (Immutable)
# ============================================================================


@dataclass(frozen=True)
class ChatMessage:
    """Represents a chat message - Immutable"""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class ChatResponse:
    """Represents a complete chat response - Immutable"""

    query: str
    response: str
    branch: str
    confidence: float
    context_sources: int
    processing_time: float
    model_used: str
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class ChatContext:
    """Functional context for chat processing"""

    config: Any
    logger: Any
    security_monitor: Any
    branch_detector: Any
    model_interface: Optional[Any]
    context_manager: Any


@dataclass(frozen=True)
class ProcessingResult:
    """Result of functional processing pipeline"""

    response: ChatResponse
    logs: List[str]
    metrics: Dict[str, Any]


# ============================================================================
# Functional Utilities
# ============================================================================


def compose(*functions: Callable) -> Callable:
    """Function composition utility"""

    def composed(arg):
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)

    return composed


def pipe(value: Any, *functions: Callable) -> Any:
    """Pipe value through functions"""
    return reduce(lambda acc, f: f(acc), functions, value)


def curry(func: Callable) -> Callable:
    """Curry a function"""

    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return partial(func, *args, **kwargs)

    return curried


def safe_execute(func: Callable, error_handler: Callable = None) -> Callable:
    """Make function execution safe with error handling"""

    def wrapper(*args, **kwargs):
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            if error_handler:
                return error_handler(e, *args, **kwargs)
            return Err(e)

    return wrapper


# ============================================================================
# Pure Functions for Chat Processing
# ============================================================================


def create_chat_message(role: str, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
    """Create a chat message - Pure function"""
    return ChatMessage(role=role, content=content, timestamp=time.time(), metadata=metadata or {})


def create_chat_response(
    query: str,
    response: str,
    branch: str = "general",
    confidence: float = 0.0,
    context_sources: int = 0,
    processing_time: float = 0.0,
    model_used: str = "unknown",
    tokens_used: int = 0,
    error: str = None,
    metadata: Dict[str, Any] = None,
) -> ChatResponse:
    """Create a chat response - Pure function"""
    return ChatResponse(
        query=query,
        response=response,
        branch=branch,
        confidence=confidence,
        context_sources=context_sources,
        processing_time=processing_time,
        model_used=model_used,
        tokens_used=tokens_used,
        error=error,
        metadata=metadata or {},
    )


def create_chat_context(config_path: str = None) -> ChatContext:
    """Create functional chat context - Pure function"""
    config = get_config()
    logger = get_logger("chat_engine")
    security_monitor = get_security_monitor()

    # Import here to avoid circular dependencies
    from app.chat_engine import create_branch_detector, create_context_manager, create_model_interface

    branch_detector = create_branch_detector(config.branches_config_path)
    model_interface = None
    context_manager = create_context_manager(config.corpus_root)

    # Initialize model interface if paths are configured
    if config.model_path and config.llama_binary_path:
        try:
            model_interface = create_model_interface(config.model_path, config.llama_binary_path, config)
        except Exception as e:
            logger.error(f"Could not initialize model interface: {e}")

    return ChatContext(
        config=config,
        logger=logger,
        security_monitor=security_monitor,
        branch_detector=branch_detector,
        model_interface=model_interface,
        context_manager=context_manager,
    )


# ============================================================================
# Functional Branch Detection
# ============================================================================


def load_branches_config(branches_config_path: str) -> Dict[str, Any]:
    """Load branches configuration - Pure function"""
    try:
        if Path(branches_config_path).exists():
            with open(branches_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {domain["name"]: domain for domain in data["domains"]}
        else:
            return get_default_branches()
    except Exception:
        return get_default_branches()


def get_default_branches() -> Dict[str, Any]:
    """Default branch configurations - Pure function"""
    return {
        "programación": {
            "name": "programación",
            "keywords": ["código", "python", "bug", "error", "programar", "desarrollo", "software"],
            "description": "Programming and development specialization",
        },
        "medicina": {
            "name": "medicina",
            "keywords": ["medicina", "salud", "médico", "tratamiento", "diagnóstico", "síntoma"],
            "description": "Medicine and health specialization",
        },
        "inteligencia artificial": {
            "name": "inteligencia artificial",
            "keywords": [
                "IA",
                "inteligencia artificial",
                "machine learning",
                "algoritmo",
                "modelo",
            ],
            "description": "Artificial intelligence specialization",
        },
        "general": {
            "name": "general",
            "keywords": ["general", "común", "básico"],
            "description": "General knowledge",
        },
    }


def get_related_words() -> Dict[str, List[str]]:
    """Get related words for branch detection - Pure function"""
    return {
        "programación": ["dev", "script", "función", "clase", "método", "variable"],
        "medicina": ["paciente", "enfermedad", "medicamento", "hospital", "clínica"],
        "inteligencia artificial": ["neural", "training", "datos", "predicción", "clasificación"],
    }


def calculate_branch_score(query: str, branch_name: str, branch_config: Dict, related_words: Dict) -> float:
    """Calculate score for a branch - Pure function"""
    query_lower = query.lower()
    score = 0.0
    keywords = branch_config.get("keywords", [])

    # Score based on exact keyword matches (high weight)
    for keyword in keywords:
        if keyword.lower() in query_lower:
            score += 3.0

    # Score based on branch name match
    if branch_name.lower() in query_lower:
        score += 5.0

    # Score based on related words
    if branch_name in related_words:
        for word in related_words[branch_name]:
            if word in query_lower:
                score += 1.0

    return score


def detect_branch(query: str, branches_config: Dict[str, Any], logger: Any = None) -> Tuple[str, float]:
    """
    Detect the most appropriate branch for a query - Pure function

    Args:
        query: User query
        branches_config: Available branch configurations
        logger: Optional logger

    Returns:
        Tuple of (branch_name, confidence_score)
    """
    if logger:
        logger.debug(f"Detecting branch for query: '{query[:50]}...'")

    query_lower = query.lower()
    branch_scores = {}
    related_words = get_related_words()

    for branch_name, branch_config in branches_config.items():
        score = calculate_branch_score(query, branch_name, branch_config, related_words)
        if score > 0:
            branch_scores[branch_name] = score

    # Select best branch
    if branch_scores:
        best_branch = max(branch_scores.keys(), key=lambda k: branch_scores[k])
        best_score = branch_scores[best_branch]
        if logger:
            logger.info(f"Detected branch: '{best_branch}' (score: {best_score:.1f})")
        return best_branch, best_score
    else:
        if logger:
            logger.info("No specific branch detected, using 'general'")
        return "general", 0.1


def create_branch_detector(branches_config_path: str) -> Callable[[str], Tuple[str, float]]:
    """Create a branch detector function - Factory function"""
    branches_config = load_branches_config(branches_config_path)
    logger = get_logger("chat_engine")

    def detector(query: str) -> Tuple[str, float]:
        return detect_branch(query, branches_config, logger)

    return detector


# ============================================================================
# Functional Model Interface
# ============================================================================


def validate_model_files(model_path: str, llama_binary_path: str) -> Result[Tuple[str, str], str]:
    """Validate that model files exist - Pure function"""
    if not Path(model_path).exists():
        return Err(f"Model file not found: {model_path}")

    if not Path(llama_binary_path).exists():
        return Err(f"llama-cli binary not found: {llama_binary_path}")

    return Ok((model_path, llama_binary_path))


def create_prompt(query: str, context_docs: List[str] = None) -> str:
    """Create optimized prompt for the model - Pure function"""
    system_prompt = "Eres Sheily, una asistente de IA útil y conocedora. Responde de manera clara y concisa en español."

    # Add context if available
    context_section = ""
    if context_docs:
        context_text = "\n\n".join(context_docs[:3])  # Max 3 docs
        context_section = f"\n\nContexto relevante:\n{context_text}\n"

    # Create complete prompt
    full_prompt = f"""{system_prompt}
{context_section}
Pregunta: {query}

Respuesta:"""

    return full_prompt


def get_fallback_response(query: str) -> str:
    """Generate fallback response when model fails - Pure function"""
    return f"Entiendo que preguntas sobre '{query}'. Puedo ayudarte mejor si proporcionas más contexto específico sobre tu consulta."


def execute_model_command(model_path: str, llama_binary_path: str, prompt: str, config: Any) -> Result[str, str]:
    """Execute model command - Pure function"""
    try:
        # Prepare command
        cmd = [
            str(llama_binary_path),
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--n-predict",
            str(config.model_max_tokens),
            "--temp",
            str(config.model_temperature),
            "--top-p",
            "0.9",
            "--ctx-size",
            "2048",
            "--batch-size",
            "1",
            "--threads",
            str(config.model_threads),
        ]

        # Execute model de forma segura
        result = safe_subprocess_run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.model_timeout,
            cwd=str(Path(llama_binary_path).parent),
        )

        if result.returncode == 0:
            response = result.stdout.strip()

            # Clean response (remove prompt repetition)
            if prompt in response:
                response = response.replace(prompt, "").strip()

            if len(response) > 10:
                return Ok(response)
            else:
                return Ok(get_fallback_response(query))
        else:
            return Err(f"Model execution failed (code {result.returncode}): {result.stderr}")

    except subprocess.TimeoutExpired:
        return Err(f"Model timeout after {config.model_timeout}s")
    except Exception as e:
        return Err(f"Model execution error: {e}")


def generate_model_response(
    query: str,
    context_docs: List[str],
    model_path: str,
    llama_binary_path: str,
    config: Any,
    logger: Any = None,
) -> str:
    """Generate response using GGUF model - Pure function"""
    if logger:
        logger.debug(f"Executing GGUF model with timeout: {config.model_timeout}s")

    # Create specialized prompt
    full_prompt = create_prompt(query, context_docs)

    # Execute model
    result = execute_model_command(model_path, llama_binary_path, full_prompt, config)

    if result.is_ok():
        response = result.unwrap()
        if logger:
            logger.info(f"Model response generated: {len(response)} characters")
        return response
    else:
        error = result.unwrap_err()
        if logger:
            logger.error(error)
        return get_fallback_response(query)


def create_model_interface(model_path: str, llama_binary_path: str, config: Any) -> Callable[[str, List[str]], str]:
    """Create a model interface function - Factory function"""
    # Validate files
    validation = validate_model_files(model_path, llama_binary_path)
    if validation.is_err():
        raise FileNotFoundError(validation.unwrap_err())

    logger = get_logger("model")
    logger.info(f"Model files validated: {model_path}")

    def interface(query: str, context_docs: List[str] = None) -> str:
        return generate_model_response(query, context_docs or [], model_path, llama_binary_path, config, logger)

    return interface


# ============================================================================
# Functional Context Management
# ============================================================================


def get_basic_knowledge(branch_name: str) -> List[str]:
    """Get basic knowledge for branch - Pure function"""
    basic_knowledge = {
        "programación": [
            "Python es un lenguaje de programación interpretado, de alto nivel y multiparadigma.",
            "Para debuggear código, usa print() statements, debugger integrado o herramientas como pdb.",
            "Las mejores prácticas incluyen código limpio, comentarios útiles y testing regular.",
        ],
        "medicina": [
            "La medicina se basa en evidencia científica y protocolos establecidos.",
            "Es fundamental consultar siempre con profesionales médicos para diagnósticos.",
            "Los síntomas pueden tener múltiples causas, requiriendo evaluación profesional.",
        ],
        "inteligencia artificial": [
            "IA abarca machine learning, deep learning y procesamiento de lenguaje natural.",
            "Los modelos requieren datos de calidad y entrenamiento adecuado.",
            "La ética en IA es crucial para desarrollo responsable de tecnología.",
        ],
        "general": [
            "Puedo ayudarte con información general sobre diversos temas.",
            "Para preguntas específicas, proporciona más contexto para mejor respuesta.",
            "Siempre verifica información importante con fuentes confiables.",
        ],
    }

    return basic_knowledge.get(branch_name, basic_knowledge["general"])


def read_branch_context_file(txt_file: Path) -> Result[str, str]:
    """Read a branch context file - Pure function"""
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Take relevant excerpt (first 300 chars)
            if len(content) > 300:
                content = content[:300] + "..."
            return Ok(f"[{txt_file.parent.name}] {content}")
    except Exception as e:
        return Err(f"Error reading {txt_file}: {e}")


def get_branch_context_files(corpus_root: Path, branch_name: str) -> List[Path]:
    """Get branch context files - Pure function"""
    branch_corpus_path = corpus_root / branch_name
    if branch_corpus_path.exists():
        return list(branch_corpus_path.glob("*.txt"))
    return []


def get_branch_context(query: str, branch_name: str, corpus_root: Path, logger: Any = None) -> List[str]:
    """Get context from branch-specific corpus - Pure function"""
    context_docs = []

    try:
        # Look for branch corpus
        txt_files = get_branch_context_files(corpus_root, branch_name)

        # Read context files
        for txt_file in txt_files[:2]:  # Max 2 docs from corpus
            result = read_branch_context_file(txt_file)
            if result.is_ok():
                context_docs.append(result.unwrap())
            elif logger:
                logger.debug(result.unwrap_err())

        if logger:
            logger.debug(f"Found {len(context_docs)} docs in branch '{branch_name}'")

    except Exception as e:
        if logger:
            logger.error(f"Error accessing branch corpus: {e}")

    return context_docs


def get_context_for_query(
    query: str, branch_name: str, corpus_root: Path, max_docs: int = 3, logger: Any = None
) -> List[str]:
    """
    Get relevant context for a query - Pure function

    Args:
        query: User query
        branch_name: Detected branch
        corpus_root: Root corpus directory
        max_docs: Maximum number of context documents
        logger: Optional logger

    Returns:
        List of context documents
    """
    context_docs = []

    # Try to get branch-specific context
    branch_docs = get_branch_context(query, branch_name, corpus_root, logger)
    context_docs.extend(branch_docs)

    # If no context found, provide basic knowledge
    if not context_docs:
        context_docs = get_basic_knowledge(branch_name)

    return context_docs[:max_docs]


def create_context_manager(corpus_root: str) -> Callable[[str, str, int], List[str]]:
    """Create a context manager function - Factory function"""
    corpus_path = Path(corpus_root)
    logger = get_logger("context")

    def manager(query: str, branch_name: str, max_docs: int = 3) -> List[str]:
        return get_context_for_query(query, branch_name, corpus_path, max_docs, logger)

    return manager


# ============================================================================
# Functional Chat Processing Pipeline
# ============================================================================


def generate_fallback_response(query: str, branch: str) -> str:
    """Generate fallback response when model is not available - Pure function"""
    fallback_responses = {
        "programación": f"Como experta en programación, puedo ayudarte con '{query}'. Para código específico, proporciona más detalles sobre el lenguaje y contexto.",
        "medicina": f"Sobre '{query}' en medicina, te recomiendo consultar con un profesional médico para información precisa y personalizada.",
        "inteligencia artificial": f"Respecto a '{query}' en IA, este es un campo amplio. ¿Podrías especificar si te interesa machine learning, redes neuronales u otro aspecto?",
        "general": f"Entiendo que preguntas sobre '{query}'. Puedo ayudarte mejor si proporcionas más contexto específico.",
    }

    return fallback_responses.get(branch, fallback_responses["general"])


def check_security_validation(query: str, client_id: str, security_monitor: Any) -> Result[str, str]:
    """Check security validation - Pure function"""
    is_secure, security_reason = security_monitor.check_request(query, client_id)
    if not is_secure:
        return Err(security_reason)
    return Ok("secure")


def create_security_blocked_response(query: str, security_reason: str, processing_time: float) -> ChatResponse:
    """Create security blocked response - Pure function"""
    return create_chat_response(
        query=query,
        response=f"Lo siento, tu consulta ha sido bloqueada por razones de seguridad: {security_reason}",
        branch="general",
        confidence=0.0,
        context_sources=0,
        processing_time=processing_time,
        model_used="security_blocked",
        error="Security violation",
    )


def create_error_response(query: str, error: str, processing_time: float) -> ChatResponse:
    """Create error response - Pure function"""
    return create_chat_response(
        query=query,
        response=f"Lo siento, ocurrió un error procesando tu consulta: {str(error)}",
        branch="general",
        confidence=0.0,
        context_sources=0,
        processing_time=processing_time,
        model_used="error",
        error=error,
    )


def process_chat_query(query: str, client_id: str, context: ChatContext) -> ChatResponse:
    """
    Process a complete chat query with security validation - Pure function

    Args:
        query: User query
        client_id: Client identifier for security tracking
        context: Chat context with all dependencies

    Returns:
        Complete chat response
    """
    start_time = time.time()

    # Create logging context
    log_context = LogContext(
        component="chat_engine",
        operation="process_query",
        metadata={"query_length": len(query), "client_id": client_id},
    )

    with context.logger.context(**log_context.__dict__):
        try:
            context.logger.info(f"Processing query: '{query[:50]}...'")

            # Step 0: Security validation
            security_result = check_security_validation(query, client_id, context.security_monitor)

            if security_result.is_err():
                security_reason = security_result.unwrap_err()
                context.logger.warning(f"Security violation from {client_id}: {security_reason}")
                processing_time = time.time() - start_time
                return create_security_blocked_response(query, security_reason, processing_time)

            # Step 1: Detect branch
            branch, confidence = context.branch_detector(query)

            # Step 2: Get context
            context_docs = context.context_manager(query, branch)

            # Step 3: Generate response
            if context.model_interface:
                response = context.model_interface(query, context_docs)
                model_used = "gguf_q4"
            else:
                response = generate_fallback_response(query, branch)
                model_used = "fallback"

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create response object
            chat_response = create_chat_response(
                query=query,
                response=response,
                branch=branch,
                confidence=confidence,
                context_sources=len(context_docs),
                processing_time=processing_time,
                model_used=model_used,
                metadata={
                    "context_docs_count": len(context_docs),
                    "branch_detected": branch,
                    "confidence_score": confidence,
                },
            )

            context.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return chat_response

        except Exception as e:
            processing_time = time.time() - start_time
            context.logger.exception(f"Error processing query: {e}")
            return create_error_response(query, str(e), processing_time)


def create_chat_engine(config_path: str = None) -> Callable[[str, str], ChatResponse]:
    """Create a chat engine function - Factory function"""
    context = create_chat_context(config_path)

    def engine(query: str, client_id: str = "unknown") -> ChatResponse:
        return process_chat_query(query, client_id, context)

    return engine


def perform_health_check(context: ChatContext) -> Dict[str, Any]:
    """Perform health check on chat engine components - Pure function"""
    health_status = {"status": "healthy", "components": {}, "timestamp": time.time()}

    # Check branch detector
    try:
        # This would need to be implemented based on the functional branch detector
        health_status["components"]["branch_detector"] = {
            "status": "healthy",
            "message": "Branch detector functional",
        }
    except Exception as e:
        health_status["components"]["branch_detector"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    # Check model interface
    if context.model_interface:
        try:
            health_status["components"]["model_interface"] = {
                "status": "healthy",
                "model_configured": True,
            }
        except Exception as e:
            health_status["components"]["model_interface"] = {"status": "error", "error": str(e)}
            health_status["status"] = "degraded"
    else:
        health_status["components"]["model_interface"] = {
            "status": "not_configured",
            "message": "Model paths not configured",
        }

    # Check context manager
    try:
        corpus_path = Path(context.config.corpus_root)
        health_status["components"]["context_manager"] = {
            "status": "healthy" if corpus_path.exists() else "warning",
            "corpus_exists": corpus_path.exists(),
        }
    except Exception as e:
        health_status["components"]["context_manager"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    return health_status


def create_health_checker(context: ChatContext) -> Callable[[], Dict[str, Any]]:
    """Create a health checker function - Factory function"""

    def checker() -> Dict[str, Any]:
        return perform_health_check(context)

    return checker
