#!/usr/bin/env python3
"""
Test Data Fixtures
==================
Datos de ejemplo para testing.
"""

# Sample training data
SAMPLE_TRAINING_EXAMPLES = [
    {
        "instruction": "¿Qué es Python?",
        "response": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.",
    },
    {
        "instruction": "Explica qué es Machine Learning",
        "response": "Machine Learning es una rama de la inteligencia artificial que permite a las máquinas aprender de datos.",
    },
    {
        "instruction": "¿Qué es una función en programación?",
        "response": "Una función es un bloque de código reutilizable que realiza una tarea específica.",
    },
]

# Sample branches data
SAMPLE_BRANCHES = [
    {
        "name": "physics",
        "description": "Physics and quantum mechanics",
        "keywords": ["physics", "quantum", "mechanics"],
        "examples": 1824,
    },
    {
        "name": "medicine",
        "description": "Medical knowledge and health",
        "keywords": ["medicine", "health", "medical"],
        "examples": 1380,
    },
    {
        "name": "programming",
        "description": "Programming and software development",
        "keywords": ["programming", "code", "software"],
        "examples": 2156,
    },
]

# Sample configuration
SAMPLE_CONFIG = {
    "system_name": "Sheily AI Test",
    "host": "127.0.0.1",
    "port": 8000,
    "cors_origins": ["http://localhost:3000"],
    "debug": True,
}

# Sample security events
SAMPLE_SECURITY_EVENTS = [
    {
        "event_type": "rate_limit_exceeded",
        "severity": "MEDIUM",
        "description": "Client exceeded rate limit",
        "source_ip": "192.168.1.100",
        "timestamp": "2025-10-29T10:00:00Z",
    },
    {
        "event_type": "blocked_keyword",
        "severity": "HIGH",
        "description": "Blocked dangerous keyword in query",
        "source_ip": "192.168.1.101",
        "timestamp": "2025-10-29T10:05:00Z",
    },
]

# Sample API responses
SAMPLE_API_RESPONSES = {
    "chat_success": {
        "response": "La capital de Francia es París.",
        "branch": "geography",
        "confidence": 0.95,
        "processing_time_ms": 234,
    },
    "error_rate_limit": {"error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests"}},
}

# Sample health check data
SAMPLE_HEALTH_DATA = {
    "status": "healthy",
    "timestamp": "2025-10-29T10:00:00Z",
    "components": {
        "system": {"status": "healthy", "latency_ms": 5},
        "disk": {"status": "healthy", "percent_used": 45},
        "memory": {"status": "healthy", "percent_used": 60},
    },
}
