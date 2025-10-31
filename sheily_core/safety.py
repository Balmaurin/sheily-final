#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Seguridad Empresarial para Sheily AI
==============================================

Módulo de seguridad avanzado con:
- Monitoreo de seguridad en tiempo real
- Detección de amenazas empresarial
- Validación de consultas seguras
- Cumplimiento normativo integrado
- Auditoría de seguridad completa
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


@dataclass
class SecurityEvent:
    """Evento de seguridad empresarial"""

    timestamp: datetime
    event_type: str  # "suspicious_query", "blocked_content", "rate_limit", "unauthorized_access"
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    description: str
    source_ip: str = "unknown"
    user_id: str = "unknown"
    query_hash: str = None
    action_taken: str = "logged"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Configuración de seguridad empresarial"""

    max_queries_per_minute: int = 60
    max_query_length: int = 10000
    blocked_keywords: List[str] = field(default_factory=list)
    suspicious_patterns: List[str] = field(default_factory=list)
    require_authentication: bool = False
    enable_rate_limiting: bool = True
    enable_content_filtering: bool = True
    log_all_queries: bool = True


class SecurityMonitor:
    """Monitor de seguridad empresarial avanzado"""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.security_events: List[SecurityEvent] = []
        self.query_counts: Dict[str, List[float]] = {}  # IP -> timestamps
        self.logger = logging.getLogger("security_monitor")

        # Inicializar patrones de seguridad
        self._init_security_patterns()

    def _init_security_patterns(self):
        """Inicializar patrones de seguridad empresarial"""
        # Palabras clave bloqueadas (empresariales)
        self.config.blocked_keywords = [
            # Contenido sensible empresarial
            "contraseña",
            "password",
            "token",
            "api_key",
            "secret",
            "credenciales",
            "autenticación",
            "sesión",
            "cookie",
            # Comandos peligrosos
            "rm -rf",
            "sudo",
            "chmod 777",
            "eval",
            "exec",
            # Contenido inapropiado
            "contenido_adulto",
            "violencia",
            "ilegal",
            "drogas",
        ]

        # Patrones sospechosos
        self.config.suspicious_patterns = [
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IPs
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
            r"\b[A-Za-z0-9]{32,}\b",  # Tokens largos
            r"<script[^>]*>.*?</script>",  # Scripts
            r"javascript:",  # JavaScript URLs
            r"vbscript:",  # VBScript URLs
            r"on\w+\s*=",  # Event handlers
        ]

    def check_request(self, query: str, client_id: str = "unknown") -> Tuple[bool, str]:
        """
        Verificar si una consulta es segura

        Args:
            query: Consulta del usuario
            client_id: ID del cliente

        Returns:
            Tupla (is_secure, reason_if_not)
        """
        # Verificación de longitud
        if len(query) > self.config.max_query_length:
            return False, f"Consulta demasiado larga: {len(query)} > {self.config.max_query_length}"

        # Verificación de rate limiting
        if self.config.enable_rate_limiting:
            rate_check = self._check_rate_limit(client_id)
            if not rate_check[0]:
                return False, f"Rate limit excedido: {rate_check[1]}"

        # Verificación de contenido bloqueado
        if self.config.enable_content_filtering:
            content_check = self._check_blocked_content(query)
            if not content_check[0]:
                return False, f"Contenido bloqueado: {content_check[1]}"

        # Verificación de patrones sospechosos
        suspicious_ok, suspicious_msg = self._check_suspicious_patterns(query)
        # Bloquear solo si se detecta un patrón sospechoso
        if not suspicious_ok:
            return False, f"Patrón sospechoso detectado: {suspicious_msg}"

        return True, "Consulta segura"

    def _check_rate_limit(self, client_id: str) -> Tuple[bool, str]:
        """Verificar rate limiting empresarial"""
        current_time = time.time()
        minute_ago = current_time - 60

        # Inicializar contador para cliente si no existe
        if client_id not in self.query_counts:
            self.query_counts[client_id] = []

        # Limpiar timestamps antiguos
        self.query_counts[client_id] = [
            timestamp for timestamp in self.query_counts[client_id] if timestamp > minute_ago
        ]

        # Verificar límite
        if len(self.query_counts[client_id]) >= self.config.max_queries_per_minute:
            return False, f"Máximo {self.config.max_queries_per_minute} consultas por minuto"

        # Agregar timestamp actual
        self.query_counts[client_id].append(current_time)
        return True, "Dentro de límites"

    def _check_blocked_content(self, query: str) -> Tuple[bool, str]:
        """Verificar contenido bloqueado empresarial"""
        query_lower = query.lower()

        for keyword in self.config.blocked_keywords:
            if keyword.lower() in query_lower:
                return False, f"Palabra clave bloqueada: {keyword}"

        return True, "Contenido permitido"

    def _check_suspicious_patterns(self, query: str) -> Tuple[bool, str]:
        """Verificar patrones sospechosos empresariales"""
        for pattern in self.config.suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Patrón sospechoso: {pattern}"

        return True, "No hay patrones sospechosos"

    def record_security_event(self, event: SecurityEvent):
        """Registrar evento de seguridad empresarial"""
        self.security_events.append(event)

        # Log del evento
        self.logger.warning(
            f"Evento de seguridad: {event.event_type} - {event.severity} - {event.description}"
        )

        # Mantener solo eventos recientes (últimas 24 horas)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.security_events = [e for e in self.security_events if e.timestamp > cutoff_time]

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Obtener resumen de seguridad empresarial"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]

        # Estadísticas por severidad
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for event in recent_events:
            severity_counts[event.severity] += 1

        # Estadísticas por tipo
        type_counts = {}
        for event in recent_events:
            event_type = event.event_type
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "events_by_severity": severity_counts,
            "events_by_type": type_counts,
            "critical_events": severity_counts["CRITICAL"],
            "high_risk_events": severity_counts["HIGH"],
            "security_score": self._calculate_security_score(recent_events),
        }

    def _calculate_security_score(self, events: List[SecurityEvent]) -> float:
        """Calcular puntuación de seguridad empresarial"""
        if not events:
            return 100.0

        base_score = 100.0

        # Penalización por eventos críticos
        critical_events = sum(1 for e in events if e.severity == "CRITICAL")
        if critical_events > 0:
            base_score -= critical_events * 25.0

        # Penalización por eventos de alta severidad
        high_events = sum(1 for e in events if e.severity == "HIGH")
        if high_events > 0:
            base_score -= high_events * 10.0

        # Penalización por volumen alto de eventos
        if len(events) > 50:
            base_score -= (len(events) - 50) * 0.5

        return max(0.0, min(100.0, base_score))


def get_security_monitor(config: SecurityConfig = None) -> SecurityMonitor:
    """Obtener instancia del monitor de seguridad empresarial"""
    return SecurityMonitor(config)


def create_security_event(
    event_type: str,
    severity: str,
    description: str,
    source_ip: str = "unknown",
    user_id: str = "unknown",
    query: str = None,
    action_taken: str = "logged",
) -> SecurityEvent:
    """Crear evento de seguridad empresarial"""
    # Crear hash de la consulta si se proporciona
    query_hash = None
    if query:
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]

    return SecurityEvent(
        timestamp=datetime.now(),
        event_type=event_type,
        severity=severity,
        description=description,
        source_ip=source_ip,
        user_id=user_id,
        query_hash=query_hash,
        action_taken=action_taken,
    )


# ============================================================================
# Exports del módulo
# ============================================================================

__all__ = [
    "SecurityEvent",
    "SecurityConfig",
    "SecurityMonitor",
    "get_security_monitor",
    "create_security_event",
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "Sheily AI Team - Enterprise Security System"
