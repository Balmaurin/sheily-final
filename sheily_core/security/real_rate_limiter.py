#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Rate Limiter - NO MOCK
============================
Sistema de rate limiting funcional y en memoria.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClientRecord:
    """Registro de cliente para rate limiting"""

    requests: list = field(default_factory=list)
    blocked_until: float = 0.0
    total_requests: int = 0
    blocked_count: int = 0


class RealRateLimiter:
    """
    Rate Limiter REAL (no mock)
    ============================

    Implementa rate limiting funcional en memoria con:
    - Límite de requests por minuto
    - Sliding window algorithm
    - Bloqueo temporal
    - Estadísticas por cliente
    """

    def __init__(
        self, max_requests_per_minute: int = 60, block_duration_seconds: int = 60, cleanup_interval: int = 300
    ):
        """
        Inicializar rate limiter

        Args:
            max_requests_per_minute: Límite de requests por minuto
            block_duration_seconds: Duración del bloqueo en segundos
            cleanup_interval: Intervalo de limpieza de registros antiguos
        """
        self.max_requests = max_requests_per_minute
        self.block_duration = block_duration_seconds
        self.cleanup_interval = cleanup_interval

        self._clients: Dict[str, ClientRecord] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()

        logger.info(
            f"RealRateLimiter inicializado: "
            f"max_requests={max_requests_per_minute}/min, "
            f"block_duration={block_duration_seconds}s"
        )

    def check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verificar rate limit para cliente

        Args:
            client_id: Identificador del cliente (IP, user_id, etc.)

        Returns:
            Tupla (permitido, mensaje_error)
        """
        with self._lock:
            now = time.time()

            # Cleanup periódico
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_old_records(now)

            # Obtener o crear registro
            if client_id not in self._clients:
                self._clients[client_id] = ClientRecord()

            client = self._clients[client_id]

            # 1. Verificar si está bloqueado
            if client.blocked_until > now:
                remaining = int(client.blocked_until - now)
                logger.warning(f"Client {client_id} bloqueado por {remaining}s " f"(bloqueos: {client.blocked_count})")
                return False, f"Rate limit exceeded. Blocked for {remaining} seconds"

            # 2. Limpiar requests antiguos (sliding window)
            window_start = now - 60  # Ventana de 1 minuto
            client.requests = [req_time for req_time in client.requests if req_time > window_start]

            # 3. Verificar límite
            if len(client.requests) >= self.max_requests:
                # Excedió el límite - bloquear
                client.blocked_until = now + self.block_duration
                client.blocked_count += 1

                logger.warning(
                    f"Client {client_id} excedió rate limit: " f"{len(client.requests)}/{self.max_requests} requests"
                )

                return False, f"Rate limit exceeded: {self.max_requests} requests/minute"

            # 4. Registrar request
            client.requests.append(now)
            client.total_requests += 1

            # Log cada 10 requests
            if client.total_requests % 10 == 0:
                logger.debug(
                    f"Client {client_id}: {len(client.requests)} requests en ventana actual, "
                    f"{client.total_requests} total"
                )

            return True, None

    def get_client_stats(self, client_id: str) -> Dict:
        """
        Obtener estadísticas de un cliente

        Args:
            client_id: ID del cliente

        Returns:
            Estadísticas del cliente
        """
        with self._lock:
            if client_id not in self._clients:
                return {"exists": False, "total_requests": 0, "current_window_requests": 0, "is_blocked": False}

            client = self._clients[client_id]
            now = time.time()

            # Limpiar ventana
            window_start = now - 60
            current_requests = [req for req in client.requests if req > window_start]

            return {
                "exists": True,
                "total_requests": client.total_requests,
                "current_window_requests": len(current_requests),
                "is_blocked": client.blocked_until > now,
                "blocked_until": client.blocked_until if client.blocked_until > now else None,
                "blocked_count": client.blocked_count,
                "remaining_capacity": max(0, self.max_requests - len(current_requests)),
            }

    def reset_client(self, client_id: str) -> bool:
        """
        Resetear rate limit de un cliente

        Args:
            client_id: ID del cliente

        Returns:
            True si se reseteó
        """
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Client {client_id} reseteado")
                return True
            return False

    def get_global_stats(self) -> Dict:
        """
        Obtener estadísticas globales

        Returns:
            Estadísticas del sistema
        """
        with self._lock:
            now = time.time()

            total_clients = len(self._clients)
            blocked_clients = sum(1 for c in self._clients.values() if c.blocked_until > now)
            total_requests = sum(c.total_requests for c in self._clients.values())

            # Requests activos en ventana actual
            active_requests = 0
            window_start = now - 60
            for client in self._clients.values():
                active_requests += sum(1 for req in client.requests if req > window_start)

            return {
                "total_clients": total_clients,
                "blocked_clients": blocked_clients,
                "active_clients": total_clients - blocked_clients,
                "total_requests": total_requests,
                "active_window_requests": active_requests,
                "max_requests_per_minute": self.max_requests,
                "block_duration_seconds": self.block_duration,
            }

    def _cleanup_old_records(self, now: float):
        """Limpiar registros antiguos"""
        # Remover clientes inactivos por más de 1 hora
        cleanup_threshold = now - 3600

        to_remove = [
            client_id
            for client_id, record in self._clients.items()
            if (not record.requests or max(record.requests) < cleanup_threshold) and record.blocked_until < now
        ]

        for client_id in to_remove:
            del self._clients[client_id]

        if to_remove:
            logger.debug(f"Cleanup: removidos {len(to_remove)} clientes inactivos")

        self._last_cleanup = now


# Instancia global
_rate_limiter: Optional[RealRateLimiter] = None
_limiter_lock = Lock()


def get_rate_limiter(max_requests_per_minute: int = 60, **kwargs) -> RealRateLimiter:
    """
    Obtener instancia global del rate limiter

    Args:
        max_requests_per_minute: Límite de requests
        **kwargs: Argumentos adicionales

    Returns:
        Instancia de RealRateLimiter
    """
    global _rate_limiter

    with _limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = RealRateLimiter(max_requests_per_minute=max_requests_per_minute, **kwargs)

    return _rate_limiter


# Exports
__all__ = ["RealRateLimiter", "ClientRecord", "get_rate_limiter"]
