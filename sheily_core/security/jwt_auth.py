#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWT Authentication System - REAL Implementation
================================================
Sistema de autenticación JWT funcional y seguro.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

try:
    import jwt
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """Payload del token JWT"""

    user_id: str
    role: str
    exp: int
    iat: int

    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {"user_id": self.user_id, "role": self.role, "exp": self.exp, "iat": self.iat}


class JWTAuthManager:
    """
    Gestor de autenticación JWT REAL
    ================================

    Implementación completa de JWT con:
    - Generación de tokens
    - Validación de tokens
    - Manejo de expiración
    - Refresh tokens
    - Roles y permisos
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiry_minutes: int = 60,
        refresh_expiry_days: int = 30,
    ):
        """
        Inicializar gestor JWT

        Args:
            secret_key: Clave secreta (usa SECRET_KEY del .env si no se provee)
            algorithm: Algoritmo JWT (default: HS256)
            token_expiry_minutes: Minutos de validez del token
            refresh_expiry_days: Días de validez del refresh token
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT no está instalado. Instalar con: pip install PyJWT")

        # Obtener SECRET_KEY del entorno o usar la provista
        self.secret_key = secret_key or os.getenv("SECRET_KEY")

        if not self.secret_key or self.secret_key == "change_this_in_production":
            raise ValueError("SECRET_KEY no configurada o usa valor por defecto. " "Configurar SECRET_KEY en .env")

        if len(self.secret_key) < 32:
            logger.warning("SECRET_KEY es muy corta. Recomendado: 32+ caracteres")

        self.algorithm = algorithm
        self.token_expiry_minutes = token_expiry_minutes
        self.refresh_expiry_days = refresh_expiry_days

        logger.info(
            f"JWT Auth Manager inicializado: "
            f"algorithm={algorithm}, "
            f"token_expiry={token_expiry_minutes}m, "
            f"refresh_expiry={refresh_expiry_days}d"
        )

    def generate_token(self, user_id: str, role: str = "user", additional_claims: Optional[Dict] = None) -> str:
        """
        Generar token JWT

        Args:
            user_id: ID del usuario
            role: Rol del usuario (user, trainer, admin)
            additional_claims: Claims adicionales opcionales

        Returns:
            Token JWT firmado
        """
        now = datetime.utcnow()
        exp = now + timedelta(minutes=self.token_expiry_minutes)

        payload = {"user_id": user_id, "role": role, "iat": int(now.timestamp()), "exp": int(exp.timestamp())}

        # Añadir claims adicionales si existen
        if additional_claims:
            payload.update(additional_claims)

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            logger.debug(f"Token generado para user_id={user_id}, role={role}")
            return token

        except Exception as e:
            logger.error(f"Error generando token: {e}")
            raise

    def generate_refresh_token(self, user_id: str, role: str = "user") -> str:
        """
        Generar refresh token (validez más larga)

        Args:
            user_id: ID del usuario
            role: Rol del usuario

        Returns:
            Refresh token JWT
        """
        now = datetime.utcnow()
        exp = now + timedelta(days=self.refresh_expiry_days)

        payload = {
            "user_id": user_id,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "type": "refresh",  # Marcar como refresh token
        }

        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            logger.debug(f"Refresh token generado para user_id={user_id}")
            return token

        except Exception as e:
            logger.error(f"Error generando refresh token: {e}")
            raise

    def validate_token(self, token: str) -> Tuple[bool, Optional[TokenPayload], str]:
        """
        Validar token JWT

        Args:
            token: Token JWT a validar

        Returns:
            Tupla (válido, payload, mensaje_error)
        """
        if not token:
            return False, None, "Token vacío"

        # Limpiar prefijo Bearer si existe
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            # Decodificar y validar
            payload_dict = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verificar campos requeridos
            required_fields = ["user_id", "role", "exp", "iat"]
            for field in required_fields:
                if field not in payload_dict:
                    return False, None, f"Campo requerido '{field}' no encontrado"

            # Crear payload tipado
            payload = TokenPayload(
                user_id=payload_dict["user_id"],
                role=payload_dict["role"],
                exp=payload_dict["exp"],
                iat=payload_dict["iat"],
            )

            logger.debug(f"Token válido para user_id={payload.user_id}")
            return True, payload, "Token válido"

        except ExpiredSignatureError:
            logger.warning("Token expirado")
            return False, None, "Token expirado"

        except InvalidTokenError as e:
            logger.warning(f"Token inválido: {e}")
            return False, None, f"Token inválido: {str(e)}"

        except Exception as e:
            logger.error(f"Error validando token: {e}")
            return False, None, f"Error de validación: {str(e)}"

    def refresh_access_token(self, refresh_token: str) -> Tuple[bool, Optional[str], str]:
        """
        Generar nuevo access token usando refresh token

        Args:
            refresh_token: Refresh token válido

        Returns:
            Tupla (éxito, nuevo_token, mensaje)
        """
        # Validar refresh token
        valid, payload, error = self.validate_token(refresh_token)

        if not valid:
            return False, None, f"Refresh token inválido: {error}"

        # Verificar que sea un refresh token
        try:
            payload_dict = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])

            if payload_dict.get("type") != "refresh":
                return False, None, "No es un refresh token"

            # Generar nuevo access token
            new_token = self.generate_token(user_id=payload.user_id, role=payload.role)

            logger.info(f"Access token renovado para user_id={payload.user_id}")
            return True, new_token, "Token renovado exitosamente"

        except Exception as e:
            logger.error(f"Error renovando token: {e}")
            return False, None, f"Error: {str(e)}"

    def check_permission(self, payload: TokenPayload, required_role: str) -> bool:
        """
        Verificar si el usuario tiene el rol requerido

        Args:
            payload: Payload del token
            required_role: Rol requerido (user, trainer, admin)

        Returns:
            True si tiene permiso
        """
        # Jerarquía de roles
        role_hierarchy = {"user": 1, "trainer": 2, "admin": 3}

        user_level = role_hierarchy.get(payload.role, 0)
        required_level = role_hierarchy.get(required_role, 999)

        has_permission = user_level >= required_level

        if not has_permission:
            logger.warning(
                f"Permiso denegado: user_id={payload.user_id}, " f"role={payload.role}, required={required_role}"
            )

        return has_permission


# Instancia global (singleton)
_jwt_manager: Optional[JWTAuthManager] = None


def get_jwt_manager(secret_key: Optional[str] = None, **kwargs) -> JWTAuthManager:
    """
    Obtener instancia global del JWT manager

    Args:
        secret_key: Clave secreta opcional
        **kwargs: Argumentos adicionales para JWTAuthManager

    Returns:
        Instancia de JWTAuthManager
    """
    global _jwt_manager

    if _jwt_manager is None:
        _jwt_manager = JWTAuthManager(secret_key=secret_key, **kwargs)

    return _jwt_manager


def require_auth(required_role: str = "user"):
    """
    Decorador para requerir autenticación en endpoints

    Args:
        required_role: Rol mínimo requerido

    Usage:
        @require_auth("trainer")
        def protected_endpoint(payload: TokenPayload):
            # payload contiene información del usuario autenticado
            pass
    """

    def decorator(func):
        def wrapper(token: str, *args, **kwargs):
            jwt_manager = get_jwt_manager()

            # Validar token
            valid, payload, error = jwt_manager.validate_token(token)

            if not valid:
                raise PermissionError(f"Autenticación fallida: {error}")

            # Verificar permisos
            if not jwt_manager.check_permission(payload, required_role):
                raise PermissionError(f"Rol insuficiente. Requerido: {required_role}, " f"actual: {payload.role}")

            # Ejecutar función con payload
            return func(payload, *args, **kwargs)

        return wrapper

    return decorator


# Exports
__all__ = ["JWTAuthManager", "TokenPayload", "get_jwt_manager", "require_auth", "JWT_AVAILABLE"]
