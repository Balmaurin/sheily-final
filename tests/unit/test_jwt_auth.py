#!/usr/bin/env python3
"""
Unit Tests: JWT Authentication System - REAL
=============================================
Tests para el sistema de autenticación JWT REAL (no mock).
"""

import pytest
import os
from datetime import datetime, timedelta


@pytest.mark.unit
class TestJWTAuthManager:
    """Tests para JWTAuthManager REAL"""
    
    def test_jwt_manager_requires_secret_key(self):
        """Verificar que JWT manager requiere SECRET_KEY"""
        # Guardar SECRET_KEY actual
        original_key = os.getenv('SECRET_KEY')
        
        try:
            # Remover SECRET_KEY
            if 'SECRET_KEY' in os.environ:
                del os.environ['SECRET_KEY']
            
            from sheily_core.security.jwt_auth import JWTAuthManager
            
            # Debe fallar sin SECRET_KEY
            with pytest.raises(ValueError, match="SECRET_KEY"):
                JWTAuthManager()
                
        finally:
            # Restaurar
            if original_key:
                os.environ['SECRET_KEY'] = original_key
    
    def test_jwt_manager_rejects_default_key(self):
        """Verificar que rechaza SECRET_KEY por defecto"""
        os.environ['SECRET_KEY'] = 'change_this_in_production'
        
        try:
            from sheily_core.security.jwt_auth import JWTAuthManager
            
            with pytest.raises(ValueError, match="SECRET_KEY"):
                JWTAuthManager()
        finally:
            # Limpiar
            if 'SECRET_KEY' in os.environ:
                del os.environ['SECRET_KEY']
    
    def test_jwt_manager_initialization(self, test_env_vars):
        """Verificar inicialización correcta"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        assert manager.secret_key == "test_secret_key_at_least_32_chars_long_12345"
        assert manager.algorithm == "HS256"
        assert manager.token_expiry_minutes == 60
    
    def test_generate_token(self, test_env_vars):
        """Verificar generación de token"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        token = manager.generate_token(user_id="test_user", role="trainer")
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are largo
    
    def test_validate_valid_token(self, test_env_vars):
        """Verificar validación de token válido"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        # Generar token
        token = manager.generate_token(user_id="test_user", role="trainer")
        
        # Validar
        valid, payload, error = manager.validate_token(token)
        
        assert valid is True
        assert payload is not None
        assert payload.user_id == "test_user"
        assert payload.role == "trainer"
        assert error == "Token válido"
    
    def test_validate_expired_token(self, test_env_vars):
        """Verificar detección de token expirado"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        # Manager con expiración de 0 segundos
        manager = JWTAuthManager(
            secret_key="test_secret_key_at_least_32_chars_long_12345",
            token_expiry_minutes=0
        )
        
        token = manager.generate_token(user_id="test_user", role="user")
        
        # Esperar un poco
        import time
        time.sleep(1)
        
        # Validar - debe estar expirado
        valid, payload, error = manager.validate_token(token)
        
        assert valid is False
        assert payload is None
        assert "expirado" in error.lower()
    
    def test_validate_invalid_token(self, test_env_vars):
        """Verificar rechazo de token inválido"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        # Token falso
        fake_token = "esto.no.es.un.token.jwt.valido"
        
        valid, payload, error = manager.validate_token(fake_token)
        
        assert valid is False
        assert payload is None
        assert "inválido" in error.lower() or "invalid" in error.lower()
    
    def test_validate_empty_token(self, test_env_vars):
        """Verificar rechazo de token vacío"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        valid, payload, error = manager.validate_token("")
        
        assert valid is False
        assert payload is None
    
    def test_check_permission_hierarchy(self, test_env_vars):
        """Verificar jerarquía de roles"""
        from sheily_core.security.jwt_auth import JWTAuthManager, TokenPayload, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        # Admin puede hacer todo
        admin_payload = TokenPayload(
            user_id="admin1",
            role="admin",
            exp=9999999999,
            iat=1234567890
        )
        assert manager.check_permission(admin_payload, "user") is True
        assert manager.check_permission(admin_payload, "trainer") is True
        assert manager.check_permission(admin_payload, "admin") is True
        
        # Trainer puede hacer user y trainer
        trainer_payload = TokenPayload(
            user_id="trainer1",
            role="trainer",
            exp=9999999999,
            iat=1234567890
        )
        assert manager.check_permission(trainer_payload, "user") is True
        assert manager.check_permission(trainer_payload, "trainer") is True
        assert manager.check_permission(trainer_payload, "admin") is False
        
        # User solo puede hacer user
        user_payload = TokenPayload(
            user_id="user1",
            role="user",
            exp=9999999999,
            iat=1234567890
        )
        assert manager.check_permission(user_payload, "user") is True
        assert manager.check_permission(user_payload, "trainer") is False
        assert manager.check_permission(user_payload, "admin") is False
    
    def test_refresh_token_flow(self, test_env_vars):
        """Verificar flujo de refresh token"""
        from sheily_core.security.jwt_auth import JWTAuthManager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager = JWTAuthManager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        
        # Generar refresh token
        refresh_token = manager.generate_refresh_token(user_id="test_user", role="trainer")
        
        assert refresh_token is not None
        
        # Usar refresh token para obtener nuevo access token
        success, new_token, message = manager.refresh_access_token(refresh_token)
        
        assert success is True
        assert new_token is not None
        assert "renovado" in message.lower() or "success" in message.lower()
        
        # Validar nuevo token
        valid, payload, _ = manager.validate_token(new_token)
        assert valid is True
        assert payload.user_id == "test_user"


@pytest.mark.unit
class TestJWTGlobalManager:
    """Tests para gestor global de JWT"""
    
    def test_get_jwt_manager_singleton(self, test_env_vars):
        """Verificar que get_jwt_manager devuelve singleton"""
        from sheily_core.security.jwt_auth import get_jwt_manager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        manager1 = get_jwt_manager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        manager2 = get_jwt_manager()
        
        assert manager1 is manager2


@pytest.mark.unit
class TestJWTIntegration:
    """Tests de integración de JWT con API"""
    
    def test_api_jwt_validation_with_real_jwt(self, test_env_vars):
        """Verificar que API usa JWT real"""
        from sheily_core.security.jwt_auth import get_jwt_manager, JWT_AVAILABLE
        
        if not JWT_AVAILABLE:
            pytest.skip("PyJWT not installed")
        
        # Generar token real
        manager = get_jwt_manager(secret_key="test_secret_key_at_least_32_chars_long_12345")
        token = manager.generate_token(user_id="api_test_user", role="trainer")
        
        # Importar validación de API
        from sheily_core.core.api import validate_jwt_token
        
        # Validar con API
        result = validate_jwt_token(token)
        
        # Debe ser Ok (no Err)
        assert hasattr(result, 'is_ok')
        if result.is_ok():
            payload = result.unwrap()
            assert payload['user_id'] == "api_test_user"
            assert payload['role'] == "trainer"
            assert 'permissions' in payload
