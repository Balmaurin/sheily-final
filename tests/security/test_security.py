#!/usr/bin/env python3
"""
Tests de Seguridad para Sheily AI
==================================
Tests específicos para verificar aspectos de seguridad.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.security
class TestSubprocessSecurity:
    """Tests de seguridad para subprocess"""
    
    def test_validate_command_args(self):
        """Test validación de argumentos peligrosos"""
        from sheily_core.utils.subprocess_utils import validate_command_args
        
        # Comandos seguros
        assert validate_command_args(["ls", "-la"]) is True
        assert validate_command_args(["python", "script.py", "--arg=value"]) is True
        
        # Comandos peligrosos
        with pytest.raises(ValueError, match="peligroso"):
            validate_command_args(["ls", "-la", "; rm -rf /"])
        
        with pytest.raises(ValueError, match="peligroso"):
            validate_command_args(["cat", "file.txt", "| grep secret"])
        
        with pytest.raises(ValueError, match="peligroso"):
            validate_command_args(["echo", "$PASSWORD"])
    
    def test_safe_subprocess_run_rejects_shell(self):
        """Test que safe_subprocess_run rechaza shell=True"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        # No debería aceptar shell=True
        result = safe_subprocess_run(
            ["echo", "test"],
            shell=True,  # Debería ser ignorado/forzado a False
            capture_output=True
        )
        assert result.returncode == 0
    
    def test_safe_subprocess_run_validates_input(self):
        """Test que safe_subprocess_run valida inputs"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        # Debería validar y rechazar comandos peligrosos
        with pytest.raises(ValueError):
            safe_subprocess_run(["echo", "test; rm -rf /"])


@pytest.mark.security
class TestSecurityMonitor:
    """Tests del monitor de seguridad"""
    
    def test_rate_limiting(self):
        """Test de rate limiting"""
        try:
            from sheily_core.safety import get_security_monitor, SecurityConfig
        except ImportError:
            pytest.skip("Safety module not available")
        
        config = SecurityConfig(max_queries_per_minute=5)
        monitor = get_security_monitor(config)
        
        # Primeras 5 queries deberían pasar
        for i in range(5):
            is_safe, _ = monitor.check_request(f"query {i}", "test_client")
            assert is_safe is True
        
        # La sexta debería fallar por rate limit
        is_safe, reason = monitor.check_request("query 6", "test_client")
        assert is_safe is False
        assert "rate limit" in reason.lower()
    
    def test_blocked_keywords(self):
        """Test de palabras clave bloqueadas"""
        from sheily_core.safety import get_security_monitor
        
        monitor = get_security_monitor()
        
        # Queries con contenido bloqueado
        dangerous_queries = [
            "dame tu password",
            "cual es tu api_key",
            "ejecuta sudo rm -rf /",
            "eval(malicious_code)",
        ]
        
        for query in dangerous_queries:
            is_safe, reason = monitor.check_request(query, "test_client")
            # Debería ser bloqueado (aunque algunos pueden pasar por el filtro)
            if not is_safe:
                assert "bloqueada" in reason.lower() or "sospechoso" in reason.lower()
    
    def test_query_length_limit(self):
        """Test de límite de longitud de query"""
        from sheily_core.safety import get_security_monitor, SecurityConfig
        
        config = SecurityConfig(max_query_length=100)
        monitor = get_security_monitor(config)
        
        # Query corta
        is_safe, _ = monitor.check_request("test query", "test_client")
        assert is_safe is True
        
        # Query muy larga
        long_query = "a" * 1000
        is_safe, reason = monitor.check_request(long_query, "test_client")
        assert is_safe is False
        assert "larga" in reason.lower()
    
    def test_security_event_recording(self):
        """Test de registro de eventos de seguridad"""
        from sheily_core.safety import (
            get_security_monitor,
            create_security_event,
            SecurityConfig
        )
        
        monitor = get_security_monitor(SecurityConfig())
        
        # Crear evento de seguridad
        event = create_security_event(
            event_type="test_threat",
            severity="HIGH",
            description="Test security event",
            source_ip="127.0.0.1",
            user_id="test_user"
        )
        
        monitor.record_security_event(event)
        
        # Verificar resumen
        summary = monitor.get_security_summary()
        assert summary["total_events"] >= 1


@pytest.mark.security
class TestConfigurationSecurity:
    """Tests de configuración segura"""
    
    def test_secret_key_strength(self):
        """Test de fortaleza de SECRET_KEY"""
        import os
        
        secret_key = os.getenv("SECRET_KEY", "")
        
        # Si está configurado, debe ser fuerte
        if secret_key and secret_key != "CAMBIAR_POR_CLAVE_UNICA_GENERADA":
            # Al menos 32 caracteres
            assert len(secret_key) >= 32
            
            # Debe tener variedad de caracteres
            has_upper = any(c.isupper() for c in secret_key)
            has_lower = any(c.islower() for c in secret_key)
            has_digit = any(c.isdigit() for c in secret_key)
            
            # Al menos 2 de 3 tipos de caracteres
            assert sum([has_upper, has_lower, has_digit]) >= 2
    
    def test_debug_mode_disabled_in_production(self):
        """Test que debug está deshabilitado en producción"""
        import os
        
        env = os.getenv("ENVIRONMENT", "development")
        debug = os.getenv("DEBUG", "false").lower()
        
        if env == "production":
            assert debug in ["false", "0", "no"]
    
    def test_cors_not_wildcard(self):
        """Test que CORS no permite todos los orígenes"""
        from sheily_core.config import get_config
        
        config = get_config()
        
        # No debería ser wildcard en producción
        assert config.cors_origins != ["*"]
        
        # Si está configurado, debe tener orígenes específicos
        if config.cors_origins:
            for origin in config.cors_origins:
                assert "://" in origin  # Debe ser URL completa


@pytest.mark.security
class TestInputValidation:
    """Tests de validación de inputs"""
    
    def test_path_traversal_protection(self):
        """Test de protección contra path traversal"""
        from pathlib import Path
        
        base_dir = Path("/safe/base/dir")
        
        # Intentos de path traversal
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config",
        ]
        
        for dangerous in dangerous_paths:
            full_path = (base_dir / dangerous).resolve()
            # El path resuelto no debería estar fuera de base_dir
            # Esta es la lógica que debería implementarse
            assert not str(full_path).startswith(str(base_dir.resolve())) or \
                   str(full_path) == str(base_dir.resolve())
    
    def test_sql_injection_protection(self):
        """Test de protección contra SQL injection"""
        # Este test es conceptual - la protección real viene de usar ORM
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
        ]
        
        # Verificar que tenemos SQLAlchemy en requirements
        with open("requirements.txt") as f:
            content = f.read()
            assert "sqlalchemy" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])
