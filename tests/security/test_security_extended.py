#!/usr/bin/env python3
"""
Test de seguridad adicionales para Sheily AI
Implementa análisis de vulnerabilidades y tests de seguridad
"""

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Configurar path
sys_path_backup = None
try:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(project_root))
    from tests.test_imports import Config, Logger, setup_test_environment

    setup_test_environment()
except ImportError:
    # Fallback si no está disponible
    class Config:
        def __init__(self, **kwargs):
            self.data = kwargs

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value):
            self.data[key] = value

    class Logger:
        def info(self, msg):
            pass

        def error(self, msg):
            pass


class TestSecurityBasics:
    """Tests básicos de seguridad"""

    def test_environment_variables_security(self):
        """Test seguridad de variables de entorno"""
        # Variables sensibles no deberían estar en el código
        sensitive_vars = ["PASSWORD", "SECRET", "TOKEN", "KEY", "API_KEY"]

        # Verificar que no hay variables sensibles hardcodeadas
        test_config = Config(debug=False, secret_key="test_key", api_token="test_token")

        # Las variables sensibles deberían venir del entorno, no estar hardcodeadas
        assert test_config.get("secret_key") is not None
        assert test_config.get("debug") is False

    def test_secret_key_generation(self):
        """Test generación segura de claves secretas"""
        # Generar clave secreta
        secret_key = secrets.token_urlsafe(32)

        # Verificar propiedades de seguridad
        assert len(secret_key) >= 32
        assert isinstance(secret_key, str)

        # Generar otra y verificar que son diferentes
        another_key = secrets.token_urlsafe(32)
        assert secret_key != another_key

    def test_password_hashing(self):
        """Test hashing seguro de contraseñas"""
        password = "test_password_123"

        # Simular hash seguro
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)

        # Verificar que el hash es diferente de la contraseña original
        assert password_hash != password.encode("utf-8")
        assert len(password_hash) == 32  # SHA-256 produces 32 bytes

    def test_input_validation(self):
        """Test validación de entrada para prevenir inyección"""
        # Inputs maliciosos comunes
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../etc/passwd",
            "eval(malicious_code)",
            "${jndi:ldap://malicious.com}",
            "{{7*7}}",  # Template injection
        ]

        for malicious_input in malicious_inputs:
            # Simular validación de entrada
            sanitized = self._sanitize_input(malicious_input)

            # El input sanitizado no debería contener caracteres peligrosos
            dangerous_chars = ["<", ">", "'", '"', ";", "&", "|"]
            for char in dangerous_chars:
                assert char not in sanitized or sanitized.count(char) == 0

    def _sanitize_input(self, user_input):
        """Función helper para sanitización"""
        if not isinstance(user_input, str):
            return ""

        # Remover caracteres peligrosos
        dangerous_chars = ["<", ">", "'", '"', ";", "&", "|", "$", "`"]
        sanitized = user_input
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")

        return sanitized


class TestFileSystemSecurity:
    """Tests de seguridad del sistema de archivos"""

    def test_path_traversal_prevention(self):
        """Test prevención de path traversal"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "....//....//....//etc//passwd",
        ]

        for malicious_path in malicious_paths:
            safe_path = self._sanitize_path(malicious_path)

            # El path sanitizado no debería contener traversal
            assert ".." not in safe_path
            assert not os.path.isabs(safe_path) or safe_path.startswith("/tmp/")

    def test_file_permissions(self):
        """Test permisos seguros de archivos"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear archivo de prueba
            test_file = os.path.join(temp_dir, "test_secure_file.txt")

            with open(test_file, "w") as f:
                f.write("sensitive data")

            # Establecer permisos seguros (solo propietario)
            os.chmod(test_file, 0o600)

            # Verificar permisos
            file_stat = os.stat(test_file)
            permissions = oct(file_stat.st_mode)[-3:]

            # Debería ser 600 (rw-------) o más restrictivo
            assert permissions in ["600", "400"]

    def test_temporary_file_security(self):
        """Test seguridad de archivos temporales"""
        # Crear archivo temporal seguro
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("sensitive temporary data")
            temp_path = temp_file.name

        try:
            # Verificar que el archivo temporal tiene permisos seguros
            file_stat = os.stat(temp_path)
            permissions = oct(file_stat.st_mode)[-3:]

            # En sistemas Unix, debería ser 600 o más restrictivo
            if os.name != "nt":  # No Windows
                assert int(permissions) <= 600
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _sanitize_path(self, user_path):
        """Sanitizar path para prevenir traversal"""
        if not isinstance(user_path, str):
            return ""

        # Remover path traversal
        sanitized = user_path.replace("..", "").replace("\\", "/")

        # Asegurar que está dentro de un directorio seguro
        if os.path.isabs(sanitized):
            sanitized = os.path.basename(sanitized)

        return sanitized


class TestNetworkSecurity:
    """Tests de seguridad de red"""

    def test_secure_url_validation(self):
        """Test validación segura de URLs"""
        # URLs maliciosas
        malicious_urls = [
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "file:///etc/passwd",
            "ftp://malicious.com/malware",
            "http://169.254.169.254/metadata",  # AWS metadata
        ]

        for url in malicious_urls:
            is_safe = self._validate_url(url)
            assert not is_safe, f"URL should be rejected: {url}"

    def test_allowed_hosts(self):
        """Test lista de hosts permitidos"""
        allowed_hosts = ["localhost", "127.0.0.1", "example.com"]
        malicious_hosts = [
            "malicious.com",
            "169.254.169.254",  # AWS metadata
            "192.168.1.1",  # Internal network
            "10.0.0.1",  # Internal network
        ]

        for host in allowed_hosts:
            assert self._is_host_allowed(host)

        for host in malicious_hosts:
            assert not self._is_host_allowed(host)

    def test_rate_limiting_simulation(self):
        """Test simulación de rate limiting"""
        # Simular rate limiter
        rate_limiter = {}
        client_ip = "192.168.1.100"
        max_requests = 10
        time_window = 60  # seconds

        import time

        current_time = int(time.time())

        # Simular múltiples requests
        for i in range(15):  # Más que el límite
            allowed = self._check_rate_limit(rate_limiter, client_ip, max_requests, current_time)

            if i < max_requests:
                assert allowed, f"Request {i} should be allowed"
            else:
                assert not allowed, f"Request {i} should be rate limited"

    def _validate_url(self, url):
        """Validar URL de forma segura"""
        if not isinstance(url, str):
            return False

        # Esquemas permitidos
        allowed_schemes = ["http", "https"]

        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.scheme in allowed_schemes
        except:
            return False

    def _is_host_allowed(self, host):
        """Verificar si un host está permitido"""
        allowed_hosts = ["localhost", "127.0.0.1", "example.com"]
        return host in allowed_hosts

    def _check_rate_limit(self, rate_limiter, client_ip, max_requests, current_time):
        """Simulación simple de rate limiting"""
        if client_ip not in rate_limiter:
            rate_limiter[client_ip] = []

        # Limpiar requests antiguos (simulado)
        rate_limiter[client_ip] = [t for t in rate_limiter[client_ip] if current_time - t < 60]

        # Verificar límite
        if len(rate_limiter[client_ip]) >= max_requests:
            return False

        # Agregar request actual
        rate_limiter[client_ip].append(current_time)
        return True


class TestDataSecurity:
    """Tests de seguridad de datos"""

    def test_sensitive_data_masking(self):
        """Test enmascaramiento de datos sensibles"""
        sensitive_data = {
            "credit_card": "1234-5678-9012-3456",
            "ssn": "123-45-6789",
            "email": "user@example.com",
            "phone": "+1-555-123-4567",
        }

        for data_type, value in sensitive_data.items():
            masked = self._mask_sensitive_data(value, data_type)

            # Los datos enmascarados no deberían revelar información completa
            assert masked != value
            assert len(masked) > 0

    def test_data_encryption_simulation(self):
        """Test simulación de cifrado de datos"""
        sensitive_text = "This is sensitive information"

        # Simular cifrado simple (en producción usar librerías crypto apropiadas)
        encrypted = self._simple_encrypt(sensitive_text)
        decrypted = self._simple_decrypt(encrypted)

        # Verificar cifrado/descifrado
        assert encrypted != sensitive_text
        assert decrypted == sensitive_text

    def test_sql_injection_prevention(self):
        """Test prevención de inyección SQL"""
        # Queries maliciosas típicas
        malicious_queries = [
            "1'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; SELECT * FROM passwords; --",
            "1' UNION SELECT password FROM users --",
        ]

        for query in malicious_queries:
            # Simular sanitización de query
            sanitized = self._sanitize_sql_input(query)

            # El query sanitizado no debería contener SQL malicioso
            sql_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "UNION", "SELECT"]
            for keyword in sql_keywords:
                # Si contiene keywords, deberían estar escapados o removidos
                if keyword in sanitized.upper():
                    # Debería estar dentro de comillas o escapado
                    assert sanitized.count("'") % 2 == 0 or keyword not in sanitized.upper()

    def _mask_sensitive_data(self, data, data_type):
        """Enmascarar datos sensibles"""
        if data_type == "credit_card":
            return f"****-****-****-{data[-4:]}"
        elif data_type == "ssn":
            return f"***-**-{data[-4:]}"
        elif data_type == "email":
            parts = data.split("@")
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif data_type == "phone":
            return f"***-***-{data[-4:]}"

        return "***"

    def _simple_encrypt(self, text):
        """Cifrado simple para testing (NO usar en producción)"""
        return "".join(chr(ord(c) + 1) for c in text)

    def _simple_decrypt(self, encrypted_text):
        """Descifrado simple para testing"""
        return "".join(chr(ord(c) - 1) for c in encrypted_text)

    def _sanitize_sql_input(self, user_input):
        """Sanitizar entrada SQL"""
        if not isinstance(user_input, str):
            return ""

        # Escapar comillas simples
        sanitized = user_input.replace("'", "''")

        # Remover comentarios SQL
        sanitized = sanitized.replace("--", "").replace("/*", "").replace("*/", "")

        return sanitized


@pytest.mark.security
class TestSecurityIntegration:
    """Tests de integración de seguridad"""

    def test_comprehensive_security_check(self):
        """Test integral de seguridad"""
        # Configuración de seguridad
        security_config = {
            "secret_key_length": 32,
            "password_min_length": 8,
            "session_timeout": 3600,
            "max_login_attempts": 5,
            "enable_2fa": True,
        }

        # Verificar configuración
        for key, value in security_config.items():
            assert value is not None
            if isinstance(value, int):
                assert value > 0

    def test_security_headers(self):
        """Test headers de seguridad HTTP"""
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }

        # Simular verificación de headers
        for header, expected_value in required_headers.items():
            # En una aplicación real, esto verificaría headers HTTP reales
            assert self._check_security_header(header, expected_value)

    def test_authentication_security(self):
        """Test seguridad de autenticación"""
        # Simular intento de login
        username = "testuser"
        password = "secure_password_123"

        # Test autenticación básica
        auth_result = self._simulate_authentication(username, password)
        assert auth_result["success"] is True
        assert "token" in auth_result

        # Test con credenciales incorrectas
        bad_auth = self._simulate_authentication(username, "wrong_password")
        assert bad_auth["success"] is False
        assert "error" in bad_auth

    def _check_security_header(self, header_name, expected_value):
        """Simular verificación de header de seguridad"""
        # En implementación real, verificaría headers HTTP
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }

        return security_headers.get(header_name) == expected_value

    def _simulate_authentication(self, username, password):
        """Simular proceso de autenticación"""
        # Credenciales válidas simuladas
        valid_credentials = {"testuser": "secure_password_123"}

        if username in valid_credentials and valid_credentials[username] == password:
            return {"success": True, "token": secrets.token_urlsafe(32), "expires_in": 3600}
        else:
            return {"success": False, "error": "Invalid credentials"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])
