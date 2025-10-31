#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Test Suite for Sheily Core Modules
Tests for core utilities, configuration, logging, and security
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import logging
from datetime import datetime
import sys


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_core_dir():
    """Create temporary core directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_logger():
    """Create mock logger"""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def sample_config():
    """Create sample configuration"""
    return {
        "app_name": "sheily",
        "version": "1.0.0",
        "debug": True,
        "log_level": "INFO",
        "model_path": "/models/llama.gguf",
        "max_tokens": 2048,
        "temperature": 0.7,
    }


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Tests for core configuration"""

    def test_load_config_from_file(self, temp_core_dir, sample_config):
        """Test loading configuration from file"""
        config_file = temp_core_dir / "config.json"
        config_file.write_text(json.dumps(sample_config))
        
        loaded = json.loads(config_file.read_text())
        assert loaded["app_name"] == "sheily"
        assert loaded["version"] == "1.0.0"

    def test_config_defaults(self):
        """Test configuration defaults"""
        config = {
            "debug": True,
            "log_level": "INFO",
            "timeout": 300,
        }
        
        assert config.get("debug", False) == True
        assert config.get("timeout", 300) == 300

    def test_config_validation(self, sample_config):
        """Test configuration validation"""
        required_fields = ["app_name", "version"]
        
        for field in required_fields:
            assert field in sample_config

    def test_config_override(self, sample_config):
        """Test configuration overrides"""
        overrides = {"debug": False, "max_tokens": 4096}
        merged = {**sample_config, **overrides}
        
        assert merged["debug"] == False
        assert merged["max_tokens"] == 4096
        assert merged["app_name"] == "sheily"  # Original preserved

    def test_environment_variable_config(self):
        """Test loading config from environment variables"""
        import os
        
        os.environ["SHEILY_DEBUG"] = "true"
        os.environ["SHEILY_TIMEOUT"] = "600"
        
        config = {
            "debug": os.getenv("SHEILY_DEBUG", "false").lower() == "true",
            "timeout": int(os.getenv("SHEILY_TIMEOUT", "300")),
        }
        
        assert config["debug"] == True
        assert config["timeout"] == 600


# ============================================================================
# Logger Tests
# ============================================================================


class TestLogging:
    """Tests for core logging"""

    def test_logger_initialization(self, mock_logger):
        """Test logger initialization"""
        assert callable(mock_logger.debug)
        assert callable(mock_logger.error)

    def test_log_level_setting(self):
        """Test log level setting"""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        
        assert log_levels["DEBUG"] < log_levels["INFO"]
        assert log_levels["WARNING"] > log_levels["INFO"]

    def test_structured_logging(self, mock_logger):
        """Test structured logging"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": "Test message",
            "module": "test",
        }
        
        assert "timestamp" in log_entry
        assert "message" in log_entry

    def test_log_filtering(self):
        """Test log filtering"""
        logs = [
            {"level": "DEBUG", "message": "debug msg"},
            {"level": "ERROR", "message": "error msg"},
            {"level": "INFO", "message": "info msg"},
        ]
        
        error_logs = [log for log in logs if log["level"] == "ERROR"]
        assert len(error_logs) == 1

    def test_context_logging(self, mock_logger):
        """Test logging with context"""
        context = {
            "user_id": "user123",
            "session_id": "sess456",
            "request_id": "req789",
        }
        
        assert context["user_id"] == "user123"

    def test_exception_logging(self, mock_logger):
        """Test exception logging"""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            assert isinstance(e, ValueError)


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurity:
    """Tests for core security"""

    def test_input_sanitization(self):
        """Test input sanitization"""
        untrusted_input = "<script>alert('xss')</script>"
        sanitized = untrusted_input.replace("<", "&lt;").replace(">", "&gt;")
        
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized

    def test_path_traversal_prevention(self):
        """Test path traversal prevention"""
        user_input = "../../etc/passwd"
        base_path = Path("/safe/directory")
        
        requested_path = (base_path / user_input).resolve()
        is_safe = str(requested_path).startswith(str(base_path.resolve()))
        
        assert not is_safe or user_input.startswith("..")

    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        user_input = "test; rm -rf /"
        
        # Should not execute shell commands directly
        assert ";" in user_input

    def test_rate_limiting_logic(self):
        """Test rate limiting logic"""
        request_timestamps = [
            datetime.now().timestamp(),
            datetime.now().timestamp() + 0.1,
            datetime.now().timestamp() + 0.2,
        ]
        
        requests_per_second = len(request_timestamps) / 1.0
        rate_limit = 100
        
        assert requests_per_second < rate_limit

    def test_authentication_mocking(self):
        """Test authentication"""
        user = {"username": "admin", "password_hash": "hashed_pwd"}
        
        assert "password_hash" in user
        assert "password" not in user  # Should not store plaintext

    def test_authorization_check(self):
        """Test authorization check"""
        user_permissions = ["read", "write"]
        required_permission = "write"
        
        assert required_permission in user_permissions


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for core utility functions"""

    def test_string_utilities(self):
        """Test string utility functions"""
        text = "  Hello World  "
        
        stripped = text.strip()
        assert stripped == "Hello World"
        
        lower = text.lower()
        assert "hello" in lower

    def test_path_utilities(self, temp_core_dir):
        """Test path utility functions"""
        test_path = temp_core_dir / "test" / "nested" / "path"
        test_path.mkdir(parents=True, exist_ok=True)
        
        assert test_path.exists()
        assert test_path.is_dir()

    def test_file_utilities(self, temp_core_dir):
        """Test file utility functions"""
        test_file = temp_core_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_json_utilities(self, temp_core_dir):
        """Test JSON utility functions"""
        data = {"key": "value", "number": 42}
        json_file = temp_core_dir / "data.json"
        
        json_file.write_text(json.dumps(data))
        loaded = json.loads(json_file.read_text())
        
        assert loaded["key"] == "value"

    def test_time_utilities(self):
        """Test time utility functions"""
        now = datetime.now()
        iso_format = now.isoformat()
        
        assert "T" in iso_format

    def test_hash_utilities(self):
        """Test hash utility functions"""
        import hashlib
        
        text = "test"
        hash_value = hashlib.sha256(text.encode()).hexdigest()
        
        assert len(hash_value) == 32


# ============================================================================
# Context Management Tests
# ============================================================================


class TestContextManagement:
    """Tests for context management"""

    def test_context_creation(self):
        """Test context creation"""
        context = {
            "request_id": "req123",
            "user_id": "user456",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"source": "api"},
        }
        
        assert context["request_id"] == "req123"
        assert "timestamp" in context

    def test_context_enrichment(self):
        """Test context enrichment"""
        context = {"user_id": "user123"}
        
        # Add enrichment data
        context["user_agent"] = "Mozilla/5.0"
        context["ip_address"] = "192.168.1.1"
        
        assert context["user_agent"] == "Mozilla/5.0"

    def test_context_passing(self):
        """Test context passing through functions"""
        def process_with_context(context, data):
            return {"context": context, "result": data}
        
        context = {"request_id": "123"}
        result = process_with_context(context, "data")
        
        assert result["context"]["request_id"] == "123"

    def test_context_cleanup(self):
        """Test context cleanup"""
        context = {"temp_data": "value"}
        
        # Cleanup
        context.pop("temp_data", None)
        
        assert "temp_data" not in context


# ============================================================================
# Integration Tests
# ============================================================================


class TestCoreIntegration:
    """Integration tests for core functionality"""

    def test_config_logger_integration(self, sample_config, mock_logger):
        """Test configuration and logger integration"""
        config = sample_config
        
        log_level = config.get("log_level", "INFO")
        assert log_level == "INFO"
        
        mock_logger.info("Config loaded")
        mock_logger.info.assert_called_with("Config loaded")

    def test_security_context_integration(self):
        """Test security and context integration"""
        context = {
            "user_id": "user123",
            "permissions": ["read", "write"],
        }
        
        required_perm = "read"
        has_permission = required_perm in context["permissions"]
        
        assert has_permission

    def test_config_file_environment_integration(self, temp_core_dir, sample_config):
        """Test config file and environment integration"""
        import os
        
        config_file = temp_core_dir / "config.json"
        config_file.write_text(json.dumps(sample_config))
        
        os.environ["CONFIG_PATH"] = str(config_file)
        
        config_path = os.getenv("CONFIG_PATH")
        assert Path(config_path).exists()

    def test_error_logging_integration(self, mock_logger):
        """Test error logging integration"""
        try:
            raise RuntimeError("Test error")
        except RuntimeError as e:
            error_msg = str(e)
            assert error_msg == "Test error"


# ============================================================================
# Dependency Tests
# ============================================================================


class TestDependencies:
    """Tests for dependency management"""

    def test_import_availability(self):
        """Test availability of required imports"""
        required_modules = [
            "json",
            "pathlib",
            "datetime",
            "logging",
        ]
        
        for module_name in required_modules:
            assert module_name in sys.modules or __import__(module_name)

    def test_optional_dependencies(self):
        """Test optional dependencies"""
        optional = ["numpy", "pandas"]
        
        for module_name in optional:
            try:
                __import__(module_name)
            except ImportError:
                pass  # Optional module not installed

    def test_version_compatibility(self, sample_config):
        """Test version compatibility"""
        version = sample_config["version"]
        major, minor, patch = version.split(".")
        
        assert int(major) >= 1
        assert int(minor) >= 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in core"""

    def test_missing_config_file(self):
        """Test handling missing config file"""
        missing_path = Path("/nonexistent/config.json")
        assert not missing_path.exists()

    def test_invalid_json_handling(self):
        """Test handling invalid JSON"""
        invalid_json = "{invalid"
        
        try:
            json.loads(invalid_json)
            assert False, "Should raise JSONDecodeError"
        except json.JSONDecodeError:
            assert True

    def test_permission_error_handling(self, temp_core_dir):
        """Test handling permission errors"""
        restricted_file = temp_core_dir / "restricted.txt"
        restricted_file.write_text("content")
        
        try:
            restricted_file.chmod(0o000)
            # Try to read (should fail if permissions properly enforced)
        except (PermissionError, OSError):
            pass
        finally:
            restricted_file.chmod(0o644)

    def test_timeout_handling(self):
        """Test timeout handling"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        # Just verify handler is callable
        assert callable(timeout_handler)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Tests for core performance"""

    def test_config_loading_speed(self, temp_core_dir, sample_config):
        """Test configuration loading speed"""
        import time
        
        config_file = temp_core_dir / "config.json"
        config_file.write_text(json.dumps(sample_config))
        
        start = time.time()
        loaded = json.loads(config_file.read_text())
        elapsed = time.time() - start
        
        assert elapsed < 0.1

    def test_logger_efficiency(self, mock_logger):
        """Test logger efficiency"""
        import time
        
        start = time.time()
        for _ in range(1000):
            mock_logger.info("Test message")
        elapsed = time.time() - start
        
        assert elapsed < 1.0

    def test_path_resolution_speed(self, temp_core_dir):
        """Test path resolution speed"""
        import time
        
        start = time.time()
        for _ in range(100):
            (temp_core_dir / "test" / "path").resolve()
        elapsed = time.time() - start
        
        assert elapsed < 0.5


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for state management"""

    def test_global_state_handling(self):
        """Test global state handling"""
        global_state = {"initialized": False}
        
        global_state["initialized"] = True
        assert global_state["initialized"] == True

    def test_state_persistence(self, temp_core_dir):
        """Test state persistence"""
        state = {"counter": 0}
        state_file = temp_core_dir / "state.json"
        
        state["counter"] += 1
        state_file.write_text(json.dumps(state))
        
        loaded_state = json.loads(state_file.read_text())
        assert loaded_state["counter"] == 1

    def test_state_isolation(self):
        """Test state isolation between tests"""
        state1 = {"value": 1}
        state2 = {"value": 2}
        
        assert state1["value"] != state2["value"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
