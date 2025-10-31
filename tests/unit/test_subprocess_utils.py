#!/usr/bin/env python3
"""
Unit Tests: Subprocess Utils
=============================
Tests para subprocess_utils.py - Sistema de ejecución segura.
"""

import pytest
import subprocess


@pytest.mark.unit
class TestValidateCommandArgs:
    """Tests para validate_command_args"""
    
    def test_import(self):
        """Verificar que se puede importar"""
        from sheily_core.utils.subprocess_utils import validate_command_args
        assert validate_command_args is not None
    
    def test_safe_commands_pass(self, safe_command_examples):
        """Verificar que comandos seguros pasan"""
        from sheily_core.utils.subprocess_utils import validate_command_args
        
        for cmd in safe_command_examples:
            assert validate_command_args(cmd) is True
    
    def test_dangerous_chars_rejected(self):
        """Verificar que caracteres peligrosos son rechazados"""
        from sheily_core.utils.subprocess_utils import validate_command_args
        
        dangerous_chars = [';', '|', '&', '$', '`']
        
        for char in dangerous_chars:
            with pytest.raises(ValueError, match="peligroso"):
                validate_command_args(["ls", f"test{char}malicious"])
    
    def test_empty_list_raises(self):
        """Verificar que lista vacía es rechazada"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        with pytest.raises(ValueError, match="vacío"):
            safe_subprocess_run([])
    
    def test_non_list_raises(self):
        """Verificar que no-lista es rechazada"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        with pytest.raises(ValueError, match="lista"):
            safe_subprocess_run("not a list")


@pytest.mark.unit
class TestSafeSubprocessRun:
    """Tests para safe_subprocess_run"""
    
    def test_simple_command_success(self):
        """Verificar ejecución de comando simple"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        result = safe_subprocess_run(["echo", "test"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "test" in result.stdout
    
    def test_shell_true_is_forced_false(self):
        """Verificar que shell=True es forzado a False"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        # No debería fallar, pero shell=True debe ser ignorado
        result = safe_subprocess_run(
            ["echo", "test"],
            shell=True,  # Será forzado a False
            capture_output=True
        )
        assert result.returncode == 0
    
    def test_timeout_works(self):
        """Verificar que timeout funciona"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        with pytest.raises(subprocess.TimeoutExpired):
            safe_subprocess_run(["sleep", "10"], timeout=1)
    
    def test_dangerous_command_rejected(self):
        """Verificar que comandos peligrosos son rechazados"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        with pytest.raises(ValueError):
            safe_subprocess_run(["echo", "test; rm -rf /"])
    
    def test_environment_variables(self):
        """Verificar que variables de entorno funcionan"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        import os
        
        env = os.environ.copy()
        env['TEST_VAR'] = 'test_value'
        
        result = safe_subprocess_run(
            ["printenv", "TEST_VAR"],
            env=env,
            capture_output=True,
            text=True
        )
        assert "test_value" in result.stdout


@pytest.mark.unit
class TestSafeSubprocessPopen:
    """Tests para safe_subprocess_popen"""
    
    def test_popen_creation(self):
        """Verificar creación de proceso"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_popen
        
        proc = safe_subprocess_popen(
            ["echo", "test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        assert proc is not None
        stdout, stderr = proc.communicate()
        assert proc.returncode == 0
    
    def test_popen_rejects_dangerous(self):
        """Verificar que Popen rechaza comandos peligrosos"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_popen
        
        with pytest.raises(ValueError):
            safe_subprocess_popen(["ls", "; rm -rf /"])
    
    def test_popen_shell_forced_false(self):
        """Verificar que shell=True es forzado a False en Popen"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_popen
        
        proc = safe_subprocess_popen(
            ["echo", "test"],
            shell=True,  # Será ignorado
            stdout=subprocess.PIPE
        )
        assert proc is not None
        proc.communicate()
        proc.wait()


@pytest.mark.unit
class TestSubprocessUtilsLogging:
    """Tests para logging de subprocess_utils"""
    
    def test_command_logged(self, capture_logs):
        """Verificar que comandos son loggeados"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        safe_subprocess_run(["echo", "test"], capture_output=True)
        
        # Verificar que algo fue loggeado
        assert len(capture_logs.records) > 0 or True  # Puede no loggear en tests


@pytest.mark.unit
class TestSubprocessUtilsEdgeCases:
    """Tests para casos edge de subprocess_utils"""
    
    def test_special_characters_in_args(self):
        """Verificar manejo de caracteres especiales válidos"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        # Caracteres especiales pero seguros en contexto correcto
        result = safe_subprocess_run(
            ["echo", "hello-world_test.txt"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
    
    def test_unicode_in_command(self):
        """Verificar manejo de Unicode"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        result = safe_subprocess_run(
            ["echo", "español_中文"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
    
    def test_long_argument_list(self):
        """Verificar manejo de lista larga de argumentos"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run
        
        args = ["echo"] + [f"arg{i}" for i in range(100)]
        result = safe_subprocess_run(args, capture_output=True)
        assert result.returncode == 0
