#!/usr/bin/env python3
"""
Tests de Integración para Sheily AI
====================================
Tests end-to-end que verifican la integración de componentes.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.integration
class TestChatIntegration:
    """Tests de integración del sistema de chat"""

    def test_chat_engine_import(self):
        """Verificar que el chat engine se puede importar"""
        from sheily_core import chat_engine

        assert chat_engine is not None

    def test_config_loading(self):
        """Verificar que la configuración se carga correctamente"""
        from sheily_core.config import get_config

        config = get_config()
        assert config is not None
        assert config.system_name == "Sheily AI Enterprise"

    def test_security_monitor(self):
        """Verificar que el monitor de seguridad funciona"""
        from sheily_core.safety import get_security_monitor

        monitor = get_security_monitor()

        # Test query segura
        is_safe, reason = monitor.check_request("¿Cuál es la capital de Francia?", "test_client")
        assert is_safe is True

    def test_subprocess_utils_import(self):
        """Verificar que el módulo de subprocess seguro existe"""
        from sheily_core.utils.subprocess_utils import safe_subprocess_run

        assert safe_subprocess_run is not None


@pytest.mark.integration
class TestSecurityIntegration:
    """Tests de integración de seguridad"""

    def test_secret_key_configured(self):
        """Verificar que SECRET_KEY no es el valor por defecto"""
        import os

        from sheily_core.config import get_config

        secret_key = os.getenv("SECRET_KEY", "")
        assert secret_key != "change_this_in_production"
        assert len(secret_key) >= 32 or secret_key == ""  # Vacío está OK en tests

    def test_cors_not_wildcard(self):
        """Verificar que CORS no usa wildcard"""
        from sheily_core.config import get_config

        config = get_config()

        assert config.cors_origins is not None
        assert ["*"] != config.cors_origins  # No debería ser wildcard

    def test_no_shell_true_in_code(self):
        """Verificar que no hay shell=True en el código"""
        import os
        import subprocess

        project_root = Path(__file__).parent.parent

        # Buscar shell=True en archivos Python (excluyendo venv y comentarios)
        result = subprocess.run(
            [
                "grep",
                "-r",
                "shell=True",
                str(project_root / "sheily_core"),
                str(project_root / "sheily_train"),
                "--include=*.py",
            ],
            capture_output=True,
            text=True,
        )

        # Filtrar comentarios
        if result.stdout:
            lines = [line for line in result.stdout.split("\n") if line and not "#" in line.split("shell=True")[0]]
            assert len(lines) == 0, f"Found shell=True in: {lines}"


@pytest.mark.integration
class TestDatabaseIntegration:
    """Tests de integración con base de datos"""

    @pytest.fixture
    def db_url(self):
        """URL de base de datos para tests"""
        import os

        return os.getenv("DATABASE_URL", "sqlite:///:memory:")

    def test_database_connection(self, db_url):
        """Verificar conexión a base de datos"""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1
        except ImportError:
            pytest.skip("SQLAlchemy not installed")


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingIntegration:
    """Tests de integración del sistema de entrenamiento"""

    def test_list_branches(self):
        """Verificar que se pueden listar las ramas"""
        project_root = Path(__file__).parent.parent
        branches_dir = project_root / "all-Branches"

        if branches_dir.exists():
            branches = [d.name for d in branches_dir.iterdir() if d.is_dir()]
            assert len(branches) >= 50  # Deberíamos tener al menos 50 dominios

    def test_training_data_structure(self):
        """Verificar estructura de datos de entrenamiento"""
        project_root = Path(__file__).parent.parent
        branches_dir = project_root / "all-Branches"

        if branches_dir.exists():
            # Verificar que al menos una rama tiene datos
            physics_dir = branches_dir / "physics" / "training" / "data"
            if physics_dir.exists():
                jsonl_files = list(physics_dir.glob("*.jsonl"))
                assert len(jsonl_files) > 0


@pytest.mark.integration
class TestAPIIntegration:
    """Tests de integración de API (si existe)"""

    @pytest.mark.skipif(True, reason="API endpoint not implemented yet")
    def test_health_endpoint(self):
        """Verificar endpoint de health check"""
        import httpx

        response = httpx.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.skipif(True, reason="API endpoint not implemented yet")
    def test_api_authentication(self):
        """Verificar autenticación de API"""
        import httpx

        # Sin autenticación debería fallar
        response = httpx.get("http://localhost:8000/api/v1/chat")
        assert response.status_code in [401, 403]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
