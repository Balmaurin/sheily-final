#!/usr/bin/env python3
"""Pytest configuration and fixtures tailored for Anthropology AI tests."""

import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANTHROPOLOGY_ROOT = PROJECT_ROOT / "all-Branches" / "anthropology"
SRC_PATH = ANTHROPOLOGY_ROOT / "src"

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_configure(config):
    """Register custom markers used across the project."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: Tests requiring model files")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return absolute path to repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def anthropology_root() -> Path:
    """Return root path of the anthropology branch."""
    return ANTHROPOLOGY_ROOT


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_env_vars(monkeypatch):
    """Set baseline environment variables for tests."""
    test_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


@pytest.fixture
def sample_text():
    """Text snippet reused in multiple tests."""
    return "Este es un texto de prueba para el sistema Anthropology AI."


@pytest.fixture
def sample_training_data(temp_dir) -> Path:
    """Create a lightweight training dataset in JSONL format."""
    import json

    data_file = temp_dir / "training_data.jsonl"
    samples = [
        {"instruction": "¿Qué es antropología?", "response": "La antropología estudia al ser humano."},
        {"instruction": "Define cultura", "response": "Conjunto de conocimientos, creencias y costumbres."},
    ]

    with data_file.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return data_file


@pytest.fixture
def mock_config():
    """Simple mock configuration object."""
    from dataclasses import dataclass, field

    @dataclass
    class MockConfig:
        system_name: str = "Anthropology AI Test"
        host: str = "127.0.0.1"
        port: int = 8000
        cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])
        debug: bool = True

    return MockConfig()
