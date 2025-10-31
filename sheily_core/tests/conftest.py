#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Test Configuration for Sheily AI System
===============================================

Comprehensive pytest configuration with:
- Custom fixtures for all refactored components
- Test data factories
- Performance benchmarking setup
- Security testing utilities
- Temporary file and directory management
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app.chat_engine import ChatEngine
from app.rag_engine import RAGEngine

from sheily_core.config import SheilyConfig
from sheily_core.logger import get_logger
from sheily_core.safety import SecurityMonitor

# Global test configuration
TEST_CONFIG = {
    "performance_thresholds": {
        "initialization_time": 1.0,  # seconds
        "query_processing_time": 0.5,  # seconds
        "memory_increase_limit": 50,  # MB
    },
    "security_settings": {
        "enable_rate_limiting": True,
        "enable_input_validation": True,
        "max_requests_per_user": 100,
    },
    "test_data": {
        "sample_documents": [
            "Python is a programming language for data science and AI.",
            "Machine learning algorithms require quality data.",
            "Deep learning uses neural networks for complex tasks.",
            "Natural language processing helps understand human text.",
        ],
        "sample_queries": [
            "What is Python?",
            "How does machine learning work?",
            "Tell me about deep learning",
            "What is natural language processing?",
        ],
    },
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG


@pytest.fixture
def temp_dir():
    """Provide temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_corpus(temp_dir):
    """Create test corpus with sample documents"""
    corpus_path = temp_dir / "test_corpus"
    corpus_path.mkdir()

    # Create branch directories
    branches = ["programación", "inteligencia artificial", "medicina", "general"]
    for branch in branches:
        branch_path = corpus_path / branch
        branch_path.mkdir()

        # Create sample documents for each branch
        for i in range(3):
            doc_file = branch_path / f"doc_{i}.txt"
            doc_file.write_text(f"Sample document {i} for {branch} branch with relevant content.")

    return corpus_path


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing"""
    config = SheilyConfig()
    config.host = "test.localhost"
    config.port = 8888
    config.debug = True
    config.max_search_results = 3
    config.similarity_threshold = 0.1  # Lower for testing

    return config


@pytest.fixture
def security_monitor():
    """Provide security monitor for testing"""
    return SecurityMonitor()


@pytest.fixture
def chat_engine():
    """Provide chat engine for testing"""
    return ChatEngine()


@pytest.fixture
def rag_engine(test_corpus):
    """Provide RAG engine with test corpus"""
    engine = RAGEngine(str(test_corpus))
    engine.load_documents()
    return engine


@pytest.fixture
def test_document_factory():
    """Factory for creating test documents"""

    def create_document(title="Test Document", content="Test content", branch="general", **metadata):
        return {"title": title, "content": content, "branch": branch, "metadata": metadata}

    return create_document


@pytest.fixture
def test_query_factory():
    """Factory for creating test queries"""

    def create_query(text="Test query", expected_branch="general", malicious=False):
        return {"text": text, "expected_branch": expected_branch, "malicious": malicious}

    return create_query


@pytest.fixture
def benchmark_data():
    """Provide benchmark test data"""
    return {
        "documents": [f"Benchmark document {i}" for i in range(100)],
        "queries": [f"Benchmark query {i}" for i in range(50)],
        "expected_performance": {
            "max_load_time": 2.0,
            "max_search_time": 0.1,
            "max_memory_increase": 25,  # MB
        },
    }


@pytest.fixture
def clean_logs_dir():
    """Ensure clean logs directory for tests"""
    logs_dir = Path("logs")
    if logs_dir.exists():
        # Clear existing logs
        for log_file in logs_dir.glob("*.log"):
            log_file.unlink()
        for log_file in logs_dir.glob("*.jsonl"):
            log_file.unlink()

    logs_dir.mkdir(exist_ok=True)
    yield logs_dir

    # Cleanup after test
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            log_file.unlink()
        for log_file in logs_dir.glob("*.jsonl"):
            log_file.unlink()


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests"""

    class PerformanceMonitor:
        def __init__(self):
            self.start_times = {}
            self.measurements = {}

        def start_timer(self, name: str):
            """Start timing an operation"""
            self.start_times[name] = time.time()

        def end_timer(self, name: str) -> float:
            """End timing and return duration"""
            if name not in self.start_times:
                raise ValueError(f"Timer '{name}' was not started")

            duration = time.time() - self.start_times[name]
            self.measurements[name] = duration
            del self.start_times[name]

            return duration

        def get_measurement(self, name: str) -> float:
            """Get measurement by name"""
            return self.measurements.get(name, 0.0)

        def reset(self):
            """Reset all measurements"""
            self.start_times.clear()
            self.measurements.clear()

    return PerformanceMonitor()


@pytest.fixture
def test_user_factory():
    """Factory for creating test users"""

    def create_user(user_id="test_user", is_malicious=False, request_pattern="normal"):
        return {
            "user_id": user_id,
            "is_malicious": is_malicious,
            "request_pattern": request_pattern,
            "expected_behavior": "blocked" if is_malicious else "allowed",
        }

    return create_user


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    logger = get_logger("test_setup")
    logger.info("Setting up test environment")

    yield

    logger.info("Cleaning up test environment")


@pytest.fixture
def sample_branches_config(temp_dir):
    """Create sample branches configuration for testing"""
    branches_config = {
        "domains": [
            {
                "name": "programación",
                "keywords": ["python", "código", "programar", "desarrollo"],
                "description": "Programming and development",
            },
            {
                "name": "medicina",
                "keywords": ["medicina", "salud", "médico", "tratamiento"],
                "description": "Medicine and health",
            },
            {
                "name": "inteligencia artificial",
                "keywords": ["IA", "machine learning", "deep learning", "neural"],
                "description": "Artificial intelligence",
            },
        ]
    }

    config_file = temp_dir / "test_branches.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(branches_config, f, ensure_ascii=False, indent=2)

    return config_file


# Pytest hooks for enhanced test reporting
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test path and name
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath) or "security" in item.name:
            item.add_marker(pytest.mark.security)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# Global test utilities
class TestUtils:
    """Utility class for test helpers"""

    @staticmethod
    def create_test_corpus(base_path: Path, branches: List[str] = None, documents_per_branch: int = 3):
        """Create a test corpus with specified structure"""
        if branches is None:
            branches = ["programación", "medicina", "inteligencia artificial", "general"]

        corpus_path = base_path / "test_corpus"
        corpus_path.mkdir(exist_ok=True)

        for branch in branches:
            branch_path = corpus_path / branch
            branch_path.mkdir(exist_ok=True)

            for i in range(documents_per_branch):
                doc_file = branch_path / f"test_doc_{i}.txt"
                doc_file.write_text(f"Test document {i} for {branch} branch with sample content.")

        return corpus_path

    @staticmethod
    def measure_performance(func, *args, **kwargs):
        """Measure performance of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        return result, end_time - start_time

    @staticmethod
    def assert_performance_threshold(duration: float, threshold: float, operation: str):
        """Assert that operation completed within threshold"""
        assert duration <= threshold, f"{operation} took {duration:.3f}s, exceeds threshold {threshold:.3f}s"


# Make TestUtils available globally
test_utils = TestUtils()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
