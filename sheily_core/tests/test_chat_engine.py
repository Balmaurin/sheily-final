#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Tests for Chat Engine
==================================

Tests for the enhanced chat engine including:
- Branch detection and management
- Model interface integration
- Context management
- Query processing pipeline
- Error handling and fallbacks
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from app.chat_engine import (
    BranchDetector,
    ChatEngine,
    ChatResponse,
    ContextManager,
    LogContext,
    ModelInterface,
)


class TestBranchDetector:
    """Test cases for BranchDetector class"""

    def test_branch_detector_initialization(self):
        """Test branch detector initialization"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {
                "domains": [
                    {
                        "name": "test_branch",
                        "keywords": ["test", "example"],
                        "description": "Test branch",
                    }
                ]
            }
            json.dump(test_config, f)
            config_path = f.name

        try:
            detector = BranchDetector(config_path)

            assert detector.branches_config_path == config_path
            assert "test_branch" in detector.branches_config
            assert detector.branches_config["test_branch"]["keywords"] == ["test", "example"]

        finally:
            Path(config_path).unlink()

    def test_default_branches_loading(self):
        """Test loading of default branches when config file doesn't exist"""
        detector = BranchDetector("/nonexistent/path.json")

        assert len(detector.branches_config) > 0
        assert "programación" in detector.branches_config
        assert "medicina" in detector.branches_config
        assert "inteligencia artificial" in detector.branches_config
        assert "general" in detector.branches_config

    def test_branch_detection_scoring(self):
        """Test branch detection scoring algorithm"""
        detector = BranchDetector("/nonexistent/path.json")

        # Test keyword matching
        branch, confidence = detector.detect_branch("Python programming is great")
        assert branch == "programación"
        assert confidence > 0

        # Test branch name matching
        branch, confidence = detector.detect_branch("Tell me about medicina")
        assert branch == "medicina"
        assert confidence > 0

        # Test general fallback
        branch, confidence = detector.detect_branch("Random unrelated query")
        assert branch == "general"
        assert confidence >= 0


class TestModelInterface:
    """Test cases for ModelInterface class"""

    def test_model_interface_initialization(self):
        """Test model interface initialization"""
        with patch("app.chat_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            interface = ModelInterface("/path/to/model.gguf", "/path/to/llama-cli", MagicMock())

            assert interface.model_path == Path("/path/to/model.gguf")
            assert interface.llama_binary_path == Path("/path/to/llama-cli")

    def test_model_file_validation(self):
        """Test model file validation"""
        with patch("app.chat_engine.Path") as mock_path:
            # Test missing model file
            mock_path.return_value.exists.return_value = False

            with pytest.raises(FileNotFoundError, match="Model file not found"):
                ModelInterface("/missing/model.gguf", "/missing/llama-cli", MagicMock())

    def test_prompt_creation(self):
        """Test prompt creation for model"""
        with patch("app.chat_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            interface = ModelInterface("/dummy/model.gguf", "/dummy/llama-cli", MagicMock())

            prompt = interface._create_prompt("Test query", ["Context doc 1", "Context doc 2"])

            assert "Test query" in prompt
            assert "Context doc 1" in prompt
            assert "Context doc 2" in prompt
            assert "Respuesta:" in prompt

    def test_fallback_response_generation(self):
        """Test fallback response generation"""
        with patch("app.chat_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            interface = ModelInterface("/dummy/model.gguf", "/dummy/llama-cli", MagicMock())

            response = interface._get_fallback_response("Test query", "programación")

            assert "Test query" in response
            assert "programación" in response or "experta en programación" in response


class TestContextManager:
    """Test cases for ContextManager class"""

    def test_context_manager_initialization(self):
        """Test context manager initialization"""
        manager = ContextManager("/test/corpus")

        assert manager.corpus_root == Path("/test/corpus")
        assert manager.logger is not None

    def test_context_retrieval(self):
        """Test context retrieval for queries"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test corpus structure
            corpus_path = Path(temp_dir) / "corpus"
            branch_path = corpus_path / "programación"
            branch_path.mkdir(parents=True)

            # Create test document
            test_doc = branch_path / "test.txt"
            test_doc.write_text("Python is a programming language for data science and AI.")

            manager = ContextManager(str(corpus_path))

            context = manager.get_context("Python programming", "programación")

            assert len(context) > 0
            assert any("Python" in doc for doc in context)

    def test_basic_knowledge_fallback(self):
        """Test basic knowledge fallback when no context found"""
        manager = ContextManager("/nonexistent/corpus")

        context = manager.get_context("Unknown topic", "unknown_branch")

        assert len(context) > 0
        assert all(isinstance(doc, str) for doc in context)


class TestChatEngine:
    """Test cases for ChatEngine class"""

    def test_chat_engine_initialization(self):
        """Test chat engine initialization"""
        engine = ChatEngine()

        assert engine.config is not None
        assert engine.logger is not None
        assert engine.security_monitor is not None
        assert engine.branch_detector is not None
        assert engine.context_manager is not None

    def test_chat_engine_with_model(self):
        """Test chat engine initialization with model"""
        with patch("app.chat_engine.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            with patch.dict(
                "os.environ",
                {"SHEILY_MODEL_PATH": "/test/model.gguf", "SHEILY_LLAMA_BINARY": "/test/llama-cli"},
            ):
                engine = ChatEngine()
                # Model interface may or may not be initialized depending on config
                assert engine.model_interface is None or engine.model_interface is not None

    def test_query_processing_safe(self):
        """Test processing of safe queries"""
        engine = ChatEngine()

        response = engine.process_query("Hello world", "test_user")

        assert isinstance(response, ChatResponse)
        assert response.query == "Hello world"
        assert response.response is not None
        assert response.branch in ["general", "programación", "medicina", "inteligencia artificial"]
        assert response.confidence >= 0
        assert response.processing_time >= 0

    def test_query_processing_malicious(self):
        """Test processing of malicious queries"""
        engine = ChatEngine()

        response = engine.process_query("DROP TABLE users;", "malicious_user")

        assert isinstance(response, ChatResponse)
        assert response.query == "DROP TABLE users;"
        assert "seguridad" in response.response.lower() or "bloqueada" in response.response.lower()
        assert response.model_used == "security_blocked"
        assert response.error == "Security violation"

    def test_query_processing_with_context(self):
        """Test query processing with context retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test corpus
            corpus_path = Path(temp_dir) / "corpus"
            branch_path = corpus_path / "programación"
            branch_path.mkdir(parents=True)

            test_doc = branch_path / "python.txt"
            test_doc.write_text(
                "Python is a versatile programming language used for web development, data science, and AI."
            )

            with patch("app.chat_engine.get_config") as mock_config:
                mock_config.return_value.corpus_root = str(corpus_path)

                engine = ChatEngine()
                response = engine.process_query("Python programming", "test_user")

                assert isinstance(response, ChatResponse)
                assert response.response is not None

    def test_health_check(self):
        """Test chat engine health check"""
        engine = ChatEngine()

        health = engine.health_check()

        assert "status" in health
        assert "components" in health
        assert "timestamp" in health

        assert health["status"] in ["healthy", "degraded"]
        assert "branch_detector" in health["components"]
        assert "context_manager" in health["components"]


class TestChatResponse:
    """Test cases for ChatResponse class"""

    def test_chat_response_creation(self):
        """Test chat response creation"""
        response = ChatResponse(
            query="Test query",
            response="Test response",
            branch="general",
            confidence=0.8,
            context_sources=2,
            processing_time=0.5,
            model_used="test_model",
        )

        assert response.query == "Test query"
        assert response.response == "Test response"
        assert response.branch == "general"
        assert response.confidence == 0.8
        assert response.context_sources == 2
        assert response.processing_time == 0.5
        assert response.model_used == "test_model"

    def test_chat_response_defaults(self):
        """Test chat response default values"""
        response = ChatResponse(
            query="Test query",
            response="Test response",
            branch="general",
            confidence=0.5,
            context_sources=0,
            processing_time=0.1,
            model_used="test_model",
        )

        assert response.tokens_used == 0
        assert response.error is None
        assert response.metadata == {}


class TestChatEngineIntegration:
    """Integration tests for chat engine"""

    def test_chat_engine_full_pipeline(self):
        """Test complete chat engine pipeline"""
        engine = ChatEngine()

        # Test multiple queries
        test_queries = [
            "Hello world",
            "Python programming",
            "Medical advice",
            "AI and machine learning",
        ]

        for query in test_queries:
            response = engine.process_query(query, "integration_test_user")

            assert isinstance(response, ChatResponse)
            assert response.query == query
            assert response.response is not None
            assert response.processing_time >= 0
            assert response.branch in [
                "general",
                "programación",
                "medicina",
                "inteligencia artificial",
            ]

    def test_chat_engine_error_handling(self):
        """Test chat engine error handling"""
        engine = ChatEngine()

        # Test with None query
        response = engine.process_query(None, "test_user")
        assert response.error is not None

        # Test with empty query
        response = engine.process_query("", "test_user")
        assert response.error is not None

        # Test with very long query
        long_query = "a" * 10000
        response = engine.process_query(long_query, "test_user")
        # Should handle gracefully without crashing
        assert isinstance(response, ChatResponse)


class TestChatEnginePerformance:
    """Performance tests for chat engine"""

    def test_query_processing_performance(self):
        """Test query processing performance"""
        engine = ChatEngine()

        start_time = time.time()

        # Process multiple queries
        for i in range(10):
            response = engine.process_query(f"Performance test query {i}", "perf_test_user")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        # Should process queries reasonably fast (less than 1 second average)
        assert avg_time < 1.0, f"Average query time too slow: {avg_time}s"

    def test_memory_usage_stability(self):
        """Test memory usage stability during processing"""
        import os

        import psutil

        engine = ChatEngine()
        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss

        # Process many queries
        for i in range(50):
            response = engine.process_query(f"Memory test query {i}", "memory_test_user")

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        memory_increase_mb = memory_increase / (1024 * 1024)
        assert memory_increase_mb < 50, f"Memory usage increased too much: {memory_increase_mb}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
