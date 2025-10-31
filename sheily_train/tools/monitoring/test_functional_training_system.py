#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Sheily AI Functional Training System
================================================================

This module provides comprehensive unit tests for all functional training components:
- Adapter system testing
- Training engine validation
- GGUF integration verification
- Data preparation testing
- Training orchestration validation
- Safety mechanism testing
- Integration testing between components

Tests are designed for production readiness with:
- 100% functional programming compliance
- Immutable data structure validation
- Pure function testing
- Error handling verification
- Performance benchmarking
"""

import json

# Add project root to path for imports
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from result import Err, Ok


class TestAdapterSystem(unittest.TestCase):
    """Test adapter system functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.adapters import (
            compose_adapter_configs,
            create_adapter_config,
            create_adapter_state,
            validate_adapter_config,
        )

        self.create_adapter_config = create_adapter_config
        self.validate_adapter_config = validate_adapter_config
        self.compose_adapter_configs = compose_adapter_configs
        self.create_adapter_state = create_adapter_state

    def test_adapter_config_creation(self):
        """Test adapter configuration creation"""
        config = self.create_adapter_config(
            adapter_type="LORA",
            base_model="microsoft/Phi-3-mini-4k-instruct",
            target_modules=["qkv_proj", "o_proj"],
            rank=8,
            alpha=16,
            dropout=0.05,
        )

        self.assertIsNotNone(config.adapter_id)
        self.assertEqual(config.adapter_type, "LORA")
        self.assertEqual(config.rank, 8)
        self.assertEqual(config.alpha, 16)
        self.assertEqual(len(config.target_modules), 2)

    def test_adapter_config_validation(self):
        """Test adapter configuration validation"""
        # Valid configuration
        valid_config = self.create_adapter_config(
            adapter_type="LORA",
            base_model="test_model",
            target_modules=["test_module"],
            rank=8,
            alpha=16,
        )

        result = self.validate_adapter_config(valid_config)
        self.assertTrue(result.is_ok())

        # Invalid configuration - negative rank
        invalid_config = self.create_adapter_config(
            adapter_type="LORA",
            base_model="test_model",
            target_modules=["test_module"],
            rank=-1,
            alpha=16,
        )

        result = self.validate_adapter_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_adapter_composition(self):
        """Test adapter composition functionality"""
        base_config = self.create_adapter_config(
            adapter_type="LORA",
            base_model="test_model",
            target_modules=["qkv_proj"],
            rank=8,
            alpha=16,
        )

        overlay_config = self.create_adapter_config(
            adapter_type="LORA",
            base_model="test_model",
            target_modules=["o_proj"],
            rank=16,
            alpha=32,
        )

        composed = self.compose_adapter_configs(base_config, overlay_config)

        self.assertEqual(len(composed.target_modules), 2)  # Should merge modules
        self.assertEqual(composed.rank, 16)  # Should take higher rank
        self.assertEqual(composed.alpha, 32)  # Should take higher alpha

    def test_adapter_state_creation(self):
        """Test adapter state creation"""
        state = self.create_adapter_state(
            adapter_id="test_adapter",
            branch_name="test_branch",
            language="EN",
            training_iterations=100,
            knowledge_base_size=1000,
        )

        self.assertEqual(state.adapter_id, "test_adapter")
        self.assertEqual(state.branch_name, "test_branch")
        self.assertEqual(state.language, "EN")
        self.assertEqual(state.training_iterations, 100)
        self.assertEqual(state.knowledge_base_size, 1000)


class TestTrainingEngine(unittest.TestCase):
    """Test training engine functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.training import (
            calculate_training_progress,
            create_training_config,
            create_training_state,
            validate_training_config,
        )

        self.create_training_config = create_training_config
        self.validate_training_config = validate_training_config
        self.create_training_state = create_training_state
        self.calculate_training_progress = calculate_training_progress

    def test_training_config_creation(self):
        """Test training configuration creation"""
        config = self.create_training_config(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            adapter_config={"rank": 8, "alpha": 16},
            training_corpus={"source": "test"},
            hyperparameters={"learning_rate": 1e-4, "epochs": 3},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        self.assertIsNotNone(config.training_id)
        self.assertEqual(config.model_name, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(config.language, "EN")
        self.assertEqual(config.hyperparameters["learning_rate"], 1e-4)

    def test_training_config_validation(self):
        """Test training configuration validation"""
        # Valid configuration
        valid_config = self.create_training_config(
            model_name="test_model",
            adapter_config={"rank": 8},
            training_corpus={"data": []},
            hyperparameters={"learning_rate": 1e-4, "batch_size": 16},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        result = self.validate_training_config(valid_config)
        self.assertTrue(result.is_ok())

        # Invalid configuration - negative learning rate
        invalid_config = self.create_training_config(
            model_name="test_model",
            adapter_config={"rank": 8},
            training_corpus={"data": []},
            hyperparameters={"learning_rate": -1, "batch_size": 16},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        result = self.validate_training_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_training_state_creation(self):
        """Test training state creation"""
        state = self.create_training_state(
            training_id="test_training",
            total_epochs=10,
            total_iterations=1000,
            checkpoint_path="test_checkpoint.json",
        )

        self.assertEqual(state.training_id, "test_training")
        self.assertEqual(state.total_epochs, 10)
        self.assertEqual(state.total_iterations, 1000)
        self.assertEqual(state.current_epoch, 0)
        self.assertEqual(state.current_iteration, 0)

    def test_training_progress_calculation(self):
        """Test training progress calculation"""
        state = self.create_training_state(
            training_id="test_training",
            total_epochs=10,
            total_iterations=100,
            checkpoint_path="test_checkpoint.json",
        )

        # Initial progress should be 0
        progress = self.calculate_training_progress(state)
        self.assertEqual(progress, 0.0)

        # Update state to half way
        from sheily_core.llm_engine.training import update_training_state

        updated_state = update_training_state(state, current_epoch=5, current_iteration=50)

        progress = self.calculate_training_progress(updated_state)
        self.assertEqual(progress, 0.5)  # Exactly halfway


class TestGGUFIntegration(unittest.TestCase):
    """Test GGUF integration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.gguf_integration import (
            create_gguf_model_config,
            create_gguf_model_state,
            load_gguf_model_functional,
            validate_gguf_config,
        )

        self.create_gguf_model_config = create_gguf_model_config
        self.validate_gguf_config = validate_gguf_config
        self.create_gguf_model_state = create_gguf_model_state
        self.load_gguf_model_functional = load_gguf_model_functional

    def test_gguf_model_config_creation(self):
        """Test GGUF model configuration creation"""
        config = self.create_gguf_model_config(
            model_path="models/gguf/llama-3.2.gguf",
            quantization_type="Q4_K_M",
            context_length=4096,
            metadata={"test_model": True},
        )

        self.assertIsNotNone(config.model_id)
        self.assertEqual(config.quantization_type, "Q4_K_M")
        self.assertEqual(config.context_length, 4096)
        self.assertTrue("test_model" in config.metadata)

    def test_gguf_model_state_creation(self):
        """Test GGUF model state creation"""
        state = self.create_gguf_model_state(
            model_id="test_gguf_model",
            loaded=True,
            memory_usage=1024 * 1024 * 1024,  # 1GB
            current_branch="general",
            language_mode="EN",
        )

        self.assertEqual(state.model_id, "test_gguf_model")
        self.assertTrue(state.loaded)
        self.assertEqual(state.current_branch, "general")
        self.assertEqual(state.language_mode, "EN")

    def test_gguf_model_loading_functional(self):
        """Test functional GGUF model loading"""
        # Test with non-existent file (should fail gracefully)
        result = self.load_gguf_model_functional(
            model_path="non_existent_model.gguf", context_length=4096
        )

        self.assertTrue(result.is_err())
        self.assertIn("not found", str(result.unwrap_err()).lower())

    def test_gguf_memory_requirements(self):
        """Test GGUF memory requirements calculation"""
        from sheily_core.llm_engine.gguf_integration import calculate_gguf_memory_requirements

        config = self.create_gguf_model_config(
            model_path="test.gguf", model_size=1024 * 1024 * 1024  # 1GB
        )

        memory_reqs = calculate_gguf_memory_requirements(config)

        self.assertIn("model_memory_mb", memory_reqs)
        self.assertIn("context_memory_mb", memory_reqs)
        self.assertIn("kv_cache_mb", memory_reqs)
        self.assertIn("total_memory_mb", memory_reqs)
        self.assertGreater(memory_reqs["total_memory_mb"], 0)


class TestDataPreparation(unittest.TestCase):
    """Test data preparation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.data_preparation import (
            calculate_dataset_statistics,
            create_data_preparation_config,
            create_prepared_dataset,
            validate_language_separation,
        )

        self.create_data_preparation_config = create_data_preparation_config
        self.create_prepared_dataset = create_prepared_dataset
        self.calculate_dataset_statistics = calculate_dataset_statistics
        self.validate_language_separation = validate_language_separation

    def test_data_preparation_config_creation(self):
        """Test data preparation configuration creation"""
        config = self.create_data_preparation_config(
            language="EN",
            corpus_type="domain_specific",
            chunk_size=512,
            chunk_overlap=64,
            max_documents=1000,
        )

        self.assertIsNotNone(config.config_id)
        self.assertEqual(config.language, "EN")
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 64)

    def test_prepared_dataset_creation(self):
        """Test prepared dataset creation"""
        mock_documents = [
            {"content": "Test document 1", "language": "EN", "domain": "test"},
            {"content": "Test document 2", "language": "EN", "domain": "test"},
        ]

        dataset = self.create_prepared_dataset(
            language="EN", branch_name="test_branch", documents=mock_documents
        )

        self.assertEqual(dataset.language, "EN")
        self.assertEqual(dataset.branch_name, "test_branch")
        self.assertEqual(len(dataset.documents), 2)
        self.assertGreater(dataset.vocabulary_size, 0)

    def test_dataset_statistics_calculation(self):
        """Test dataset statistics calculation"""
        documents = [
            {"content": "This is a test document with some content"},
            {"content": "Another test document with different content"},
        ]

        stats = self.calculate_dataset_statistics(documents)

        self.assertEqual(stats["total_documents"], 2)
        self.assertGreater(stats["total_characters"], 0)
        self.assertGreater(stats["total_words"], 0)
        self.assertGreater(stats["vocabulary_size"], 0)

    def test_language_separation_validation(self):
        """Test language separation validation"""
        en_documents = [
            {"content": "English document", "language": "EN"},
            {"content": "Another English document", "language": "EN"},
        ]

        es_documents = [
            {"content": "Documento en español", "language": "ES"},
            {"content": "Otro documento en español", "language": "ES"},
        ]

        # Valid separation
        result = self.validate_language_separation(en_documents, es_documents)
        self.assertTrue(result.is_ok())

        # Invalid separation - English corpus contains Spanish
        contaminated_en = [
            {"content": "English document", "language": "EN"},
            {"content": "Documento en español", "language": "ES"},  # Wrong language
        ]

        result = self.validate_language_separation(contaminated_en, es_documents)
        self.assertTrue(result.is_err())


class TestTrainingOrchestrator(unittest.TestCase):
    """Test training orchestrator functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.training_orchestrator import (
            calculate_session_progress,
            create_training_orchestration_config,
            create_training_session,
            validate_orchestration_config,
        )

        self.create_training_orchestration_config = create_training_orchestration_config
        self.validate_orchestration_config = validate_orchestration_config
        self.create_training_session = create_training_session
        self.calculate_session_progress = calculate_session_progress

    def test_orchestration_config_creation(self):
        """Test orchestration configuration creation"""
        config = self.create_training_orchestration_config(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            branches_to_train=["general", "anthropology"],
            languages=["EN", "ES"],
            max_iterations=5,
            metadata={"test_orchestration": True},
        )

        self.assertIsNotNone(config.orchestration_id)
        self.assertEqual(config.model_name, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(len(config.branches_to_train), 2)
        self.assertEqual(len(config.languages), 2)

    def test_orchestration_config_validation(self):
        """Test orchestration configuration validation"""
        # Valid configuration
        valid_config = self.create_training_orchestration_config(
            model_name="test_model",
            branches_to_train=["general"],
            languages=["EN"],
            max_iterations=5,
        )

        result = self.validate_orchestration_config(valid_config)
        self.assertTrue(result.is_ok())

        # Invalid configuration - no branches
        invalid_config = self.create_training_orchestration_config(
            model_name="test_model",
            branches_to_train=[],  # Empty branches
            languages=["EN"],
            max_iterations=5,
        )

        result = self.validate_orchestration_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_training_session_creation(self):
        """Test training session creation"""
        config = self.create_training_orchestration_config(
            model_name="test_model",
            branches_to_train=["general"],
            languages=["EN"],
            max_iterations=5,
        )

        session = self.create_training_session(config)

        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.orchestration_config, config)
        self.assertEqual(session.current_iteration, 0)
        self.assertEqual(len(session.trained_branches), 0)

    def test_session_progress_calculation(self):
        """Test session progress calculation"""
        config = self.create_training_orchestration_config(
            model_name="test_model",
            branches_to_train=["general", "anthropology"],
            languages=["EN", "ES"],
            max_iterations=10,
        )

        session = self.create_training_session(config)

        # Initial progress should be 0
        progress = self.calculate_session_progress(session)
        self.assertEqual(progress, 0.0)

        # Update session to partial completion
        from sheily_core.llm_engine.training_orchestrator import update_training_session

        updated_session = update_training_session(
            session, current_iteration=5, trained_branches=["general"], completed_languages=["EN"]
        )

        progress = self.calculate_session_progress(updated_session)
        # Should be partial progress: (0.5 + 0.5 + 0.5) / 3 = 0.5
        self.assertGreater(progress, 0.0)
        self.assertLess(progress, 1.0)


class TestTrainingRouter(unittest.TestCase):
    """Test training router functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.training_router import (
            calculate_route_priority,
            create_training_route,
            create_training_route_request,
            select_best_training_route,
        )

        self.create_training_route = create_training_route
        self.create_training_route_request = create_training_route_request
        self.calculate_route_priority = calculate_route_priority
        self.select_best_training_route = select_best_training_route

    def test_training_route_creation(self):
        """Test training route creation"""
        route = self.create_training_route(
            source_branch="general", target_branch="anthropology", language="EN", priority=5
        )

        self.assertIsNotNone(route.route_id)
        self.assertEqual(route.source_branch, "general")
        self.assertEqual(route.target_branch, "anthropology")
        self.assertEqual(route.language, "EN")
        self.assertEqual(route.priority, 5)

    def test_training_route_request_creation(self):
        """Test training route request creation"""
        request = self.create_training_route_request(
            query="anthropology question", language="EN", branch_name="general"
        )

        self.assertIsNotNone(request.request_id)
        self.assertEqual(request.query, "anthropology question")
        self.assertEqual(request.language, "EN")
        self.assertEqual(request.branch_name, "general")

    def test_route_priority_calculation(self):
        """Test route priority calculation"""
        route = self.create_training_route(
            source_branch="general", target_branch="anthropology", language="EN", priority=5
        )

        request = self.create_training_route_request(
            query="anthropology question", language="EN", branch_name="general"
        )

        priority = self.calculate_route_priority(route, request)

        # Should have base priority + language bonus + branch bonus
        self.assertGreater(priority, 5)  # Base priority

    def test_best_route_selection(self):
        """Test best route selection"""
        routes = [
            self.create_training_route("general", "anthropology", "EN", 5),
            self.create_training_route("general", "philosophy", "EN", 3),
            self.create_training_route("general", "programming", "EN", 7),
        ]

        request = self.create_training_route_request(
            query="programming question", language="EN", branch_name="general"
        )

        best_route = self.select_best_training_route(request, routes)

        self.assertIsNotNone(best_route)
        # Should select the route with highest priority for the given context
        self.assertEqual(best_route.target_branch, "programming")


class TestTrainingDependencies(unittest.TestCase):
    """Test training dependency management"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.training_deps import (
            check_package_security,
            create_production_dependency_context,
            create_training_dependency_context,
            resolve_package_dependency,
        )

        self.create_training_dependency_context = create_training_dependency_context
        self.create_production_dependency_context = create_production_dependency_context
        self.check_package_security = check_package_security
        self.resolve_package_dependency = resolve_package_dependency

    def test_training_dependency_context_creation(self):
        """Test training dependency context creation"""
        context = self.create_training_dependency_context()

        self.assertTrue(context.training_mode)
        self.assertIn("torch", context.allowed_packages)
        self.assertIn("transformers", context.allowed_packages)
        self.assertNotIn("gradio", context.blocked_packages)

    def test_production_dependency_context_creation(self):
        """Test production dependency context creation"""
        context = self.create_production_dependency_context()

        self.assertFalse(context.training_mode)
        self.assertNotIn("torch", context.allowed_packages)
        self.assertIn("torch", context.blocked_packages)

    def test_package_security_check(self):
        """Test package security checking"""
        training_context = self.create_training_dependency_context()
        production_context = self.create_production_dependency_context()

        # Torch should be allowed in training mode
        self.assertTrue(self.check_package_security("torch", training_context))

        # Torch should be blocked in production mode
        self.assertFalse(self.check_package_security("torch", production_context))

    def test_dependency_resolution(self):
        """Test dependency resolution"""
        context = self.create_training_dependency_context()

        # Test direct resolution
        result = self.resolve_package_dependency("torch", context)
        self.assertTrue(result.is_ok())
        self.assertEqual(result.unwrap(), "torch")

        # Test blocked package
        production_context = self.create_production_dependency_context()
        result = self.resolve_package_dependency("torch", production_context)
        self.assertTrue(result.is_err())


class TestIntegrationBetweenComponents(unittest.TestCase):
    """Test integration between training system components"""

    def test_adapter_training_integration(self):
        """Test integration between adapters and training engine"""
        from sheily_core.adapters import create_adapter_config
        from sheily_core.llm_engine.training import create_training_config

        # Create adapter config
        adapter_config = create_adapter_config(
            adapter_type="LORA", base_model="test_model", target_modules=["qkv_proj"], rank=8
        )

        # Create training config with adapter
        training_config = create_training_config(
            model_name="test_model",
            adapter_config={"rank": adapter_config.rank, "alpha": adapter_config.alpha},
            training_corpus={"data": []},
            hyperparameters={"learning_rate": 1e-4},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        self.assertEqual(training_config.adapter_config["rank"], 8)
        self.assertEqual(training_config.adapter_config["alpha"], 16)  # Default alpha

    def test_gguf_adapter_integration(self):
        """Test integration between GGUF models and adapters"""
        from sheily_core.adapters import create_adapter_config
        from sheily_core.llm_engine.gguf_integration import (
            create_gguf_adapter_composition,
            create_gguf_model_state,
        )

        # Create GGUF model state
        model_state = create_gguf_model_state(
            model_id="test_gguf", loaded=True, adapter_attached=False
        )

        # Create adapter config
        adapter_config = {
            "general": {
                "adapter_id": "general_adapter",
                "type": "LORA",
                "description": "General purpose adapter",
            }
        }

        # Create composition function
        composition_func = create_gguf_adapter_composition()

        # Test composition
        result = composition_func(model_state, adapter_config, "general")

        self.assertTrue(result.is_ok())
        composed_state = result.unwrap()
        self.assertTrue(composed_state.adapter_attached)

    def test_full_training_pipeline_integration(self):
        """Test full training pipeline integration"""
        from sheily_core.llm_engine.data_preparation import create_data_preparation_config
        from sheily_core.llm_engine.training import create_training_config
        from sheily_core.llm_engine.training_orchestrator import (
            create_training_orchestration_config,
        )

        # Create data preparation config
        data_config = create_data_preparation_config(
            language="EN", corpus_type="domain_specific", max_documents=100
        )

        # Create training config
        training_config = create_training_config(
            model_name="test_model",
            adapter_config={"rank": 8},
            training_corpus={"prepared": True},
            hyperparameters={"epochs": 3},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        # Create orchestration config
        orch_config = create_training_orchestration_config(
            model_name="test_model",
            branches_to_train=["test_branch"],
            languages=["EN"],
            max_iterations=3,
            data_preparation_config={"max_documents": data_config.max_documents},
            training_config={"epochs": training_config.hyperparameters["epochs"]},
        )

        # Verify integration
        self.assertEqual(orch_config.languages, ["EN"])
        self.assertEqual(orch_config.branches_to_train, ["test_branch"])
        self.assertEqual(orch_config.data_preparation_config["max_documents"], 100)


class TestErrorHandling(unittest.TestCase):
    """Test error handling across all components"""

    def test_adapter_error_handling(self):
        """Test error handling in adapter system"""
        from sheily_core.adapters import create_adapter_config, validate_adapter_config

        # Test with invalid adapter config
        invalid_config = create_adapter_config(
            adapter_type="LORA",
            base_model="",  # Invalid empty model
            target_modules=[],  # Invalid empty modules
            rank=-1,  # Invalid negative rank
        )

        result = validate_adapter_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_training_error_handling(self):
        """Test error handling in training engine"""
        from sheily_core.llm_engine.training import create_training_config, validate_training_config

        # Test with invalid training config
        invalid_config = create_training_config(
            model_name="",  # Invalid empty model
            adapter_config={},  # Invalid empty adapter config
            training_corpus={},  # Invalid empty training data
            hyperparameters={"learning_rate": -1},  # Invalid negative learning rate
            output_dir="",
            language="EN",
            branch_name="test",
        )

        result = validate_training_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_gguf_error_handling(self):
        """Test error handling in GGUF integration"""
        from sheily_core.llm_engine.gguf_integration import validate_gguf_safety

        # Test with non-existent file
        safety_config = {"max_file_size_mb": 1000}
        result = validate_gguf_safety("non_existent_file.gguf", safety_config)

        self.assertTrue(result.is_err())


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of training system"""

    def test_adapter_creation_performance(self):
        """Test adapter creation performance"""
        from sheily_core.adapters import create_adapter_config

        start_time = time.time()

        # Create multiple adapters
        for i in range(100):
            config = create_adapter_config(
                adapter_type="LORA",
                base_model=f"test_model_{i}",
                target_modules=[f"module_{j}" for j in range(10)],
                rank=8,
                alpha=16,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast (< 1 second for 100 adapters)
        self.assertLess(duration, 1.0)
        print(f"Adapter creation performance: {duration:.3f}s for 100 adapters")

    def test_training_config_performance(self):
        """Test training configuration performance"""
        from sheily_core.llm_engine.training import create_training_config

        start_time = time.time()

        # Create multiple training configs
        for i in range(50):
            config = create_training_config(
                model_name=f"test_model_{i}",
                adapter_config={"rank": 8},
                training_corpus={"data": list(range(100))},
                hyperparameters={"learning_rate": 1e-4, "epochs": 3},
                output_dir=f"test_output_{i}",
                language="EN",
                branch_name=f"test_branch_{i}",
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast (< 1 second for 50 configs)
        self.assertLess(duration, 1.0)
        print(f"Training config performance: {duration:.3f}s for 50 configs")

    def test_immutability_enforcement(self):
        """Test that all data structures are properly immutable"""
        from sheily_core.adapters import create_adapter_config
        from sheily_core.llm_engine.gguf_integration import create_gguf_model_config
        from sheily_core.llm_engine.training import create_training_config

        # Test adapter config immutability
        config = create_adapter_config(
            adapter_type="LORA", base_model="test_model", target_modules=["test_module"], rank=8
        )

        # Should not be able to modify frozen dataclass
        with self.assertRaises(AttributeError):
            config.rank = 16

        # Test training config immutability
        training_config = create_training_config(
            model_name="test_model",
            adapter_config={"rank": 8},
            training_corpus={"data": []},
            hyperparameters={"learning_rate": 1e-4},
            output_dir="test_output",
            language="EN",
            branch_name="test_branch",
        )

        with self.assertRaises(AttributeError):
            training_config.model_name = "modified_model"

        # Test GGUF config immutability
        gguf_config = create_gguf_model_config(model_path="test.gguf", quantization_type="Q4_K_M")

        with self.assertRaises(AttributeError):
            gguf_config.quantization_type = "Q5_K_M"


def run_comprehensive_training_tests():
    """Run all comprehensive training system tests"""
    # Create test suite
    test_classes = [
        TestAdapterSystem,
        TestTrainingEngine,
        TestGGUFIntegration,
        TestDataPreparation,
        TestTrainingOrchestrator,
        TestTrainingRouter,
        TestTrainingDependencies,
        TestIntegrationBetweenComponents,
        TestErrorHandling,
        TestPerformanceCharacteristics,
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors

    print(f"\n{'='*60}")
    print("COMPREHENSIVE TRAINING SYSTEM TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {passed/total_tests*100:.1f}%")

    if failures > 0:
        print(f"\nFAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if errors > 0:
        print(f"\nERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_training_tests()
    print(f"\nOverall Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1)
