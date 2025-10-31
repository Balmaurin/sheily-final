#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for LoRA Training System
=================================================

This module provides comprehensive unit tests for LoRA training functionality:
- LoRA configuration and validation testing
- Multi-branch training pipeline testing
- GGUF integration testing
- API endpoint testing for LoRA operations
- Docker containerization testing
- Performance and scalability testing

Production-ready tests for enterprise deployment.
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


class TestLoRATrainingConfig(unittest.TestCase):
    """Test LoRA training configuration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import create_lora_training_config, validate_lora_training_config

        self.create_lora_training_config = create_lora_training_config
        self.validate_lora_training_config = validate_lora_training_config

    def test_lora_config_creation(self):
        """Test LoRA training configuration creation"""
        config = self.create_lora_training_config(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            branches_to_train=["general", "anthropology", "philosophy"],
            languages=["EN", "ES"],
            lora_rank=8,
            lora_alpha=16,
            num_epochs=3,
            batch_size=16,
            learning_rate=1e-4,
        )

        self.assertIsNotNone(config.config_id)
        self.assertEqual(config.model_name, "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(len(config.branches_to_train), 3)
        self.assertEqual(len(config.languages), 2)

    def test_lora_config_validation(self):
        """Test LoRA training configuration validation"""
        # Valid configuration
        valid_config = self.create_lora_training_config(
            model_name="test_model",
            branches_to_train=["general"],
            languages=["EN"],
            lora_rank=8,
            lora_alpha=16,
            num_epochs=3,
        )

        result = self.validate_lora_training_config(valid_config)
        self.assertTrue(result.is_ok())

        # Invalid configuration - negative rank
        invalid_config = self.create_lora_training_config(
            model_name="test_model",
            branches_to_train=["general"],
            languages=["EN"],
            lora_rank=-1,  # Invalid
            lora_alpha=16,
            num_epochs=3,
        )

        result = self.validate_lora_training_config(invalid_config)
        self.assertTrue(result.is_err())

    def test_lora_config_immutability(self):
        """Test that LoRA configurations are immutable"""
        config = self.create_lora_training_config(model_name="test_model", lora_rank=8)

        # Should not be able to modify frozen dataclass
        with self.assertRaises(AttributeError):
            config.lora_rank = 16

        with self.assertRaises(AttributeError):
            config.model_name = "modified_model"


class TestLoRAAdapterState(unittest.TestCase):
    """Test LoRA adapter state management"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import create_lora_adapter_state, update_lora_adapter_state

        self.create_lora_adapter_state = create_lora_adapter_state
        self.update_lora_adapter_state = update_lora_adapter_state

    def test_adapter_state_creation(self):
        """Test LoRA adapter state creation"""
        state = self.create_lora_adapter_state(
            adapter_id="test_adapter",
            branch_name="anthropology",
            language="EN",
            total_epochs=3,
            adapter_path="test_path",
        )

        self.assertEqual(state.adapter_id, "test_adapter")
        self.assertEqual(state.branch_name, "anthropology")
        self.assertEqual(state.language, "EN")
        self.assertEqual(state.total_epochs, 3)
        self.assertEqual(state.current_epoch, 0)
        self.assertEqual(state.training_status, "initialized")

    def test_adapter_state_updates(self):
        """Test immutable adapter state updates"""
        initial_state = self.create_lora_adapter_state(
            adapter_id="test_adapter",
            branch_name="anthropology",
            language="EN",
            total_epochs=3,
            adapter_path="test_path",
        )

        # Update state
        updated_state = self.update_lora_adapter_state(
            initial_state,
            training_status="training",
            current_epoch=1,
            current_loss=0.5,
            best_loss=0.5,
        )

        # Original state should be unchanged
        self.assertEqual(initial_state.current_epoch, 0)
        self.assertEqual(initial_state.training_status, "initialized")

        # Updated state should have new values
        self.assertEqual(updated_state.current_epoch, 1)
        self.assertEqual(updated_state.training_status, "training")
        self.assertEqual(updated_state.current_loss, 0.5)
        self.assertEqual(updated_state.best_loss, 0.5)


class TestLoRATrainingExecution(unittest.TestCase):
    """Test LoRA training execution"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import (
            create_multibranch_lora_trainer,
            create_production_lora_trainer,
            execute_branch_lora_training,
        )

        self.create_production_lora_trainer = create_production_lora_trainer
        self.create_multibranch_lora_trainer = create_multibranch_lora_trainer
        self.execute_branch_lora_training = execute_branch_lora_training

    def test_branch_lora_training_execution(self):
        """Test individual branch LoRA training"""
        from sheily_core.llm_engine.lora_training import create_lora_adapter_state, create_lora_training_config

        # Create test configuration
        config = create_lora_training_config(
            model_name="test_model",
            branches_to_train=["anthropology"],
            languages=["EN"],
            lora_rank=8,
            num_epochs=2,  # Reduced for testing
        )

        # Create adapter state
        adapter_state = create_lora_adapter_state(
            adapter_id="anthropology_EN_lora_r8",
            branch_name="anthropology",
            language="EN",
            total_epochs=2,
            adapter_path="test_path",
        )

        # Execute training
        result = self.execute_branch_lora_training(config, adapter_state)

        self.assertTrue(result.is_ok())
        final_state = result.unwrap()

        self.assertEqual(final_state.training_status, "completed")
        self.assertEqual(final_state.current_epoch, 2)
        self.assertGreater(final_state.training_time, 0)

    def test_multibranch_lora_training(self):
        """Test multi-branch LoRA training"""
        config = create_lora_training_config(
            model_name="test_model",
            branches_to_train=["anthropology", "philosophy"],
            languages=["EN", "ES"],
            lora_rank=8,
            num_epochs=2,  # Reduced for testing
        )

        # Create trainer
        trainer = self.create_multibranch_lora_trainer()

        # Execute training
        result = trainer(config)

        self.assertTrue(result.is_ok())
        session = result.unwrap()

        self.assertEqual(session.current_status, "completed")
        self.assertEqual(session.overall_progress, 1.0)
        self.assertEqual(len(session.branches_completed), 4)  # 2 branches × 2 languages

    def test_lora_training_error_handling(self):
        """Test error handling in LoRA training"""
        from sheily_core.llm_engine.lora_training import create_lora_training_config

        # Create invalid configuration
        invalid_config = create_lora_training_config(
            model_name="",  # Invalid empty model
            branches_to_train=[],  # Invalid empty branches
            languages=["EN"],
            lora_rank=-1,  # Invalid negative rank
            num_epochs=3,
        )

        trainer = self.create_multibranch_lora_trainer()
        result = trainer(invalid_config)

        self.assertTrue(result.is_err())


class TestGGUFLoRAIntegration(unittest.TestCase):
    """Test GGUF LoRA integration"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import create_gguf_lora_integration

        self.create_gguf_lora_integration = create_gguf_lora_integration

    def test_gguf_lora_integration_creation(self):
        """Test GGUF LoRA integration setup"""
        from sheily_core.llm_engine.lora_training import create_lora_training_config

        config = create_lora_training_config(
            model_name="test_model", branches_to_train=["general"], languages=["EN"], lora_rank=8
        )

        integration_func = self.create_gguf_lora_integration()
        result = integration_func(config)

        self.assertTrue(result.is_ok())
        integration_data = result.unwrap()

        self.assertTrue(integration_data["gguf_model_loaded"])
        self.assertTrue(integration_data["lora_training_ready"])
        self.assertIn("lora_parameters", integration_data)
        self.assertEqual(integration_data["lora_parameters"]["rank"], 8)


class TestLoRAApiIntegration(unittest.TestCase):
    """Test LoRA API integration"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import (
            process_lora_adapter_info_request,
            process_lora_training_request,
        )

        self.process_lora_training_request = process_lora_training_request
        self.process_lora_adapter_info_request = process_lora_adapter_info_request

    def test_lora_training_api_request(self):
        """Test LoRA training API request processing"""
        request_data = {
            "branches": ["anthropology", "philosophy"],
            "languages": ["EN", "ES"],
            "lora_rank": 8,
            "num_epochs": 2,
        }

        result = self.process_lora_training_request(request_data)

        self.assertTrue(result.is_ok())
        response_data = result.unwrap()

        self.assertTrue(response_data["success"])
        self.assertIn("session_id", response_data)
        self.assertEqual(response_data["branches_trained"], 4)  # 2 × 2

    def test_lora_adapter_info_api_request(self):
        """Test LoRA adapter info API request processing"""
        result = self.process_lora_adapter_info_request("anthropology", "EN")

        self.assertTrue(result.is_ok())
        adapter_info = result.unwrap()

        self.assertEqual(adapter_info["branch"], "anthropology")
        self.assertEqual(adapter_info["language"], "EN")
        self.assertIn("adapters", adapter_info)
        self.assertIn("branch_performance", adapter_info)


class TestDockerIntegration(unittest.TestCase):
    """Test Docker containerization"""

    def setUp(self):
        """Set up test fixtures"""
        from sheily_core.llm_engine.lora_training import create_docker_compose_config, create_docker_training_config

        self.create_docker_training_config = create_docker_training_config
        self.create_docker_compose_config = create_docker_compose_config

    def test_docker_training_config_creation(self):
        """Test Docker training configuration creation"""
        config = self.create_docker_training_config()

        self.assertIn("dockerfile", config)
        self.assertIn("base_image", config)
        self.assertIn("python_version", config)
        self.assertIn("volumes", config)
        self.assertIn("environment", config)
        self.assertIn("resource_limits", config)

        self.assertEqual(config["python_version"], "3.9")
        self.assertEqual(config["base_image"], "nvidia/cuda:11.8-devel-ubuntu22.04")

    def test_docker_compose_config_creation(self):
        """Test Docker Compose configuration creation"""
        config = self.create_docker_compose_config()

        self.assertIn("version", config)
        self.assertIn("services", config)
        self.assertIn("networks", config)

        # Check main service configuration
        services = config["services"]
        self.assertIn("sheily-training", services)

        training_service = services["sheily-training"]
        self.assertIn("build", training_service)
        self.assertIn("environment", training_service)
        self.assertIn("volumes", training_service)
        self.assertIn("healthcheck", training_service)

        # Check GPU configuration
        deploy = training_service.get("deploy", {})
        reservations = deploy.get("resources", {}).get("reservations", {})
        devices = reservations.get("devices", [])
        self.assertGreater(len(devices), 0)


class TestLoRATrainingPerformance(unittest.TestCase):
    """Test LoRA training performance characteristics"""

    def test_lora_config_creation_performance(self):
        """Test LoRA configuration creation performance"""
        from sheily_core.llm_engine.lora_training import create_lora_training_config

        start_time = time.time()

        # Create multiple configurations
        for i in range(50):
            config = create_lora_training_config(
                model_name=f"test_model_{i}",
                branches_to_train=[f"branch_{j}" for j in range(5)],
                languages=["EN", "ES"],
                lora_rank=8,
                num_epochs=3,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast (< 1 second for 50 configs)
        self.assertLess(duration, 1.0)
        print(f"LoRA config creation: {duration:.3f}s for 50 configs")

    def test_adapter_state_update_performance(self):
        """Test adapter state update performance"""
        from sheily_core.llm_engine.lora_training import create_lora_adapter_state, update_lora_adapter_state

        # Create initial state
        initial_state = create_lora_adapter_state(
            adapter_id="test_adapter",
            branch_name="test_branch",
            language="EN",
            total_epochs=10,
            adapter_path="test_path",
        )

        start_time = time.time()

        # Update state multiple times
        current_state = initial_state
        for epoch in range(10):
            current_state = update_lora_adapter_state(
                current_state,
                training_status="training",
                current_epoch=epoch + 1,
                current_loss=0.5 - (epoch * 0.05),
                training_time=current_state.training_time + 60.0,
            )

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast (< 0.5 seconds for 10 updates)
        self.assertLess(duration, 0.5)
        print(f"Adapter state updates: {duration:.3f}s for 10 updates")


class TestLoRAIntegrationWithExistingSystem(unittest.TestCase):
    """Test LoRA integration with existing system components"""

    def test_adapter_lora_integration(self):
        """Test integration between adapter system and LoRA training"""
        from sheily_core.adapters import create_adapter_config
        from sheily_core.llm_engine.lora_training import create_lora_training_config

        # Create base adapter config
        adapter_config = create_adapter_config(
            adapter_type="LORA",
            base_model="test_model",
            target_modules=["qkv_proj", "o_proj"],
            rank=8,
            alpha=16,
        )

        # Create LoRA training config
        lora_config = create_lora_training_config(
            model_name="test_model",
            branches_to_train=["general"],
            languages=["EN"],
            lora_rank=adapter_config.rank,
            lora_alpha=adapter_config.alpha,
            target_modules=adapter_config.target_modules,
        )

        # Verify integration
        self.assertEqual(lora_config.lora_rank, adapter_config.rank)
        self.assertEqual(lora_config.lora_alpha, adapter_config.alpha)
        self.assertEqual(lora_config.target_modules, adapter_config.target_modules)

    def test_gguf_lora_integration(self):
        """Test integration between GGUF models and LoRA training"""
        from sheily_core.llm_engine.gguf_integration import create_gguf_model_config
        from sheily_core.llm_engine.lora_training import create_lora_training_config

        # Create GGUF model config
        gguf_config = create_gguf_model_config(
            model_path="models/gguf/llama-3.2.gguf", quantization_type="Q4_K_M", context_length=4096
        )

        # Create LoRA training config
        lora_config = create_lora_training_config(
            model_name="test_model",
            gguf_model_path=gguf_config.model_path,
            branches_to_train=["general"],
            languages=["EN"],
            lora_rank=8,
        )

        # Verify integration
        self.assertEqual(lora_config.gguf_model_path, gguf_config.model_path)
        self.assertEqual(lora_config.branches_to_train, ["general"])
        self.assertEqual(lora_config.languages, ["EN"])


def run_lora_training_tests():
    """Run all LoRA training system tests"""
    # Create test suite
    test_classes = [
        TestLoRATrainingConfig,
        TestLoRAAdapterState,
        TestLoRATrainingExecution,
        TestGGUFLoRAIntegration,
        TestLoRAApiIntegration,
        TestDockerIntegration,
        TestLoRATrainingPerformance,
        TestLoRAIntegrationWithExistingSystem,
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
    print("LoRA TRAINING SYSTEM TEST RESULTS")
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
    success = run_lora_training_tests()
    print(f"\nOverall Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1)
