#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Script for Sheily AI Functional Training System
====================================================

This script provides automated setup and installation for the training system:
- Environment configuration
- Dependency verification
- Directory structure creation
- Configuration file generation
- System validation
- Production deployment preparation

Features:
- Zero-dependency core installation
- Functional programming compliance
- Comprehensive error handling
- Production-ready configuration
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from result import Err, Ok, Result

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TrainingSystemInstaller:
    """Functional training system installer"""

    def __init__(self):
        self.project_root = project_root
        self.setup_config = None
        self.installation_log = []

    def log_step(self, message: str):
        """Log installation step"""
        timestamp = self._get_timestamp()
        log_entry = f"[{timestamp}] {message}"
        self.installation_log.append(log_entry)
        print(log_entry)

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import time

        return time.strftime("%Y-%m-%d %H:%M:%S")

    def verify_python_version(self) -> Result[bool, str]:
        """Verify Python version compatibility"""
        self.log_step("Verifying Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return Err(f"Python 3.8+ required, found {version.major}.{version.minor}")

        self.log_step(f"✅ Python {version.major}.{version.minor}.{version.micro} compatible")
        return Ok(True)

    def verify_project_structure(self) -> Result[Dict[str, Any], str]:
        """Verify project structure exists"""
        self.log_step("Verifying project structure...")

        required_paths = [
            "sheily_core",
            "sheily_core/llm_engine",
            "branches",
            "corpus_EN",
            "corpus_ES",
            "tests",
            "scripts",
        ]

        missing_paths = []
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                missing_paths.append(path)

        if missing_paths:
            return Err(f"Missing required paths: {missing_paths}")

        self.log_step("✅ Project structure verified")
        return Ok({"verified": True, "required_paths": required_paths})

    def create_directory_structure(self) -> Result[Dict[str, Any], str]:
        """Create necessary directory structure"""
        self.log_step("Creating directory structure...")

        directories_to_create = [
            "training_output",
            "training_output/checkpoints",
            "training_output/logs",
            "training_output/adapters",
            "models",
            "models/gguf",
            "config",
            "logs",
            "reports",
        ]

        created_dirs = []
        for dir_path in directories_to_create:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
                self.log_step(f"   Created: {dir_path}")
            except Exception as e:
                return Err(f"Failed to create directory {dir_path}: {e}")

        return Ok({"created_directories": created_dirs})

    def generate_configuration_files(self) -> Result[Dict[str, Any], str]:
        """Generate default configuration files"""
        self.log_step("Generating configuration files...")

        # Training configuration
        training_config = {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "branches_to_train": ["general", "anthropology", "philosophy", "programming"],
            "languages": ["EN", "ES"],
            "max_iterations": 5,
            "checkpoint_interval": 2,
            "evaluation_interval": 3,
            "adapter_composition_strategy": "incremental",
            "data_preparation": {
                "chunk_size": 512,
                "chunk_overlap": 64,
                "max_documents": 1000,
                "filtering_criteria": {"min_length": 100, "min_quality_score": 0.7},
            },
            "training": {"epochs": 3, "batch_size": 16, "learning_rate": 1e-4, "warmup_steps": 100},
            "output": {
                "base_dir": "training_output",
                "checkpoints_dir": "checkpoints",
                "logs_dir": "logs",
                "adapters_dir": "adapters",
            },
        }

        # API configuration
        api_config = {
            "host": "localhost",
            "port": 8004,
            "debug": False,
            "enable_cors": True,
            "allowed_origins": ["*"],
            "api_key_required": False,
            "api_key": "",
            "rate_limit": 100,
            "timeout": 30,
        }

        # Save configurations
        config_files = {"training_config.json": training_config, "api_config.json": api_config}

        created_configs = []
        for filename, config_data in config_files.items():
            config_path = self.project_root / filename
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                created_configs.append(filename)
                self.log_step(f"   Generated: {filename}")
            except Exception as e:
                return Err(f"Failed to create {filename}: {e}")

        return Ok({"created_configs": created_configs, "config_data": config_files})

    def verify_dependencies(self) -> Result[Dict[str, Any], str]:
        """Verify system dependencies"""
        self.log_step("Verifying dependencies...")

        # Core dependencies (should be available)
        core_deps = ["json", "os", "sys", "time", "pathlib"]

        # Optional dependencies for enhanced functionality
        optional_deps = ["torch", "transformers", "datasets", "accelerate"]

        core_status = {}
        for dep in core_deps:
            try:
                __import__(dep)
                core_status[dep] = "✅ Available"
            except ImportError:
                core_status[dep] = "❌ Missing"

        optional_status = {}
        for dep in optional_deps:
            try:
                __import__(dep)
                optional_status[dep] = "✅ Available"
            except ImportError:
                optional_status[dep] = "⚠️  Optional - Not Available"

        all_deps = {**core_status, **optional_status}

        # Check for critical missing dependencies
        critical_missing = [dep for dep, status in core_status.items() if "❌" in status]

        if critical_missing:
            return Err(f"Critical dependencies missing: {critical_missing}")

        self.log_step("✅ Dependencies verified")
        return Ok(
            {"dependencies": all_deps, "core_deps": core_deps, "optional_deps": optional_deps}
        )

    def validate_installation(self) -> Result[Dict[str, Any], str]:
        """Validate complete installation"""
        self.log_step("Validating installation...")

        validation_checks = []

        # Check if modules can be imported
        try:
            from sheily_core.llm_engine import (
                create_adapter_config,
                create_training_config,
                load_sheily_q4_model,
                start_api_server_functional,
            )

            validation_checks.append("✅ Core modules importable")
        except ImportError as e:
            return Err(f"Module import failed: {e}")

        # Check if configuration files exist
        config_files = ["training_config.json", "api_config.json"]
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                validation_checks.append(f"✅ {config_file} exists")
            else:
                validation_checks.append(f"❌ {config_file} missing")

        # Check if directories exist
        required_dirs = ["training_output", "models", "logs"]
        for dir_name in required_dirs:
            if (self.project_root / dir_name).exists():
                validation_checks.append(f"✅ {dir_name}/ directory exists")
            else:
                validation_checks.append(f"❌ {dir_name}/ directory missing")

        return Ok({"validation_checks": validation_checks})

    def run_setup(self) -> Result[Dict[str, Any], str]:
        """Run complete setup process"""
        self.log_step("🚀 Starting Sheily AI Training System Setup...")

        setup_steps = [
            ("Python Version Check", self.verify_python_version),
            ("Project Structure Verification", self.verify_project_structure),
            ("Directory Structure Creation", self.create_directory_structure),
            ("Configuration Generation", self.generate_configuration_files),
            ("Dependency Verification", self.verify_dependencies),
            ("Installation Validation", self.validate_installation),
        ]

        results = {}
        for step_name, step_func in setup_steps:
            try:
                result = step_func()
                if result.is_ok():
                    results[step_name] = {"success": True, "data": result.unwrap()}
                else:
                    results[step_name] = {"success": False, "error": result.unwrap_err()}
                    return Err(f"Setup failed at step '{step_name}': {result.unwrap_err()}")
            except Exception as e:
                results[step_name] = {"success": False, "error": str(e)}
                return Err(f"Setup failed at step '{step_name}': {e}")

        self.log_step("✅ Setup completed successfully")
        return Ok({"setup_results": results, "installation_log": self.installation_log})

    def create_startup_script(self) -> Result[str, str]:
        """Create startup script for easy system launch"""
        self.log_step("Creating startup script...")

        startup_script = """#!/bin/bash
# Sheily AI Training System Startup Script
# ========================================

echo "🚀 Starting Sheily AI Training System..."

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if configuration exists
if [ ! -f "training_config.json" ]; then
    echo "❌ Configuration file not found. Running setup..."
    python scripts/setup_training_system.py
fi

# Start API server (optional)
echo "🌐 Starting API server..."
python -c "
from sheily_core.llm_engine import start_api_server_functional
result = start_api_server_functional()
print(f'API Server: {\"✅ Started\" if result[\"success\"] else \"❌ Failed\"}')
"

# Show available commands
echo ""
echo "📋 Available Commands:"
echo "  python -m sheily_core.llm_engine --init     # Initialize system"
echo "  python -m sheily_core.llm_engine --status   # Check status"
echo "  python -m sheily_core.llm_engine --train    # Start training"
echo "  python test_training_system.py              # Run tests"
echo "  python test_gguf_integration.py             # Test GGUF integration"
echo ""
echo "✅ Sheily AI Training System is ready!"
"""

        script_path = self.project_root / "start_training_system.sh"
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(startup_script)

            # Make executable
            script_path.chmod(0o755)

            self.log_step(f"✅ Startup script created: {script_path}")
            return Ok(str(script_path))

        except Exception as e:
            return Err(f"Failed to create startup script: {e}")

    def create_requirements_file(self) -> Result[str, str]:
        """Create requirements file for the training system"""
        self.log_step("Creating requirements file...")

        # Core requirements for training system
        requirements_content = """# Sheily AI Training System Requirements
# ======================================

# Core dependencies (functional programming)
result>=0.14.0

# Data processing
pathlib>=1.0.1

# Optional: For enhanced GGUF support
# torch>=2.0.0
# transformers>=4.30.0
# datasets>=2.10.0
# accelerate>=0.20.0
# peft>=0.4.0

# Optional: For API server
# fastapi>=0.100.0
# uvicorn>=0.23.0

# Development and testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Type checking
mypy>=1.0.0

# Code formatting
black>=23.0.0
isort>=5.12.0
"""

        requirements_path = self.project_root / "requirements_training.txt"
        try:
            with open(requirements_path, "w", encoding="utf-8") as f:
                f.write(requirements_content)

            self.log_step(f"✅ Requirements file created: {requirements_path}")
            return Ok(str(requirements_path))

        except Exception as e:
            return Err(f"Failed to create requirements file: {e}")


def main():
    """Main setup function"""
    print("🔧 Sheily AI Training System Setup")
    print("=" * 50)

    installer = TrainingSystemInstaller()

    try:
        # Run setup
        result = installer.run_setup()

        if result.is_ok():
            setup_data = result.unwrap()

            print("\n" + "=" * 50)
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("=" * 50)

            # Create additional setup artifacts
            startup_script_result = installer.create_startup_script()
            if startup_script_result.is_ok():
                print(f"✅ Startup script: {startup_script_result.unwrap()}")

            requirements_result = installer.create_requirements_file()
            if requirements_result.is_ok():
                print(f"✅ Requirements file: {requirements_result.unwrap()}")

            print("\n📋 Next Steps:")
            print("1. Review configuration files: training_config.json, api_config.json")
            print("2. Run tests: python test_training_system.py")
            print("3. Start system: python -m sheily_core.llm_engine --init")
            print("4. Begin training: python -m sheily_core.llm_engine --train")

            print(f"\n📊 Installation completed in {len(installer.installation_log)} steps")
            return 0

        else:
            print(f"\n❌ SETUP FAILED: {result.unwrap_err()}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ SETUP ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
