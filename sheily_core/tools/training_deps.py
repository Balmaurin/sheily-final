#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional Training Dependencies Manager for LLM Engine
=======================================================

This module provides functional dependency management for training:
- Immutable dependency configurations
- Pure functions for dependency operations
- Integration with depswitch for security
- Functional dependency pipelines
- Composable dependency strategies
"""

import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from result import Err, Ok, Result

# ============================================================================
# Functional Data Types for Training Dependencies
# ============================================================================


@dataclass(frozen=True)
class TrainingDependency:
    """Immutable training dependency"""

    name: str
    version: str
    package_type: str  # ml, data, utils, etc.
    security_level: str  # safe, restricted, blocked
    alternatives: List[str]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class DependencyContext:
    """Immutable dependency context"""

    training_mode: bool
    allowed_packages: List[str]
    blocked_packages: List[str]
    dependency_map: Dict[str, str]
    security_config: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DependencyResolution:
    """Immutable dependency resolution result"""

    request_id: str
    requested_package: str
    resolved_package: str
    resolution_method: str
    security_check: bool
    alternatives_considered: List[str]
    metadata: Dict[str, Any]
    timestamp: str


# ============================================================================
# Pure Functions for Dependency Management
# ============================================================================


def create_training_dependency(
    name: str,
    version: str,
    package_type: str,
    security_level: str = "restricted",
    alternatives: List[str] = None,
    config: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> TrainingDependency:
    """Create training dependency - Pure function"""
    return TrainingDependency(
        name=name,
        version=version,
        package_type=package_type,
        security_level=security_level,
        alternatives=alternatives or [],
        config=config or {},
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def create_dependency_context(
    training_mode: bool = False,
    allowed_packages: List[str] = None,
    blocked_packages: List[str] = None,
    dependency_map: Dict[str, str] = None,
    security_config: Dict[str, Any] = None,
) -> DependencyContext:
    """Create dependency context - Pure function"""
    return DependencyContext(
        training_mode=training_mode,
        allowed_packages=allowed_packages or [],
        blocked_packages=blocked_packages or [],
        dependency_map=dependency_map or {},
        security_config=security_config or {},
        metadata={"created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )


def create_dependency_resolution(
    request_id: str,
    requested_package: str,
    resolved_package: str,
    resolution_method: str,
    security_check: bool,
    alternatives_considered: List[str] = None,
    metadata: Dict[str, Any] = None,
) -> DependencyResolution:
    """Create dependency resolution - Pure function"""
    return DependencyResolution(
        request_id=request_id,
        requested_package=requested_package,
        resolved_package=resolved_package,
        resolution_method=resolution_method,
        security_check=security_check,
        alternatives_considered=alternatives_considered or [],
        metadata=metadata or {},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def check_package_security(package_name: str, context: DependencyContext) -> bool:
    """Check if package is allowed in current context - Pure function"""
    # Check blocked packages first
    if package_name in context.blocked_packages:
        return False

    # In training mode, allow training-specific packages
    if context.training_mode:
        training_packages = [
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "datasets",
            "tokenizers",
            "diffusers",
            "safetensors",
        ]
        if package_name in training_packages:
            return True

    # Check explicitly allowed packages
    if package_name in context.allowed_packages:
        return True

    return False


def resolve_package_dependency(package_name: str, context: DependencyContext) -> Result[str, str]:
    """Resolve package dependency - Pure function"""
    # Check if package is allowed
    if not check_package_security(package_name, context):
        return Err(f"Package {package_name} is not allowed in current context")

    # Check if there's a mapped alternative
    if package_name in context.dependency_map:
        return Ok(context.dependency_map[package_name])

    # Return original package if allowed
    return Ok(package_name)


def validate_dependency_context(context: DependencyContext) -> Result[DependencyContext, str]:
    """Validate dependency context - Pure function"""
    if context.training_mode and not context.allowed_packages:
        return Err("Training mode requires explicit allowed packages")

    if len(context.blocked_packages) > 100:
        return Err("Too many blocked packages specified")

    return Ok(context)


def compose_dependency_contexts(
    base_context: DependencyContext, overlay_context: DependencyContext
) -> DependencyContext:
    """Compose dependency contexts - Pure function"""
    # Merge allowed packages
    merged_allowed = list(set(base_context.allowed_packages + overlay_context.allowed_packages))

    # Merge blocked packages
    merged_blocked = list(set(base_context.blocked_packages + overlay_context.blocked_packages))

    # Overlay dependency map
    merged_map = {**base_context.dependency_map, **overlay_context.dependency_map}

    # Merge security config
    merged_security = {**base_context.security_config, **overlay_context.security_config}

    return DependencyContext(
        training_mode=base_context.training_mode or overlay_context.training_mode,
        allowed_packages=merged_allowed,
        blocked_packages=merged_blocked,
        dependency_map=merged_map,
        security_config=merged_security,
        metadata={
            **base_context.metadata,
            **overlay_context.metadata,
            "composed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )


# ============================================================================
# Training-Specific Dependency Management
# ============================================================================


def create_training_dependency_context() -> DependencyContext:
    """Create training-specific dependency context - Factory function"""
    # Training-allowed packages (safe for training environment)
    training_allowed = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "tokenizers",
        "diffusers",
        "safetensors",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "requests",
        "pyyaml",
    ]

    # Production-blocked packages (not allowed in training)
    training_blocked = [
        "gradio",
        "streamlit",
        "jupyter",
        "ipython",
        "notebook",
        "fastapi",
        "uvicorn",
        "flask",
        "django",
    ]

    # Safe alternatives for training
    training_map = {
        "requests": "sheily_core.depswitch.providers.http_urllib",
        "yaml": "sheily_core.depswitch.providers.yaml_tomljson",
        "numpy": "sheily_core.depswitch.providers.guard_fail",  # Use alternative implementations
    }

    return create_dependency_context(
        training_mode=True,
        allowed_packages=training_allowed,
        blocked_packages=training_blocked,
        dependency_map=training_map,
        security_config={
            "require_verification": True,
            "allow_network_access": False,
            "max_package_size_mb": 100,
            "allowed_sources": ["pypi", "internal"],
        },
    )


def create_production_dependency_context() -> DependencyContext:
    """Create production dependency context - Factory function"""
    # Production-safe packages only
    production_allowed = [
        "sheily_core.depswitch.providers.http_urllib",
        "sheily_core.depswitch.providers.yaml_tomljson",
    ]

    # Block all ML/AI packages in production
    production_blocked = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "tokenizers",
        "diffusers",
        "safetensors",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "requests",
        "pyyaml",
        "gradio",
        "streamlit",
        "jupyter",
        "ipython",
        "notebook",
        "fastapi",
        "uvicorn",
        "flask",
        "django",
    ]

    return create_dependency_context(
        training_mode=False,
        allowed_packages=production_allowed,
        blocked_packages=production_blocked,
        dependency_map={},  # No remapping in production
        security_config={
            "require_verification": True,
            "allow_network_access": False,
            "max_package_size_mb": 10,
            "allowed_sources": ["internal_only"],
        },
    )


# ============================================================================
# Functional Dependency Resolution Pipeline
# ============================================================================


def create_dependency_resolver() -> Callable[[str, DependencyContext], Result[DependencyResolution, str]]:
    """Create dependency resolver - Factory function"""

    def resolver(package_name: str, context: DependencyContext) -> Result[DependencyResolution, str]:
        request_id = f"dep_{int(time.time())}_{hash(package_name) % 10000}"

        # Resolve package
        resolution_result = resolve_package_dependency(package_name, context)

        if resolution_result.is_err():
            return Err(resolution_result.unwrap_err())

        resolved_package = resolution_result.unwrap()

        # Create resolution record
        resolution = create_dependency_resolution(
            request_id=request_id,
            requested_package=package_name,
            resolved_package=resolved_package,
            resolution_method="direct" if package_name == resolved_package else "mapped",
            security_check=check_package_security(package_name, context),
            alternatives_considered=context.dependency_map.get(package_name, []),
            metadata={
                "context_mode": "training" if context.training_mode else "production",
                "security_level": context.security_config.get("require_verification", False),
            },
        )

        return Ok(resolution)

    return resolver


def create_dependency_validation_pipeline() -> Callable[[DependencyContext], Result[DependencyContext, str]]:
    """Create dependency validation pipeline - Factory function"""

    def pipeline(context: DependencyContext) -> Result[DependencyContext, str]:
        return validate_dependency_context(context)

    return pipeline


def create_safe_import_executor() -> Callable[[str, DependencyContext], Result[Any, str]]:
    """Create safe import executor - Factory function"""

    def executor(package_name: str, context: DependencyContext) -> Result[Any, str]:
        # Resolve dependency first
        resolver = create_dependency_resolver()
        resolution_result = resolver(package_name, context)

        if resolution_result.is_err():
            return Err(resolution_result.unwrap_err())

        resolution = resolution_result.unwrap()
        final_package = resolution.resolved_package

        try:
            # Check if it's a depswitch provider
            if final_package.startswith("sheily_core.depswitch.providers."):
                # Use depswitch mechanism
                if "depswitch" in sys.modules:
                    # This would integrate with actual depswitch
                    pass
                else:
                    return Err(f"Depswitch not available for provider: {final_package}")
            else:
                # Regular import
                module = __import__(final_package)
                return Ok(module)

        except ImportError as e:
            return Err(f"Failed to import {final_package}: {e}")
        except Exception as e:
            return Err(f"Import error for {final_package}: {e}")

        return Ok(None)

    return executor


# ============================================================================
# Training Dependency Context Management
# ============================================================================


def initialize_training_dependencies() -> Result[DependencyContext, str]:
    """Initialize training dependencies - Pure function"""
    try:
        # Create training context
        context = create_training_dependency_context()

        # Validate context
        validation_result = validate_dependency_context(context)
        if validation_result.is_err():
            return Err(validation_result.unwrap_err())

        return Ok(context)

    except Exception as e:
        return Err(f"Failed to initialize training dependencies: {e}")


def switch_to_training_mode() -> Result[DependencyContext, str]:
    """Switch to training mode dependencies - Pure function"""
    return initialize_training_dependencies()


def switch_to_production_mode() -> Result[DependencyContext, str]:
    """Switch to production mode dependencies - Pure function"""
    try:
        context = create_production_dependency_context()

        validation_result = validate_dependency_context(context)
        if validation_result.is_err():
            return Err(validation_result.unwrap_err())

        return Ok(context)

    except Exception as e:
        return Err(f"Failed to initialize production dependencies: {e}")


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================


def get_training_dependencies_functional() -> Dict[str, Any]:
    """Get training dependencies using functional approach - Legacy compatibility"""
    try:
        context_result = initialize_training_dependencies()

        if context_result.is_ok():
            context = context_result.unwrap()
            return {
                "success": True,
                "training_mode": context.training_mode,
                "allowed_packages": context.allowed_packages,
                "blocked_packages": context.blocked_packages,
                "dependency_map": context.dependency_map,
                "security_config": context.security_config,
                "metadata": context.metadata,
            }
        else:
            return {
                "success": False,
                "error": context_result.unwrap_err(),
                "fallback_mode": "production",
            }

    except Exception as e:
        return {"success": False, "error": str(e), "fallback_mode": "production"}


def resolve_dependency_functional(package_name: str, training_mode: bool = False) -> Dict[str, Any]:
    """Resolve dependency using functional approach - Legacy compatibility"""
    try:
        # Create appropriate context
        if training_mode:
            context = create_training_dependency_context()
        else:
            context = create_production_dependency_context()

        # Resolve dependency
        resolver = create_dependency_resolver()
        result = resolver(package_name, context)

        if result.is_ok():
            resolution = result.unwrap()
            return {
                "success": True,
                "request_id": resolution.request_id,
                "requested_package": resolution.requested_package,
                "resolved_package": resolution.resolved_package,
                "resolution_method": resolution.resolution_method,
                "security_check": resolution.security_check,
                "alternatives_considered": resolution.alternatives_considered,
                "metadata": resolution.metadata,
                "timestamp": resolution.timestamp,
            }
        else:
            return {
                "success": False,
                "error": result.unwrap_err(),
                "requested_package": package_name,
                "fallback_package": package_name,  # Fallback to original
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "requested_package": package_name,
            "fallback_package": package_name,
        }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data types
    "TrainingDependency",
    "DependencyContext",
    "DependencyResolution",
    # Pure functions
    "create_training_dependency",
    "create_dependency_context",
    "create_dependency_resolution",
    "check_package_security",
    "resolve_package_dependency",
    "validate_dependency_context",
    "compose_dependency_contexts",
    # Factory functions
    "create_training_dependency_context",
    "create_production_dependency_context",
    "create_dependency_resolver",
    "create_dependency_validation_pipeline",
    "create_safe_import_executor",
    # Context management
    "initialize_training_dependencies",
    "switch_to_training_mode",
    "switch_to_production_mode",
    # Legacy compatibility
    "get_training_dependencies_functional",
    "resolve_dependency_functional",
]
