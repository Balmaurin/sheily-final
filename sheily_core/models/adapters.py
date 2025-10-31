#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional Adapters Module for Sheily AI System
==============================================

This module provides functional composition patterns for adapter management:
- Immutable adapter configurations
- Functional adapter composition
- Pure functions for adapter operations
- Composable adapter pipelines
- Zero-dependency implementation for adapter handling
"""

import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from result import Err, Ok, Result

# Version del módulo
__version__ = "1.0.0"

# ============================================================================
# Functional Data Types for Adapters
# ============================================================================


@dataclass(frozen=True)
class AdapterConfig:
    """Immutable adapter configuration"""

    adapter_id: str
    adapter_type: str  # LORA, PEFT, etc.
    base_model: str
    target_modules: List[str]
    rank: int
    alpha: int
    dropout: float
    task_type: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AdapterState:
    """Immutable adapter state"""

    adapter_id: str
    branch_name: str
    language: str
    training_iterations: int
    knowledge_base_size: int
    performance_metrics: Dict[str, float]
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AdapterComposition:
    """Immutable adapter composition"""

    composition_id: str
    base_adapter: AdapterConfig
    composed_adapters: List[AdapterConfig]
    composition_strategy: str
    merged_config: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class AdapterContext:
    """Functional context for adapter operations"""

    adapters_path: Path
    adapter_configs: Dict[str, AdapterConfig]
    adapter_states: Dict[str, AdapterState]
    compositions: Dict[str, AdapterComposition]
    logger: Any


# ============================================================================
# Pure Functions for Adapter Operations
# ============================================================================


def create_adapter_config(
    adapter_type: str,
    base_model: str,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    task_type: str = "CAUSAL_LM",
    config: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> AdapterConfig:
    """Create adapter configuration - Pure function"""
    return AdapterConfig(
        adapter_id=f"adapter_{int(time.time())}_{hash(str(target_modules)) % 10000}",
        adapter_type=adapter_type,
        base_model=base_model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        task_type=task_type,
        config=config or {},
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def create_adapter_state(
    adapter_id: str,
    branch_name: str,
    language: str,
    training_iterations: int = 0,
    knowledge_base_size: int = 0,
    performance_metrics: Dict[str, float] = None,
    state: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> AdapterState:
    """Create adapter state - Pure function"""
    return AdapterState(
        adapter_id=adapter_id,
        branch_name=branch_name,
        language=language,
        training_iterations=training_iterations,
        knowledge_base_size=knowledge_base_size,
        performance_metrics=performance_metrics or {},
        state=state or {},
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def load_adapter_config_from_file(config_path: Path) -> Optional[AdapterConfig]:
    """Load adapter configuration from file - Pure function"""
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return AdapterConfig(
            adapter_id=data["adapter_id"],
            adapter_type=data["adapter_type"],
            base_model=data["base_model"],
            target_modules=data["target_modules"],
            rank=data["rank"],
            alpha=data["alpha"],
            dropout=data["dropout"],
            task_type=data["task_type"],
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
    except Exception:
        return None


def save_adapter_config_to_file(config: AdapterConfig, config_path: Path) -> Result[Path, str]:
    """Save adapter configuration to file - Pure function"""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "adapter_id": config.adapter_id,
            "adapter_type": config.adapter_type,
            "base_model": config.base_model,
            "target_modules": config.target_modules,
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "task_type": config.task_type,
            "config": config.config,
            "metadata": config.metadata,
            "created_at": config.created_at,
            "updated_at": config.updated_at,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return Ok(config_path)
    except Exception as e:
        return Err(f"Failed to save adapter config: {e}")


def compose_adapter_configs(
    base_config: AdapterConfig, overlay_config: AdapterConfig
) -> AdapterConfig:
    """Compose two adapter configurations - Pure function"""
    # Merge target modules
    merged_modules = list(set(base_config.target_modules + overlay_config.target_modules))

    # Merge configurations
    merged_config = {**base_config.config, **overlay_config.config}

    # Use highest rank and alpha for better capacity
    final_rank = max(base_config.rank, overlay_config.rank)
    final_alpha = max(base_config.alpha, overlay_config.alpha)

    return AdapterConfig(
        adapter_id=f"composed_{base_config.adapter_id}_{overlay_config.adapter_id}",
        adapter_type=base_config.adapter_type,
        base_model=base_config.base_model,
        target_modules=merged_modules,
        rank=final_rank,
        alpha=final_alpha,
        dropout=min(base_config.dropout, overlay_config.dropout),  # Use lower dropout
        task_type=base_config.task_type,
        config=merged_config,
        metadata={
            **base_config.metadata,
            **overlay_config.metadata,
            "composed_from": [base_config.adapter_id, overlay_config.adapter_id],
            "composition_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def validate_adapter_config(config: AdapterConfig) -> Result[AdapterConfig, str]:
    """Validate adapter configuration - Pure function"""
    if not config.adapter_id:
        return Err("Adapter ID cannot be empty")

    if config.rank <= 0:
        return Err("Adapter rank must be positive")

    if not config.target_modules:
        return Err("Target modules cannot be empty")

    if config.dropout < 0 or config.dropout >= 1:
        return Err("Dropout must be between 0 and 1")

    return Ok(config)


def update_adapter_state(
    current_state: AdapterState,
    training_iterations: int = None,
    knowledge_base_size: int = None,
    performance_metrics: Dict[str, float] = None,
    state: Dict[str, Any] = None,
) -> AdapterState:
    """Update adapter state immutably - Pure function"""
    return AdapterState(
        adapter_id=current_state.adapter_id,
        branch_name=current_state.branch_name,
        language=current_state.language,
        training_iterations=training_iterations
        if training_iterations is not None
        else current_state.training_iterations,
        knowledge_base_size=knowledge_base_size
        if knowledge_base_size is not None
        else current_state.knowledge_base_size,
        performance_metrics=performance_metrics
        if performance_metrics is not None
        else current_state.performance_metrics,
        state=state if state is not None else current_state.state,
        metadata={**current_state.metadata, "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
        created_at=current_state.created_at,
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def calculate_adapter_memory_size(config: AdapterConfig) -> int:
    """Calculate approximate memory size for adapter - Pure function"""
    # Rough estimation based on rank and target modules
    base_memory_per_module = 1024 * 1024  # 1MB per module baseline
    rank_multiplier = config.rank * 1000  # Additional memory per rank unit

    total_memory = len(config.target_modules) * (base_memory_per_module + rank_multiplier)
    return total_memory


def filter_adapters_by_language(
    adapters: List[AdapterConfig], language: str
) -> List[AdapterConfig]:
    """Filter adapters by language - Pure function"""
    return [adapter for adapter in adapters if adapter.metadata.get("language") == language]


def sort_adapters_by_performance(
    adapters: List[AdapterConfig], states: Dict[str, AdapterState]
) -> List[AdapterConfig]:
    """Sort adapters by performance metrics - Pure function"""

    def get_avg_performance(adapter_id: str) -> float:
        state = states.get(adapter_id)
        if not state or not state.performance_metrics:
            return 0.0
        return sum(state.performance_metrics.values()) / len(state.performance_metrics)

    # Sort by average performance (descending)
    return sorted(adapters, key=lambda a: get_avg_performance(a.adapter_id), reverse=True)


# ============================================================================
# Functional Composition Patterns for Adapters
# ============================================================================


def create_adapter_processor(adapters_path: Path) -> Callable[[str], Optional[AdapterConfig]]:
    """Create adapter processor function - Factory function"""

    def processor(adapter_id: str) -> Optional[AdapterConfig]:
        config_path = adapters_path / f"{adapter_id}_config.json"
        return load_adapter_config_from_file(config_path)

    return processor


def create_adapter_composer(
    strategy: str = "merge",
) -> Callable[[AdapterConfig, AdapterConfig], AdapterConfig]:
    """Create adapter composer function - Factory function"""
    if strategy == "merge":

        def composer(base: AdapterConfig, overlay: AdapterConfig) -> AdapterConfig:
            return compose_adapter_configs(base, overlay)

        return composer
    else:
        # Default to base adapter
        def composer(base: AdapterConfig, overlay: AdapterConfig) -> AdapterConfig:
            return base

        return composer


def create_adapter_validator() -> Callable[[AdapterConfig], Result[AdapterConfig, str]]:
    """Create adapter validator function - Factory function"""

    def validator(config: AdapterConfig) -> Result[AdapterConfig, str]:
        return validate_adapter_config(config)

    return validator


def create_adapter_filter(language: str) -> Callable[[List[AdapterConfig]], List[AdapterConfig]]:
    """Create adapter filter function - Factory function"""

    def filter_func(adapters: List[AdapterConfig]) -> List[AdapterConfig]:
        return filter_adapters_by_language(adapters, language)

    return filter_func


# ============================================================================
# Functional Pipeline Composition for Adapters
# ============================================================================


def create_adapter_pipeline(
    adapters_path: Path, composition_strategy: str = "merge"
) -> Callable[[str, str], Result[AdapterConfig, str]]:
    """Create complete adapter processing pipeline - Factory function"""
    processor = create_adapter_processor(adapters_path)
    composer = create_adapter_composer(composition_strategy)
    validator = create_adapter_validator()

    def pipeline(base_adapter_id: str, overlay_adapter_id: str) -> Result[AdapterConfig, str]:
        # Load base adapter
        base_adapter = processor(base_adapter_id)
        if not base_adapter:
            return Err(f"Base adapter not found: {base_adapter_id}")

        # Load overlay adapter
        overlay_adapter = processor(overlay_adapter_id)
        if not overlay_adapter:
            return Err(f"Overlay adapter not found: {overlay_adapter_id}")

        # Compose adapters
        composed_adapter = composer(base_adapter, overlay_adapter)

        # Validate result
        validation_result = validator(composed_adapter)
        return validation_result

    return pipeline


def create_incremental_training_pipeline(
    adapters_path: Path, language: str
) -> Callable[[List[str]], Result[List[AdapterConfig], str]]:
    """Create incremental training pipeline - Factory function"""
    filter_func = create_adapter_filter(language)
    processor = create_adapter_processor(adapters_path)

    def pipeline(adapter_ids: List[str]) -> Result[List[AdapterConfig], str]:
        # Load all adapters
        adapters = []
        for adapter_id in adapter_ids:
            adapter = processor(adapter_id)
            if adapter:
                adapters.append(adapter)
            else:
                return Err(f"Adapter not found: {adapter_id}")

        # Filter by language
        filtered_adapters = filter_func(adapters)

        return Ok(filtered_adapters)

    return pipeline


# ============================================================================
# Adapter Context Management
# ============================================================================


def create_adapter_context(adapters_path: Path) -> AdapterContext:
    """Create adapter context - Pure function"""
    return AdapterContext(
        adapters_path=adapters_path,
        adapter_configs={},
        adapter_states={},
        compositions={},
        logger=None,  # Will be injected when needed
    )


def register_adapter_in_context(
    context: AdapterContext, config: AdapterConfig, state: AdapterState = None
) -> AdapterContext:
    """Register adapter in context - Pure function"""
    new_configs = {**context.adapter_configs, config.adapter_id: config}
    new_states = context.adapter_states

    if state:
        new_states = {**context.adapter_states, state.adapter_id: state}

    return AdapterContext(
        adapters_path=context.adapters_path,
        adapter_configs=new_configs,
        adapter_states=new_states,
        compositions=context.compositions,
        logger=context.logger,
    )


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================


def create_lora_config(
    base_model: str,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> Dict[str, Any]:
    """Create LoRA configuration - Legacy compatibility"""
    config = create_adapter_config(
        adapter_type="LORA",
        base_model=base_model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        task_type="CAUSAL_LM",
    )

    return {
        "adapter_id": config.adapter_id,
        "adapter_type": config.adapter_type,
        "base_model": config.base_model,
        "target_modules": config.target_modules,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "task_type": config.task_type,
        "config": config.config,
        "metadata": config.metadata,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    }


def save_adapter_config_legacy(config: Dict[str, Any], file_path: str) -> bool:
    """Save adapter configuration - Legacy compatibility"""
    try:
        adapter_config = AdapterConfig(
            adapter_id=config["adapter_id"],
            adapter_type=config["adapter_type"],
            base_model=config["base_model"],
            target_modules=config["target_modules"],
            rank=config["rank"],
            alpha=config["alpha"],
            dropout=config["dropout"],
            task_type=config["task_type"],
            config=config.get("config", {}),
            metadata=config.get("metadata", {}),
            created_at=config["created_at"],
            updated_at=config["updated_at"],
        )

        path = Path(file_path)
        result = save_adapter_config_to_file(adapter_config, path)
        return result.is_ok()
    except Exception:
        return False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data types
    "AdapterConfig",
    "AdapterState",
    "AdapterComposition",
    "AdapterContext",
    # Pure functions
    "create_adapter_config",
    "create_adapter_state",
    "load_adapter_config_from_file",
    "save_adapter_config_to_file",
    "compose_adapter_configs",
    "validate_adapter_config",
    "update_adapter_state",
    "calculate_adapter_memory_size",
    "filter_adapters_by_language",
    "sort_adapters_by_performance",
    # Factory functions
    "create_adapter_processor",
    "create_adapter_composer",
    "create_adapter_validator",
    "create_adapter_filter",
    "create_adapter_pipeline",
    "create_incremental_training_pipeline",
    # Context management
    "create_adapter_context",
    "register_adapter_in_context",
    # Legacy compatibility
    "create_lora_config",
    "save_adapter_config_legacy",
]

# Log de inicialización del módulo
print(f"✅ Módulo adapters v{__version__} inicializado correctamente (Paradigma Funcional)")
