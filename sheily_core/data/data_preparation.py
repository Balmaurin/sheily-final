#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functional Data Preparation Module for LLM Engine
=================================================

This module provides functional data preparation for training:
- Immutable data preparation configurations
- Pure functions for data processing
- Linguistic separation between EN/ES
- Integration with corpus_EN and corpus_ES
- Composable data preparation pipelines
"""

import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from result import Err, Ok, Result

# ============================================================================
# Functional Data Types for Data Preparation
# ============================================================================


@dataclass(frozen=True)
class DataPreparationConfig:
    """Immutable data preparation configuration"""

    config_id: str
    language: str
    corpus_type: str  # domain_specific, general, mixed
    chunk_size: int
    chunk_overlap: int
    max_documents: int
    filtering_criteria: Dict[str, Any]
    preprocessing_steps: List[str]
    output_format: str
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class PreparedDataset:
    """Immutable prepared dataset"""

    dataset_id: str
    language: str
    branch_name: str
    documents: List[Dict[str, Any]]
    total_chunks: int
    vocabulary_size: int
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class CorpusIntegration:
    """Immutable corpus integration settings"""

    corpus_root: Path
    en_corpus_path: Path
    es_corpus_path: Path
    domain_configs: Dict[str, Dict[str, Any]]
    language_separation: bool
    cross_contamination_check: bool
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DataPreparationContext:
    """Functional context for data preparation operations"""

    config: DataPreparationConfig
    corpus_integration: CorpusIntegration
    prepared_datasets: Dict[str, PreparedDataset]
    logger: Any


# ============================================================================
# Pure Functions for Data Preparation
# ============================================================================


def create_data_preparation_config(
    language: str,
    corpus_type: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    max_documents: int = 1000,
    filtering_criteria: Dict[str, Any] = None,
    preprocessing_steps: List[str] = None,
    output_format: str = "jsonl",
    metadata: Dict[str, Any] = None,
) -> DataPreparationConfig:
    """Create data preparation configuration - Pure function"""
    return DataPreparationConfig(
        config_id=f"prep_{int(time.time())}_{hash(f'{language}_{corpus_type}') % 10000}",
        language=language,
        corpus_type=corpus_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_documents=max_documents,
        filtering_criteria=filtering_criteria or {},
        preprocessing_steps=preprocessing_steps or ["normalize", "filter", "chunk"],
        output_format=output_format,
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def create_prepared_dataset(
    language: str,
    branch_name: str,
    documents: List[Dict[str, Any]],
    statistics: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
) -> PreparedDataset:
    """Create prepared dataset - Pure function"""
    # Calculate statistics if not provided
    if not statistics:
        statistics = calculate_dataset_statistics(documents)

    return PreparedDataset(
        dataset_id=f"dataset_{int(time.time())}_{hash(f'{language}_{branch_name}') % 10000}",
        language=language,
        branch_name=branch_name,
        documents=documents,
        total_chunks=sum(len(doc.get("chunks", [])) for doc in documents),
        vocabulary_size=statistics.get("vocabulary_size", 0),
        statistics=statistics,
        metadata=metadata or {},
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def create_corpus_integration(
    corpus_root: Path,
    en_corpus_path: str = "corpus_EN",
    es_corpus_path: str = "corpus_ES",
    domain_configs: Dict[str, Dict[str, Any]] = None,
    language_separation: bool = True,
    cross_contamination_check: bool = True,
) -> CorpusIntegration:
    """Create corpus integration - Pure function"""
    return CorpusIntegration(
        corpus_root=corpus_root,
        en_corpus_path=Path(corpus_root) / en_corpus_path,
        es_corpus_path=Path(corpus_root) / es_corpus_path,
        domain_configs=domain_configs or {},
        language_separation=language_separation,
        cross_contamination_check=cross_contamination_check,
        metadata={"created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )


def calculate_dataset_statistics(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate dataset statistics - Pure function"""
    if not documents:
        return {
            "total_documents": 0,
            "total_characters": 0,
            "total_words": 0,
            "avg_document_length": 0,
            "vocabulary_size": 0,
        }

    total_docs = len(documents)
    all_text = " ".join(doc.get("content", "") for doc in documents)
    words = all_text.split()
    vocabulary = set(words)

    return {
        "total_documents": total_docs,
        "total_characters": len(all_text),
        "total_words": len(words),
        "avg_document_length": len(all_text) / total_docs,
        "vocabulary_size": len(vocabulary),
        "calculated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def validate_language_separation(
    en_documents: List[Dict[str, Any]], es_documents: List[Dict[str, Any]]
) -> Result[bool, str]:
    """Validate language separation - Pure function"""
    # Check for cross-contamination
    en_languages = set(doc.get("language", "EN") for doc in en_documents)
    es_languages = set(doc.get("language", "ES") for doc in es_documents)

    if "ES" in en_languages:
        return Err("English corpus contains Spanish documents")

    if "EN" in es_languages:
        return Err("Spanish corpus contains English documents")

    return Ok(True)


def filter_documents_by_criteria(documents: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter documents by criteria - Pure function"""
    filtered_docs = documents

    # Filter by minimum length
    if "min_length" in criteria:
        min_len = criteria["min_length"]
        filtered_docs = [doc for doc in filtered_docs if len(doc.get("content", "")) >= min_len]

    # Filter by maximum length
    if "max_length" in criteria:
        max_len = criteria["max_length"]
        filtered_docs = [doc for doc in filtered_docs if len(doc.get("content", "")) <= max_len]

    # Filter by domains
    if "allowed_domains" in criteria:
        allowed_domains = set(criteria["allowed_domains"])
        filtered_docs = [doc for doc in filtered_docs if doc.get("domain") in allowed_domains]

    # Filter by quality score
    if "min_quality_score" in criteria:
        min_score = criteria["min_quality_score"]
        filtered_docs = [doc for doc in filtered_docs if doc.get("quality_score", 0) >= min_score]

    return filtered_docs


def chunk_document_content(content: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk document content - Pure function"""
    if not content or chunk_size <= 0:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size

        # Find word boundary
        if end < len(content):
            last_space = content.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = max(start + 1, end - overlap)

    return chunks


def preprocess_document(document: Dict[str, Any], config: DataPreparationConfig) -> Dict[str, Any]:
    """Preprocess single document - Pure function"""
    processed_doc = document.copy()

    # Apply preprocessing steps
    for step in config.preprocessing_steps:
        if step == "normalize":
            # Normalize text
            content = processed_doc.get("content", "")
            processed_doc["content"] = " ".join(content.split())  # Normalize whitespace

        elif step == "filter":
            # Apply filtering criteria
            if config.filtering_criteria:
                # Simple filtering - would need more sophisticated logic
                pass

        elif step == "chunk":
            # Chunk content
            content = processed_doc.get("content", "")
            chunks = chunk_document_content(content, config.chunk_size, config.chunk_overlap)
            processed_doc["chunks"] = chunks
            processed_doc["chunk_count"] = len(chunks)

    return processed_doc


# ============================================================================
# Language-Specific Data Loading
# ============================================================================


def load_english_corpus_data(
    corpus_integration: CorpusIntegration, branch_name: str, config: DataPreparationConfig
) -> Result[List[Dict[str, Any]], str]:
    """Load English corpus data - Pure function"""
    try:
        if not corpus_integration.en_corpus_path.exists():
            return Err(f"English corpus path does not exist: {corpus_integration.en_corpus_path}")

        # Load branch-specific data
        branch_path = corpus_integration.en_corpus_path / branch_name

        if not branch_path.exists():
            return Err(f"Branch not found in English corpus: {branch_name}")

        documents = []

        # Load domain configuration
        domain_config_file = branch_path / "domain_config.yaml"
        if domain_config_file.exists():
            # This would integrate with corpus_EN module
            # For now, create mock documents
            mock_docs = [
                {
                    "id": f"en_{branch_name}_{i}",
                    "content": f"English document {i} for {branch_name} branch",
                    "language": "EN",
                    "domain": branch_name,
                    "source": "corpus_EN",
                    "quality_score": 0.85 + (i * 0.01),
                }
                for i in range(min(10, config.max_documents))  # Mock data
            ]
            documents.extend(mock_docs)

        return Ok(documents)

    except Exception as e:
        return Err(f"Failed to load English corpus data: {e}")


def load_spanish_corpus_data(
    corpus_integration: CorpusIntegration, branch_name: str, config: DataPreparationConfig
) -> Result[List[Dict[str, Any]], str]:
    """Load Spanish corpus data - Pure function"""
    try:
        if not corpus_integration.es_corpus_path.exists():
            return Err(f"Spanish corpus path does not exist: {corpus_integration.es_corpus_path}")

        # Load branch-specific data
        branch_path = corpus_integration.es_corpus_path / branch_name

        if not branch_path.exists():
            return Err(f"Branch not found in Spanish corpus: {branch_name}")

        documents = []

        # Load domain configuration
        domain_config_file = branch_path / "domain_config.yaml"
        if domain_config_file.exists():
            # This would integrate with corpus_ES module
            # For now, create mock documents
            mock_docs = [
                {
                    "id": f"es_{branch_name}_{i}",
                    "content": f"Documento en español {i} para la rama {branch_name}",
                    "language": "ES",
                    "domain": branch_name,
                    "source": "corpus_ES",
                    "quality_score": 0.85 + (i * 0.01),
                }
                for i in range(min(10, config.max_documents))  # Mock data
            ]
            documents.extend(mock_docs)

        return Ok(documents)

    except Exception as e:
        return Err(f"Failed to load Spanish corpus data: {e}")


# ============================================================================
# Functional Data Preparation Pipelines
# ============================================================================


def create_language_specific_loader(
    language: str,
) -> Callable[[CorpusIntegration, str, DataPreparationConfig], Result[List[Dict[str, Any]], str]]:
    """Create language-specific data loader - Factory function"""
    if language == "EN":

        def loader(
            integration: CorpusIntegration, branch: str, config: DataPreparationConfig
        ) -> Result[List[Dict[str, Any]], str]:
            return load_english_corpus_data(integration, branch, config)

        return loader
    elif language == "ES":

        def loader(
            integration: CorpusIntegration, branch: str, config: DataPreparationConfig
        ) -> Result[List[Dict[str, Any]], str]:
            return load_spanish_corpus_data(integration, branch, config)

        return loader
    else:

        def loader(
            integration: CorpusIntegration, branch: str, config: DataPreparationConfig
        ) -> Result[List[Dict[str, Any]], str]:
            return Err(f"Unsupported language: {language}")

        return loader


def create_document_preprocessor() -> Callable[[List[Dict[str, Any]], DataPreparationConfig], List[Dict[str, Any]]]:
    """Create document preprocessor - Factory function"""

    def preprocessor(documents: List[Dict[str, Any]], config: DataPreparationConfig) -> List[Dict[str, Any]]:
        return [preprocess_document(doc, config) for doc in documents]

    return preprocessor


def create_dataset_filter() -> Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]:
    """Create dataset filter - Factory function"""

    def filter_func(documents: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        return filter_documents_by_criteria(documents, criteria)

    return filter_func


def create_data_preparation_pipeline(
    language: str,
) -> Callable[[CorpusIntegration, str, DataPreparationConfig], Result[PreparedDataset, str]]:
    """Create complete data preparation pipeline - Factory function"""
    loader = create_language_specific_loader(language)
    preprocessor = create_document_preprocessor()
    filter_func = create_dataset_filter()

    def pipeline(
        integration: CorpusIntegration, branch_name: str, config: DataPreparationConfig
    ) -> Result[PreparedDataset, str]:
        # Load raw documents
        load_result = loader(integration, branch_name, config)
        if load_result.is_err():
            return Err(load_result.unwrap_err())

        documents = load_result.unwrap()

        # Apply filtering
        if config.filtering_criteria:
            documents = filter_func(documents, config.filtering_criteria)

        # Limit documents
        documents = documents[: config.max_documents]

        # Preprocess documents
        processed_documents = preprocessor(documents, config)

        # Create prepared dataset
        dataset = create_prepared_dataset(
            language=language,
            branch_name=branch_name,
            documents=processed_documents,
            metadata={
                "preparation_config": config.config_id,
                "corpus_integration": integration.metadata,
                "preprocessing_steps": config.preprocessing_steps,
            },
        )

        return Ok(dataset)

    return pipeline


# ============================================================================
# Data Preparation Context Management
# ============================================================================


def create_data_preparation_context(corpus_root: Path, language_separation: bool = True) -> DataPreparationContext:
    """Create data preparation context - Pure function"""
    # Create corpus integration
    corpus_integration = create_corpus_integration(corpus_root=corpus_root, language_separation=language_separation)

    # Create default config
    default_config = create_data_preparation_config(
        language="EN",
        corpus_type="general",
        metadata={"context_created": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )

    return DataPreparationContext(
        config=default_config,
        corpus_integration=corpus_integration,
        prepared_datasets={},
        logger=None,
    )


def register_prepared_dataset(context: DataPreparationContext, dataset: PreparedDataset) -> DataPreparationContext:
    """Register prepared dataset in context - Pure function"""
    new_datasets = {**context.prepared_datasets, dataset.dataset_id: dataset}

    return DataPreparationContext(
        config=context.config,
        corpus_integration=context.corpus_integration,
        prepared_datasets=new_datasets,
        logger=context.logger,
    )


# ============================================================================
# Legacy Compatibility Functions
# ============================================================================


def prepare_training_corpus_functional(
    language: str, branch_name: str, corpus_root: str = ".", max_documents: int = 1000
) -> Dict[str, Any]:
    """Prepare training data using functional approach - Legacy compatibility"""
    try:
        # Create corpus integration
        corpus_integration = create_corpus_integration(corpus_root=Path(corpus_root), language_separation=True)

        # Create preparation config
        config = create_data_preparation_config(
            language=language,
            corpus_type="domain_specific",
            max_documents=max_documents,
            metadata={"legacy_call": True},
        )

        # Create preparation pipeline
        pipeline = create_data_preparation_pipeline(language)

        # Execute pipeline
        result = pipeline(corpus_integration, branch_name, config)

        if result.is_ok():
            dataset = result.unwrap()
            return {
                "success": True,
                "dataset_id": dataset.dataset_id,
                "language": dataset.language,
                "branch_name": dataset.branch_name,
                "total_documents": len(dataset.documents),
                "total_chunks": dataset.total_chunks,
                "vocabulary_size": dataset.vocabulary_size,
                "statistics": dataset.statistics,
                "metadata": dataset.metadata,
                "created_at": dataset.created_at,
            }
        else:
            return {
                "success": False,
                "error": result.unwrap_err(),
                "language": language,
                "branch_name": branch_name,
                "fallback_documents": 0,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "language": language,
            "branch_name": branch_name,
            "fallback_documents": 0,
        }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data types
    "DataPreparationConfig",
    "PreparedDataset",
    "CorpusIntegration",
    "DataPreparationContext",
    # Pure functions
    "create_data_preparation_config",
    "create_prepared_dataset",
    "create_corpus_integration",
    "calculate_dataset_statistics",
    "validate_language_separation",
    "filter_documents_by_criteria",
    "chunk_document_content",
    "preprocess_document",
    # Language-specific functions
    "load_english_corpus_data",
    "load_spanish_corpus_data",
    # Factory functions
    "create_language_specific_loader",
    "create_document_preprocessor",
    "create_dataset_filter",
    "create_data_preparation_pipeline",
    # Context management
    "create_data_preparation_context",
    "register_prepared_dataset",
    # Legacy compatibility
    "prepare_training_corpus_functional",
]
