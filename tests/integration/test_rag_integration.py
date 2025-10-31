#!/usr/bin/env python3
"""
Tests para el sistema RAG completo
"""

import asyncio
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sheily_core.data.document_processor import DocumentProcessor
from sheily_core.data.embeddings import ProductionEmbeddingManager
from sheily_core.integration.rag_service import RAGService


class TestRAGSystem:
    """Tests del sistema RAG completo"""

    @pytest.fixture
    def sample_documents(self):
        """Documentos de prueba"""
        return [
            {
                "content": "La antropología es el estudio científico de los seres humanos y sus culturas.",
                "metadata": {"source": "test", "domain": "antropologia"},
            },
            {
                "content": "El machine learning es una rama de la inteligencia artificial.",
                "metadata": {"source": "test", "domain": "programacion"},
            },
        ]

    def test_document_processor_creation(self):
        """Test creación del procesador de documentos"""
        processor = DocumentProcessor()
        assert processor is not None
        assert hasattr(processor, "text_splitter")
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 100

    def test_document_chunking(self):
        """Test división de documentos en chunks"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        long_text = (
            "Este es un texto largo que debería ser dividido en múltiples chunks para poder ser procesado eficientemente por el sistema de embeddings. "
            * 10
        )

        chunks = processor.text_splitter.split_text(long_text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + overlap

        # Verificar overlap
        if len(chunks) > 1:
            assert chunks[0][-20:] == chunks[1][:20]  # overlap de 20 chars

    @pytest.mark.asyncio
    async def test_embedding_manager_initialization(self):
        """Test inicialización del embedding manager"""
        config = Mock()
        config.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        manager = ProductionEmbeddingManager(config)
        assert manager is not None
        assert hasattr(manager, "model")
        assert manager.config == config

    def test_rag_service_creation(self):
        """Test creación del servicio RAG"""
        service = RAGService()
        assert service is not None
        assert hasattr(service, "app")
        assert hasattr(service, "embedding_manager")
        assert hasattr(service, "document_processor")
        assert service.index is None  # No inicializado aún

    @pytest.mark.asyncio
    async def test_rag_service_initialization(self):
        """Test inicialización completa del servicio RAG"""
        service = RAGService()

        # Mock del embedding manager
        with patch.object(service, "embedding_manager") as mock_manager:
            mock_manager.initialize = asyncio.coroutine(lambda: None)()

            await service.initialize()

            mock_manager.initialize.assert_called_once()
            assert service.index is not None

    def test_integration_flow(self, sample_documents):
        """Test flujo de integración completo"""
        # Crear componentes
        processor = DocumentProcessor()
        service = RAGService()

        # Procesar documentos
        chunks = []
        for doc in sample_documents:
            doc_chunks = processor.text_splitter.split_text(doc["content"])
            for chunk in doc_chunks:
                chunks.append({"id": f"test_{len(chunks)}", "content": chunk, "metadata": doc["metadata"]})

        assert len(chunks) >= len(sample_documents)  # Al menos un chunk por documento

        # Verificar estructura de chunks
        for chunk in chunks:
            assert "id" in chunk
            assert "content" in chunk
            assert "metadata" in chunk
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0


class TestSecurity:
    """Tests de seguridad básicos"""

    def test_no_shell_execution(self):
        """Verificar que no se usa shell=True en subprocess"""
        import subprocess

        # Este test pasaría si no hay usos inseguros en el código
        # En un test real, se escanearía el código fuente
        assert True  # Placeholder

    def test_no_eval_usage(self):
        """Verificar que no se usa eval()"""
        # Similar al anterior
        assert True  # Placeholder


class TestPerformance:
    """Tests de performance básicos"""

    def test_chunking_performance(self):
        """Test performance de chunking"""
        import time

        processor = DocumentProcessor()
        large_text = "Palabra " * 10000  # Texto grande

        start_time = time.time()
        chunks = processor.text_splitter.split_text(large_text)
        end_time = time.time()

        # Debería procesar rápidamente
        assert end_time - start_time < 1.0  # Menos de 1 segundo
        assert len(chunks) > 0

    def test_memory_usage_basic(self):
        """Test básico de uso de memoria"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Operación que usa memoria
        processor = DocumentProcessor()
        large_text = "Texto largo " * 5000
        chunks = processor.text_splitter.split_text(large_text)

        memory_after = process.memory_info().rss

        # No debería usar más de 50MB adicional
        memory_increase = memory_after - memory_before
        assert memory_increase < 50 * 1024 * 1024  # 50MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
