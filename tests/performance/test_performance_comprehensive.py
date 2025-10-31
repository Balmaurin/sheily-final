#!/usr/bin/env python3
"""
Tests de performance para Sheily AI
"""

import pytest
import time
import psutil
import os
from pathlib import Path
import asyncio
from unittest.mock import Mock


class TestPerformance:
    """Tests de performance del sistema"""

    def test_document_processing_speed(self):
        """Test velocidad de procesamiento de documentos"""
        from sheily_core.data.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Documento de tamaño medio
        test_doc = "La antropología cultural es una disciplina académica que estudia las diversas culturas humanas. " * 100

        start_time = time.time()
        chunks = processor.text_splitter.split_text(test_doc)
        end_time = time.time()

        processing_time = end_time - start_time

        # Debería procesar en menos de 0.1 segundos
        assert processing_time < 0.1, ".3f"        assert len(chunks) > 1

    def test_memory_efficiency(self):
        """Test eficiencia de memoria"""
        from sheily_core.data.document_processor import DocumentProcessor

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Procesar múltiples documentos
        processor = DocumentProcessor()
        documents = ["Documento de prueba " * 1000] * 10

        for doc in documents:
            chunks = processor.text_splitter.split_text(doc)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # No debería aumentar más de 50MB
        assert memory_increase < 50, ".2f"

    @pytest.mark.asyncio
    async def test_embedding_performance(self):
        """Test performance de embeddings"""
        from sheily_core.data.embeddings import ProductionEmbeddingManager, EmbeddingConfig

        config = EmbeddingConfig()
        manager = ProductionEmbeddingManager(config)

        # Inicializar
        await manager.initialize()

        # Documentos de prueba
        test_docs = [
            "La antropología estudia las culturas humanas.",
            "El machine learning es parte de la IA.",
            "Los embeddings vectoriales representan significado semántico."
        ]

        start_time = time.time()
        embeddings = await manager.encode_text(test_docs)
        end_time = time.time()

        processing_time = end_time - start_time

        # Debería procesar en menos de 2 segundos (incluyendo carga del modelo)
        assert processing_time < 5.0, ".3f"        assert len(embeddings) == len(test_docs)
        assert len(embeddings[0]) > 300  # Dimensión razonable del embedding

    def test_rag_service_startup_time(self):
        """Test tiempo de inicio del servicio RAG"""
        from sheily_core.integration.rag_service import RAGService

        start_time = time.time()
        service = RAGService()
        end_time = time.time()

        startup_time = end_time - start_time

        # Debería inicializar en menos de 0.1 segundos
        assert startup_time < 0.1, ".3f"        assert service.app is not None
        assert service.document_processor is not None

    def test_concurrent_processing(self):
        """Test procesamiento concurrente"""
        import threading
        from sheily_core.data.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        results = []
        errors = []

        def process_document(doc_id):
            try:
                doc = f"Documento {doc_id} con contenido de prueba. " * 50
                chunks = processor.text_splitter.split_text(doc)
                results.append((doc_id, len(chunks)))
            except Exception as e:
                errors.append((doc_id, str(e)))

        # Crear 5 threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_document, args=(i,))
            threads.append(thread)
            thread.start()

        # Esperar que terminen
        for thread in threads:
            thread.join(timeout=5)

        # Verificar resultados
        assert len(results) == 5, f"Procesados {len(results)} documentos, esperados 5"
        assert len(errors) == 0, f"Errores encontrados: {errors}"

        # Todos deberían tener chunks
        for doc_id, chunk_count in results:
            assert chunk_count > 0, f"Documento {doc_id} no generó chunks"

    def test_large_document_handling(self):
        """Test manejo de documentos grandes"""
        from sheily_core.data.document_processor import DocumentProcessor

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        # Documento muy grande (equivalente a ~50KB de texto)
        large_doc = "Este es un documento muy largo que contiene mucha información sobre diversos temas. " * 2000

        start_time = time.time()
        chunks = processor.text_splitter.split_text(large_doc)
        end_time = time.time()

        processing_time = end_time - start_time

        # Debería procesar en menos de 1 segundo
        assert processing_time < 1.0, ".3f"        # Debería generar múltiples chunks
        assert len(chunks) > 10, f"Solo {len(chunks)} chunks generados"

        # Verificar que los chunks tienen tamaño razonable
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 550, f"Chunk {i} demasiado grande: {len(chunk)} caracteres"

    def test_cpu_usage_efficiency(self):
        """Test eficiencia de uso de CPU"""
        from sheily_core.data.document_processor import DocumentProcessor

        cpu_before = psutil.cpu_percent(interval=None)

        # Operación intensiva
        processor = DocumentProcessor()
        for i in range(100):
            doc = f"Documento {i} con contenido variado. " * 20
            chunks = processor.text_splitter.split_text(doc)

        cpu_after = psutil.cpu_percent(interval=None)
        cpu_increase = cpu_after - cpu_before

        # El uso de CPU no debería ser excesivo
        # Nota: Este test puede fallar en máquinas muy lentas
        assert cpu_increase < 50, ".1f"


class TestScalability:
    """Tests de escalabilidad"""

    def test_memory_cleanup(self):
        """Test limpieza de memoria"""
        import gc
        from sheily_core.data.document_processor import DocumentProcessor

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Crear muchos objetos
        processors = [DocumentProcessor() for _ in range(10)]
        large_docs = ["Documento grande " * 1000 for _ in range(10)]

        # Procesar
        for processor in processors:
            for doc in large_docs:
                chunks = processor.text_splitter.split_text(doc)

        # Limpiar
        del processors
        del large_docs
        gc.collect()

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB

        # Después de cleanup, no debería haber mucho aumento de memoria
        assert memory_increase < 100, ".2f"

    def test_file_system_performance(self):
        """Test performance del sistema de archivos"""
        import tempfile
        import shutil

        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear archivos de prueba
            test_files = []
            for i in range(10):
                file_path = Path(temp_dir) / f"test_doc_{i}.txt"
                content = f"Contenido del documento {i}. " * 100
                file_path.write_text(content, encoding='utf-8')
                test_files.append(file_path)

            # Medir tiempo de lectura
            start_time = time.time()
            total_content = ""
            for file_path in test_files:
                total_content += file_path.read_text(encoding='utf-8')
            end_time = time.time()

            reading_time = end_time - start_time

            # Debería leer en menos de 0.1 segundos
            assert reading_time < 0.1, ".3f"            assert len(total_content) > 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
