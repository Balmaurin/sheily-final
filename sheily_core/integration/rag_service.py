#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Service - Microservicio de Recuperación Aumentada por Generación
===================================================================

Servicio FastAPI que expone el sistema RAG completo:
- Procesamiento de documentos con RecursiveCharacterTextSplitter
- Generación de embeddings con sentence-transformers multilingüe
- Almacenamiento y búsqueda en ChromaDB

Endpoints:
- POST /process_corpus: Procesar y indexar corpus completo
- POST /add_document: Agregar documento individual
- POST /search: Buscar documentos similares
- GET /health: Estado del servicio
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sheily_core.data.document_processor import DocumentProcessor, process_all_branches_corpus
from sheily_core.data.embeddings import ProductionEmbeddingManager, EmbeddingConfig


# Modelos Pydantic para requests/responses
class ProcessCorpusRequest(BaseModel):
    corpus_path: str
    domains: Optional[List[str]] = None
    collection_name: str = "sheily_rag"


class AddDocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict] = None
    collection_name: str = "sheily_rag"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: str = "sheily_rag"


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: Dict
    score: float


class RAGService:
    """
    Servicio RAG que integra DocumentProcessor, EmbeddingManager y FAISS
    """

    def __init__(self):
        self.app = FastAPI(title="Sheily RAG Service", version="1.0.0")

        # FAISS index y almacenamiento de datos
        self.index = None
        self.index_ids = []  # Lista de IDs correspondientes
        self.index_metadatas = []  # Lista de metadatas correspondientes
        self.index_contents = []  # Lista de contenidos correspondientes

        self.embedding_manager = None
        self.document_processor = DocumentProcessor()

        # Inicializar rutas
        self._setup_routes()

    async def initialize(self):
        """Inicializar el embedding manager"""
        config = EmbeddingConfig()
        self.embedding_manager = ProductionEmbeddingManager(config)
        await self.embedding_manager.initialize()

        # Inicializar FAISS index (dimensión se determinará con el primer embedding)
        self.index = None

        print("✅ RAG Service inicializado con FAISS")

    def _add_to_faiss_index(self, embeddings: np.ndarray, ids: List[str], metadatas: List[Dict], contents: List[str]):
        """Agregar embeddings al índice FAISS"""
        if self.index is None:
            # Crear índice FAISS con dimensión correcta
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Agregar al índice
        self.index.add(embeddings.astype(np.float32))

        # Agregar datos correspondientes
        self.index_ids.extend(ids)
        self.index_metadatas.extend(metadatas)
        self.index_contents.extend(contents)

    def _setup_routes(self):
        """Configurar endpoints FastAPI"""

        @self.app.post("/process_corpus")
        async def process_corpus(request: ProcessCorpusRequest):
            """Procesar e indexar un corpus completo"""
            try:
                # Procesar documentos
                chunks = process_all_branches_corpus(request.corpus_path, request.domains)

                if not chunks:
                    raise HTTPException(status_code=404, detail="No se encontraron documentos para procesar")

                # Generar embeddings para todos los chunks
                print(f"Generando embeddings para {len(chunks)} chunks...")
                contents = [chunk["content"] for chunk in chunks]
                embeddings = await self.embedding_manager.encode_text(contents)

                # Agregar al índice FAISS
                self._add_to_faiss_index(
                    embeddings=np.array(embeddings),
                    ids=[chunk["id"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks],
                    contents=contents
                )

                return {
                    "status": "success",
                    "chunks_processed": len(chunks),
                    "total_indexed": len(self.index_ids)
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error procesando corpus: {str(e)}")

        @self.app.post("/add_document")
        async def add_document(request: AddDocumentRequest):
            """Agregar un documento individual"""
            try:
                # Crear chunks del documento
                chunks = self.document_processor.text_splitter.split_text(request.content)

                if not chunks:
                    raise HTTPException(status_code=400, detail="Contenido vacío o inválido")

                # Generar embeddings
                embeddings = await self.embedding_manager.encode_text(chunks)

                # Preparar metadata
                base_metadata = request.metadata or {}
                base_metadata["source"] = "manual_add"

                # Crear IDs únicos
                import time
                timestamp = int(time.time())
                ids = [f"manual_{timestamp}_{i}" for i in range(len(chunks))]

                # Preparar metadatas para cada chunk
                metadatas = []
                for i, chunk in enumerate(chunks):
                    metadata = base_metadata.copy()
                    metadata["chunk_index"] = i
                    metadata["total_chunks"] = len(chunks)
                    metadatas.append(metadata)

                # Agregar al índice FAISS
                self._add_to_faiss_index(
                    embeddings=np.array(embeddings),
                    ids=ids,
                    metadatas=metadatas,
                    contents=chunks
                )

                return {
                    "status": "success",
                    "chunks_added": len(chunks),
                    "total_indexed": len(self.index_ids)
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error agregando documento: {str(e)}")

        @self.app.post("/search")
        async def search(request: SearchRequest) -> List[SearchResult]:
            """Buscar documentos similares"""
            try:
                if self.index is None or self.index.ntotal == 0:
                    return []

                # Generar embedding de la query
                query_embedding = await self.embedding_manager.encode_text(request.query)
                query_embedding = np.array([query_embedding]).astype(np.float32)

                # Buscar en FAISS
                scores, indices = self.index.search(query_embedding, request.top_k)

                # Formatear resultados
                search_results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.index_ids):  # Validar índice
                        search_results.append(SearchResult(
                            id=self.index_ids[idx],
                            content=self.index_contents[idx],
                            metadata=self.index_metadatas[idx],
                            score=float(score)  # FAISS devuelve similitud coseno normalizada
                        ))

                return search_results

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

        @self.app.get("/health")
        async def health():
            """Estado del servicio"""
            return {
                "status": "healthy",
                "embedding_model": self.embedding_manager.config.model_name if self.embedding_manager else None,
                "vector_store": "FAISS",
                "total_indexed": len(self.index_ids) if self.index_ids else 0,
                "index_dimension": self.index.d if self.index else None
            }


# Instancia global del servicio
rag_service = RAGService()


async def main():
    """Punto de entrada principal"""
    try:
        print("🔄 Inicializando RAG Service...")
        await rag_service.initialize()
        print("✅ RAG Service listo para recibir conexiones")

        # Ejecutar servidor
        import uvicorn
        config = uvicorn.Config(
            rag_service.app,
            host="0.0.0.0",
            port=8002,  # Cambiado de 8001 a 8002
            log_level="info"
        )
        server = uvicorn.Server(config)

        print("🌐 Iniciando servidor HTTP en http://0.0.0.0:8002")
        await server.serve()

    except Exception as e:
        print(f"❌ Error fatal en RAG Service: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
