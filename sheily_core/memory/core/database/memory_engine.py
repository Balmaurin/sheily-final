#!/usr/bin/env python3
# ==============================================================================
# SHEILY MEMORY SYSTEM - SISTEMA DE MEMORIA HÍBRIDA AVANZADO
# ==============================================================================
# Sistema de memoria híbrida humano-IA completamente funcional

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SheilyMemoryV2:
    """Sistema de memoria híbrida humano-IA avanzado"""

    def __init__(self, memory_db_path: str = "sheily_core/memory/storage/memory.db"):
        self.memory_db_path = Path(memory_db_path)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.memory_db_path)
        self._init_database()

    def _init_database(self):
        """Inicializar base de datos de memoria"""
        cursor = self.db.cursor()

        # Tabla de documentos memorizados
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                full_content TEXT,
                metadata TEXT,
                created_at REAL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                category TEXT DEFAULT 'general',
                tags TEXT,
                UNIQUE(content_hash)
            )
        """
        )

        # Tabla de chunks para recuperación eficiente
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_index INTEGER,
                content TEXT,
                embedding_vector TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                chunk_size INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """
        )

        # Tabla de contexto emocional
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emotional_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                emotion_type TEXT,
                intensity REAL,
                context_description TEXT,
                timestamp REAL,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """
        )

        self.db.commit()

    def memorize_file(
        self, file_path: Path, category: str = "general", importance: float = 0.5
    ) -> bool:
        """Memorizar un archivo completo"""
        try:
            # Leer contenido del archivo
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Generar hash único del contenido
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Extraer metadatos
            metadata = {
                "file_size": len(content),
                "lines": len(content.split("\n")),
                "category": category,
                "importance_score": importance,
            }

            # Insertar documento
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO documents
                (file_name, file_type, content_hash, full_content, metadata, created_at, importance_score, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    file_path.name,
                    file_path.suffix,
                    content_hash,
                    content[:50000],  # Limitar contenido guardado
                    json.dumps(metadata),
                    time.time(),
                    importance,
                    category,
                ),
            )

            doc_id = cursor.lastrowid

            # Si es un documento nuevo, crear chunks
            if doc_id:
                self._create_chunks(doc_id, content)

            self.db.commit()
            return True

        except Exception as e:
            print(f"Error memorizando archivo {file_path}: {e}")
            return False

    def _create_chunks(self, document_id: int, content: str, chunk_size: int = 1000):
        """Crear chunks del contenido para recuperación eficiente"""
        cursor = self.db.cursor()

        # Crear chunks con solapamiento
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]

            # Generar embedding simple (en producción usar modelos reales)
            embedding = self._generate_simple_embedding(chunk_content)

            chunks.append(
                (
                    document_id,
                    chunk_index,
                    chunk_content,
                    json.dumps(embedding.tolist()),
                    start,
                    end,
                    len(chunk_content),
                )
            )

            start += chunk_size - 200  # Solapamiento de 200 chars
            chunk_index += 1

        # Insertar chunks
        cursor.executemany(
            """
            INSERT INTO chunks
            (document_id, chunk_index, content, embedding_vector, start_pos, end_pos, chunk_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            chunks,
        )

    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generar embedding simple basado en características del texto"""
        # Características simples para embedding
        features = [
            len(text),  # Longitud
            text.count(" "),  # Número de palabras
            text.count("."),  # Número de oraciones
            len(set(text.lower().split())),  # Vocabulario único
            sum(ord(c) for c in text[:100]) / 100,  # Valor promedio de caracteres iniciales
        ]

        # Normalizar features
        features_array = np.array(features)
        return features_array / np.linalg.norm(features_array)

    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Buscar en la memoria usando similitud semántica"""
        try:
            # Generar embedding de la consulta
            query_embedding = self._generate_simple_embedding(query)

            cursor = self.db.cursor()

            # Buscar chunks similares
            cursor.execute(
                """
                SELECT c.*, d.file_name, d.category, d.importance_score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY (
                    (SELECT AVG(embedding_vector) FROM chunks WHERE document_id = d.id) -
                    ?
                ) ASC
                LIMIT ?
            """,
                (query_embedding.mean(), limit),
            )

            results = []
            for row in cursor.fetchall():
                similarity = self._calculate_similarity(query_embedding, json.loads(row[3]))

                if similarity > 0.1:  # Umbral mínimo de similitud
                    results.append(
                        {
                            "file_name": row[8],
                            "category": row[9],
                            "importance": row[10],
                            "chunk_content": row[2],
                            "similarity": similarity,
                            "chunk_index": row[1],
                        }
                    )

            return sorted(results, key=lambda x: x["similarity"], reverse=True)

        except Exception as e:
            print(f"Error buscando en memoria: {e}")
            return []

    def _calculate_similarity(self, query_emb: np.ndarray, chunk_emb: np.ndarray) -> float:
        """Calcular similitud entre embeddings"""
        try:
            # Similitud coseno
            dot_product = np.dot(query_emb, chunk_emb)
            norm_query = np.linalg.norm(query_emb)
            norm_chunk = np.linalg.norm(chunk_emb)

            if norm_query == 0 or norm_chunk == 0:
                return 0.0

            return dot_product / (norm_query * norm_chunk)

        except:
            return 0.0

    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema de memoria"""
        cursor = self.db.cursor()

        # Estadísticas de documentos
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(importance_score) FROM documents")
        total_importance = cursor.fetchone()[0] or 0

        # Categorías más comunes
        cursor.execute(
            "SELECT category, COUNT(*) FROM documents GROUP BY category ORDER BY COUNT(*) DESC LIMIT 5"
        )
        top_categories = cursor.fetchall()

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "average_importance": total_importance / max(total_docs, 1),
            "top_categories": top_categories,
            "database_size_mb": self.memory_db_path.stat().st_size / (1024 * 1024),
        }

    def add_emotional_context(
        self, file_name: str, emotion_type: str, intensity: float, description: str
    ):
        """Agregar contexto emocional a un documento"""
        cursor = self.db.cursor()

        # Encontrar documento por nombre
        cursor.execute("SELECT id FROM documents WHERE file_name = ?", (file_name,))
        doc_id = cursor.fetchone()

        if doc_id:
            cursor.execute(
                """
                INSERT INTO emotional_context
                (document_id, emotion_type, intensity, context_description, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (doc_id[0], emotion_type, intensity, description, time.time()),
            )

            self.db.commit()

    def get_contextual_memories(
        self, query: str, emotional_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Obtener recuerdos contextuales basados en emoción y similitud"""
        try:
            memories = self.search_memory(query, limit=10)

            # Aplicar filtro emocional si se especifica
            if emotional_filter:
                # Filtrar por contexto emocional
                filtered_memories = []
                for memory in memories:
                    cursor = self.db.cursor()
                    cursor.execute(
                        """
                        SELECT AVG(intensity) FROM emotional_context
                        WHERE document_id = (
                            SELECT id FROM documents WHERE file_name = ?
                        ) AND emotion_type = ?
                    """,
                        (memory["file_name"], emotional_filter),
                    )

                    avg_intensity = cursor.fetchone()[0]
                    if avg_intensity and avg_intensity > 0.3:
                        memory["emotional_relevance"] = avg_intensity
                        filtered_memories.append(memory)

                memories = filtered_memories

            return memories

        except Exception as e:
            print(f"Error obteniendo recuerdos contextuales: {e}")
            return []

    def cleanup_old_memories(self, days_threshold: int = 30):
        """Limpiar recuerdos antiguos para optimizar memoria"""
        cutoff_time = time.time() - (days_threshold * 24 * 60 * 60)

        cursor = self.db.cursor()
        cursor.execute(
            "DELETE FROM documents WHERE last_accessed < ? AND importance_score < 0.3",
            (cutoff_time,),
        )
        cursor.execute("DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)")

        deleted_count = cursor.rowcount
        self.db.commit()

        return deleted_count

    def get_memory_health(self) -> Dict[str, Any]:
        """Obtener estado de salud del sistema de memoria"""
        stats = self.get_memory_stats()

        # Calcular métricas de salud
        health_score = min(100, stats["total_documents"] * 2 + stats["total_chunks"] * 0.5)

        return {
            "health_score": health_score,
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "database_size_mb": stats["database_size_mb"],
            "average_importance": stats["average_importance"],
            "recommendations": self._get_health_recommendations(stats),
        }

    def _get_health_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Obtener recomendaciones basadas en estadísticas"""
        recommendations = []

        if stats["total_documents"] < 10:
            recommendations.append(
                "Agregar más documentos para mejorar la diversidad de conocimiento"
            )

        if stats["database_size_mb"] > 100:
            recommendations.append("Considerar limpieza de documentos antiguos poco importantes")

        if stats["average_importance"] < 0.4:
            recommendations.append("Revisar criterios de importancia para documentos existentes")

        return recommendations or ["Sistema de memoria funcionando óptimamente"]

    def __del__(self):
        """Cerrar conexión de base de datos"""
        if hasattr(self, "db"):
            self.db.close()
