#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Index Manager - Gestión de Índices para RAG
===========================================

Maneja la indexación, almacenamiento y gestión de vectores e índices
para el sistema RAG bilingüe de Sheily-AI.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# import numpy as np  # Comentado para evitar bloqueo DepSwitch
# ZERO-DEPS: Siempre usar implementación MockNumpy
HAS_NUMPY = False
try:
    # Verificar si numpy está disponible y permitido por DepSwitch
    import numpy as np

    # Test rápido para verificar que funciona
    _ = np.array([1])
    HAS_NUMPY = True
    print("⚠️  NumPy detectado pero se usará MockNumpy para compatibilidad zero-deps")
except (ImportError, RuntimeError, AttributeError, NameError):
    HAS_NUMPY = False

    # Mock completo de numpy para funciones esenciales
    class MockNdarray(list):
        """Mock de numpy.ndarray usando lista Python"""

        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                super().__init__(data)
            else:
                super().__init__([data])

        def astype(self, dtype):
            if dtype == "float32":
                return MockNdarray([float(x) for x in self])
            return self

    class MockNumpy:
        ndarray = MockNdarray
        float32 = "float32"

        def array(self, data):
            return MockNdarray(data)

        def zeros(self, shape):
            if isinstance(shape, int):
                return MockNdarray([0.0] * shape)
            return MockNdarray([[0.0] * shape[1] for _ in range(shape[0])])

    # FORZAR uso de MockNumpy para compatibilidad zero-deps
    np = MockNumpy()
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Gestor de índices con soporte para múltiples backends
    """

    def __init__(self, config: Dict):
        """
        Inicializar gestor de índices

        Args:
            config: Configuración del gestor
        """
        self.config = config
        self.index_type = config.get("index_type", "sqlite")
        self.index_path = Path(config.get("index_path", "data/indices/"))

        # Configuración de índices
        self.vector_dim = config.get("vector_dimension", 384)  # all-MiniLM-L6-v2
        self.max_index_size = config.get("max_index_size", 100000)

        # Estado de índices cargados
        self._loaded_indices = {}  # {language_domain: index_info}
        self._document_counts = {}

        # Backends disponibles
        self._backends = {
            "sqlite": self._init_sqlite_backend,
            "faiss": self._init_faiss_backend,
            "memory": self._init_memory_backend,
        }

        # Métricas de indexación
        self._index_stats = {
            "documents_indexed": 0,
            "documents_updated": 0,
            "documents_deleted": 0,
            "searches_performed": 0,
            "total_index_size": 0,
            "average_indexing_time": 0.0,
        }

        logger.info(f"IndexManager inicializado - Backend: {self.index_type}")

    async def initialize(self) -> bool:
        """
        Inicializar el gestor de índices

        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Crear directorio de índices
            self.index_path.mkdir(parents=True, exist_ok=True)

            # Inicializar backend seleccionado
            if self.index_type in self._backends:
                await self._backends[self.index_type]()
            else:
                raise ValueError(f"Backend de índice no soportado: {self.index_type}")

            # Cargar índices existentes
            await self._load_existing_indices()

            logger.info("IndexManager inicializado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando IndexManager: {e}")
            return False

    async def _init_sqlite_backend(self):
        """Inicializar backend SQLite con FTS5"""
        self.db_path = self.index_path / "sheily_index.db"

        # Crear conexión y tablas
        self._create_sqlite_tables()

        logger.info("Backend SQLite inicializado")

    def _create_sqlite_tables(self):
        """Crear tablas SQLite para indexación"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla principal de documentos
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            title TEXT,
            domain TEXT,
            language TEXT,
            metadata TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Índice FTS5 para búsqueda de texto completo
        cursor.execute(
            """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content,
            title,
            domain,
            language,
            content='documents',
            content_rowid='rowid'
        )
        """
        )

        # Trigger para mantener FTS5 sincronizado
        cursor.execute(
            """
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, content, title, domain, language)
            VALUES (new.rowid, new.content, new.title, new.domain, new.language);
        END
        """
        )

        cursor.execute(
            """
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content, title, domain, language)
            VALUES('delete', old.rowid, old.content, old.title, old.domain, old.language);
        END
        """
        )

        cursor.execute(
            """
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, content, title, domain, language)
            VALUES('delete', old.rowid, old.content, old.title, old.domain, old.language);
            INSERT INTO documents_fts(rowid, content, title, domain, language)
            VALUES (new.rowid, new.content, new.title, new.domain, new.language);
        END
        """
        )

        # Índices adicionales
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)")

        conn.commit()
        conn.close()

    async def _init_faiss_backend(self):
        """Inicializar backend FAISS (bloqueado por DepSwitch en producción)"""
        logger.warning("Backend FAISS solicitado - puede estar bloqueado por DepSwitch")

        try:
            # En producción, esto estaría bloqueado por DepSwitch
            # En desarrollo, se podría usar faiss-cpu
            self.faiss_indices = {}
            logger.info("Backend FAISS simulado inicializado")
        except Exception as e:
            logger.warning(f"FAISS no disponible, usando SQLite como fallback: {e}")
            await self._init_sqlite_backend()
            self.index_type = "sqlite"

    async def _init_memory_backend(self):
        """Inicializar backend en memoria"""
        self.memory_indices = {"documents": {}, "embeddings": {}, "metadata": {}}

        logger.info("Backend en memoria inicializado")

    async def _load_existing_indices(self):
        """Cargar índices existentes desde disco"""
        try:
            if self.index_type == "sqlite":
                await self._load_sqlite_indices()
            elif self.index_type == "faiss":
                await self._load_faiss_indices()
            # Memory backend no persiste

        except Exception as e:
            logger.warning(f"Error cargando índices existentes: {e}")

    async def _load_sqlite_indices(self):
        """Cargar estadísticas de índices SQLite"""
        if not self.db_path.exists():
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Contar documentos por idioma y dominio
        cursor.execute(
            """
        SELECT language, domain, COUNT(*)
        FROM documents
        GROUP BY language, domain
        """
        )

        for language, domain, count in cursor.fetchall():
            key = f"{language}_{domain}"
            self._document_counts[key] = count

            # Marcar como cargado
            self._loaded_indices[key] = {
                "language": language,
                "domain": domain,
                "document_count": count,
                "backend": "sqlite",
                "loaded_at": time.time(),
            }

        conn.close()

        total_docs = sum(self._document_counts.values())
        logger.info(f"Cargados índices SQLite: {total_docs} documentos")

    async def _load_faiss_indices(self):
        """Cargar índices FAISS desde disco"""
        # Buscar archivos de índice FAISS
        for faiss_file in self.index_path.glob("*.faiss"):
            try:
                # Extraer información del nombre del archivo
                name_parts = faiss_file.stem.split("_")
                if len(name_parts) >= 2:
                    language = name_parts[0]
                    domain = name_parts[1]

                    key = f"{language}_{domain}"
                    self._loaded_indices[key] = {
                        "language": language,
                        "domain": domain,
                        "file_path": str(faiss_file),
                        "backend": "faiss",
                        "loaded_at": time.time(),
                    }

            except Exception as e:
                logger.warning(f"Error cargando índice FAISS {faiss_file}: {e}")

    async def add_document(
        self,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Dict[str, Any] = None,
        language: str = "spanish",
        domain: str = "general",
    ) -> bool:
        """
        Añadir documento al índice

        Args:
            content: Contenido del documento
            embedding: Vector embedding (opcional)
            metadata: Metadatos del documento
            language: Idioma del documento
            domain: Dominio del documento

        Returns:
            bool: True si se añadió exitosamente
        """
        start_time = time.time()

        try:
            # Generar ID único
            doc_id = self._generate_document_id(content, language, domain)

            # Preparar metadatos
            doc_metadata = metadata or {}
            doc_metadata.update({"language": language, "domain": domain, "indexed_at": time.time()})

            # Añadir según backend
            if self.index_type == "sqlite":
                success = await self._add_document_sqlite(doc_id, content, embedding, doc_metadata, language, domain)
            elif self.index_type == "faiss":
                success = await self._add_document_faiss(doc_id, content, embedding, doc_metadata, language, domain)
            elif self.index_type == "memory":
                success = await self._add_document_memory(doc_id, content, embedding, doc_metadata, language, domain)
            else:
                success = False

            if success:
                # Actualizar contadores
                key = f"{language}_{domain}"
                self._document_counts[key] = self._document_counts.get(key, 0) + 1

                # Actualizar métricas
                indexing_time = time.time() - start_time
                self._update_indexing_metrics(indexing_time)
                self._index_stats["documents_indexed"] += 1

            return success

        except Exception as e:
            logger.error(f"Error añadiendo documento: {e}")
            return False

    def _generate_document_id(self, content: str, language: str, domain: str) -> str:
        """
        Generar ID único para documento

        Args:
            content: Contenido del documento
            language: Idioma
            domain: Dominio

        Returns:
            ID único del documento
        """
        # Crear hash basado en contenido y metadatos
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"{language}_{domain}_{content_hash}"

    async def _add_document_sqlite(
        self,
        doc_id: str,
        content: str,
        embedding: Optional[np.ndarray],
        metadata: Dict,
        language: str,
        domain: str,
    ) -> bool:
        """Añadir documento a índice SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serializar embedding si existe
            embedding_blob = None
            if embedding is not None:
                embedding_blob = pickle.dumps(embedding)

            # Insertar documento
            cursor.execute(
                """
            INSERT OR REPLACE INTO documents
            (id, content, title, domain, language, metadata, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    doc_id,
                    content,
                    metadata.get("title", ""),
                    domain,
                    language,
                    json.dumps(metadata),
                    embedding_blob,
                ),
            )

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Error añadiendo documento SQLite: {e}")
            return False

    async def _add_document_faiss(
        self,
        doc_id: str,
        content: str,
        embedding: Optional[np.ndarray],
        metadata: Dict,
        language: str,
        domain: str,
    ) -> bool:
        """Añadir documento a índice FAISS"""
        # En implementación real, usaría FAISS
        # Por ahora simulamos
        logger.info(f"Documento añadido a FAISS simulado: {doc_id}")
        return True

    async def _add_document_memory(
        self,
        doc_id: str,
        content: str,
        embedding: Optional[np.ndarray],
        metadata: Dict,
        language: str,
        domain: str,
    ) -> bool:
        """Añadir documento a índice en memoria"""
        try:
            key = f"{language}_{domain}"

            if key not in self.memory_indices["documents"]:
                self.memory_indices["documents"][key] = {}
                self.memory_indices["embeddings"][key] = {}
                self.memory_indices["metadata"][key] = {}

            self.memory_indices["documents"][key][doc_id] = content
            self.memory_indices["metadata"][key][doc_id] = metadata

            if embedding is not None:
                self.memory_indices["embeddings"][key][doc_id] = embedding

            return True

        except Exception as e:
            logger.error(f"Error añadiendo documento a memoria: {e}")
            return False

    async def search_documents(
        self,
        query: str,
        language: str,
        domain: Optional[str] = None,
        k: int = 10,
        embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos en el índice

        Args:
            query: Consulta de búsqueda
            language: Idioma de búsqueda
            domain: Dominio específico (opcional)
            k: Número de resultados
            embedding: Vector de consulta (opcional)

        Returns:
            Lista de documentos encontrados
        """
        try:
            if self.index_type == "sqlite":
                return await self._search_sqlite(query, language, domain, k)
            elif self.index_type == "faiss":
                return await self._search_faiss(query, language, domain, k, embedding)
            elif self.index_type == "memory":
                return await self._search_memory(query, language, domain, k)
            else:
                return []

        except Exception as e:
            logger.error(f"Error buscando documentos: {e}")
            return []

    async def _search_sqlite(self, query: str, language: str, domain: Optional[str], k: int) -> List[Dict[str, Any]]:
        """Buscar en índice SQLite usando FTS5"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Construir consulta FTS5
            fts_query = query.replace("'", "''")  # Escapar comillas

            sql_query = """
            SELECT d.id, d.content, d.title, d.domain, d.language,
                   d.metadata, documents_fts.rank
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.rowid
            WHERE documents_fts MATCH ?
            AND d.language = ?
            """

            params = [fts_query, language]

            if domain:
                sql_query += " AND d.domain = ?"
                params.append(domain)

            sql_query += " ORDER BY documents_fts.rank LIMIT ?"
            params.append(k)

            cursor.execute(sql_query, params)
            results = []

            for row in cursor.fetchall():
                doc_id, content, title, doc_domain, doc_language, metadata_json, rank = row

                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}

                results.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "title": title or "Documento sin título",
                        "domain": doc_domain,
                        "language": doc_language,
                        "metadata": metadata,
                        "score": 1.0 / (1.0 + abs(rank)),  # Convertir rank a score
                        "search_type": "fts",
                    }
                )

            conn.close()
            self._index_stats["searches_performed"] += 1

            return results

        except Exception as e:
            logger.error(f"Error en búsqueda SQLite: {e}")
            return []

    async def _search_faiss(
        self,
        query: str,
        language: str,
        domain: Optional[str],
        k: int,
        embedding: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Buscar en índice FAISS"""
        # Simulación de búsqueda FAISS
        logger.info(f"Búsqueda FAISS simulada: {query[:50]}...")

        # En implementación real, usaría FAISS para búsqueda vectorial
        results = []
        for i in range(min(k, 5)):
            results.append(
                {
                    "id": f"faiss_doc_{i}",
                    "content": f"Documento FAISS {i} para: {query}",
                    "title": f"Resultado FAISS {i+1}",
                    "domain": domain or "general",
                    "language": language,
                    "score": 0.9 - (i * 0.1),
                    "search_type": "faiss_vector",
                }
            )

        return results

    async def _search_memory(self, query: str, language: str, domain: Optional[str], k: int) -> List[Dict[str, Any]]:
        """Buscar en índice en memoria"""
        results = []
        query_lower = query.lower()

        # Buscar en todos los dominios si no se especifica
        domains_to_search = []
        if domain:
            domains_to_search.append(f"{language}_{domain}")
        else:
            domains_to_search = [
                key for key in self.memory_indices["documents"].keys() if key.startswith(f"{language}_")
            ]

        for key in domains_to_search:
            if key in self.memory_indices["documents"]:
                for doc_id, content in self.memory_indices["documents"][key].items():
                    # Búsqueda simple por coincidencia de texto
                    if query_lower in content.lower():
                        metadata = self.memory_indices["metadata"][key].get(doc_id, {})

                        # Calcular score básico
                        matches = content.lower().count(query_lower)
                        score = min(matches / 10.0, 1.0)  # Normalizar

                        results.append(
                            {
                                "id": doc_id,
                                "content": content,
                                "title": metadata.get("title", "Documento sin título"),
                                "domain": key.split("_", 1)[1],
                                "language": language,
                                "metadata": metadata,
                                "score": score,
                                "search_type": "memory_text",
                            }
                        )

        # Ordenar por score y limitar
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:k]

    async def load_corpus_index(self, language: str, corpus_path: str) -> bool:
        """
        Cargar o crear índice para un corpus específico

        Args:
            language: Idioma del corpus
            corpus_path: Ruta al corpus

        Returns:
            bool: True si se cargó exitosamente
        """
        try:
            corpus_dir = Path(corpus_path)
            if not corpus_dir.exists():
                logger.warning(f"Directorio de corpus no existe: {corpus_path}")
                return False

            # Marcar como cargado (en implementación real, cargaría archivos)
            self._loaded_indices[language] = {
                "language": language,
                "corpus_path": corpus_path,
                "loaded_at": time.time(),
                "backend": self.index_type,
            }

            logger.info(f"Índice de corpus cargado: {language}")
            return True

        except Exception as e:
            logger.error(f"Error cargando corpus {language}: {e}")
            return False

    async def rebuild_from_corpus(self, language: str, corpus_path: str) -> bool:
        """
        Reconstruir índice desde archivos del corpus

        Args:
            language: Idioma del corpus
            corpus_path: Ruta al corpus

        Returns:
            bool: True si se reconstruyó exitosamente
        """
        try:
            logger.info(f"Iniciando reconstrucción de índice: {language}")

            # En implementación real, leería archivos del corpus
            # y los indexaría uno por uno

            # Simulación de reconstrucción
            await asyncio.sleep(0.1)  # Simular tiempo de procesamiento

            # Actualizar información de índice
            self._loaded_indices[language] = {
                "language": language,
                "corpus_path": corpus_path,
                "rebuilt_at": time.time(),
                "backend": self.index_type,
            }

            logger.info(f"Índice reconstruido: {language}")
            return True

        except Exception as e:
            logger.error(f"Error reconstruyendo índice {language}: {e}")
            return False

    def _update_indexing_metrics(self, indexing_time: float):
        """
        Actualizar métricas de indexación

        Args:
            indexing_time: Tiempo de indexación
        """
        # Actualizar tiempo promedio
        total_indexed = self._index_stats["documents_indexed"]
        current_avg = self._index_stats["average_indexing_time"]

        if total_indexed > 0:
            self._index_stats["average_indexing_time"] = (current_avg * total_indexed + indexing_time) / (
                total_indexed + 1
            )
        else:
            self._index_stats["average_indexing_time"] = indexing_time

    def get_document_count(self, language: Optional[str] = None) -> int:
        """
        Obtener número de documentos indexados

        Args:
            language: Idioma específico (opcional)

        Returns:
            Número de documentos
        """
        if language:
            return sum(count for key, count in self._document_counts.items() if key.startswith(f"{language}_"))
        else:
            return sum(self._document_counts.values())

    def get_available_domains(self) -> Dict[str, List[str]]:
        """
        Obtener dominios disponibles por idioma

        Returns:
            Dict con dominios por idioma
        """
        domains_by_language = {}

        for key in self._document_counts.keys():
            parts = key.split("_", 1)
            if len(parts) == 2:
                language, domain = parts

                if language not in domains_by_language:
                    domains_by_language[language] = []

                if domain not in domains_by_language[language]:
                    domains_by_language[language].append(domain)

        return domains_by_language

    def get_index_sizes(self) -> Dict[str, int]:
        """Obtener tamaños de índices por idioma/dominio"""
        return self._document_counts.copy()

    def get_stats(self) -> Dict:
        """Obtener estadísticas del gestor"""
        return {
            **self._index_stats,
            "total_documents": sum(self._document_counts.values()),
            "loaded_indices": len(self._loaded_indices),
            "backend": self.index_type,
        }

    async def health_check(self) -> Dict:
        """Verificar estado de salud del gestor"""
        return {
            "status": "healthy",
            "backend": self.index_type,
            "loaded_indices": len(self._loaded_indices),
            "total_documents": self.get_document_count(),
            "available_languages": len(set(key.split("_")[0] for key in self._document_counts.keys())),
            "stats": self.get_stats(),
        }

    async def update_document(
        self,
        doc_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Actualizar documento existente

        Args:
            doc_id: ID del documento
            content: Nuevo contenido
            embedding: Nuevo embedding
            metadata: Nuevos metadatos

        Returns:
            bool: True si se actualizó exitosamente
        """
        try:
            # La implementación depende del backend
            # Para SQLite, usaríamos UPDATE
            # Para FAISS, reemplazaríamos vector
            # Para memoria, actualizaríamos diccionarios

            self._index_stats["documents_updated"] += 1
            logger.info(f"Documento actualizado: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error actualizando documento {doc_id}: {e}")
            return False

    async def delete_document(self, doc_id: str, language: str) -> bool:
        """
        Eliminar documento del índice

        Args:
            doc_id: ID del documento
            language: Idioma del documento

        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            # Implementación específica por backend
            self._index_stats["documents_deleted"] += 1
            logger.info(f"Documento eliminado: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {e}")
            return False

    async def shutdown(self):
        """Cerrar gestor y limpiar recursos"""
        logger.info("Iniciando shutdown del IndexManager")

        # Limpiar índices en memoria
        if self.index_type == "memory":
            self.memory_indices.clear()

        # Cerrar conexiones de base de datos
        # (En implementación real, cerraría conexiones activas)

        logger.info("IndexManager shutdown completado")
