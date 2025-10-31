#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Operaciones de Archivo - Manejo de archivos, uploads y backups
Extraído de main.py para mejorar la organización del código
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List


class FileOperations:
    """Clase para operaciones de archivo"""

    def __init__(self, corpus_root: str = "corpus_ES"):
        self.corpus_root = corpus_root

    def handle_upload_document(self, post_data: bytes, rag_pipeline=None) -> Dict[str, Any]:
        """Subir documento al corpus"""
        try:
            data = json.loads(post_data.decode("utf-8"))

            filename = data.get("filename", "").strip()
            content = data.get("content", "").strip()
            category = data.get("category", "general").strip()

            if not filename or not content:
                return {"error": "filename y content son requeridos"}

            if not rag_pipeline:
                return {"error": "Pipeline RAG no disponible"}

            if not filename.endswith((".txt", ".md")):
                filename += ".txt"

            category_path = Path(self.corpus_root) / category
            category_path.mkdir(exist_ok=True, parents=True)

            file_path = category_path / filename

            if file_path.exists():
                return {"error": f"El archivo {filename} ya existe en {category}"}

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            if rag_pipeline:
                rag_pipeline.load_documents()

            return {
                "message": "Documento subido exitosamente",
                "filename": filename,
                "category": category,
                "path": str(file_path),
                "size_bytes": len(content.encode("utf-8")),
                "documents_total": len(rag_pipeline.documents) if rag_pipeline else 0,
            }

        except json.JSONDecodeError:
            return {"error": "JSON inválido"}
        except Exception as e:
            return {"error": f"Error subiendo documento: {str(e)}"}

    def handle_delete_document(self, doc_id: str, rag_pipeline=None) -> Dict[str, Any]:
        """Eliminar documento del corpus"""
        try:
            if not rag_pipeline:
                return {"error": "Pipeline RAG no disponible"}

            target_path = None
            for doc in rag_pipeline.documents:
                if os.path.basename(doc["path"]) == doc_id:
                    target_path = doc["path"]
                    break

            if not target_path:
                return {"error": f"Documento {doc_id} no encontrado"}

            try:
                os.remove(target_path)
            except FileNotFoundError:
                return {"error": f"Archivo físico {doc_id} no encontrado"}

            if rag_pipeline:
                rag_pipeline.load_documents()

            return {
                "message": "Documento eliminado exitosamente",
                "deleted_file": doc_id,
                "documents_remaining": len(rag_pipeline.documents) if rag_pipeline else 0,
            }

        except Exception as e:
            return {"error": f"Error eliminando documento: {str(e)}"}

    def handle_update_document(
        self, doc_id: str, put_data: bytes, rag_pipeline=None
    ) -> Dict[str, Any]:
        """Actualizar documento completo"""
        try:
            data = json.loads(put_data.decode("utf-8"))
            new_content = data.get("content", "").strip()

            if not new_content:
                return {"error": "Content required"}

            target_path = self._find_document_path(doc_id, rag_pipeline)
            if not target_path:
                return {"error": f"Document {doc_id} not found"}

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            if rag_pipeline:
                rag_pipeline.load_documents()

            return {
                "message": "Document updated successfully",
                "document_id": doc_id,
                "new_size": len(new_content),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error actualizando documento: {str(e)}"}

    def handle_patch_document(
        self, doc_id: str, patch_data: bytes, rag_pipeline=None
    ) -> Dict[str, Any]:
        """Aplicar cambios parciales a documento"""
        try:
            data = json.loads(patch_data.decode("utf-8"))
            operation = data.get("operation", "append")
            content = data.get("content", "").strip()

            if not content:
                return {"error": "Content required"}

            target_path = self._find_document_path(doc_id, rag_pipeline)
            if not target_path:
                return {"error": f"Document {doc_id} not found"}

            with open(target_path, "r", encoding="utf-8") as f:
                current_content = f.read()

            if operation == "append":
                new_content = current_content + "\n\n" + content
            elif operation == "prepend":
                new_content = content + "\n\n" + current_content
            elif operation == "replace_section":
                section_start = data.get("section_start", "")
                section_end = data.get("section_end", "")
                new_content = self._replace_section(
                    current_content, content, section_start, section_end
                )
            else:
                return {"error": "Invalid operation"}

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            if rag_pipeline:
                rag_pipeline.load_documents()

            return {
                "message": f"Document patched successfully with {operation}",
                "document_id": doc_id,
                "operation": operation,
                "new_size": len(new_content),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error aplicando patch: {str(e)}"}

    def handle_batch_upload(self, post_data: bytes, rag_pipeline=None) -> Dict[str, Any]:
        """Subida masiva de documentos"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            documents = data.get("documents", [])
            category = data.get("category", "general")
            overwrite = data.get("overwrite", False)

            if not documents:
                return {"error": "Documents array required"}

            upload_results = []
            successful_uploads = 0
            failed_uploads = 0

            for doc in documents:
                try:
                    filename = doc.get("filename", "")
                    content = doc.get("content", "")

                    if not filename or not content:
                        failed_uploads += 1
                        upload_results.append(
                            {
                                "filename": filename,
                                "status": "error",
                                "error": "Missing filename or content",
                            }
                        )
                        continue

                    result = self._upload_single_document(filename, content, category, overwrite)

                    if result["status"] == "success":
                        successful_uploads += 1
                    else:
                        failed_uploads += 1

                    upload_results.append(result)

                except Exception as e:
                    failed_uploads += 1
                    upload_results.append(
                        {
                            "filename": doc.get("filename", "unknown"),
                            "status": "error",
                            "error": str(e),
                        }
                    )

            if successful_uploads > 0 and rag_pipeline:
                rag_pipeline.load_documents()

            return {
                "message": "Batch upload completed",
                "total_documents": len(documents),
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "upload_results": upload_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error en batch upload: {str(e)}"}

    def handle_create_backup(self, post_data: bytes) -> Dict[str, Any]:
        """Crear backup del sistema"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            backup_type = data.get("type", "full")
            backup_name = data.get("name", f"backup_{int(time.time())}")

            backup_result = self._create_system_backup(backup_type, backup_name)

            return {
                "message": "Backup created successfully",
                "backup_name": backup_name,
                "backup_type": backup_type,
                "backup_info": backup_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error creando backup: {str(e)}"}

    def handle_export_corpus(self, post_data: bytes, rag_pipeline=None) -> Dict[str, Any]:
        """Exportar corpus completo"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            export_format = data.get("format", "json")
            include_metadata = data.get("include_metadata", True)

            export_result = self._export_corpus_data(export_format, include_metadata, rag_pipeline)

            return {
                "message": "Corpus exported successfully",
                "export_format": export_format,
                "export_info": export_result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error exportando corpus: {str(e)}"}

    def handle_export_config(self, post_data: bytes) -> Dict[str, Any]:
        """Exportar configuración del sistema"""
        try:
            config_data = {
                "corpus_root": self.corpus_root,
                "server_config": {
                    "host": getattr(self, "_host", "localhost"),
                    "port": getattr(self, "_port", 8003),
                },
                "rag_config": {
                    "max_results": getattr(self, "_max_results", 5),
                    "similarity_threshold": getattr(self, "_similarity_threshold", 0.3),
                },
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0.0",
            }

            return {
                "message": "Configuration exported successfully",
                "config": config_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error exportando config: {str(e)}"}

    def handle_export_embeddings(self, post_data: bytes, rag_pipeline=None) -> Dict[str, Any]:
        """Exportar embeddings y vectores"""
        try:
            data = json.loads(post_data.decode("utf-8"))
            export_format = data.get("format", "json")

            if not rag_pipeline:
                return {"error": "RAG pipeline not available"}

            embeddings_data = self._export_embeddings_data(export_format, rag_pipeline)

            return {
                "message": "Embeddings exported successfully",
                "export_format": export_format,
                "embeddings_count": len(rag_pipeline.documents) if rag_pipeline else 0,
                "embeddings_data": embeddings_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"error": f"Error exportando embeddings: {str(e)}"}

    def handle_reindex(self, rag_pipeline=None) -> Dict[str, Any]:
        """Reindexar corpus completo"""
        try:
            if not rag_pipeline:
                return {"error": "Pipeline RAG no disponible"}

            old_count = len(rag_pipeline.documents)
            rag_pipeline.load_documents()
            new_count = len(rag_pipeline.documents)

            return {
                "message": "Reindexación completada",
                "documents_before": old_count,
                "documents_after": new_count,
                "vocabulary_size": len(rag_pipeline.embedder.vocabulary)
                if rag_pipeline.embedder.fitted
                else 0,
            }

        except Exception as e:
            return {"error": f"Error en reindexación: {str(e)}"}

    def _upload_single_document(
        self, filename: str, content: str, category: str, overwrite: bool
    ) -> Dict:
        """Subir un documento individual"""
        try:
            category_path = Path(self.corpus_root) / category
            category_path.mkdir(exist_ok=True)

            file_path = category_path / filename

            if os.path.exists(file_path) and not overwrite:
                return {
                    "filename": filename,
                    "status": "skipped",
                    "reason": "File exists and overwrite=False",
                }

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "filename": filename,
                "status": "success",
                "path": str(file_path),
                "size": len(content.encode("utf-8")),
            }

        except Exception as e:
            return {"filename": filename, "status": "error", "error": str(e)}

    def _find_document_path(self, doc_id: str, rag_pipeline=None) -> str:
        """Buscar la ruta de un documento por ID"""
        if not rag_pipeline:
            return ""

        for doc in rag_pipeline.documents:
            if isinstance(doc, dict) and os.path.basename(doc.get("path", "")) == doc_id:
                return doc["path"]
        return ""

    def _replace_section(
        self, content: str, new_section: str, section_start: str, section_end: str
    ) -> str:
        """Reemplazar una sección específica del contenido"""
        if not section_start or not section_end:
            return content + "\n\n" + new_section

        start_index = content.find(section_start)
        end_index = content.find(section_end)

        if start_index != -1 and end_index != -1 and end_index > start_index:
            return content[:start_index] + new_section + content[end_index + len(section_end) :]
        else:
            return content + "\n\n" + new_section

    def _create_system_backup(self, backup_type: str, backup_name: str) -> Dict:
        """Crear backup del sistema"""
        backup_info = {
            "type": backup_type,
            "name": backup_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files_backed_up": [],
        }

        try:
            backup_dir = f"backups/{backup_name}"
            os.makedirs(backup_dir, exist_ok=True)

            if backup_type in ["full", "corpus_only"]:
                if os.path.exists(self.corpus_root):
                    corpus_backup = f"{backup_dir}/corpus"
                    shutil.copytree(self.corpus_root, corpus_backup, dirs_exist_ok=True)
                    backup_info["files_backed_up"].append("corpus")

            if backup_type in ["full", "config_only"]:
                config_backup = f"{backup_dir}/config.json"
                config_data = {
                    "corpus_root": self.corpus_root,
                    "backup_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(config_backup, "w") as f:
                    json.dump(config_data, f, indent=2)
                backup_info["files_backed_up"].append("config")

            backup_info["status"] = "success"
            backup_info["size"] = self._calculate_backup_size(backup_dir)

        except Exception as e:
            backup_info["status"] = "error"
            backup_info["error"] = str(e)

        return backup_info

    def _export_corpus_data(
        self, export_format: str, include_metadata: bool, rag_pipeline=None
    ) -> Dict:
        """Exportar datos del corpus"""
        if not rag_pipeline:
            return {"error": "No RAG pipeline available"}

        corpus_data = {
            "documents": [],
            "total_documents": len(rag_pipeline.documents),
            "export_format": export_format,
        }

        for doc in rag_pipeline.documents:
            doc_entry = {
                "content": doc if isinstance(doc, str) else str(doc),
                "length": len(str(doc)),
            }

            if include_metadata:
                doc_entry.update({"word_count": len(str(doc).split()), "char_count": len(str(doc))})

            corpus_data["documents"].append(doc_entry)

        return corpus_data

    def _export_embeddings_data(self, export_format: str, rag_pipeline=None) -> Dict:
        """Exportar datos de embeddings"""
        embeddings_info = {
            "format": export_format,
            "embeddings_available": bool(rag_pipeline and hasattr(rag_pipeline, "embedder")),
            "documents_count": len(rag_pipeline.documents) if rag_pipeline else 0,
        }

        if rag_pipeline and hasattr(rag_pipeline.embedder, "vocabulary"):
            embeddings_info.update(
                {
                    "vocabulary_size": len(getattr(rag_pipeline.embedder, "vocabulary", {})),
                    "tfidf_available": True,
                }
            )

        return embeddings_info

    def _calculate_backup_size(self, backup_dir: str) -> int:
        """Calcular el tamaño del backup"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(backup_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
