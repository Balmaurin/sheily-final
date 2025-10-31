#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocumentProcessor para RAG System - Adaptado para JSONL
===============================================

Procesa documentos JSONL de los corpus en all-Branches/,
extrayendo contenido y dividiéndolo en chunks usando RecursiveCharacterTextSplitter.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Procesador de documentos adaptado para archivos JSONL en corpus/spanish/
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Inicializar el procesador con configuración de chunking
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Priorizar párrafos, luego frases
        )

    def process_corpus(self, corpus_path: str) -> List[Dict]:
        """
        Procesar todos los archivos JSONL en el corpus
        Args:
            corpus_path: Ruta al directorio corpus (ej: all-Branches/antropologia/corpus)
        Returns:
            Lista de diccionarios con chunks
        """
        processed_chunks = []

        corpus_path = Path(corpus_path)

        # Buscar carpetas de idiomas (spanish/ como corpus_ES)
        language_dirs = {
            "spanish": "ES",
            "english": "EN"
        }

        for lang_dir, lang_code in language_dirs.items():
            lang_path = corpus_path / lang_dir
            if lang_path.exists() and lang_path.is_dir():
                print(f"Procesando corpus {lang_code}: {lang_path}")
                chunks = self._process_language_corpus(lang_path, lang_code)
                processed_chunks.extend(chunks)

        return processed_chunks

    def _process_language_corpus(self, lang_path: Path, language: str) -> List[Dict]:
        """
        Procesar archivos JSONL en una carpeta de idioma
        """
        chunks = []

        # Buscar todos los archivos .jsonl
        for jsonl_file in lang_path.glob("*.jsonl"):
            print(f"  Procesando archivo: {jsonl_file.name}")
            file_chunks = self._process_jsonl_file(jsonl_file, language)
            chunks.extend(file_chunks)

        return chunks

    def _process_jsonl_file(self, file_path: Path, language: str) -> List[Dict]:
        """
        Procesar un archivo JSONL individual
        """
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Dividir por líneas pero considerar objetos JSON multilínea
            lines = content.split('\n')
            current_json = ""
            brace_count = 0

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                current_json += line

                # Contar llaves para detectar objetos JSON completos
                brace_count += line.count('{') - line.count('}')

                # Si tenemos un objeto JSON completo
                if brace_count == 0 and current_json.strip():
                    try:
                        doc = json.loads(current_json)
                        content_text = doc.get("content", "").strip()
                        if not content_text:
                            current_json = ""
                            continue

                        # Extraer metadata del documento JSON
                        metadata = {
                            "source": str(file_path),
                            "language": language,
                            "title": doc.get("title", ""),
                            "domain": doc.get("domain", ""),
                            "category": doc.get("category", ""),
                            "keywords": doc.get("keywords", []),
                            "date": doc.get("date", ""),
                            "line_number": line_num
                        }

                        # Dividir el contenido en chunks
                        text_chunks = self.text_splitter.split_text(content_text)

                        # Crear chunks con metadata
                        for i, chunk_text in enumerate(text_chunks):
                            chunk_id = f"{file_path.stem}_{line_num}_{i}"
                            chunk = {
                                "id": chunk_id,
                                "content": chunk_text,
                                "metadata": metadata
                            }
                            chunks.append(chunk)

                        current_json = ""  # Reset para siguiente objeto

                    except json.JSONDecodeError as e:
                        print(f"    Error parseando objeto JSON en línea {line_num} de {file_path}: {e}")
                        current_json = ""  # Reset en error
                        continue

        except Exception as e:
            print(f"    Error procesando archivo {file_path}: {e}")

        return chunks


def process_all_branches_corpus(all_branches_path: str, domains: Optional[List[str]] = None) -> List[Dict]:
    """
    Procesar corpus de todas las ramas en all-Branches/
    Args:
        all_branches_path: Ruta a all-Branches/
        domains: Lista opcional de dominios a procesar (ej: ['antropologia', 'programacion'])
    Returns:
        Lista completa de chunks de todas las ramas
    """
    all_chunks = []
    processor = DocumentProcessor()

    branches_path = Path(all_branches_path)

    if not branches_path.exists():
        raise ValueError(f"Directorio all-Branches no encontrado: {all_branches_path}")

    # Obtener lista de dominios (ramas)
    if domains:
        domain_dirs = [branches_path / domain for domain in domains if (branches_path / domain).exists()]
    else:
        domain_dirs = [d for d in branches_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Procesando {len(domain_dirs)} dominios...")

    for domain_dir in domain_dirs:
        corpus_path = domain_dir / "corpus"
        if corpus_path.exists():
            print(f"\nProcesando dominio: {domain_dir.name}")
            domain_chunks = processor.process_corpus(str(corpus_path))
            print(f"  Chunks generados: {len(domain_chunks)}")
            all_chunks.extend(domain_chunks)
        else:
            print(f"  Corpus no encontrado para {domain_dir.name}")

    print(f"\nTotal de chunks generados: {len(all_chunks)}")
    return all_chunks


# Ejemplo de uso
if __name__ == "__main__":
    # Procesar todas las ramas
    all_chunks = process_all_branches_corpus("all-Branches")

    # Mostrar estadísticas
    print(f"\nEstadísticas:")
    print(f"Total chunks: {len(all_chunks)}")

    languages = {}
    domains = {}

    for chunk in all_chunks:
        lang = chunk["metadata"]["language"]
        domain = chunk["metadata"]["domain"]

        languages[lang] = languages.get(lang, 0) + 1
        domains[domain] = domains.get(domain, 0) + 1

    print(f"Por idioma: {languages}")
    print(f"Por dominio: {domains}")

    # Mostrar un ejemplo de chunk
    if all_chunks:
        print(f"\nEjemplo de chunk:")
        print(f"ID: {all_chunks[0]['id']}")
        print(f"Contenido (primeros 200 chars): {all_chunks[0]['content'][:200]}...")
        print(f"Metadata: {all_chunks[0]['metadata']}")
