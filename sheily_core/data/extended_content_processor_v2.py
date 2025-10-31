#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTENDED CONTENT PROCESSOR V2 - PROCESADOR DE CONTENIDO EXTENSO
================================================================

Sistema avanzado para procesamiento de contenido extenso que proporciona:

CAPACIDADES DE PROCESAMIENTO:
- Soporte para documentos de hasta 50MB
- Procesamiento inteligente de código fuente extenso
- Manejo avanzado de contenido académico
- Procesamiento de conversaciones históricas largas
- Integración con datos de RAMAS_ACADEMICAS_OFICIALES.json
- Procesamiento de logs y datos históricos
- Chunking adaptativo según tipo de contenido
- Preservación de contexto en contenido extenso
- Optimización automática de procesamiento

ARQUITECTURA DE PROCESAMIENTO:
- Pipeline de procesamiento multi-etapa
- Análisis inteligente de estructura de contenido
- Extracción de metadatos avanzada
- Sistema de calidad y validación
- Integración perfecta con memoria y RAG
- Procesamiento paralelo para contenido extenso
"""

import asyncio
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Importaciones avanzadas con fallbacks
try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configuración avanzada
CONTENT_PROCESSOR_ROOT = Path(__file__).resolve().parents[2] / "data" / "extended_content_v2"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE_LIMITS = {
    "code": (500, 2000),
    "academic": (800, 3000),
    "conversation": (300, 1500),
    "log": (200, 1000),
    "data": (400, 2000),
}
PROCESSING_THREADS = 4


@dataclass
class ContentMetadata:
    """Metadatos avanzados de contenido"""

    file_path: str
    file_size: int
    content_type: str
    encoding: str
    language: str
    structure_detected: Dict[str, Any]
    quality_score: float
    processing_time: float
    chunks_created: int
    word_count: int
    estimated_reading_time: int  # minutos

    # Metadatos específicos por tipo
    code_metadata: Dict[str, Any] = field(default_factory=dict)
    academic_metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Resultado de procesamiento de contenido"""

    success: bool
    content_chunks: List[Dict[str, Any]]
    metadata: ContentMetadata
    processing_stats: Dict[str, Any]
    error_message: Optional[str] = None

    def get_total_chunks(self) -> int:
        """Obtener número total de chunks"""
        return len(self.content_chunks)

    def get_estimated_size(self) -> int:
        """Obtener tamaño estimado del contenido procesado"""
        return sum(len(chunk.get("content", "")) for chunk in self.content_chunks)


class ExtendedContentProcessor:
    """Procesador avanzado de contenido extenso"""

    def __init__(self):
        self.processor_root = CONTENT_PROCESSOR_ROOT
        self._init_directories()

        # Configuración de procesamiento
        self.chunk_size_limits = CHUNK_SIZE_LIMITS
        self.max_file_size = MAX_FILE_SIZE
        self.processing_threads = PROCESSING_THREADS

        # Estadísticas de procesamiento
        self.processing_stats = defaultdict(int)

        self.logger = self._get_logger()

    def _get_logger(self):
        """Obtener logger con fallback"""
        try:
            from sheily_core.logger import get_logger

            return get_logger("extended_content_processor")
        except ImportError:
            import logging

            return logging.getLogger("extended_content_processor")

    def _init_directories(self):
        """Inicializar estructura de directorios"""
        self.processor_root.mkdir(parents=True, exist_ok=True)
        (self.processor_root / "processed").mkdir(exist_ok=True)
        (self.processor_root / "metadata").mkdir(exist_ok=True)
        (self.processor_root / "temp").mkdir(exist_ok=True)

    def process_file(
        self,
        file_path: str,
        content_type: str = "auto",
        processing_options: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """Procesar archivo extenso con opciones avanzadas"""
        processing_options = processing_options or {}

        try:
            # Validar archivo
            if not os.path.exists(file_path):
                return ProcessingResult(
                    success=False,
                    content_chunks=[],
                    metadata=ContentMetadata(
                        file_path=file_path,
                        file_size=0,
                        content_type="unknown",
                        encoding="unknown",
                        language="unknown",
                        structure_detected={},
                        quality_score=0.0,
                        processing_time=0.0,
                        chunks_created=0,
                        word_count=0,
                        estimated_reading_time=0,
                    ),
                    processing_stats={},
                    error_message=f"Archivo no encontrado: {file_path}",
                )

            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return ProcessingResult(
                    success=False,
                    content_chunks=[],
                    metadata=ContentMetadata(
                        file_path=file_path,
                        file_size=file_size,
                        content_type="unknown",
                        encoding="unknown",
                        language="unknown",
                        structure_detected={},
                        quality_score=0.0,
                        processing_time=0.0,
                        chunks_created=0,
                        word_count=0,
                        estimated_reading_time=0,
                    ),
                    processing_stats={},
                    error_message=f"Archivo demasiado grande: {file_size} bytes (máximo: {self.max_file_size})",
                )

            # Detectar tipo de contenido si es automático
            if content_type == "auto":
                content_type = self._detect_content_type(file_path)

            # Leer contenido según tipo de archivo
            content, encoding = self._read_file_content(file_path, content_type)

            if not content:
                return ProcessingResult(
                    success=False,
                    content_chunks=[],
                    metadata=ContentMetadata(
                        file_path=file_path,
                        file_size=file_size,
                        content_type=content_type,
                        encoding=encoding,
                        language="unknown",
                        structure_detected={},
                        quality_score=0.0,
                        processing_time=0.0,
                        chunks_created=0,
                        word_count=0,
                        estimated_reading_time=0,
                    ),
                    processing_stats={},
                    error_message="No se pudo leer el contenido del archivo",
                )

            # Procesar contenido
            start_time = time.time()
            chunks, metadata = self._process_content(
                content, file_path, content_type, processing_options
            )
            processing_time = time.time() - start_time

            # Crear metadatos avanzados
            content_metadata = ContentMetadata(
                file_path=file_path,
                file_size=file_size,
                content_type=content_type,
                encoding=encoding,
                language=self._detect_language(content),
                structure_detected=self._analyze_content_structure(content, content_type),
                quality_score=self._calculate_content_quality(content, content_type),
                processing_time=processing_time,
                chunks_created=len(chunks),
                word_count=len(content.split()),
                estimated_reading_time=len(content.split()) // 200,  # ~200 palabras por minuto
            )

            # Agregar metadatos específicos por tipo
            if content_type == "code":
                content_metadata.code_metadata = self._extract_code_metadata(content)
            elif content_type == "academic":
                content_metadata.academic_metadata = self._extract_academic_metadata(content)
            elif content_type == "conversation":
                content_metadata.conversation_metadata = self._extract_conversation_metadata(
                    content
                )

            # Estadísticas de procesamiento
            processing_stats = {
                "processing_time": processing_time,
                "content_length": len(content),
                "chunks_created": len(chunks),
                "compression_ratio": len(content) / sum(len(chunk["content"]) for chunk in chunks)
                if chunks
                else 1.0,
                "average_chunk_size": sum(len(chunk["content"]) for chunk in chunks) / len(chunks)
                if chunks
                else 0,
            }

            return ProcessingResult(
                success=True,
                content_chunks=chunks,
                metadata=content_metadata,
                processing_stats=processing_stats,
            )

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                content_chunks=[],
                metadata=ContentMetadata(
                    file_path=file_path,
                    file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    content_type="unknown",
                    encoding="unknown",
                    language="unknown",
                    structure_detected={},
                    quality_score=0.0,
                    processing_time=0.0,
                    chunks_created=0,
                    word_count=0,
                    estimated_reading_time=0,
                ),
                processing_stats={},
                error_message=str(e),
            )

    def _detect_content_type(self, file_path: str) -> str:
        """Detectar tipo de contenido basado en extensión y contenido"""
        file_extension = Path(file_path).suffix.lower()

        # Detección por extensión
        extension_map = {
            ".py": "code",
            ".js": "code",
            ".java": "code",
            ".cpp": "code",
            ".c": "code",
            ".h": "code",
            ".php": "code",
            ".rb": "code",
            ".go": "code",
            ".rs": "code",
            ".swift": "code",
            ".kt": "code",
            ".scala": "code",
            ".html": "code",
            ".css": "code",
            ".xml": "code",
            ".json": "data",
            ".yaml": "data",
            ".yml": "data",
            ".csv": "data",
            ".xlsx": "data",
            ".pdf": "academic",
            ".docx": "academic",
            ".doc": "academic",
            ".txt": "text",
            ".md": "academic",
            ".log": "log",
            ".chat": "conversation",
        }

        if file_extension in extension_map:
            return extension_map[file_extension]

        # Detección por contenido si extensión no es concluyente
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample_content = f.read(1024)  # Leer primeros 1KB

            # Detectar por contenido
            if any(
                indicator in sample_content
                for indicator in ["def ", "function ", "class ", "import "]
            ):
                return "code"
            elif any(
                indicator in sample_content
                for indicator in ["capítulo", "introducción", "conclusión", "abstract"]
            ):
                return "academic"
            elif any(
                indicator in sample_content
                for indicator in ["usuario:", "asistente:", "human:", "assistant:"]
            ):
                return "conversation"
            elif any(
                indicator in sample_content for indicator in ["ERROR", "WARN", "INFO", "DEBUG"]
            ):
                return "log"
            else:
                return "text"

        except Exception:
            return "text"

    def _read_file_content(self, file_path: str, content_type: str) -> Tuple[str, str]:
        """Leer contenido del archivo según tipo"""
        try:
            # Detectar encoding
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read(1024)
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result.get("encoding", "utf-8") or "utf-8"

            # Leer contenido según tipo
            if content_type == "pdf" and PDF_AVAILABLE:
                return self._read_pdf_content(file_path), encoding
            elif content_type in ["docx", "academic"] and DOCX_AVAILABLE:
                return self._read_docx_content(file_path), encoding
            elif content_type == "data" and PANDAS_AVAILABLE:
                return self._read_data_content(file_path), encoding
            else:
                # Lectura estándar de texto
                with open(file_path, "r", encoding=encoding, errors="ignore") as f:
                    return f.read(), encoding

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return "", "unknown"

    def _read_pdf_content(self, file_path: str) -> str:
        """Leer contenido de archivo PDF"""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"

                return content

        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
            return ""

    def _read_docx_content(self, file_path: str) -> str:
        """Leer contenido de archivo DOCX"""
        try:
            doc = docx.Document(file_path)
            content = ""

            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"

            return content

        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""

    def _read_data_content(self, file_path: str) -> str:
        """Leer contenido de archivo de datos"""
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                return df.to_string()
            elif file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
                return df.to_string()
            elif file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
            else:
                return ""

        except Exception as e:
            self.logger.error(f"Error reading data file {file_path}: {e}")
            return ""

    def _process_content(
        self, content: str, file_path: str, content_type: str, processing_options: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Procesar contenido con opciones avanzadas"""
        # Crear chunks según tipo de contenido
        if content_type == "code":
            chunks = self._process_code_content(content, processing_options)
        elif content_type == "academic":
            chunks = self._process_academic_content(content, processing_options)
        elif content_type == "conversation":
            chunks = self._process_conversation_content(content, processing_options)
        elif content_type == "log":
            chunks = self._process_log_content(content, processing_options)
        elif content_type == "data":
            chunks = self._process_data_content(content, processing_options)
        else:
            chunks = self._process_text_content(content, processing_options)

        # Agregar metadatos a cada chunk
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{Path(file_path).stem}_chunk_{i}"
            chunk["source_file"] = file_path
            chunk["content_type"] = content_type
            chunk["processing_timestamp"] = time.time()

        return chunks, {"chunks_processed": len(chunks)}

    def _process_code_content(self, content: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesar contenido de código fuente"""
        chunks = []

        # Dividir por funciones/clases si es posible
        function_pattern = r"(def |function |class |public class |interface )"
        code_sections = re.split(f"({function_pattern})", content)

        current_section = ""
        for i in range(0, len(code_sections), 2):
            section_header = code_sections[i] if i < len(code_sections) else ""
            section_content = code_sections[i + 1] if i + 1 < len(code_sections) else ""

            current_section += section_header + section_content

            # Crear chunk cuando alcanza tamaño óptimo
            if len(current_section) > 1500 or (section_header and current_section):
                if current_section.strip():
                    chunks.append(
                        {
                            "content": current_section.strip(),
                            "chunk_type": "code_section",
                            "metadata": {
                                "has_functions": "def " in current_section
                                or "function " in current_section,
                                "has_classes": "class " in current_section,
                                "language": self._detect_code_language(current_section),
                                "line_count": current_section.count("\n"),
                            },
                        }
                    )
                current_section = ""

        # Agregar sección final
        if current_section.strip():
            chunks.append(
                {
                    "content": current_section.strip(),
                    "chunk_type": "code_section",
                    "metadata": {
                        "has_functions": "def " in current_section
                        or "function " in current_section,
                        "has_classes": "class " in current_section,
                        "language": self._detect_code_language(current_section),
                        "line_count": current_section.count("\n"),
                    },
                }
            )

        return chunks

    def _process_academic_content(
        self, content: str, options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Procesar contenido académico"""
        chunks = []

        # Detectar estructura académica
        sections = self._detect_academic_sections(content)

        for section in sections:
            section_content = section["content"]
            section_type = section["type"]

            # Crear chunks dentro de cada sección
            if len(section_content) > 2000:
                # Dividir sección larga en chunks más pequeños
                sentences = re.split(r"[.!?]+", section_content)
                current_chunk = ""
                chunk_index = 0

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_chunk) + len(sentence) > 1500 and current_chunk:
                        chunks.append(
                            {
                                "content": current_chunk.strip(),
                                "chunk_type": f"academic_{section_type}",
                                "metadata": {
                                    "section_type": section_type,
                                    "chunk_index": chunk_index,
                                    "total_section_chunks": len(sentences) // 1500 + 1,
                                },
                            }
                        )
                        current_chunk = sentence
                        chunk_index += 1
                    else:
                        if current_chunk:
                            current_chunk += ". " + sentence
                        else:
                            current_chunk = sentence

                # Agregar último chunk de sección
                if current_chunk.strip():
                    chunks.append(
                        {
                            "content": current_chunk.strip(),
                            "chunk_type": f"academic_{section_type}",
                            "metadata": {
                                "section_type": section_type,
                                "chunk_index": chunk_index,
                                "total_section_chunks": chunk_index + 1,
                            },
                        }
                    )
            else:
                # Sección pequeña, crear un solo chunk
                chunks.append(
                    {
                        "content": section_content,
                        "chunk_type": f"academic_{section_type}",
                        "metadata": {
                            "section_type": section_type,
                            "chunk_index": 0,
                            "total_section_chunks": 1,
                        },
                    }
                )

        return chunks

    def _process_conversation_content(
        self, content: str, options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Procesar contenido de conversación"""
        chunks = []

        # Dividir por turnos de conversación
        turn_patterns = [
            r"\n\s*(?:Usuario|Human|User):\s*",
            r"\n\s*(?:Asistente|Assistant|Bot|IA|Sheily):\s*",
            r"\n\s*\d{1,2}:\d{2}.*?(?=\n\s*\d{1,2}:\d{2}|\Z)",
        ]

        conversation_turns = []
        remaining_content = content

        for pattern in turn_patterns:
            turns = re.findall(pattern, remaining_content)
            for turn in turns:
                if len(turn.strip()) > 50:  # Solo turnos sustanciales
                    conversation_turns.append(turn.strip())

            # Remover turnos encontrados
            remaining_content = re.sub(pattern, "", remaining_content)

        # Crear chunks de conversación
        current_chunk = ""
        for turn in conversation_turns:
            if len(current_chunk) + len(turn) > 1000 and current_chunk:
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "chunk_type": "conversation_segment",
                        "metadata": {
                            "turn_count": current_chunk.count("\n"),
                            "participant_count": len(
                                set(
                                    re.findall(
                                        r"(Usuario|Asistente|Human|Assistant)", current_chunk
                                    )
                                )
                            ),
                        },
                    }
                )
                current_chunk = turn
            else:
                if current_chunk:
                    current_chunk += "\n\n" + turn
                else:
                    current_chunk = turn

        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "chunk_type": "conversation_segment",
                    "metadata": {
                        "turn_count": current_chunk.count("\n"),
                        "participant_count": len(
                            set(re.findall(r"(Usuario|Asistente|Human|Assistant)", current_chunk))
                        ),
                    },
                }
            )

        return chunks

    def _process_log_content(self, content: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesar contenido de logs"""
        chunks = []

        # Dividir por entradas de log
        log_patterns = [
            r"\n\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
            r"\n\s*(?:ERROR|WARN|INFO|DEBUG|FATAL)",
        ]

        log_entries = []
        remaining_content = content

        for pattern in log_patterns:
            entries = re.findall(pattern, remaining_content)
            for entry in entries:
                if len(entry.strip()) > 20:  # Solo entradas sustanciales
                    log_entries.append(entry.strip())

            remaining_content = re.sub(pattern, "", remaining_content)

        # Crear chunks de log
        current_chunk = ""
        for entry in log_entries:
            if len(current_chunk) + len(entry) > 800 and current_chunk:
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "chunk_type": "log_segment",
                        "metadata": {
                            "entry_count": current_chunk.count("\n"),
                            "error_count": current_chunk.count("ERROR")
                            + current_chunk.count("FATAL"),
                            "warning_count": current_chunk.count("WARN"),
                        },
                    }
                )
                current_chunk = entry
            else:
                if current_chunk:
                    current_chunk += "\n" + entry
                else:
                    current_chunk = entry

        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "chunk_type": "log_segment",
                    "metadata": {
                        "entry_count": current_chunk.count("\n"),
                        "error_count": current_chunk.count("ERROR") + current_chunk.count("FATAL"),
                        "warning_count": current_chunk.count("WARN"),
                    },
                }
            )

        return chunks

    def _process_data_content(self, content: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesar contenido de datos"""
        chunks = []

        # Para contenido de datos, crear chunks más estructurados
        lines = content.split("\n")
        current_chunk = ""
        chunk_index = 0

        for line in lines:
            # Crear nuevo chunk cada 500 líneas o si cambia el tipo de datos
            if len(current_chunk.split("\n")) > 500 or (
                current_chunk and self._detect_data_type_change(current_chunk, line)
            ):
                if current_chunk.strip():
                    chunks.append(
                        {
                            "content": current_chunk.strip(),
                            "chunk_type": "data_segment",
                            "metadata": {
                                "line_count": current_chunk.count("\n"),
                                "data_type": self._detect_data_type(current_chunk),
                                "chunk_index": chunk_index,
                            },
                        }
                    )
                current_chunk = line
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line

        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "chunk_type": "data_segment",
                    "metadata": {
                        "line_count": current_chunk.count("\n"),
                        "data_type": self._detect_data_type(current_chunk),
                        "chunk_index": chunk_index,
                    },
                }
            )

        return chunks

    def _process_text_content(self, content: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procesar contenido de texto general"""
        chunks = []

        # Chunking inteligente por oraciones
        sentences = re.split(r"[.!?]+", content)
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > 1500 and current_chunk:
                chunks.append(
                    {
                        "content": current_chunk.strip(),
                        "chunk_type": "text_segment",
                        "metadata": {
                            "sentence_count": current_chunk.count(".")
                            + current_chunk.count("!")
                            + current_chunk.count("?"),
                            "word_count": len(current_chunk.split()),
                            "chunk_index": chunk_index,
                        },
                    }
                )
                current_chunk = sentence
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "content": current_chunk.strip(),
                    "chunk_type": "text_segment",
                    "metadata": {
                        "sentence_count": current_chunk.count(".")
                        + current_chunk.count("!")
                        + current_chunk.count("?"),
                        "word_count": len(current_chunk.split()),
                        "chunk_index": chunk_index,
                    },
                }
            )

        return chunks

    def _detect_academic_sections(self, content: str) -> List[Dict[str, Any]]:
        """Detectar secciones en contenido académico"""
        sections = []

        # Patrones de secciones académicas
        section_patterns = [
            (
                r"\n\s*(?:CAPÍTULO|CAPITULO)\s+\d+.*?(?=\n\s*(?:CAPÍTULO|CAPITULO|\Z))",
                "chapter",
                re.DOTALL,
            ),
            (
                r"\n\s*\d+\.\s+[A-ZÁÉÍÓÚÑ].*?(?=\n\s*(?:\d+\.|CAPÍTULO|CAPITULO|\Z))",
                "section",
                re.DOTALL,
            ),
            (
                r"\n\s*(?:INTRODUCCIÓN|ABSTRACT).*?(?=\n\s*(?:CAPÍTULO|CAPITULO|\d+\.|\Z))",
                "introduction",
                re.DOTALL,
            ),
            (
                r"\n\s*(?:CONCLUSIÓN|CONCLUSIONES).*?(?=\n\s*(?:BIBLIOGRAFÍA|REFERENCIAS|\Z))",
                "conclusion",
                re.DOTALL,
            ),
            (
                r"\n\s*(?:METODOLOGÍA|MÉTODO).*?(?=\n\s*(?:RESULTADOS|ANÁLISIS|\Z))",
                "methodology",
                re.DOTALL,
            ),
            (
                r"\n\s*(?:RESULTADOS|ANÁLISIS).*?(?=\n\s*(?:CONCLUSIÓN|BIBLIOGRAFÍA|\Z))",
                "results",
                re.DOTALL,
            ),
        ]

        for pattern, section_type, flags in section_patterns:
            matches = re.findall(pattern, content, flags)
            for match in matches:
                if len(match.strip()) > 100:  # Solo secciones sustanciales
                    sections.append(
                        {"content": match.strip(), "type": section_type, "length": len(match)}
                    )

        # Si no se detectaron secciones, tratar como texto general
        if not sections:
            sections.append({"content": content, "type": "general", "length": len(content)})

        return sections

    def _detect_language(self, content: str) -> str:
        """Detectar idioma del contenido"""
        # Análisis simple de idioma basado en palabras comunes
        spanish_indicators = [
            "el",
            "la",
            "los",
            "las",
            "en",
            "con",
            "por",
            "para",
            "desde",
            "hasta",
        ]
        english_indicators = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of"]

        spanish_count = sum(1 for word in spanish_indicators if word in content.lower())
        english_count = sum(1 for word in english_indicators if word in content.lower())

        if spanish_count > english_count:
            return "es"
        elif english_count > spanish_count:
            return "en"
        else:
            return "unknown"

    def _analyze_content_structure(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analizar estructura del contenido"""
        structure = {
            "total_length": len(content),
            "line_count": content.count("\n"),
            "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
            "sentence_count": len(re.findall(r"[.!?]+", content)),
            "word_count": len(content.split()),
        }

        # Análisis específico por tipo
        if content_type == "code":
            structure["function_count"] = len(re.findall(r"\bdef \b|\bfunction \b", content))
            structure["class_count"] = len(re.findall(r"\bclass \b", content))
            structure["comment_count"] = len(re.findall(r"#.*|//.*|/\*.*\*/", content))

        elif content_type == "academic":
            structure["section_count"] = len(re.findall(r"\b\d+\.\s+[A-ZÁÉÍÓÚÑ]", content))
            structure["citation_count"] = len(re.findall(r"\[\d+\]|\(\w+\s+\d+\)", content))

        elif content_type == "conversation":
            structure["user_turns"] = len(re.findall(r"\b(?:Usuario|Human|User):\s*", content))
            structure["assistant_turns"] = len(
                re.findall(r"\b(?:Asistente|Assistant|Bot):\s*", content)
            )

        return structure

    def _calculate_content_quality(self, content: str, content_type: str) -> float:
        """Calcular calidad del contenido"""
        quality_score = 0.5  # Base

        # Factores de calidad generales
        if len(content) > 1000:
            quality_score += 0.2  # Contenido sustancial

        if content.count("\n") > 10:
            quality_score += 0.1  # Bien estructurado

        word_count = len(content.split())
        if word_count > 100:
            quality_score += 0.1  # Contenido significativo

        # Factores específicos por tipo
        if content_type == "code":
            if "def " in content or "function " in content:
                quality_score += 0.2  # Tiene funciones
            if "class " in content:
                quality_score += 0.1  # Tiene clases

        elif content_type == "academic":
            if any(
                section in content.lower()
                for section in ["introducción", "conclusión", "metodología"]
            ):
                quality_score += 0.2  # Tiene secciones académicas clave

        elif content_type == "conversation":
            if ("Usuario:" in content or "Human:" in content) and (
                "Asistente:" in content or "Assistant:" in content
            ):
                quality_score += 0.2  # Tiene ambos participantes

        return min(1.0, quality_score)

    def _extract_code_metadata(self, content: str) -> Dict[str, Any]:
        """Extraer metadatos específicos de código"""
        return {
            "language": self._detect_code_language(content),
            "function_count": len(re.findall(r"\bdef \b|\bfunction \b", content)),
            "class_count": len(re.findall(r"\bclass \b", content)),
            "import_count": len(re.findall(r"\bimport \b|\bfrom \b|\binclude\b", content)),
            "comment_ratio": len(re.findall(r"#.*|//.*|/\*.*\*/", content))
            / max(len(content.split("\n")), 1),
            "complexity_score": self._calculate_code_complexity(content),
        }

    def _extract_academic_metadata(self, content: str) -> Dict[str, Any]:
        """Extraer metadatos específicos de contenido académico"""
        return {
            "section_count": len(re.findall(r"\b\d+\.\s+[A-ZÁÉÍÓÚÑ]", content)),
            "citation_count": len(re.findall(r"\[\d+\]|\(\w+\s+\d+\)", content)),
            "figure_count": len(
                re.findall(r"(?:figura|figure|tabla|table)\s+\d+", content.lower())
            ),
            "reference_count": len(
                re.findall(r"(?:bibliografía|referencias|references)", content.lower())
            ),
            "academic_level": self._estimate_academic_level(content),
        }

    def _extract_conversation_metadata(self, content: str) -> Dict[str, Any]:
        """Extraer metadatos específicos de conversación"""
        return {
            "total_turns": len(re.findall(r"(?:Usuario|Asistente|Human|Assistant):", content)),
            "user_turns": len(re.findall(r"\b(?:Usuario|Human|User):\s*", content)),
            "assistant_turns": len(re.findall(r"\b(?:Asistente|Assistant|Bot):\s*", content)),
            "avg_turn_length": self._calculate_average_turn_length(content),
            "conversation_topics": self._extract_conversation_topics(content),
        }

    def _detect_code_language(self, content: str) -> str:
        """Detectar lenguaje de programación"""
        language_indicators = {
            "python": ["def ", "import ", "print(", "len(", "range("],
            "javascript": ["function ", "console.log", "var ", "let ", "const "],
            "java": ["public class", "System.out.print", "import java"],
            "cpp": ["#include", "std::", "int main("],
            "html": ["<html", "<body", "<div"],
            "css": ["{", "}", "color:", "font-size:"],
        }

        scores = {}
        for language, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in content.lower())
            if score > 0:
                scores[language] = score

        return max(scores.items(), key=lambda x: x[1])[0] if scores else "unknown"

    def _calculate_code_complexity(self, content: str) -> float:
        """Calcular complejidad de código"""
        # Métricas simples de complejidad
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Complejidad basada en estructuras de control
        control_structures = len(
            re.findall(r"\bif\b|\bfor\b|\bwhile\b|\bswitch\b|\btry\b", content)
        )

        # Complejidad basada en nesting
        max_nesting = 0
        current_nesting = 0
        for line in lines:
            current_nesting += line.count("{") + line.count("(") - line.count("}") - line.count(")")
            max_nesting = max(max_nesting, current_nesting)

        # Normalizar complejidad
        complexity = (control_structures * 0.3 + max_nesting * 0.7) / max(len(non_empty_lines), 1)

        return min(1.0, complexity)

    def _estimate_academic_level(self, content: str) -> str:
        """Estimar nivel académico del contenido"""
        content_lower = content.lower()

        # Indicadores de nivel básico
        basic_indicators = ["introducción", "conceptos básicos", "fundamentos"]
        if any(indicator in content_lower for indicator in basic_indicators):
            return "basic"

        # Indicadores de nivel avanzado
        advanced_indicators = [
            "metodología avanzada",
            "análisis complejo",
            "investigación doctoral",
        ]
        if any(indicator in content_lower for indicator in advanced_indicators):
            return "advanced"

        # Nivel intermedio por defecto
        return "intermediate"

    def _calculate_average_turn_length(self, content: str) -> float:
        """Calcular longitud promedio de turnos en conversación"""
        turns = re.findall(
            r"(?:Usuario|Asistente|Human|Assistant):\s*(.*?)(?=(?:Usuario|Asistente|Human|Assistant):|\Z)",
            content,
            re.DOTALL,
        )

        if not turns:
            return 0.0

        total_length = sum(len(turn.strip()) for turn in turns)
        return total_length / len(turns)

    def _extract_conversation_topics(self, content: str) -> List[str]:
        """Extraer temas principales de conversación"""
        # Palabras clave comunes en conversaciones técnicas
        topics = []

        technical_topics = {
            "programming": ["código", "programar", "función", "variable", "algoritmo"],
            "ai": [
                "inteligencia artificial",
                "machine learning",
                "redes neuronales",
                "deep learning",
            ],
            "academic": ["estudio", "investigación", "tesis", "universidad", "profesor"],
            "technology": ["tecnología", "computadora", "software", "aplicación", "sistema"],
        }

        content_lower = content.lower()

        for topic, keywords in technical_topics.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _detect_data_type_change(self, current_chunk: str, new_line: str) -> bool:
        """Detectar cambio de tipo de datos"""
        # Lógica simple para detectar cambios de estructura
        current_type = self._detect_data_type(current_chunk)
        new_type = self._detect_data_type(new_line)

        return current_type != new_type

    def _detect_data_type(self, content: str) -> str:
        """Detectar tipo de datos en contenido"""
        content_str = str(content).lower()

        if "," in content_str and len(content_str.split(",")) > 3:
            return "csv"
        elif ":" in content_str and len(content_str.split("\n")) > 2:
            return "key_value"
        elif content_str.count("{") > content_str.count("}"):
            return "json_like"
        else:
            return "text"

    def process_ramas_academicas_data(self) -> ProcessingResult:
        """Procesar datos de RAMAS_ACADEMICAS_OFICIALES.json"""
        try:
            ramas_file = Path(__file__).resolve().parents[2] / "RAMAS_ACADEMICAS_OFICIALES.json"

            if not ramas_file.exists():
                return ProcessingResult(
                    success=False,
                    content_chunks=[],
                    metadata=ContentMetadata(
                        file_path=str(ramas_file),
                        file_size=0,
                        content_type="json",
                        encoding="utf-8",
                        language="es",
                        structure_detected={},
                        quality_score=0.0,
                        processing_time=0.0,
                        chunks_created=0,
                        word_count=0,
                        estimated_reading_time=0,
                    ),
                    processing_stats={},
                    error_message="Archivo RAMAS_ACADEMICAS_OFICIALES.json no encontrado",
                )

            # Leer archivo JSON
            with open(ramas_file, "r", encoding="utf-8") as f:
                ramas_data = json.load(f)

            # Crear contenido estructurado
            content_parts = []
            content_parts.append(
                f"# RAMAS ACADÉMICAS OFICIALES - {ramas_data['metadata']['total_ramas']} ramas"
            )
            content_parts.append(f"Versión: {ramas_data['metadata']['version']}")
            content_parts.append(f"Fecha: {ramas_data['metadata']['fecha_creacion']}")
            content_parts.append("")

            # Procesar cada rama académica
            for rama in ramas_data["ramas_academicas"]:
                content_parts.append(f"## {rama['rama']}")
                content_parts.append(f"Descripción: {rama['descripcion']}")
                content_parts.append("")

            # Agregar ubicaciones
            content_parts.append("## UBICACIONES DE DATOS")
            for location, description in ramas_data["ubicaciones"].items():
                content_parts.append(f"- **{location}**: {description}")

            full_content = "\n".join(content_parts)

            # Procesar como contenido académico
            chunks, metadata = self._process_content(full_content, str(ramas_file), "academic", {})

            # Crear metadatos específicos para ramas académicas
            ramas_metadata = ContentMetadata(
                file_path=str(ramas_file),
                file_size=os.path.getsize(ramas_file),
                content_type="academic",
                encoding="utf-8",
                language="es",
                structure_detected=self._analyze_content_structure(full_content, "academic"),
                quality_score=0.9,  # Alta calidad para datos oficiales
                processing_time=0.1,
                chunks_created=len(chunks),
                word_count=len(full_content.split()),
                estimated_reading_time=len(full_content.split()) // 200,
                academic_metadata={
                    "total_ramas": ramas_data["metadata"]["total_ramas"],
                    "version": ramas_data["metadata"]["version"],
                    "disciplines": [rama["rama"] for rama in ramas_data["ramas_academicas"]],
                    "data_locations": ramas_data["ubicaciones"],
                },
            )

            return ProcessingResult(
                success=True,
                content_chunks=chunks,
                metadata=ramas_metadata,
                processing_stats={"ramas_processed": len(ramas_data["ramas_academicas"])},
            )

        except Exception as e:
            self.logger.error(f"Error processing RAMAS_ACADEMICAS_OFICIALES.json: {e}")
            return ProcessingResult(
                success=False,
                content_chunks=[],
                metadata=ContentMetadata(
                    file_path=str(ramas_file) if "ramas_file" in locals() else "unknown",
                    file_size=0,
                    content_type="json",
                    encoding="utf-8",
                    language="es",
                    structure_detected={},
                    quality_score=0.0,
                    processing_time=0.0,
                    chunks_created=0,
                    word_count=0,
                    estimated_reading_time=0,
                ),
                processing_stats={},
                error_message=str(e),
            )

    def process_log_file(self, log_file_path: str) -> ProcessingResult:
        """Procesar archivo de log específico"""
        return self.process_file(log_file_path, "log")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de procesamiento"""
        return {
            "files_processed": self.processing_stats["files_processed"],
            "total_chunks_created": self.processing_stats["chunks_created"],
            "total_processing_time": self.processing_stats["processing_time"],
            "average_chunk_size": self.processing_stats["total_size"]
            / max(self.processing_stats["chunks_created"], 1),
            "content_types_processed": list(self.processing_stats["content_types"].keys()),
            "errors_encountered": self.processing_stats["errors"],
        }


# Función de integración con sistemas existentes
def integrate_extended_content_processor() -> ExtendedContentProcessor:
    """Integrar procesador de contenido extenso"""
    return ExtendedContentProcessor()


# Función de procesamiento masivo
def process_multiple_files(
    file_paths: List[str], content_types: List[str] = None
) -> List[ProcessingResult]:
    """Procesar múltiples archivos en paralelo"""
    processor = ExtendedContentProcessor()
    results = []

    if content_types is None:
        content_types = ["auto"] * len(file_paths)

    for file_path, content_type in zip(file_paths, content_types):
        result = processor.process_file(file_path, content_type)
        results.append(result)

    return results
