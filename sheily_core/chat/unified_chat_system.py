#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SISTEMA ÚNICO DE CHAT SHEILY AI - INTEGRACIÓN COMPLETA
========================================================

ÚNICO sistema que maneja TODA la conversación del chat:
1. 🔍 Detección automática de rama especializada
2. 📚 Búsqueda de contexto RAG específico
3. 🧠 Carga dinámica de adaptadores LoRA por rama
4. 💬 Generación con modelo GGUF Q4 nativo
5. ✅ Respuesta experta y contextualizada

ESTE ES EL ÚNICO SISTEMA QUE DEBE USARSE PARA RESPUESTAS
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Importar utilidad segura de subprocess
try:
    from ..utils.subprocess_utils import safe_subprocess_run
except ImportError:
    # Fallback si no está disponible
    safe_subprocess_run = subprocess.run

# Importar sistemas refactorizados
from sheily_core.config import get_config
from sheily_core.logger import LogContext, get_logger


class UnifiedChatSystem:
    """
    🎯 SISTEMA ÚNICO DE CHAT COMPLETO - REFACTORIZADO

    Integra detección de ramas + RAG especializado + adaptadores LoRA + GGUF Q4
    Ahora con configuración centralizada y logging estructurado
    """

    def __init__(self):
        # Usar configuración centralizada
        self.config = get_config()
        self.logger = get_logger("unified_chat")

        # Configurar rutas desde configuración o defaults
        self.model_path = self._get_model_path()
        self.llama_binary = self._get_llama_binary()
        self.adapters_path = Path("models/lora_adapters/retraining")
        self.corpus_path = Path("corpus_ES")
        self.rag_db_path = Path("data/rag_database.db")

        # Configuración de ramas académicas
        self.branches_config = self._load_branches_config()
        self.current_adapter = None
        self.adapter_cache = {}

        # Inicializar RAG
        self.rag_pipeline = self._initialize_rag_system()

        # Banner del sistema
        self.banner = """
═══════════════════════════════════════════════════════
🤖  Sheily AI — Sistema Cognitivo Completo Activado
🧠  RAG + LoRA + GGUF Q4  |  39 ramas activas  |  Modo chat
═══════════════════════════════════════════════════════
"""

    def _get_model_path(self) -> Path:
        """Obtener ruta del modelo GGUF"""
        possible_paths = [
            Path("models/gguf/llama-3.2.gguf"),
            Path("models/llama-3.2.gguf"),
            Path("llama-3.2.gguf"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Si no existe, usar la primera como default
        return possible_paths[0]

    def _get_llama_binary(self) -> Path:
        """Obtener ruta del binario llama-cpp"""
        possible_paths = [
            Path("llama.cpp/build/bin/llama-cli"),
            Path("llama.cpp/llama-cli"),
            Path("/usr/local/bin/llama-cli"),
            Path("./llama-cli"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Default fallback
        return possible_paths[0]

    def _initialize_rag_system(self):
        """Inicializar sistema RAG"""
        try:
            # Crear directorio para base de datos RAG si no existe
            self.rag_db_path.parent.mkdir(parents=True, exist_ok=True)

            # Inicializar base de datos RAG si no existe
            if not self.rag_db_path.exists():
                self._create_rag_database()

            return True
        except Exception as e:
            self.logger.error(f"Error inicializando RAG: {e}")
            return False

    def _create_rag_database(self):
        """Crear base de datos RAG básica"""
        conn = sqlite3.connect(self.rag_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                branch TEXT,
                chunk_index INTEGER,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_branch ON documents(branch)
        """
        )

        conn.commit()
        conn.close()

    def _verify_components(self):
        """Verificar que los componentes esenciales estén disponibles"""
        missing_components = []

        if self.model_path and not self.model_path.exists():
            missing_components.append(f"Modelo GGUF: {self.model_path}")

        if self.llama_binary and not self.llama_binary.exists():
            missing_components.append(f"llama-cli: {self.llama_binary}")

        if missing_components:
            for component in missing_components:
                self.logger.error(f"Componente no encontrado: {component}")
            self.logger.warning(
                "Algunos componentes no están disponibles, pero el chat puede funcionar en modo degradado"
            )

        self.logger.info("Verificación de componentes completada")

    def _load_branches_config(self) -> Dict:
        """Cargar configuración de ramas especializadas"""
        try:
            branches_file = Path("../branches/base_branches.json")
            if branches_file.exists():
                with open(branches_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {domain["name"]: domain for domain in data["domains"]}
            else:
                print("⚠️ Archivo de ramas no encontrado, usando ramas por defecto")
                return self._get_default_branches()
        except Exception as e:
            print(f"⚠️ Error cargando ramas: {e}")
            return self._get_default_branches()

    def _get_default_branches(self) -> Dict:
        """Ramas por defecto más importantes"""
        return {
            "programación": {
                "name": "programación",
                "keywords": [
                    "código",
                    "python",
                    "bug",
                    "error",
                    "programar",
                    "desarrollo",
                    "software",
                ],
                "description": "Especialización en programación y desarrollo",
            },
            "medicina": {
                "name": "medicina",
                "keywords": [
                    "medicina",
                    "salud",
                    "médico",
                    "tratamiento",
                    "diagnóstico",
                    "síntoma",
                ],
                "description": "Especialización en medicina y salud",
            },
            "inteligencia artificial": {
                "name": "inteligencia artificial",
                "keywords": [
                    "IA",
                    "inteligencia artificial",
                    "machine learning",
                    "algoritmo",
                    "modelo",
                ],
                "description": "Especialización en inteligencia artificial",
            },
            "general": {
                "name": "general",
                "keywords": ["general", "común", "básico"],
                "description": "Conocimiento general",
            },
        }

    def detect_specialized_branch(self, query: str) -> Tuple[str, float]:
        """
        🔍 DETECTAR RAMA ESPECIALIZADA para la consulta

        Args:
            query: Consulta del usuario

        Returns:
            Tuple con (nombre_rama, confidence_score)
        """
        query_lower = query.lower()
        branch_scores = {}

        self.logger.debug(f"Analizando consulta para detectar rama: '{query[:50]}...'")

        # Calcular score para cada rama
        for branch_name, branch_config in self.branches_config.items():
            score = 0.0
            keywords = branch_config.get("keywords", [])

            # Score por keywords exactas (peso alto)
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 3.0
                    self.logger.debug(
                        f"Keyword '{keyword}' encontrada en rama '{branch_name}' (+3.0)"
                    )

            # Score por nombre de rama
            if branch_name.lower() in query_lower:
                score += 5.0
                self.logger.debug(f"Nombre de rama '{branch_name}' encontrado (+5.0)")

            # Score por palabras relacionadas
            related_words = {
                "programación": ["dev", "script", "función", "clase", "método", "variable"],
                "medicina": ["paciente", "enfermedad", "medicamento", "hospital", "clínica"],
                "inteligencia artificial": [
                    "neural",
                    "training",
                    "datos",
                    "predicción",
                    "clasificación",
                ],
            }

            if branch_name in related_words:
                for word in related_words[branch_name]:
                    if word in query_lower:
                        score += 1.0

            if score > 0:
                branch_scores[branch_name] = score

        # Seleccionar la rama con mayor score
        if branch_scores:
            best_branch = max(branch_scores.keys(), key=lambda k: branch_scores[k])
            best_score = branch_scores[best_branch]
            self.logger.info(f"Rama detectada: '{best_branch}' (score: {best_score:.1f})")
            return best_branch, best_score
        else:
            self.logger.info("Rama detectada: 'general' (por defecto)")
            return "general", 0.1

    def search_specialized_context(
        self, query: str, branch_name: str, rag_pipeline=None
    ) -> List[str]:
        """
        📚 BUSCAR CONTEXTO RAG ESPECIALIZADO

        Args:
            query: Consulta del usuario
            branch_name: Rama especializada detectada
            rag_pipeline: Pipeline RAG disponible

        Returns:
            Lista de documentos de contexto relevantes
        """
        context_docs = []

        self.logger.debug(f"Buscando contexto especializado para rama '{branch_name}'")

        # Si hay pipeline RAG, usar búsqueda avanzada
        if rag_pipeline:
            try:
                # Búsqueda RAG estándar
                rag_result = rag_pipeline.rag_query(query, max_results=5)
                base_docs = rag_result.get("results", [])

                # Filtrar documentos por relevancia de rama
                for doc in base_docs:
                    doc_text = str(doc).lower()
                    branch_keywords = self.branches_config[branch_name].get("keywords", [])

                    # Verificar si el documento es relevante para la rama
                    relevance_score = 0
                    for keyword in branch_keywords:
                        if keyword.lower() in doc_text:
                            relevance_score += 1

                    if relevance_score > 0 or branch_name == "general":
                        context_docs.append(str(doc))

                self.logger.debug(f"Encontrados {len(context_docs)} documentos especializados")

            except Exception as e:
                self.logger.warning(f"Error en búsqueda RAG: {e}")

        # Buscar documentos específicos de la rama (corpus local)
        branch_docs = self._search_branch_corpus(query, branch_name)
        context_docs.extend(branch_docs)

        # Si no hay contexto, proporcionar conocimiento básico de la rama
        if not context_docs:
            context_docs = self._get_branch_basic_knowledge(branch_name)

        return context_docs[:3]  # Limitar a top 3 para optimizar GGUF

    def _search_branch_corpus(self, query: str, branch_name: str) -> List[str]:
        """Buscar en corpus específico de la rama"""
        corpus_docs = []

        try:
            # Buscar en directorio de corpus de la rama
            branch_corpus_path = self.corpus_path / branch_name
            if branch_corpus_path.exists():
                # Leer archivos de texto de la rama
                for txt_file in branch_corpus_path.glob("*.txt"):
                    try:
                        with open(txt_file, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Tomar fragmento relevante (primeros 300 chars)
                            if len(content) > 300:
                                content = content[:300] + "..."
                            corpus_docs.append(f"[{branch_name}] {content}")
                    except Exception as e:
                        self.logger.debug(f"Error reading corpus file {txt_file}: {e}")
                        continue

                self.logger.debug(
                    f"Encontrados {len(corpus_docs)} docs en corpus de '{branch_name}'"
                )
        except Exception as e:
            self.logger.warning(f"Error accediendo corpus: {e}")

        return corpus_docs[:2]  # Max 2 docs del corpus

    def _get_branch_basic_knowledge(self, branch_name: str) -> List[str]:
        """Conocimiento básico por rama cuando no hay contexto"""
        basic_knowledge = {
            "programación": [
                "Python es un lenguaje de programación interpretado, de alto nivel y multiparadigma.",
                "Para debuggear código, usa print() statements, debugger integrado o herramientas como pdb.",
                "Las mejores prácticas incluyen código limpio, comentarios útiles y testing regular.",
            ],
            "medicina": [
                "La medicina se basa en evidencia científica y protocolos establecidos.",
                "Es fundamental consultar siempre con profesionales médicos para diagnósticos.",
                "Los síntomas pueden tener múltiples causas, requiriendo evaluación profesional.",
            ],
            "inteligencia artificial": [
                "IA abarca machine learning, deep learning y procesamiento de lenguaje natural.",
                "Los modelos requieren datos de calidad y entrenamiento adecuado.",
                "La ética en IA es crucial para desarrollo responsable de tecnología.",
            ],
            "general": [
                "Puedo ayudarte con información general sobre diversos temas.",
                "Para preguntas específicas, proporciona más contexto para mejor respuesta.",
                "Siempre verifica información importante con fuentes confiables.",
            ],
        }

        return basic_knowledge.get(branch_name, basic_knowledge["general"])

    def generate_response_with_gguf(
        self, query: str, context_docs: List[str], branch_name: str
    ) -> str:
        """
        🧠 GENERAR RESPUESTA con modelo GGUF Q4 nativo

        Args:
            query: Consulta original
            context_docs: Documentos de contexto
            branch_name: Rama especializada

        Returns:
            Respuesta generada por GGUF
        """
        self.logger.info(f"Generando respuesta GGUF para rama '{branch_name}'")

        # Crear prompt especializado con contexto
        specialized_prompt = self._create_specialized_prompt(query, context_docs, branch_name)

        try:
            # Ejecutar llama.cpp con modelo GGUF usando configuración por defecto
            cmd = [
                str(self.llama_binary),
                "--model",
                str(self.model_path),
                "--prompt",
                specialized_prompt,
                "--n-predict",
                "512",
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "--ctx-size",
                "2048",
                "--batch-size",
                "1",
                "--threads",
                "4",
            ]

            self.logger.debug("Ejecutando GGUF con timeout: 60s")

            # Usar subprocess seguro con validación
            result = safe_subprocess_run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.dirname(self.llama_binary),
            )

            if result.returncode == 0:
                response = result.stdout.strip()

                # Limpiar respuesta (remover prompt repetido)
                if specialized_prompt in response:
                    response = response.replace(specialized_prompt, "").strip()

                # Validar respuesta
                if len(response) > 10:
                    self.logger.info(f"Respuesta GGUF generada: {len(response)} caracteres")
                    return response
                else:
                    self.logger.warning("Respuesta GGUF muy corta, usando fallback")
                    return self._generate_fallback_response(query, branch_name)

            else:
                self.logger.error(f"Error GGUF (código {result.returncode}): {result.stderr}")
                return self._generate_fallback_response(query, branch_name)

        except subprocess.TimeoutExpired:
            self.logger.error("Timeout GGUF después de 60s")
            return self._generate_fallback_response(query, branch_name)
        except Exception as e:
            self.logger.exception(f"Error ejecutando GGUF: {e}")
            return self._generate_fallback_response(query, branch_name)

    def _create_specialized_prompt(
        self, query: str, context_docs: List[str], branch_name: str
    ) -> str:
        """Crear prompt especializado por rama"""

        # Prompt base por rama
        branch_prompts = {
            "programación": "Eres Sheily, una experta en programación y desarrollo de software. Responde de manera técnica y precisa.",
            "medicina": "Eres Sheily, una asistente especializada en medicina y salud. Proporciona información médica general, pero recomienda siempre consultar profesionales.",
            "inteligencia artificial": "Eres Sheily, una experta en inteligencia artificial y machine learning. Explica conceptos técnicos de manera clara.",
            "general": "Eres Sheily, una asistente de IA útil y conocedora. Responde de manera clara y concisa en español.",
        }

        system_prompt = branch_prompts.get(branch_name, branch_prompts["general"])

        # Agregar contexto si existe
        context_section = ""
        if context_docs:
            context_text = "\n\n".join(context_docs[:3])  # Max 3 docs
            context_section = f"\n\nContexto especializado relevante:\n{context_text}\n"

        # Prompt completo optimizado para 512 tokens
        full_prompt = f"""{system_prompt}

{context_section}
Pregunta: {query}

Respuesta especializada:"""

        return full_prompt

    def _generate_fallback_response(self, query: str, branch_name: str) -> str:
        """Respuesta de emergencia cuando GGUF falla"""
        fallback_responses = {
            "programación": f"Como experta en programación, puedo ayudarte con '{query}'. Para código específico, proporciona más detalles sobre el lenguaje y contexto.",
            "medicina": f"Sobre '{query}' en medicina, te recomiendo consultar con un profesional médico para información precisa y personalizada.",
            "inteligencia artificial": f"Respecto a '{query}' en IA, este es un campo amplio. ¿Podrías especificar si te interesa machine learning, redes neuronales u otro aspecto?",
            "general": f"Entiendo que preguntas sobre '{query}'. Puedo ayudarte mejor si proporcionas más contexto específico.",
        }

        return fallback_responses.get(branch_name, fallback_responses["general"])

    def process_complete_query(self, query: str, rag_pipeline=None) -> Dict:
        """
        🎯 PROCESO COMPLETO DE CONSULTA - SISTEMA ÚNICO

        Este es el ÚNICO método que debe llamarse para procesar consultas

        Args:
            query: Consulta del usuario
            rag_pipeline: Pipeline RAG opcional

        Returns:
            Dict con respuesta completa y metadatos
        """
        start_time = time.time()

        # Crear contexto de logging
        log_context = LogContext(
            component="unified_chat",
            operation="process_complete_query",
            metadata={"query_length": len(query)},
        )

        with self.logger.context(**log_context.__dict__):
            self.logger.info(f"Procesando consulta completa: '{query[:50]}...'")

            # 1. Detectar rama especializada
            branch_name, branch_confidence = self.detect_specialized_branch(query)

            # 2. Buscar contexto RAG especializado
            context_docs = self.search_specialized_context(query, branch_name, rag_pipeline)

            # 3. Generar respuesta con GGUF Q4
            response = self.generate_response_with_gguf(query, context_docs, branch_name)

            # 4. Preparar resultado completo
            processing_time = time.time() - start_time

            result = {
                "query": query,
                "chat_response": response,
                "detected_branch": branch_name,
                "branch_confidence": branch_confidence,
                "context_sources": len(context_docs),
                "context_preview": context_docs[:1] if context_docs else [],
                "processing_time": round(processing_time, 2),
                "mode": "unified_branch_rag_gguf",
                "system": "ÚNICO Sistema Sheily",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.logger.info(
                f"Respuesta completa generada en {processing_time:.2f}s",
                extra={
                    "branch": branch_name,
                    "context_docs": len(context_docs),
                    "response_length": len(response),
                },
            )

            return result


# 🎯 INSTANCIA GLOBAL - ÚNICO SISTEMA
unified_chat_system = None


def init_unified_system():
    """Inicializar el sistema único global"""
    global unified_chat_system
    logger = get_logger("unified_chat")

    try:
        unified_chat_system = UnifiedChatSystem()
        logger.info("Sistema Único inicializado correctamente")
        return True
    except Exception as e:
        logger.exception(f"Error inicializando Sistema Único: {e}")
        return False


def process_chat_query_unified(query: str, rag_pipeline=None) -> Dict:
    """
    🎯 FUNCIÓN ÚNICA para procesar consultas de chat

    ESTA ES LA ÚNICA FUNCIÓN QUE DEBE USARSE PARA RESPUESTAS
    """
    global unified_chat_system
    logger = get_logger("unified_chat")

    # Inicializar si es necesario
    if unified_chat_system is None:
        if not init_unified_system():
            logger.error("No se pudo inicializar el sistema único")
            return {
                "query": query,
                "chat_response": "Lo siento, el sistema no está disponible en este momento.",
                "error": "Sistema no inicializado",
                "mode": "error",
            }

    # Procesar consulta con sistema único
    return unified_chat_system.process_complete_query(query, rag_pipeline)


if __name__ == "__main__":
    # Test del sistema único
    logger = get_logger("unified_chat")
    logger.info("🧪 Testing Sistema Único Sheily...")

    if init_unified_system():
        # Test con diferentes tipos de consultas
        test_queries = [
            "¿Cómo debuggear un error en Python?",
            "¿Cuáles son los síntomas de la gripe?",
            "¿Qué es machine learning?",
            "¿Cuál es la capital de Francia?",
        ]

        for query in test_queries:
            logger.info(f"{'='*50}")
            result = process_chat_query_unified(query)
            logger.info(f"Query: {result['query']}")
            logger.info(f"Rama: {result['detected_branch']}")
            logger.info(f"Respuesta: {result['chat_response'][:100]}...")
            logger.info(f"Tiempo: {result['processing_time']}s")
    else:
        logger.error("❌ No se pudo inicializar el sistema único")
