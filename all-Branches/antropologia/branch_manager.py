#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestor de la rama antropologia - Versión Enterprise
===================================================

Maneja corpus, adaptadores LoRA, sistema RAG, memoria y modelos específicos
para la rama de antropología cultural, social y física.

Author: Sheily AI Team
Version: 2.0.0 - Enterprise Ready
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AntropologiaManager:
    """
    Gestor completo de la rama de antropología.
    
    Maneja todos los aspectos de la rama especializada incluyendo:
    - Carga y gestión de corpus multilingüe
    - Adaptadores LoRA premium optimizados
    - Sistema RAG con índices TF-IDF y Sentence Transformers
    - Integración con sistema de memoria
    - Modelos especializados y configuraciones
    - Pipeline de entrenamiento específico del dominio
    """
    
    def __init__(self, branch_path: Optional[str] = None):
        """
        Inicializar el gestor de la rama antropología.
        
        Args:
            branch_path (Optional[str]): Ruta base de la rama. Si no se proporciona,
                                         se usa el directorio actual o una ruta por defecto.
        """
        self.branch_name = "antropologia"
        
        # Determinar ruta base
        if branch_path:
            self.branch_path = Path(branch_path)
        else:
            # Intentar detectar ruta actual
            current_file = Path(__file__).resolve()
            if current_file.parent.name == "antropologia":
                self.branch_path = current_file.parent
            else:
                self.branch_path = Path("all-Branches/antropologia")
        
        logger.info(f"Inicializando AntropologiaManager en: {self.branch_path}")
        
        # Rutas específicas de la rama
        self.corpus_path = self.branch_path / "corpus"
        self.adapters_path = self.branch_path / "adapters"
        self.memory_path = self.branch_path / "memory"
        self.training_path = self.branch_path / "training"
        
        # Archivo de configuración del modelo (en raíz de la rama)
        self.model_config_path = self.branch_path / "model_config.json"
        
        # Validar estructura de directorios
        self._validate_structure()
        
        # Estado interno
        self._corpus_data = {}
        self._adapters = {}
        self._rag_indices = {}
        self._model_config = None
        self._metadata = self._load_metadata()
    
    def _validate_structure(self) -> None:
        """
        Validar que existen los directorios esenciales de la rama.
        
        Raises:
            FileNotFoundError: Si falta algún directorio crítico.
        """
        essential_dirs = [
            self.corpus_path,
            self.adapters_path,
            self.training_path
        ]
        
        for directory in essential_dirs:
            if not directory.exists():
                logger.warning(f"Directorio no encontrado: {directory}")
                # Crear directorio si no existe
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio creado: {directory}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Cargar metadatos de la rama desde archivos de configuración.
        
        Returns:
            Dict[str, Any]: Metadatos consolidados de la rama.
        """
        metadata = {
            "branch_name": self.branch_name,
            "specialization": "antropología cultural, social y física",
            "loaded": False
        }
        
        # Cargar metadata del corpus
        corpus_meta_path = self.corpus_path / "spanish" / "meta.json"
        if corpus_meta_path.exists():
            try:
                with open(corpus_meta_path, 'r', encoding='utf-8') as f:
                    metadata["corpus"] = json.load(f)
                logger.info("Metadata del corpus cargada correctamente")
            except Exception as e:
                logger.error(f"Error cargando metadata del corpus: {e}")
        
        # Cargar metadata del adaptador
        adapter_meta_path = self.adapters_path / "lora_adapters" / "current" / "metadata.json"
        if adapter_meta_path.exists():
            try:
                with open(adapter_meta_path, 'r', encoding='utf-8') as f:
                    metadata["adapter"] = json.load(f)
                logger.info("Metadata del adaptador cargada correctamente")
            except Exception as e:
                logger.error(f"Error cargando metadata del adaptador: {e}")
        
        # Cargar configuración del modelo
        if self.model_config_path.exists():
            try:
                with open(self.model_config_path, 'r', encoding='utf-8') as f:
                    metadata["model"] = json.load(f)
                logger.info("Configuración del modelo cargada correctamente")
            except Exception as e:
                logger.error(f"Error cargando configuración del modelo: {e}")
        
        return metadata
    
    def load_corpus(self, corpus_type: str = "spanish") -> Dict[str, List[Dict[str, Any]]]:
        """
        Cargar corpus específico de la rama antropología.
        
        Args:
            corpus_type (str): Tipo de corpus a cargar ('spanish', 'training', etc.)
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Datos del corpus organizados por tipo de documento.
        
        Raises:
            FileNotFoundError: Si el corpus especificado no existe.
        """
        logger.info(f"Cargando corpus tipo: {corpus_type}")
        
        corpus_dir = self.corpus_path / corpus_type
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Directorio de corpus no encontrado: {corpus_dir}")
        
        corpus_data = {}
        
        # Cargar todos los archivos JSONL del corpus
        jsonl_files = list(corpus_dir.glob("*.jsonl"))
        
        for file_path in jsonl_files:
            try:
                documents = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                doc = json.loads(line)
                                documents.append(doc)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parseando línea en {file_path.name}: {e}")
                
                corpus_data[file_path.stem] = documents
                logger.info(f"Cargados {len(documents)} documentos de {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error cargando {file_path.name}: {e}")
        
        # Cargar metadata si existe
        meta_path = corpus_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    corpus_data["_metadata"] = json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando metadata: {e}")
        
        self._corpus_data[corpus_type] = corpus_data
        logger.info(f"Corpus '{corpus_type}' cargado exitosamente con {len(corpus_data)} categorías")
        
        return corpus_data
    
    def load_adapters(self, adapter_type: str = "current") -> Dict[str, Any]:
        """
        Cargar adaptadores LoRA específicos de la rama.
        
        Args:
            adapter_type (str): Tipo de adaptador ('current', 'previous', etc.)
        
        Returns:
            Dict[str, Any]: Información y configuración del adaptador cargado.
        
        Raises:
            FileNotFoundError: Si el adaptador especificado no existe.
        """
        logger.info(f"Cargando adaptador LoRA tipo: {adapter_type}")
        
        adapter_dir = self.adapters_path / "lora_adapters" / adapter_type
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Directorio de adaptador no encontrado: {adapter_dir}")
        
        adapter_info = {
            "type": adapter_type,
            "path": str(adapter_dir),
            "loaded": False
        }
        
        # Cargar configuración del adaptador
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    adapter_info["config"] = json.load(f)
                logger.info(f"Configuración del adaptador cargada: rank={adapter_info['config'].get('r')}")
            except Exception as e:
                logger.error(f"Error cargando configuración del adaptador: {e}")
        
        # Cargar metadata del adaptador
        metadata_path = adapter_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    adapter_info["metadata"] = json.load(f)
                logger.info(f"Metadata del adaptador cargada: quality_score={adapter_info['metadata'].get('quality_score')}")
            except Exception as e:
                logger.error(f"Error cargando metadata del adaptador: {e}")
        
        # Verificar existencia del archivo del adaptador
        adapter_file = adapter_dir / "adapter.safetensors"
        if adapter_file.exists():
            adapter_info["adapter_file"] = str(adapter_file)
            adapter_info["adapter_size_mb"] = adapter_file.stat().st_size / (1024 * 1024)
            adapter_info["loaded"] = True
            logger.info(f"Adaptador encontrado: {adapter_info['adapter_size_mb']:.2f} MB")
        else:
            logger.warning(f"Archivo de adaptador no encontrado: {adapter_file}")
        
        self._adapters[adapter_type] = adapter_info
        logger.info(f"Adaptador '{adapter_type}' procesado exitosamente")
        
        return adapter_info
    
    def initialize_rag(self) -> Dict[str, Any]:
        """
        Inicializar sistema RAG específico de la rama antropología.
        
        Carga índices TF-IDF, Sentence Transformers y otros componentes necesarios
        para el sistema de Retrieval-Augmented Generation.
        
        Returns:
            Dict[str, Any]: Información sobre los índices RAG inicializados.
        """
        logger.info("Inicializando sistema RAG para antropología")
        
        rag_info = {
            "initialized": False,
            "indices": {}
        }
        
        corpus_spanish = self.corpus_path / "spanish"
        
        # Cargar índice TF-IDF
        tfidf_dir = corpus_spanish / "tfidf"
        if tfidf_dir.exists():
            try:
                vocab_path = tfidf_dir / "vocabulary.json"
                if vocab_path.exists():
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocabulary = json.load(f)
                    rag_info["indices"]["tfidf"] = {
                        "path": str(tfidf_dir),
                        "vocabulary_size": len(vocabulary),
                        "loaded": True
                    }
                    logger.info(f"Índice TF-IDF cargado: {len(vocabulary)} términos")
            except Exception as e:
                logger.error(f"Error cargando índice TF-IDF: {e}")
        
        # Cargar índice Sentence Transformers
        st_dir = corpus_spanish / "st"
        if st_dir.exists():
            try:
                index_path = st_dir / "index.json"
                if index_path.exists():
                    with open(index_path, 'r', encoding='utf-8') as f:
                        st_index = json.load(f)
                    rag_info["indices"]["sentence_transformers"] = {
                        "path": str(st_dir),
                        "loaded": True,
                        "info": st_index
                    }
                    logger.info("Índice Sentence Transformers cargado")
            except Exception as e:
                logger.error(f"Error cargando índice Sentence Transformers: {e}")
        
        # Cargar índice RAG principal
        rag_index_path = corpus_spanish / "rag_index.json"
        if rag_index_path.exists():
            try:
                with open(rag_index_path, 'r', encoding='utf-8') as f:
                    rag_index = json.load(f)
                rag_info["main_index"] = rag_index
                logger.info("Índice RAG principal cargado")
            except Exception as e:
                logger.error(f"Error cargando índice RAG principal: {e}")
        
        rag_info["initialized"] = len(rag_info["indices"]) > 0
        self._rag_indices = rag_info
        
        logger.info(f"Sistema RAG inicializado con {len(rag_info['indices'])} índices")
        return rag_info
    
    def load_memory(self) -> Dict[str, Any]:
        """
        Cargar sistema de memoria específico de la rama.
        
        Returns:
            Dict[str, Any]: Estado del sistema de memoria.
        """
        logger.info("Cargando sistema de memoria para antropología")
        
        memory_info = {
            "loaded": False,
            "integrators": []
        }
        
        # Verificar integrador de memoria
        memory_integrator_path = self.memory_path / "memory_integrator.py"
        if memory_integrator_path.exists():
            memory_info["integrators"].append({
                "name": "memory_integrator",
                "path": str(memory_integrator_path),
                "available": True
            })
            logger.info("Memory integrator encontrado")
        
        # Verificar integrador LoRA-RAG
        lora_rag_integrator_path = self.memory_path / "lora_rag_integrator.py"
        if lora_rag_integrator_path.exists():
            memory_info["integrators"].append({
                "name": "lora_rag_integrator",
                "path": str(lora_rag_integrator_path),
                "available": True
            })
            logger.info("LoRA-RAG integrator encontrado")
        
        memory_info["loaded"] = len(memory_info["integrators"]) > 0
        logger.info(f"Sistema de memoria cargado con {len(memory_info['integrators'])} integradores")
        
        return memory_info
    
    def get_specialized_model(self) -> Dict[str, Any]:
        """
        Obtener configuración del modelo especializado de la rama.
        
        Returns:
            Dict[str, Any]: Configuración completa del modelo especializado.
        """
        logger.info("Obteniendo configuración del modelo especializado")
        
        if not self.model_config_path.exists():
            logger.warning(f"Configuración del modelo no encontrada en: {self.model_config_path}")
            return {
                "available": False,
                "message": "Configuración del modelo no encontrada"
            }
        
        try:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            model_config["available"] = True
            model_config["config_path"] = str(self.model_config_path)
            
            self._model_config = model_config
            logger.info(f"Modelo especializado: {model_config.get('specialized_model', 'N/A')}")
            
            return model_config
        
        except Exception as e:
            logger.error(f"Error cargando configuración del modelo: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    def train_branch_specific(self, data_path: Optional[str] = None, 
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Entrenar específicamente para la rama de antropología.
        
        Args:
            data_path (Optional[str]): Ruta a datos de entrenamiento personalizados.
                                      Si no se proporciona, usa datos de training/ por defecto.
            config (Optional[Dict[str, Any]]): Configuración personalizada de entrenamiento.
        
        Returns:
            Dict[str, Any]: Resultado del proceso de entrenamiento.
        """
        logger.info("Iniciando entrenamiento específico de antropología")
        
        # Usar datos por defecto si no se especifica ruta
        if data_path is None:
            data_path = str(self.training_path / "train_improved.jsonl")
        
        training_result = {
            "status": "prepared",
            "data_path": data_path,
            "domain": "antropologia"
        }
        
        # Verificar existencia de datos de entrenamiento
        if not Path(data_path).exists():
            logger.error(f"Datos de entrenamiento no encontrados: {data_path}")
            training_result["status"] = "error"
            training_result["error"] = "Datos de entrenamiento no encontrados"
            return training_result
        
        # Contar ejemplos de entrenamiento
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                training_examples = [line for line in f if line.strip()]
            
            training_result["num_examples"] = len(training_examples)
            logger.info(f"Datos de entrenamiento: {len(training_examples)} ejemplos")
        except Exception as e:
            logger.error(f"Error leyendo datos de entrenamiento: {e}")
            training_result["status"] = "error"
            training_result["error"] = str(e)
            return training_result
        
        # Configuración de entrenamiento
        training_config = config or {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "lora_rank": 56,
            "lora_alpha": 112,
            "target_modules": ["c_attn", "c_proj"]
        }
        
        training_result["config"] = training_config
        training_result["status"] = "ready"
        
        logger.info("Configuración de entrenamiento preparada")
        logger.info(f"  - Ejemplos: {training_result['num_examples']}")
        logger.info(f"  - Épocas: {training_config['epochs']}")
        logger.info(f"  - LoRA Rank: {training_config['lora_rank']}")
        
        return training_result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo de la rama antropología.
        
        Returns:
            Dict[str, Any]: Estado completo de todos los componentes.
        """
        logger.info("Generando reporte de estado de la rama")
        
        status = {
            "branch_name": self.branch_name,
            "branch_path": str(self.branch_path),
            "metadata": self._metadata,
            "corpus_loaded": len(self._corpus_data) > 0,
            "adapters_loaded": len(self._adapters) > 0,
            "rag_initialized": self._rag_indices.get("initialized", False),
            "model_config_available": self._model_config is not None
        }
        
        # Estadísticas de corpus
        if self._corpus_data:
            total_docs = sum(
                len(docs) for corpus in self._corpus_data.values()
                for key, docs in corpus.items()
                if key != "_metadata" and isinstance(docs, list)
            )
            status["corpus_stats"] = {
                "types_loaded": len(self._corpus_data),
                "total_documents": total_docs
            }
        
        # Estadísticas de adaptadores
        if self._adapters:
            status["adapter_stats"] = {
                "types_loaded": len(self._adapters),
                "adapters": list(self._adapters.keys())
            }
        
        return status
    
    def initialize_all(self) -> Dict[str, Any]:
        """
        Inicializar todos los componentes de la rama.
        
        Returns:
            Dict[str, Any]: Resultado de la inicialización completa.
        """
        logger.info("=" * 70)
        logger.info(" INICIALIZACIÓN COMPLETA - RAMA ANTROPOLOGÍA ".center(70))
        logger.info("=" * 70)
        
        results = {
            "branch": self.branch_name,
            "success": True,
            "components": {}
        }
        
        # Cargar corpus
        try:
            corpus_data = self.load_corpus("spanish")
            results["components"]["corpus"] = {
                "status": "success",
                "categories": len(corpus_data)
            }
        except Exception as e:
            logger.error(f"Error cargando corpus: {e}")
            results["components"]["corpus"] = {"status": "error", "error": str(e)}
            results["success"] = False
        
        # Cargar adaptador
        try:
            adapter_info = self.load_adapters("current")
            results["components"]["adapter"] = {
                "status": "success",
                "loaded": adapter_info.get("loaded", False)
            }
        except Exception as e:
            logger.error(f"Error cargando adaptador: {e}")
            results["components"]["adapter"] = {"status": "error", "error": str(e)}
            results["success"] = False
        
        # Inicializar RAG
        try:
            rag_info = self.initialize_rag()
            results["components"]["rag"] = {
                "status": "success",
                "initialized": rag_info.get("initialized", False),
                "indices": len(rag_info.get("indices", {}))
            }
        except Exception as e:
            logger.error(f"Error inicializando RAG: {e}")
            results["components"]["rag"] = {"status": "error", "error": str(e)}
            results["success"] = False
        
        # Cargar memoria
        try:
            memory_info = self.load_memory()
            results["components"]["memory"] = {
                "status": "success",
                "loaded": memory_info.get("loaded", False)
            }
        except Exception as e:
            logger.error(f"Error cargando memoria: {e}")
            results["components"]["memory"] = {"status": "error", "error": str(e)}
        
        # Obtener modelo
        try:
            model_config = self.get_specialized_model()
            results["components"]["model"] = {
                "status": "success",
                "available": model_config.get("available", False)
            }
        except Exception as e:
            logger.error(f"Error obteniendo modelo: {e}")
            results["components"]["model"] = {"status": "error", "error": str(e)}
        
        logger.info("=" * 70)
        logger.info(f" Inicialización {'EXITOSA' if results['success'] else 'CON ERRORES'} ".center(70))
        logger.info("=" * 70)
        
        return results


# Instancia global del gestor
antropologia_manager = AntropologiaManager()


if __name__ == "__main__":
    """Ejecutar inicialización completa si se ejecuta directamente."""
    print("\n🌍 Inicializando Gestor de Rama: ANTROPOLOGÍA\n")
    
    manager = AntropologiaManager()
    results = manager.initialize_all()
    
    print("\n📊 ESTADO FINAL:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    status = manager.get_status()
    print("\n📈 ESTADÍSTICAS:")
    print(json.dumps(status, indent=2, ensure_ascii=False))
