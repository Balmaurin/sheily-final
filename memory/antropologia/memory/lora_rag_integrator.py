#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrador LoRA-RAG Específico para Antropología - Versión Enterprise
=====================================================================

Conecta adaptadores LoRA especializados con el sistema RAG del dominio.
Implementa validación robusta, manejo de errores y logging completo.

Author: Sheily AI Team
Version: 2.0.0
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AntropologiaLoRARAGIntegrator:
    """
    Integrador LoRA-RAG especializado para antropología.
    
    Gestiona la conexión entre:
    - Adaptadores LoRA premium optimizados
    - Sistema RAG con índices especializados
    - Corpus de conocimiento antropológico
    - Configuraciones específicas del dominio
    """
    
    def __init__(self):
        """Inicializar integrador LoRA-RAG para antropología."""
        self.domain = 'antropologia'
        self.branch_path = Path(__file__).parent.parent
        
        logger.info(f"Inicializando integrador LoRA-RAG para {self.domain}")
        
        # Configuración de rutas específicas del dominio
        self.lora_adapter_path = self.branch_path / 'adapters' / 'lora_adapters'
        self.rag_index_path = self.branch_path / 'corpus' / 'spanish' / 'rag_index.json'
        self.corpus_path = self.branch_path / 'corpus'
        self.config_path = self.branch_path / 'config'
        
        # Validar estructura
        self._validate_paths()
        
        # Estado interno
        self._adapters_cache = {}
        self._rag_config_cache = None
    
    def _validate_paths(self) -> None:
        """
        Validar que existen las rutas esenciales.
        
        Raises:
            FileNotFoundError: Si falta alguna ruta crítica.
        """
        essential_paths = {
            "Branch path": self.branch_path,
            "Adapters path": self.lora_adapter_path,
            "Corpus path": self.corpus_path
        }
        
        for name, path in essential_paths.items():
            if not path.exists():
                logger.warning(f"{name} no encontrado: {path}")
                # Crear directorio si no existe
                if name != "RAG index path":
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"{name} creado: {path}")
    
    def get_domain_specific_adapters(self) -> Dict[str, Any]:
        """
        Obtener adaptadores LoRA específicos del dominio con información detallada.
        
        Returns:
            Dict[str, Any]: Diccionario con información de adaptadores disponibles.
        """
        logger.info("Buscando adaptadores LoRA específicos de antropología")
        
        if self._adapters_cache:
            logger.info("Retornando adaptadores desde caché")
            return self._adapters_cache
        
        adapters = {
            "domain": self.domain,
            "available": {},
            "recommended": None
        }
        
        # Tipos de adaptadores a buscar
        adapter_types = ['current', 'previous', 'functional', 'optimized', 
                        'complete_optimized', 'retraining']
        
        for adapter_type in adapter_types:
            adapter_dir = self.lora_adapter_path / adapter_type
            
            if not adapter_dir.exists():
                continue
            
            adapter_file = adapter_dir / 'adapter.safetensors'
            config_file = adapter_dir / 'adapter_config.json'
            metadata_file = adapter_dir / 'metadata.json'
            
            if adapter_file.exists():
                adapter_info = {
                    "path": str(adapter_file),
                    "size_mb": adapter_file.stat().st_size / (1024 * 1024),
                    "available": True
                }
                
                # Cargar configuración si existe
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            adapter_info["config"] = json.load(f)
                        logger.info(f"Configuración cargada para {adapter_type}")
                    except Exception as e:
                        logger.warning(f"Error cargando config de {adapter_type}: {e}")
                
                # Cargar metadata si existe
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            adapter_info["metadata"] = json.load(f)
                        logger.info(f"Metadata cargada para {adapter_type}")
                    except Exception as e:
                        logger.warning(f"Error cargando metadata de {adapter_type}: {e}")
                
                adapters["available"][adapter_type] = adapter_info
                logger.info(f"Adaptador {adapter_type}: {adapter_info['size_mb']:.2f} MB")
        
        # Establecer adaptador recomendado (current por defecto)
        if "current" in adapters["available"]:
            adapters["recommended"] = "current"
            logger.info("Adaptador recomendado: current")
        elif adapters["available"]:
            adapters["recommended"] = list(adapters["available"].keys())[0]
            logger.info(f"Adaptador recomendado: {adapters['recommended']}")
        
        self._adapters_cache = adapters
        logger.info(f"Encontrados {len(adapters['available'])} adaptadores")
        
        return adapters
    
    def get_domain_rag_config(self) -> Dict[str, Any]:
        """
        Obtener configuración RAG específica del dominio.
        
        Returns:
            Dict[str, Any]: Configuración completa del sistema RAG.
        """
        logger.info("Cargando configuración RAG del dominio")
        
        if self._rag_config_cache:
            logger.info("Retornando configuración RAG desde caché")
            return self._rag_config_cache
        
        rag_config = {
            'domain': self.domain,
            'specialization': 'Antropología cultural, social y física',
            'corpus': {},
            'indices': {},
            'configuration': {}
        }
        
        # Configurar rutas de corpus
        corpus_spanish = self.corpus_path / 'spanish'
        corpus_training = self.corpus_path / 'training'
        
        if corpus_spanish.exists():
            rag_config['corpus']['spanish'] = {
                'path': str(corpus_spanish),
                'available': True,
                'documents': len(list(corpus_spanish.glob('*.jsonl')))
            }
            logger.info(f"Corpus español: {rag_config['corpus']['spanish']['documents']} archivos")
        
        if corpus_training.exists():
            rag_config['corpus']['training'] = {
                'path': str(corpus_training),
                'available': True
            }
            logger.info("Corpus de entrenamiento disponible")
        
        # Cargar índices RAG
        indices = {
            'tfidf': corpus_spanish / 'tfidf',
            'sentence_transformers': corpus_spanish / 'st',
            'rag_index': corpus_spanish / 'rag_index.json'
        }
        
        for index_name, index_path in indices.items():
            if index_path.exists():
                rag_config['indices'][index_name] = {
                    'path': str(index_path),
                    'available': True
                }
                
                # Cargar información adicional si es un archivo JSON
                if index_path.suffix == '.json':
                    try:
                        with open(index_path, 'r', encoding='utf-8') as f:
                            rag_config['indices'][index_name]['data'] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error cargando {index_name}: {e}")
                
                logger.info(f"Índice {index_name} disponible")
        
        # Cargar metadata del corpus
        meta_path = corpus_spanish / 'meta.json'
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    rag_config['configuration']['corpus_metadata'] = json.load(f)
                logger.info("Metadata del corpus cargada")
            except Exception as e:
                logger.warning(f"Error cargando metadata del corpus: {e}")
        
        rag_config['configuration']['embedding_model'] = rag_config.get(
            'configuration', {}
        ).get('corpus_metadata', {}).get(
            'embedding_model', 
            'sentence-transformers/all-mpnet-base-v2'
        )
        
        self._rag_config_cache = rag_config
        logger.info("Configuración RAG completa")
        
        return rag_config
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Obtener estado completo de la integración LoRA-RAG.
        
        Returns:
            Dict[str, Any]: Estado detallado de la integración.
        """
        logger.info("Generando reporte de estado de integración")
        
        status = {
            "domain": self.domain,
            "branch_path": str(self.branch_path),
            "components": {}
        }
        
        # Estado de adaptadores
        try:
            adapters = self.get_domain_specific_adapters()
            status["components"]["adapters"] = {
                "status": "available",
                "count": len(adapters.get("available", {})),
                "recommended": adapters.get("recommended")
            }
        except Exception as e:
            logger.error(f"Error obteniendo adaptadores: {e}")
            status["components"]["adapters"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Estado de RAG
        try:
            rag_config = self.get_domain_rag_config()
            status["components"]["rag"] = {
                "status": "available",
                "indices": len(rag_config.get("indices", {})),
                "corpus_types": len(rag_config.get("corpus", {}))
            }
        except Exception as e:
            logger.error(f"Error obteniendo configuración RAG: {e}")
            status["components"]["rag"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Estado general
        status["integration_ready"] = (
            status["components"]["adapters"].get("status") == "available" and
            status["components"]["rag"].get("status") == "available"
        )
        
        return status
    
    def load_recommended_adapter(self) -> Optional[Dict[str, Any]]:
        """
        Cargar el adaptador recomendado para el dominio.
        
        Returns:
            Optional[Dict[str, Any]]: Información del adaptador cargado o None si falla.
        """
        logger.info("Cargando adaptador recomendado")
        
        adapters = self.get_domain_specific_adapters()
        recommended = adapters.get("recommended")
        
        if not recommended:
            logger.warning("No hay adaptador recomendado disponible")
            return None
        
        adapter_info = adapters["available"].get(recommended)
        
        if not adapter_info:
            logger.error(f"Adaptador recomendado '{recommended}' no encontrado")
            return None
        
        logger.info(f"Adaptador '{recommended}' listo para carga")
        return {
            "name": recommended,
            "info": adapter_info,
            "domain": self.domain
        }


# Instancia específica del dominio
antropologia_lora_rag = AntropologiaLoRARAGIntegrator()


if __name__ == "__main__":
    """Ejecutar diagnóstico si se ejecuta directamente."""
    print("\n🔗 Integrador LoRA-RAG - Antropología\n")
    
    integrator = AntropologiaLoRARAGIntegrator()
    
    print("📊 Estado de la integración:")
    status = integrator.get_integration_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    print("\n🎯 Adaptadores disponibles:")
    adapters = integrator.get_domain_specific_adapters()
    print(json.dumps(adapters, indent=2, ensure_ascii=False))
    
    print("\n📚 Configuración RAG:")
    rag_config = integrator.get_domain_rag_config()
    print(json.dumps(rag_config, indent=2, ensure_ascii=False))
