"""
Universal System Manager
========================

Gestiona el sistema unificado global de Sheily:
- Corpus global sin fragmentación
- RAG universal sobre todo el conocimiento
- Adaptador LoRA único que mejora continuamente
- Auto-integración de cualquier dataset
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

logger = logging.getLogger(__name__)


class UniversalManager:
    """
    Gestor del Sistema Universal Sheily
    
    Un único sistema que maneja TODO el conocimiento sin separación por dominios.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Inicializa el gestor universal
        
        Args:
            base_path: Ruta base del sistema (por defecto: all-Branches/universal/)
        """
        if base_path is None:
            base_path = Path(__file__).parent
        
        self.base_path = Path(base_path)
        self.config_path = self.base_path / "system_config.json"
        self.corpus_path = self.base_path / "corpus" / "unified"
        self.incoming_path = self.base_path / "corpus" / "incoming"
        self.adapter_path = self.base_path / "adapters" / "universal_lora"
        self.rag_path = self.base_path / "rag"
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Estado del sistema
        self.model = None
        self.tokenizer = None
        self.corpus_index = None
        self.rag_retriever = None
        
        logger.info(f"UniversalManager inicializado en: {self.base_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración del sistema"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encuentra system_config.json en {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def initialize_all(self) -> Dict[str, Any]:
        """
        Inicializa todos los componentes del sistema universal
        
        Returns:
            Estado completo del sistema
        """
        logger.info("🚀 Inicializando Sistema Universal Sheily...")
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": self.config["system_name"],
            "version": self.config["version"],
            "components": {}
        }
        
        # 1. Inicializar corpus
        corpus_status = self._initialize_corpus()
        status["components"]["corpus"] = corpus_status
        
        # 2. Inicializar RAG
        rag_status = self._initialize_rag()
        status["components"]["rag"] = rag_status
        
        # 3. Inicializar modelo y adaptador
        model_status = self._initialize_model()
        status["components"]["model"] = model_status
        
        # 4. Procesar datos pendientes
        if self.config["corpus"]["auto_process"]:
            incoming_status = self._process_incoming_data()
            status["components"]["incoming"] = incoming_status
        
        logger.info("✅ Sistema Universal inicializado correctamente")
        return status
    
    def _initialize_corpus(self) -> Dict[str, Any]:
        """Inicializa el corpus unificado global"""
        logger.info("📚 Inicializando corpus unificado...")
        
        self.corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Contar documentos en el corpus
        corpus_files = list(self.corpus_path.glob("*.jsonl"))
        total_docs = 0
        for file in corpus_files:
            with open(file, 'r', encoding='utf-8') as f:
                total_docs += sum(1 for _ in f)
        
        return {
            "status": "initialized",
            "files": len(corpus_files),
            "total_documents": total_docs,
            "path": str(self.corpus_path)
        }
    
    def _initialize_rag(self) -> Dict[str, Any]:
        """Inicializa el sistema RAG universal"""
        logger.info("🔍 Inicializando RAG universal...")
        
        self.rag_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar si existe índice RAG previo
        rag_index_path = self.rag_path / "universal_rag_index.pkl"
        tfidf_path = self.rag_path / "tfidf_vectorizer.pkl"
        st_embeddings_path = self.rag_path / "st_embeddings.npy"
        documents_path = self.rag_path / "documents.json"
        
        if rag_index_path.exists() and documents_path.exists():
            logger.info("Cargando índice RAG universal existente...")
            
            # Cargar documentos
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Cargar TF-IDF
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    tfidf_vectorizer = pickle.load(f)
                tfidf_status = "loaded"
            else:
                tfidf_vectorizer = None
                tfidf_status = "not_found"
            
            # Cargar Sentence Transformer embeddings
            if st_embeddings_path.exists():
                st_embeddings = np.load(st_embeddings_path)
                st_status = "loaded"
            else:
                st_embeddings = None
                st_status = "not_found"
            
            return {
                "status": "loaded",
                "enabled": self.config["rag"]["enabled"],
                "top_k": self.config["rag"]["retrieval_top_k"],
                "documents_count": len(documents),
                "tfidf": tfidf_status,
                "sentence_transformer": st_status,
                "path": str(self.rag_path)
            }
        else:
            logger.info("Índice RAG no encontrado. Crear con build_rag_index()")
            
            return {
                "status": "not_indexed",
                "enabled": self.config["rag"]["enabled"],
                "top_k": self.config["rag"]["retrieval_top_k"],
                "path": str(self.rag_path),
                "action_required": "Ejecutar build_rag_index() para crear el índice"
            }
    
    def _initialize_model(self) -> Dict[str, Any]:
        """Inicializa el modelo base y el adaptador universal"""
        logger.info("🤖 Inicializando modelo y adaptador universal...")
        
        model_config = self.config["model"]
        lora_config = self.config["lora"]
        
        # Detectar dispositivo
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
            except ImportError:
                device = torch.device("cpu")
        
        logger.info(f"Dispositivo: {device}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["base_model"],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Verificar si existe adaptador entrenado
        current_adapter = self.adapter_path / "current"
        
        if current_adapter.exists() and (current_adapter / "adapter_config.json").exists():
            # Cargar modelo con adaptador existente
            logger.info("Cargando adaptador universal existente...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config["base_model"],
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                str(current_adapter),
                is_trainable=True
            )
            
            # Leer metadata
            metadata_path = current_adapter / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                training_info = {
                    "total_examples": metadata.get("total_examples_trained", 0),
                    "last_update": metadata.get("last_training", "unknown"),
                    "datasets_trained": len(metadata.get("training_history", []))
                }
            else:
                training_info = {"status": "no_metadata"}
        else:
            # Crear nuevo adaptador
            logger.info("Creando nuevo adaptador universal...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config["base_model"],
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config["r"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                target_modules=lora_config["target_modules"],
                bias=lora_config["bias"]
            )
            
            self.model = get_peft_model(base_model, peft_config)
            training_info = {"status": "new_adapter", "total_examples": 0}
        
        self.model.to(device)
        
        # Contar parámetros
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "status": "loaded",
            "base_model": model_config["base_model"],
            "device": str(device),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": f"{100 * trainable_params / total_params:.2f}%",
            "training": training_info
        }
    
    def _process_incoming_data(self) -> Dict[str, Any]:
        """Procesa automáticamente datos en corpus/incoming/"""
        logger.info("📥 Procesando datos entrantes...")
        
        self.incoming_path.mkdir(parents=True, exist_ok=True)
        
        incoming_files = list(self.incoming_path.glob("*.jsonl"))
        
        if not incoming_files:
            return {
                "status": "no_incoming_data",
                "files_processed": 0
            }
        
        processed = []
        for file in incoming_files:
            # Mover a corpus unificado
            target = self.corpus_path / file.name
            
            # Si ya existe, fusionar
            if target.exists():
                logger.info(f"Fusionando {file.name} con corpus existente...")
                # TODO: Implementar fusión inteligente
            else:
                file.rename(target)
                logger.info(f"Añadido {file.name} al corpus unificado")
            
            processed.append(file.name)
        
        return {
            "status": "processed",
            "files_processed": len(processed),
            "files": processed
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del sistema universal
        
        Returns:
            Diccionario con toda la información del sistema
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": self.config["system_name"],
            "version": self.config["version"]
        }
        
        # Estado del corpus
        corpus_files = list(self.corpus_path.glob("*.jsonl"))
        total_docs = sum(1 for f in corpus_files for _ in open(f, 'r', encoding='utf-8'))
        status["corpus"] = {
            "files": len(corpus_files),
            "total_documents": total_docs
        }
        
        # Estado del adaptador
        current_adapter = self.adapter_path / "current"
        if current_adapter.exists():
            metadata_path = current_adapter / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                status["adapter"] = {
                    "status": "trained",
                    "total_examples": metadata.get("total_examples_trained", 0),
                    "last_update": metadata.get("last_training", "unknown"),
                    "datasets": len(metadata.get("training_history", []))
                }
            else:
                status["adapter"] = {"status": "exists_no_metadata"}
        else:
            status["adapter"] = {"status": "not_trained"}
        
        # Estado del modelo
        if self.model is not None:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            status["model"] = {
                "loaded": True,
                "trainable_params": trainable,
                "total_params": total
            }
        else:
            status["model"] = {"loaded": False}
        
        return status
    
    def build_rag_index(self) -> Dict[str, Any]:
        """
        Construye el índice RAG universal desde el corpus unificado
        
        Utiliza TF-IDF y Sentence Transformers para crear índices de búsqueda.
        
        Returns:
            Información sobre el índice creado
        """
        logger.info("🔨 Construyendo índice RAG universal...")
        
        # Cargar todos los documentos del corpus
        corpus_files = list(self.corpus_path.glob("*.jsonl"))
        
        if not corpus_files:
            raise FileNotFoundError(f"No hay documentos en {self.corpus_path}")
        
        documents = []
        for file in corpus_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Saltar líneas vacías
                        continue
                    try:
                        doc = json.loads(line)
                        # Combinar instruction y output
                        text = f"{doc.get('instruction', '')} {doc.get('output', '')}"
                        documents.append({
                            "id": f"{file.stem}_{len(documents)}",
                            "text": text,
                            "metadata": {
                                "source": file.name,
                                "instruction": doc.get('instruction', ''),
                                "output": doc.get('output', '')
                            }
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parseando línea en {file.name}: {e}")
                        continue
        
        logger.info(f"📚 Cargados {len(documents)} documentos del corpus")
        
        # 1. Crear índice TF-IDF
        logger.info("🔧 Creando índice TF-IDF...")
        texts = [doc["text"] for doc in documents]
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # Guardar TF-IDF
        tfidf_path = self.rag_path / "tfidf_vectorizer.pkl"
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        logger.info(f"✅ TF-IDF: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} términos")
        
        # 2. Crear embeddings con Sentence Transformers
        logger.info("🔧 Generando embeddings con Sentence Transformers...")
        st_model = SentenceTransformer(
            self.config["rag"]["embedding_model"],
            device="cpu"  # Usar CPU para no interferir con entrenamiento
        )
        st_embeddings = st_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Guardar embeddings
        st_embeddings_path = self.rag_path / "st_embeddings.npy"
        np.save(st_embeddings_path, st_embeddings)
        
        logger.info(f"✅ Sentence Transformers: {st_embeddings.shape[0]} docs, {st_embeddings.shape[1]} dims")
        
        # 3. Guardar documentos
        documents_path = self.rag_path / "documents.json"
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(documents, indent=2, ensure_ascii=False, fp=f)
        
        # 4. Crear índice general
        rag_index = {
            "created": datetime.now().isoformat(),
            "total_documents": len(documents),
            "tfidf_features": tfidf_matrix.shape[1],
            "st_dimensions": st_embeddings.shape[1],
            "corpus_files": [f.name for f in corpus_files],
            "embedding_model": self.config["rag"]["embedding_model"]
        }
        
        rag_index_path = self.rag_path / "universal_rag_index.pkl"
        with open(rag_index_path, 'wb') as f:
            pickle.dump(rag_index, f)
        
        logger.info("✅ Índice RAG universal creado exitosamente")
        
        return {
            "status": "created",
            "total_documents": len(documents),
            "tfidf_terms": tfidf_matrix.shape[1],
            "embedding_dimensions": st_embeddings.shape[1],
            "index_path": str(self.rag_path)
        }
    
    def search_rag(self, query: str, top_k: int = 5, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Busca en el índice RAG universal
        
        Args:
            query: Texto de consulta
            top_k: Número de resultados a devolver
            method: "tfidf", "semantic" o "hybrid" (default)
        
        Returns:
            Lista de documentos más relevantes
        """
        logger.info(f"🔍 Buscando en RAG universal: '{query[:50]}...'")
        
        # Verificar que existe índice
        documents_path = self.rag_path / "documents.json"
        if not documents_path.exists():
            logger.warning("No existe índice RAG. Ejecutar build_rag_index() primero")
            return []
        
        # Cargar documentos
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        results = []
        
        if method in ["tfidf", "hybrid"]:
            # Búsqueda TF-IDF
            tfidf_path = self.rag_path / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    tfidf_vectorizer = pickle.load(f)
                
                query_vec = tfidf_vectorizer.transform([query])
                
                # Cargar matriz TF-IDF completa (simplificado, en producción usar índice)
                texts = [doc["text"] for doc in documents]
                doc_vecs = tfidf_vectorizer.transform(texts)
                
                # Calcular similitud coseno
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_vec, doc_vecs)[0]
                
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0:
                        results.append({
                            "document": documents[idx],
                            "score": float(similarities[idx]),
                            "method": "tfidf"
                        })
        
        if method in ["semantic", "hybrid"]:
            # Búsqueda semántica
            st_embeddings_path = self.rag_path / "st_embeddings.npy"
            if st_embeddings_path.exists():
                st_embeddings = np.load(st_embeddings_path)
                
                # Generar embedding de la query
                st_model = SentenceTransformer(
                    self.config["rag"]["embedding_model"],
                    device="cpu"
                )
                query_embedding = st_model.encode([query])[0]
                
                # Calcular similitud
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    st_embeddings
                )[0]
                
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > self.config["rag"]["min_score"]:
                        results.append({
                            "document": documents[idx],
                            "score": float(similarities[idx]),
                            "method": "semantic"
                        })
        
        # Si es híbrido, combinar y re-rankear
        if method == "hybrid" and len(results) > 0:
            # Combinar por documento (puede haber duplicados)
            from collections import defaultdict
            combined = defaultdict(lambda: {"score": 0, "methods": []})
            
            for result in results:
                doc_id = result["document"]["id"]
                combined[doc_id]["document"] = result["document"]
                combined[doc_id]["score"] += result["score"]
                combined[doc_id]["methods"].append(result["method"])
            
            # Convertir a lista y ordenar
            results = [
                {"document": v["document"], "score": v["score"], "methods": v["methods"]}
                for v in combined.values()
            ]
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
        
        logger.info(f"✅ Encontrados {len(results)} resultados")
        return results
    
    def add_knowledge(self, data_source: Path, auto_train: bool = False) -> Dict[str, Any]:
        """
        Añade conocimiento al sistema universal desde cualquier fuente
        
        Args:
            data_source: Archivo JSONL con nuevos datos
            auto_train: Si True, entrena automáticamente después de añadir
        
        Returns:
            Resultado de la operación
        """
        logger.info(f"📖 Añadiendo conocimiento desde: {data_source}")
        
        if not data_source.exists():
            raise FileNotFoundError(f"Fuente no encontrada: {data_source}")
        
        # Copiar a corpus unificado
        target = self.corpus_path / data_source.name
        
        # Contar ejemplos
        with open(data_source, 'r', encoding='utf-8') as f:
            examples = sum(1 for _ in f)
        
        # Copiar archivo
        import shutil
        shutil.copy2(data_source, target)
        
        logger.info(f"✅ {examples} ejemplos añadidos al corpus unificado")
        
        result = {
            "status": "added",
            "source": str(data_source),
            "examples": examples,
            "target": str(target)
        }
        
        # Auto-entrenamiento si está habilitado
        if auto_train:
            logger.info("🚀 Iniciando entrenamiento automático...")
            # TODO: Implementar auto-entrenamiento
            result["auto_train"] = "pending_implementation"
        
        return result


def main():
    """Función principal de prueba"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = UniversalManager()
    status = manager.initialize_all()
    
    print("\n" + "="*70)
    print("SISTEMA UNIVERSAL SHEILY - ESTADO COMPLETO")
    print("="*70)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print("="*70)


if __name__ == "__main__":
    main()
