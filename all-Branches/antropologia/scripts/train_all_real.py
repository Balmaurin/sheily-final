#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento REAL con Todos los Datasets - Rama Antropología
============================================================

Entrena REALMENTE con todos los datasets usando Transformers + PEFT.
REQUIERE: GPU recomendada (funcionará en CPU pero será lento)

Author: Sheily AI Team
Version: 2.0.0 - REAL TRAINING
"""

import json
import logging
import math
import shutil
import time
import torch
import torch_directml as dml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealIncrementalTrainingPipeline:
    """Pipeline de entrenamiento REAL incremental."""
    
    def __init__(self, branch_path: Optional[str] = None):
        """Inicializar pipeline."""
        self.branch_path = Path(branch_path) if branch_path else Path(__file__).parent.parent
        self.training_path = self.branch_path / 'training'
        self.adapters_path = self.branch_path / 'adapters' / 'lora_adapters'
        self.backups_path = self.adapters_path / 'backups'
        
        self.adapters_path.mkdir(parents=True, exist_ok=True)
        self.backups_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar dispositivo - DirectML para AMD GPU
        try:
            self.device = dml.device()
            self.device_type = "directml"
            logger.info(f"✓ GPU AMD DirectML detectada: {self.device}")
            logger.info("✓ Entrenamiento acelerado por GPU habilitado")
        except Exception as e:
            # Fallback a CPU si DirectML no está disponible
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Dispositivo de entrenamiento: {self.device}")
            
            if self.device_type == "cpu":
                logger.warning("⚠️ Usando CPU - El entrenamiento será LENTO")
                logger.warning("⚠️ Recomendado: GPU con al menos 8GB VRAM")
    
    def backup_adapter(self, adapter_name: str) -> Optional[str]:
        """Crear respaldo del adaptador."""
        adapter_path = self.adapters_path / adapter_name
        
        if not adapter_path.exists():
            logger.warning(f"Adaptador {adapter_name} no existe")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{adapter_name}_{timestamp}"
        backup_path = self.backups_path / backup_name
        
        try:
            shutil.copytree(adapter_path, backup_path)
            logger.info(f"✓ Backup: {backup_name}")
            return backup_name
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return None
    
    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """Obtener todos los datasets."""
        datasets = []
        
        for jsonl_file in sorted(self.training_path.glob("*.jsonl")):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                
                datasets.append({
                    "name": jsonl_file.name,
                    "path": jsonl_file,
                    "lines": lines
                })
            except Exception as e:
                logger.warning(f"Error leyendo {jsonl_file.name}: {e}")
        
        return datasets
    
    def prioritize_datasets(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Priorizar datasets."""
        priority_order = [
            "premium_complete_optimized_premium_dataset_migrated.jsonl",
            "complete_optimized_premium_dataset.jsonl",
            "premium_premium_training_dataset_migrated.jsonl",
            "adapter_premium_training_dataset.jsonl",
            "train_improved.jsonl",
            "incremental_train_improved_migrated.jsonl",
            "supplementary_improved.jsonl",
            "incremental_supplementary_improved_migrated.jsonl",
            "supplementary_data_migrated.jsonl",
            "supplementary_data.jsonl",
            "train_migrated.jsonl",
            "train.jsonl"
        ]
        
        priority_map = {name: idx for idx, name in enumerate(priority_order)}
        return sorted(datasets, key=lambda d: priority_map.get(d['name'], 999))
    
    def load_or_create_model_and_tokenizer(self, base_model: str, adapter_path: Optional[Path] = None):
        """
        Cargar modelo base y adaptador existente si aplica.
        
        Args:
            base_model: Nombre del modelo base
            adapter_path: Ruta al adaptador existente (para continuar entrenamiento)
        """
        logger.info(f"Cargando modelo: {base_model}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Configurar tokens especiales
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.padding_side is None:
            tokenizer.padding_side = "right"
        
        # Cargar modelo base
        # DirectML funciona mejor con float32, CUDA con float16
        use_fp16 = self.device_type == "cuda"
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto" if self.device_type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Si existe adaptador previo, cargarlo
        if adapter_path and adapter_path.exists():
            logger.info(f"Cargando adaptador existente para continuar entrenamiento...")
            try:
                model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
                logger.info("✓ Adaptador previo cargado (modo entrenable)")
                
                # CRÍTICO: Forzar parámetros LoRA como entrenables
                for name, param in model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                        
            except Exception as e:
                logger.warning(f"No se pudo cargar adaptador previo: {e}")
                logger.info("Creando nuevo adaptador LoRA...")
                lora_config = LoraConfig(
                    r=56,
                    lora_alpha=112,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.025,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
        else:
            logger.info("Creando nuevo adaptador LoRA...")
            lora_config = LoraConfig(
                r=56,
                lora_alpha=112,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.025,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        
        # IMPORTANTE: Mover modelo a DirectML DESPUÉS de aplicar LoRA y forzar gradientes
        if self.device_type == "directml":
            logger.info("Moviendo modelo con LoRA a GPU DirectML...")
            model = model.to(self.device)
            
            # CRÍTICO: Re-activar gradientes después del .to()
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
        
        # Mostrar parámetros entrenables
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Parámetros entrenables: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        return model, tokenizer
    
    def train_with_dataset_real(
        self,
        dataset_path: Path,
        model,
        tokenizer,
        adapter_output_path: Path,
        batch_size: int = 1,
        epochs: int = 1,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Entrenar REALMENTE con un dataset.
        
        Args:
            dataset_path: Ruta al dataset
            model: Modelo con LoRA
            tokenizer: Tokenizer
            adapter_output_path: Donde guardar el adaptador
            batch_size: Tamaño de batch (1 para CPU, 2-4 para GPU)
            epochs: Épocas de entrenamiento
            max_length: Longitud máxima de secuencia
        """
        logger.info("=" * 70)
        logger.info(f"ENTRENAMIENTO REAL: {dataset_path.name}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Cargar dataset
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    examples.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error línea {line_num}: {e}")
        
        logger.info(f"Cargados {len(examples)} ejemplos")
        
        # Preparar datos para entrenamiento
        def format_instruction(example):
            """Formatear ejemplo como prompt."""
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            return f"### Instrucción:\n{instruction}\n\n### Respuesta:\n{output}"
        
        texts = [format_instruction(ex) for ex in examples]
        dataset = Dataset.from_dict({"text": texts})
        
        # OPTIMIZACIÓN 1: Packing de secuencias (eliminar padding)
        # Tokenizar sin padding, luego empaquetar en bloques fijos
        BLOCK_SIZE = max_length  # 512 o 1024 según necesites
        
        def tokenize_no_padding(batch):
            """Tokenizar sin padding para packing."""
            output = tokenizer(
                batch["text"],
                truncation=False,
                add_special_tokens=False
            )
            return {"input_ids": output["input_ids"]}
        
        def group_texts(examples):
            """Empaquetar secuencias en bloques fijos (packing)."""
            import itertools
            # Concatenar todos los input_ids
            concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
            
            # Truncar a múltiplo de BLOCK_SIZE
            total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
            concatenated = concatenated[:total_length]
            
            # Dividir en bloques
            chunks = [concatenated[i:i + BLOCK_SIZE] 
                     for i in range(0, total_length, BLOCK_SIZE)]
            
            return {
                "input_ids": chunks,
                "labels": chunks  # Para causal LM
            }
        
        # Pre-tokenizar y empaquetar (1 sola vez)
        logger.info("Pre-tokenizando con packing (sin padding)...")
        tokenized_dataset = dataset.map(
            tokenize_no_padding,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizando"
        )
        
        tokenized_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            desc="Empaquetando"
        )
        
        logger.info(f"Bloques de entrenamiento: {len(tokenized_dataset)}")
        
        # CONFIGURAR TRAINER OPTIMIZADO PARA DIRECTML
        # Calcula pasos totales reales (batch efectivo = per_device_bs * grad_accum)
        gradient_accumulation_steps = 8
        batch_effective = batch_size * gradient_accumulation_steps
        num_training_steps = math.ceil(len(tokenized_dataset) / batch_effective) * epochs
        num_warmup_steps = int(0.03 * num_training_steps)  # warmup 3%
        
        logger.info(f"Configuración de entrenamiento:")
        logger.info(f"  - Ejemplos originales: {len(examples)}")
        logger.info(f"  - Bloques entrenamiento: {len(tokenized_dataset)}")
        logger.info(f"  - Épocas: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  - Batch efectivo: {batch_effective}")
        logger.info(f"  - Steps totales: {num_training_steps}")
        logger.info(f"  - Warmup steps: {num_warmup_steps}")
        logger.info(f"  - Dispositivo: {self.device}")
        
        # AdamW sin foreach => evita 'lerp' y el fallback a CPU
        optimizer = AdamW(
            model.parameters(),
            lr=2e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            foreach=False  # CRÍTICO para DirectML
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps
        )
        
        # Configurar TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(adapter_output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=10,
            save_strategy="no",  # Guardamos manualmente al final
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=num_warmup_steps,
            fp16=False,  # DirectML no soporta fp16
            dataloader_num_workers=0,  # 0 para DirectML
            dataloader_pin_memory=False,  # False para DirectML
        )
        
        # Data collator para packing (sin padding extra)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=None  # No padding
        )
        
        # Crear Trainer con optimizador personalizado
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),  # << Usa optimizador + scheduler personalizados
        )
        
        # ENTRENAR
        logger.info("🚀 Iniciando entrenamiento con Trainer optimizado para DirectML...")
        train_result = trainer.train()
        
        # Guardar adaptador
        model.save_pretrained(str(adapter_output_path))
        tokenizer.save_pretrained(str(adapter_output_path))
        
        training_time = time.time() - start_time
        
        # Metadata
        metadata = {
            "dataset": dataset_path.name,
            "examples": len(examples),
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "training_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "training_time_seconds": training_time,
            "device": str(self.device),  # Convertir device a string
            "device_type": self.device_type,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✓ Entrenamiento completado en {training_time:.1f}s")
        if metadata['training_loss']:
            logger.info(f"  - Loss: {metadata['training_loss']:.4f}")
        
        return metadata
    
    def train_all_real(
        self,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        batch_size: int = 1,
        epochs_per_dataset: int = 1,
        limit_datasets: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Entrenar con todos los datasets de forma REAL.
        
        Args:
            base_model: Modelo base a usar
            batch_size: Batch size (1 para CPU, 2-4 para GPU)
            epochs_per_dataset: Épocas por dataset
            limit_datasets: Limitar número de datasets (para testing)
        """
        logger.info("\n" + "=" * 70)
        logger.info("ENTRENAMIENTO REAL CON TODOS LOS DATASETS")
        logger.info("=" * 70 + "\n")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "base_model": base_model,
            "device": str(self.device),  # Convertir device a string
            "device_type": self.device_type,
            "datasets_trained": [],
            "backups_created": [],
        }
        
        # 1. Backup
        logger.info("[PASO 1/4] Creando respaldo...")
        backup = self.backup_adapter('current')
        if backup:
            results["backups_created"].append(backup)
        
        # 2. Obtener datasets
        logger.info("\n[PASO 2/4] Analizando datasets...")
        all_datasets = self.get_all_datasets()
        prioritized = self.prioritize_datasets(all_datasets)
        
        if limit_datasets:
            prioritized = prioritized[:limit_datasets]
            logger.info(f"LIMITANDO a primeros {limit_datasets} datasets (testing)")
        
        logger.info(f"\nDatasets a entrenar: {len(prioritized)}")
        for idx, ds in enumerate(prioritized, 1):
            logger.info(f"  {idx}. {ds['name']} ({ds['lines']} líneas)")
        
        # 3. Entrenar
        logger.info("\n[PASO 3/4] Entrenamiento incremental REAL...")
        
        adapter_path = self.adapters_path / 'current'
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        # Cargar modelo una sola vez
        try:
            model, tokenizer = self.load_or_create_model_and_tokenizer(
                base_model,
                adapter_path if (adapter_path / "adapter_config.json").exists() else None
            )
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
        
        # Entrenar con cada dataset
        training_history = []
        
        for idx, dataset in enumerate(prioritized, 1):
            logger.info(f"\n>>> Dataset {idx}/{len(prioritized)}")
            
            try:
                metadata = self.train_with_dataset_real(
                    dataset['path'],
                    model,
                    tokenizer,
                    adapter_path,
                    batch_size=batch_size,
                    epochs=epochs_per_dataset
                )
                
                training_history.append(metadata)
                results["datasets_trained"].append({
                    "dataset": dataset['name'],
                    "status": "success",
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.error(f"Error entrenando {dataset['name']}: {e}")
                results["datasets_trained"].append({
                    "dataset": dataset['name'],
                    "status": "error",
                    "error": str(e)
                })
        
        # Guardar metadata consolidada
        consolidated_metadata = {
            "adapter_name": "current",
            "base_model": base_model,
            "training_mode": "incremental_real",
            "total_datasets": len(prioritized),
            "total_examples": sum(d['lines'] for d in prioritized),
            "device": str(self.device),  # Convertir device a string para JSON
            "device_type": self.device_type,
            "training_history": training_history,
            "last_updated": datetime.now().isoformat()
        }
        
        metadata_path = adapter_path / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_metadata, f, indent=2, ensure_ascii=False)
        
        # 4. Rotar
        logger.info("\n[PASO 4/4] Rotando adaptadores...")
        self.rotate_adapters()
        
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "success"
        
        # Guardar resultados
        results_path = self.branch_path / "real_training_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Resultados: {results_path}")
        
        return results
    
    def rotate_adapters(self):
        """Rotar adaptadores."""
        current_path = self.adapters_path / 'current'
        previous_path = self.adapters_path / 'previous'
        
        if previous_path.exists():
            logger.info("Respaldando 'previous'...")
            self.backup_adapter('previous')
            shutil.rmtree(previous_path)
        
        if current_path.exists():
            logger.info("Rotando: current → previous")
            shutil.copytree(current_path, previous_path)
        
        logger.info("✓ Rotación completada")


def main():
    """Función principal."""
    import sys
    
    pipeline = RealIncrementalTrainingPipeline()
    
    # Argumentos de línea de comandos
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            logger.info(f"Limitando a {limit} datasets para testing")
        except:
            pass
    
    # Confirmar antes de empezar
    if pipeline.device == "cpu":
        logger.warning("\n" + "!" * 70)
        logger.warning("ADVERTENCIA: Entrenamiento en CPU será MUY LENTO")
        logger.warning("Estimado: 30-60 minutos por dataset pequeño")
        logger.warning("Estimado: 2-4 horas por dataset grande")
        logger.warning("!" * 70)
        
        response = input("\n¿Continuar con entrenamiento en CPU? (si/no): ")
        if response.lower() not in ['si', 's', 'yes', 'y']:
            logger.info("Entrenamiento cancelado por el usuario")
            return
    
    # Entrenar
    results = pipeline.train_all_real(
        batch_size=1 if pipeline.device == "cpu" else 2,
        epochs_per_dataset=1,  # 1 época por dataset es suficiente para incremental
        limit_datasets=limit
    )
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE ENTRENAMIENTO REAL")
    print("=" * 70)
    print(f"\nDatasets procesados: {len(results['datasets_trained'])}")
    
    successful = [d for d in results['datasets_trained'] if d['status'] == 'success']
    print(f"✅ Exitosos: {len(successful)}")
    print(f"❌ Fallidos: {len(results['datasets_trained']) - len(successful)}")
    print(f"\n💾 Respaldos: {len(results['backups_created'])}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
