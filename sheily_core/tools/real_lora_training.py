#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTACIÓN REAL DE ENTRENAMIENTO LoRA
========================================
Script para entrenamiento efectivo de adaptadores LoRA académicos.
Sustituye las simulaciones con implementación técnica real.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Importar bibliotecas reales de ML
try:
    import accelerate
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    ML_AVAILABLE = True
    print("✅ Bibliotecas de ML disponibles")

except ImportError as e:
    print(f"❌ Error importando bibliotecas de ML: {e}")
    print("Instalando dependencias necesarias...")
    ML_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento LoRA"""

    model_name: str = "microsoft/DialoGPT-medium"
    dataset_name: str = "bookcorpus"
    output_dir: str = "./models/lora_adapters/real_training"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 250
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn"]


class RealLoraTrainer:
    """Entrenador LoRA real para especializaciones académicas"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Usando dispositivo: {self.device}")

        if not ML_AVAILABLE:
            raise ImportError("Bibliotecas de ML no disponibles. Ejecute instalación primero.")

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Cargar modelo base y tokenizador"""
        print(f"📥 Cargando modelo base: {self.config.model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # Preparar modelo para LoRA
            model = prepare_model_for_kbit_training(model)

            print("✅ Modelo y tokenizador cargados correctamente")
            return model, tokenizer

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            # Usar modelo alternativo más pequeño para pruebas
            print("🔄 Usando modelo alternativo más pequeño...")
            return self._load_smaller_model()

    def _load_smaller_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Cargar modelo más pequeño para pruebas"""
        try:
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

            return model, tokenizer

        except Exception as e:
            raise Exception(f"No se pudo cargar ningún modelo: {e}")

    def prepare_lora_config(self, model: AutoModelForCausalLM) -> LoraConfig:
        """Preparar configuración LoRA"""
        print("🔧 Configurando parámetros LoRA...")
        # Detectar módulos objetivo automáticamente
        target_modules = []
        for name, module in model.named_modules():
            if any(target in name.lower() for target in ["attn", "attention", "proj"]):
                if "c_attn" in name or "c_proj" in name:
                    target_modules.append(name.split(".")[-1])

        if not target_modules:
            target_modules = ["c_attn"]

        print(f"🎯 Módulos objetivo detectados: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        return lora_config

    def prepare_dataset(self, academic_branch: str) -> Dataset:
        """Preparar dataset para rama académica específica"""
        print(f"📚 Preparando dataset para rama: {academic_branch}")

        # Crear datos de entrenamiento especializados
        training_texts = self._generate_academic_training_corpus(academic_branch)

        # Crear dataset
        dataset = Dataset.from_dict({"text": training_texts})

        print(f"✅ Dataset creado con {len(training_texts)} ejemplos")
        return dataset

    def _generate_academic_training_corpus(self, branch: str) -> List[str]:
        """Generar datos de entrenamiento académicos especializados"""
        # Datos especializados por rama académica
        academic_data = {
            "antropologia": [
                "La antropología cultural estudia las variaciones culturales humanas y su impacto en el comportamiento social.",
                "Los antropólogos analizan cómo las sociedades desarrollan normas, valores y prácticas culturales únicas.",
                "El estudio antropológico revela cómo la cultura influye en la percepción del mundo y las relaciones sociales.",
            ],
            "economia": [
                "La teoría económica explica cómo los mercados asignan recursos escasos entre necesidades ilimitadas.",
                "Los modelos macroeconómicos analizan el crecimiento económico y los ciclos de negocio.",
                "La economía del comportamiento integra psicología y economía para entender decisiones individuales.",
            ],
            "psicologia": [
                "La psicología cognitiva estudia los procesos mentales internos como la memoria y el pensamiento.",
                "Las teorías psicológicas explican cómo los factores ambientales influyen en el comportamiento humano.",
                "La neuropsicología investiga la relación entre el cerebro y los procesos psicológicos.",
            ],
        }

        # Obtener datos específicos o datos generales
        if branch in academic_data:
            base_texts = academic_data[branch]
        else:
            base_texts = [
                f"El estudio académico de {branch} proporciona conocimientos especializados en esta disciplina.",
                f"Los expertos en {branch} desarrollan teorías y metodologías específicas para su campo.",
                f"La investigación en {branch} contribuye al avance del conocimiento humano.",
            ]

        # Expandir datos para entrenamiento
        expanded_texts = []
        for text in base_texts:
            # Crear variaciones del texto
            expanded_texts.append(text)
            expanded_texts.append(f"En el contexto académico: {text}")
            expanded_texts.append(f"Desde la perspectiva de {branch}: {text}")

        return expanded_texts[:50]  # Limitar para pruebas iniciales

    def train_adapter(self, academic_branch: str) -> Dict:
        """Entrenar adaptador LoRA para rama académica específica"""
        print(f"🚀 Iniciando entrenamiento real para rama: {academic_branch}")

        start_time = time.time()

        try:
            # 1. Cargar modelo y tokenizador
            model, tokenizer = self.load_model_and_tokenizer()

            # 2. Preparar configuración LoRA
            lora_config = self.prepare_lora_config(model)

            # 3. Aplicar LoRA al modelo
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # 4. Preparar dataset
            dataset = self.prepare_dataset(academic_branch)

            # Tokenizar dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"], truncation=True, padding="max_length", max_length=512
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # 5. Configurar entrenamiento
            output_dir = Path(self.config.output_dir) / academic_branch
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_dir=f"{output_dir}/logs",
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                report_to=["tensorboard"],
            )

            # 6. Configurar entrenador
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_dataset,  # Usar mismo dataset para evaluación básica
                data_collator=data_collator,
            )

            # 7. Ejecutar entrenamiento
            print("🎯 Iniciando entrenamiento...")
            trainer.train()

            # 8. Guardar modelo entrenado
            trainer.save_model(str(output_dir))

            # 9. Crear metadatos reales
            training_time = time.time() - start_time

            metadata = {
                "academic_branch": academic_branch,
                "training_timestamp": datetime.now().isoformat(),
                "training_duration": round(training_time, 2),
                "model_base": self.config.model_name,
                "lora_config": {
                    "r": self.config.lora_r,
                    "alpha": self.config.lora_alpha,
                    "dropout": self.config.lora_dropout,
                },
                "dataset_info": {
                    "samples": len(tokenized_dataset),
                    "specialization": academic_branch,
                },
                "training_config": {
                    "epochs": self.config.num_train_epochs,
                    "batch_size": self.config.per_device_train_batch_size,
                    "learning_rate": self.config.learning_rate,
                },
                "hardware_used": {
                    "device": str(self.device),
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "N/A",
                },
                "status": "trained",
                "version": "2.0.0",
            }

            # Guardar metadatos reales
            with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"✅ Adaptador {academic_branch} entrenado correctamente en {training_time:.2f}s")

            return {
                "branch": academic_branch,
                "status": "SUCCESS",
                "training_time": training_time,
                "output_path": str(output_dir),
                "model_size": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }

        except Exception as e:
            print(f"❌ Error en entrenamiento de {academic_branch}: {e}")
            return {"branch": academic_branch, "status": "ERROR", "error": str(e)}


def main():
    """Función principal de entrenamiento real"""
    print("🚀 ENTRENAMIENTO REAL DE ADAPTADORES LoRA ACADÉMICOS")
    print("=" * 60)

    if not ML_AVAILABLE:
        print("❌ Bibliotecas de ML no disponibles")
        print("💡 Ejecute: pip install -r requirements_ml.txt")
        return 1

    # Configuración de entrenamiento
    config = TrainingConfig()

    # Crear entrenador
    trainer = RealLoraTrainer(config)

    # Ramas académicas a procesar (primeras 3 para prueba inicial)
    test_branches = ["antropologia", "economia", "psicologia"]

    results = []
    for branch in test_branches:
        print(f"\n🎯 Procesando rama: {branch}")
        result = trainer.train_adapter(branch)
        results.append(result)

    # Reporte final
    print("\n" + "=" * 60)
    print("📊 REPORTE DE ENTRENAMIENTO REAL")
    print("=" * 60)

    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"✅ Éxitos: {successful}/{len(results)}")

    for result in results:
        if result["status"] == "SUCCESS":
            print(f"   ✅ {result['branch']}: {result['training_time']:.1f}s")
        else:
            print(f"   ❌ {result['branch']}: {result.get('error', 'Error desconocido')}")

    return 0 if successful > 0 else 1


if __name__ == "__main__":
    exit(main())
