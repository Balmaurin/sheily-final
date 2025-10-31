#!/usr/bin/env python3
"""
ENTRENAMIENTO LoRA ACADÉMICO REAL
"""
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

print("🚀 ENTRENAMIENTO LoRA ACADÉMICO ESPECIALIZADO")
print("=" * 50)

# 1. Cargar modelo base
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"✅ Modelo cargado: {model_name}")

# 2. Configurar LoRA académico
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print(f"📊 Parámetros LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 3. Cargar datos académicos reales
academic_branches = ["antropologia", "economia", "psicologia"]
all_training_corpus = []

for branch in academic_branches:
    data_file = f"corpus_ES/{branch}/academic_data.jsonl"
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_training_corpus.append(data["text"])

print(f"📚 Datos académicos cargados: {len(all_training_corpus)} textos especializados")

# 4. Crear dataset académico
dataset = Dataset.from_dict({"text": all_training_corpus})


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(f"✅ Dataset tokenizado: {len(tokenized_dataset)} ejemplos")

# 5. Configurar entrenamiento académico
training_args = TrainingArguments(
    output_dir="./models/lora_adapters/academic_training",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    learning_rate=3e-5,
    logging_steps=5,
    save_steps=10,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 6. Ejecutar entrenamiento académico
print("🎯 Iniciando entrenamiento académico especializado...")
trainer.train()

# 7. Guardar modelo académico entrenado
trainer.save_model("./models/lora_adapters/academic_training")
print("✅ Modelo académico especializado guardado")

# 8. Crear metadatos académicos reales
metadata = {
    "training_type": "academic_specialization",
    "academic_branches": academic_branches,
    "dataset_size": len(all_training_corpus),
    "model_base": model_name,
    "lora_config": {"r": 8, "alpha": 16, "dropout": 0.05},
    "training_timestamp": "2025-10-17",
    "status": "completed",
    "specialization_level": "academic",
}

with open("./models/lora_adapters/academic_training/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Entrenamiento académico completado exitosamente")
