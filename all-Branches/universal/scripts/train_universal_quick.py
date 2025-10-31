"""
Train Universal Quick Test - Entrenamiento rápido con dataset pequeño
=====================================================================

Entrena el adaptador universal con solo 1 archivo pequeño para validación.

Uso:
    python train_universal_quick.py
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
import torch_directml

# Añadir path del sistema universal
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_manager import UniversalManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_small_dataset(corpus_path: Path, tokenizer, max_examples: int = 100):
    """
    Prepara un dataset pequeño para prueba rápida
    
    Args:
        corpus_path: Ruta al corpus unificado
        tokenizer: Tokenizador del modelo
        max_examples: Número máximo de ejemplos a usar
    
    Returns:
        Dataset tokenizado
    """
    logger.info(f"📚 Cargando dataset pequeño (máx {max_examples} ejemplos)...")
    
    # Buscar archivos de training (que tienen instruction/output)
    corpus_files = list(corpus_path.glob("*training*.jsonl"))
    
    if not corpus_files:
        # Si no hay training, buscar cualquier archivo
        corpus_files = list(corpus_path.glob("*.jsonl"))
    
    if not corpus_files:
        raise FileNotFoundError(f"No hay archivos JSONL en {corpus_path}")
    
    # Ordenar por tamaño y tomar el más pequeño
    corpus_files.sort(key=lambda f: f.stat().st_size)
    
    logger.info(f"Usando archivo: {corpus_files[0].name}")
    
    # Cargar solo el primer archivo
    dataset = load_dataset("json", data_files=str(corpus_files[0]), split="train")
    
    # Limitar ejemplos
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))
    
    logger.info(f"✅ Dataset de prueba: {len(dataset)} ejemplos")
    
    # Tokenizar
    def tokenize_function(examples):
        texts = []
        for inst, out in zip(examples["instruction"], examples["output"]):
            text = f"<|user|>\n{inst}\n<|assistant|>\n{out}\n<|endoftext|>"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,  # Más corto para ser rápido
            padding=False
        )
    
    logger.info("🔧 Tokenizando...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizando"
    )
    
    return tokenized


def train_quick(manager: UniversalManager, max_examples: int = 100, epochs: int = 1):
    """
    Entrenamiento rápido de prueba
    """
    logger.info("🚀 Iniciando entrenamiento RÁPIDO de prueba...")
    
    # Preparar dataset pequeño
    dataset = prepare_small_dataset(
        manager.corpus_path,
        manager.tokenizer,
        max_examples=max_examples
    )
    
    # Configurar entrenamiento MUY ligero
    output_dir = manager.adapter_path / "training_output_quick"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Batch más grande
        gradient_accumulation_steps=2,   # Menos acumulación
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",  # No guardar checkpoints
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        max_steps=20  # LÍMITE: solo 20 pasos para probar
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=manager.tokenizer,
        mlm=False
    )
    
    # Optimizer con foreach=False para DirectML
    from torch.optim import AdamW
    optimizer = AdamW(
        manager.model.parameters(),
        lr=2e-4,
        foreach=False  # Crítico para DirectML
    )
    
    # Trainer
    trainer = Trainer(
        model=manager.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )
    
    # Entrenar
    logger.info("🔥 Entrenando (máximo 20 pasos)...")
    start_time = datetime.now()
    
    train_result = trainer.train()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"✅ Entrenamiento completado en {duration:.0f} segundos")
    
    # Guardar adaptador
    logger.info("💾 Guardando adaptador de prueba...")
    
    current_adapter = manager.adapter_path / "current"
    current_adapter.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(str(current_adapter))
    
    # Guardar metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "total_examples_trained": len(dataset),
        "epochs": epochs,
        "max_steps": 20,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "final_loss": train_result.training_loss,
        "duration_seconds": duration,
        "training_type": "quick_test"
    }
    
    metadata_path = current_adapter / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, indent=2, ensure_ascii=False, fp=f)
    
    logger.info(f"📊 Loss final: {train_result.training_loss:.4f}")
    logger.info(f"💾 Adaptador guardado en: {current_adapter}")
    
    return {
        "status": "success",
        "examples_trained": len(dataset),
        "steps": 20,
        "final_loss": train_result.training_loss,
        "duration": duration,
        "adapter_path": str(current_adapter)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento rápido de prueba del Sistema Universal"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Número máximo de ejemplos (default: 100)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Número de épocas (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Inicializar sistema universal
    logger.info("🚀 Inicializando Sistema Universal...")
    manager = UniversalManager()
    manager.initialize_all()
    
    # Verificar GPU
    if torch.cuda.is_available():
        device_name = "CUDA"
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            device_name = f"DirectML ({device})"
        except:
            device_name = "CPU"
    
    logger.info(f"🖥️  Dispositivo: {device_name}")
    
    # Entrenar
    result = train_quick(
        manager,
        max_examples=args.max_examples,
        epochs=args.epochs
    )
    
    # Mostrar resultado
    print("\n" + "="*70)
    print("ENTRENAMIENTO RÁPIDO COMPLETADO")
    print("="*70)
    print(f"Ejemplos usados: {result['examples_trained']}")
    print(f"Pasos entrenados: {result['steps']}")
    print(f"Loss final: {result['final_loss']:.4f}")
    print(f"Duración: {result['duration']:.0f} segundos")
    print(f"Adaptador guardado en: {result['adapter_path']}")
    print("="*70)
    print("\n⚠️  NOTA: Este es un entrenamiento de PRUEBA con dataset reducido.")
    print("   Para entrenamiento completo, usa: train_universal.py")
    print("="*70)
    
    logger.info("✅ Proceso completado exitosamente")


if __name__ == "__main__":
    main()
