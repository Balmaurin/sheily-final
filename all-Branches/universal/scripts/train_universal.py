"""
Train Universal - Entrena el adaptador universal con todo el corpus
====================================================================

Entrena el adaptador LoRA único con todos los datos disponibles
en el corpus unificado. No importa el dominio de origen.

Uso:
    python train_universal.py [opciones]

Opciones:
    --batch-size: Tamaño del batch (default: 2)
    --epochs: Número de épocas (default: 3)
    --learning-rate: Learning rate (default: 2e-4)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch_directml
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Añadir path del sistema universal
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_manager import UniversalManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(corpus_path: Path, tokenizer, max_length: int = 512):
    """
    Prepara el dataset unificado para entrenamiento

    Args:
        corpus_path: Ruta al corpus unificado
        tokenizer: Tokenizador del modelo
        max_length: Longitud máxima de secuencia

    Returns:
        Dataset tokenizado listo para entrenar
    """
    logger.info("📚 Cargando corpus unificado...")

    # Buscar todos los archivos JSONL
    corpus_files = list(corpus_path.glob("*.jsonl"))

    if not corpus_files:
        raise FileNotFoundError(f"No hay archivos JSONL en {corpus_path}")

    logger.info(f"Archivos encontrados: {len(corpus_files)}")
    for file in corpus_files:
        logger.info(f"  - {file.name}")

    # Cargar todos los datasets
    all_data = []
    for file in corpus_files:
        dataset = load_dataset("json", data_files=str(file), split="train")
        all_data.append(dataset)
        logger.info(f"    {len(dataset)} ejemplos")

    # Concatenar todos los datasets
    from datasets import concatenate_datasets

    unified_dataset = concatenate_datasets(all_data)

    logger.info(f"✅ Total: {len(unified_dataset)} ejemplos en corpus unificado")

    # Tokenizar
    def tokenize_function(examples):
        # Formato: <|user|> {instruction} <|assistant|> {output}
        texts = []
        for inst, out in zip(examples["instruction"], examples["output"]):
            text = f"<|user|>\n{inst}\n<|assistant|>\n{out}\n<|endoftext|>"
            texts.append(text)

        return tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    logger.info("🔧 Tokenizando dataset...")
    tokenized = unified_dataset.map(
        tokenize_function, batched=True, remove_columns=unified_dataset.column_names, desc="Tokenizando"
    )

    return tokenized


def train_universal_adapter(
    manager: UniversalManager, batch_size: int = 2, epochs: int = 3, learning_rate: float = 2e-4
):
    """
    Entrena el adaptador universal con todo el corpus

    Args:
        manager: UniversalManager ya inicializado
        batch_size: Tamaño del batch
        epochs: Número de épocas
        learning_rate: Learning rate
    """
    logger.info("🚀 Iniciando entrenamiento del adaptador universal...")

    # Preparar dataset
    dataset = prepare_dataset(manager.corpus_path, manager.tokenizer, max_length=manager.config["model"]["max_length"])

    # Configurar entrenamiento
    training_config = manager.config["training"]
    output_dir = manager.adapter_path / "training_output"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=manager.tokenizer, mlm=False)

    # Optimizer con foreach=False para DirectML
    from torch.optim import AdamW

    optimizer = AdamW(manager.model.parameters(), lr=learning_rate, foreach=False)  # Crítico para DirectML

    # Trainer
    trainer = Trainer(
        model=manager.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    # Entrenar
    logger.info("🔥 Entrenando adaptador universal...")
    start_time = datetime.now()

    train_result = trainer.train()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(f"✅ Entrenamiento completado en {duration:.0f} segundos")

    # Guardar adaptador
    logger.info("💾 Guardando adaptador universal...")

    # Backup del adaptador anterior
    current_adapter = manager.adapter_path / "current"
    if current_adapter.exists():
        checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = manager.adapter_path / "checkpoints" / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copytree(current_adapter, checkpoint_path, dirs_exist_ok=True)
        logger.info(f"Checkpoint guardado: {checkpoint_name}")

    # Guardar nuevo adaptador
    current_adapter.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(str(current_adapter))

    # Guardar metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "total_examples_trained": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": train_result.training_loss,
        "duration_seconds": duration,
        "corpus_files": [f.name for f in manager.corpus_path.glob("*.jsonl")],
    }

    metadata_path = current_adapter / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, indent=2, ensure_ascii=False, fp=f)

    logger.info(f"📊 Loss final: {train_result.training_loss:.4f}")
    logger.info(f"💾 Adaptador guardado en: {current_adapter}")

    return {
        "status": "success",
        "examples_trained": len(dataset),
        "final_loss": train_result.training_loss,
        "duration": duration,
        "adapter_path": str(current_adapter),
    }


def main():
    parser = argparse.ArgumentParser(description="Entrena el adaptador universal de Sheily")
    parser.add_argument("--batch-size", type=int, default=2, help="Tamaño del batch (default: 2)")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")

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
    result = train_universal_adapter(
        manager, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate
    )

    # Mostrar resultado
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO DEL ADAPTADOR UNIVERSAL COMPLETADO")
    print("=" * 70)
    print(f"Ejemplos entrenados: {result['examples_trained']}")
    print(f"Loss final: {result['final_loss']:.4f}")
    print(f"Duración: {result['duration']:.0f} segundos")
    print(f"Adaptador guardado en: {result['adapter_path']}")
    print("=" * 70)

    logger.info("✅ Proceso completado exitosamente")


if __name__ == "__main__":
    main()
