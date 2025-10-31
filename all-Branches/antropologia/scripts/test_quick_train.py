#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rápido de entrenamiento con dataset pequeño
"""

import sys
from pathlib import Path

# Importar la clase del script principal
sys.path.insert(0, str(Path(__file__).parent))
from train_all_real import RealIncrementalTrainingPipeline


def main():
    """Test rápido con dataset pequeño (36 líneas)."""

    pipeline = RealIncrementalTrainingPipeline()

    print("\n" + "=" * 70)
    print("TEST RÁPIDO - Dataset pequeño (supplementary_improved.jsonl - 36 líneas)")
    print("=" * 70 + "\n")

    # Encontrar el dataset pequeño
    small_dataset = pipeline.training_path / "supplementary_improved.jsonl"

    if not small_dataset.exists():
        print(f"❌ Dataset no encontrado: {small_dataset}")
        return

    print(f"✓ Dataset encontrado: {small_dataset.name}")

    # Cargar modelo una vez
    print("\n[1/3] Cargando modelo...")
    adapter_path = pipeline.adapters_path / "current"

    try:
        model, tokenizer = pipeline.load_or_create_model_and_tokenizer(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            adapter_path if (adapter_path / "adapter_config.json").exists() else None,
        )
        print("✓ Modelo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # Entrenar
    print("\n[2/3] Entrenando...")
    try:
        result = pipeline.train_with_dataset_real(
            dataset_path=small_dataset,
            model=model,
            tokenizer=tokenizer,
            adapter_output_path=adapter_path,
            batch_size=2,  # GPU
            epochs=1,
            max_length=512,
        )

        print("\n✓ ENTRENAMIENTO EXITOSO!")
        print(f"  - Loss: {result.get('training_loss', 'N/A')}")
        print(f"  - Tiempo: {result.get('training_time_seconds', 0):.1f}s")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\n[3/3] Test completado")


if __name__ == "__main__":
    main()
