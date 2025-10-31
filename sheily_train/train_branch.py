#!/usr/bin/env python3
"""
TRAIN BRANCH - Lanzador de Entrenamiento por Rama
==================================================
Script principal para entrenar modelos usando los datos de all-Branches/

USAGE:
    python3 train_branch.py --branch physics --model llama2-7b --epochs 3
    python3 train_branch.py --branch sports --lora --output models/sports
    python3 train_branch.py --list-branches  # Ver ramas disponibles
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Paths del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
BRANCHES_DIR = PROJECT_ROOT / "all-Branches"
MODELS_OUTPUT = PROJECT_ROOT / "var" / "central_models"
LOGS_DIR = PROJECT_ROOT / "var" / "central_logs"


def list_available_branches():
    """Listar ramas disponibles para entrenamiento"""
    print("\n" + "=" * 70)
    print("  RAMAS DISPONIBLES PARA ENTRENAMIENTO")
    print("=" * 70 + "\n")

    branches = []
    for branch_dir in sorted(BRANCHES_DIR.iterdir()):
        if not branch_dir.is_dir():
            continue

        training_data = branch_dir / "training" / "data"
        if training_data.exists():
            jsonl_files = list(training_data.glob("*.jsonl"))

            if jsonl_files:
                # Count total examples
                total_examples = 0
                for jsonl_file in jsonl_files:
                    try:
                        with open(jsonl_file, "r", encoding="utf-8") as f:
                            total_examples += sum(1 for _ in f)
                    except (IOError, OSError, UnicodeDecodeError) as e:
                        # Ignorar archivos que no se pueden leer
                        pass

                branches.append({"name": branch_dir.name, "files": len(jsonl_files), "examples": total_examples})

    print(f"Total ramas con datos: {len(branches)}\n")

    for branch in branches:
        status = "✅" if branch["examples"] > 0 else "⚠️"
        print(f"{status} {branch['name']:30} | {branch['files']:2} archivos | {branch['examples']:>5} ejemplos")

    print("\n" + "=" * 70 + "\n")
    return branches


def validate_branch(branch_name: str):
    """Validar que la rama existe y tiene datos"""
    branch_path = BRANCHES_DIR / branch_name

    if not branch_path.exists():
        print(f"❌ Error: Rama '{branch_name}' no existe")
        print(f"\n💡 Ramas disponibles (ejemplos):")
        print(f"   • physics, sports, medicine, programming, finance")
        print(f"   • art, chemistry, biology, history, music")
        print(f"\n📋 Ver todas las ramas:")
        print(f"   python3 sheily_train/train_branch.py --list-branches")
        print(f"\n✅ Uso correcto:")
        print(f"   python3 sheily_train/train_branch.py --branch physics")
        print(f"   python3 sheily_train/train_branch.py --branch sports --lora")
        return False

    training_data = branch_path / "training" / "data"
    if not training_data.exists():
        print(f"❌ Error: Rama '{branch_name}' no tiene carpeta training/data/")
        return False

    jsonl_files = list(training_data.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ Error: Rama '{branch_name}' no tiene archivos .jsonl")
        return False

    print(f"✅ Rama '{branch_name}' validada:")
    print(f"   • {len(jsonl_files)} archivos de entrenamiento")

    # Count examples
    total = 0
    for f in jsonl_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                total += sum(1 for _ in file)
        except (IOError, OSError, UnicodeDecodeError) as e:
            # Ignorar archivos que no se pueden leer
            pass

    print(f"   • {total} ejemplos totales")

    return True


def prepare_training_config(branch_name: str, args):
    """Preparar configuración de entrenamiento"""
    config = {
        "branch": branch_name,
        "model": args.model,
        "data_path": str(BRANCHES_DIR / branch_name / "training" / "data"),
        "output_dir": str(MODELS_OUTPUT / branch_name),
        "logs_dir": str(LOGS_DIR),
        "training_params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_seq_length": 512,
        },
        "use_lora": args.lora,
        "lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }
        if args.lora
        else None,
        "created_at": datetime.now().isoformat(),
    }

    return config


def start_training(config: dict):
    """Iniciar entrenamiento usando el trainer real"""
    # Create output directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = Path(config["output_dir"]) / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuración guardada en: {config_file}\n")

    # Intentar importar el trainer
    try:
        from sheily_train.core.training.trainer import train_model

        # Ejecutar entrenamiento real
        success = train_model(config)
        return success

    except ImportError as e:
        print("\n" + "⚠️" * 35)
        print("  DEPENDENCIAS NO INSTALADAS")
        print("⚠️" * 35)
        print(f"\nPara entrenar modelos reales, instala las dependencias:\n")
        print(f"  pip install transformers datasets peft accelerate bitsandbytes torch\n")
        print(f"O ejecuta:\n")
        print(f"  make install\n")
        print(f"\nError: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar modelos usando datos de all-Branches/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--branch", type=str, help="Nombre de la rama a entrenar (ej: physics, sports)")

    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Modelo base a usar (default: Llama-2-7b)"
    )

    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs (default: 3)")

    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")

    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")

    parser.add_argument("--lora", action="store_true", help="Usar LoRA para entrenamiento eficiente")

    parser.add_argument("--output", type=str, help="Directorio de salida personalizado")

    parser.add_argument("--list-branches", action="store_true", help="Listar ramas disponibles")

    args = parser.parse_args()

    # List branches
    if args.list_branches:
        list_available_branches()
        return 0

    # Validate branch argument
    if not args.branch:
        print("❌ Error: Debes especificar --branch o --list-branches")
        parser.print_help()
        return 1

    # Validate branch exists and has data
    if not validate_branch(args.branch):
        return 1

    # Prepare config
    config = prepare_training_config(args.branch, args)

    # Override output if specified
    if args.output:
        config["output_dir"] = args.output

    # Start training
    success = start_training(config)

    if success:
        print("\n✅ Setup completado. Implementar trainer para entrenar realmente.")
        return 0
    else:
        print("\n❌ Error en el entrenamiento")
        return 1


if __name__ == "__main__":
    sys.exit(main())
