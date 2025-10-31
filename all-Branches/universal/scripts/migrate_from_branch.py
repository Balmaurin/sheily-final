"""
Migrate from Branch - Migra datos de una rama al sistema universal
===================================================================

Extrae corpus y datasets de cualquier rama (como antropologia)
y los integra al sistema universal.

Uso:
    python migrate_from_branch.py <branch_name> [opciones]

Ejemplos:
    python migrate_from_branch.py antropologia
    python migrate_from_branch.py astronomia --include-training
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Añadir path del sistema universal
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_manager import UniversalManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def migrate_branch_data(
    branch_name: str, manager: UniversalManager, include_training: bool = True, include_corpus: bool = True
):
    """
    Migra datos de una rama al sistema universal

    Args:
        branch_name: Nombre de la rama (ej: antropologia)
        manager: UniversalManager ya inicializado
        include_training: Incluir datasets de training/
        include_corpus: Incluir corpus/spanish/

    Returns:
        Resultado de la migración
    """
    # Localizar rama
    branch_path = manager.base_path.parent / branch_name

    if not branch_path.exists():
        raise FileNotFoundError(f"Rama no encontrada: {branch_path}")

    logger.info(f"📂 Migrando desde rama: {branch_name}")
    logger.info(f"   Ruta: {branch_path}")

    migrated = {
        "branch": branch_name,
        "timestamp": datetime.now().isoformat(),
        "corpus_files": [],
        "training_files": [],
        "total_examples": 0,
    }

    # Migrar corpus
    if include_corpus:
        corpus_spanish = branch_path / "corpus" / "spanish"

        if corpus_spanish.exists():
            logger.info("📚 Migrando corpus...")

            for jsonl_file in corpus_spanish.glob("documents_*.jsonl"):
                target_name = f"{branch_name}_corpus_{jsonl_file.name}"
                target_path = manager.corpus_path / target_name

                # Copiar archivo
                shutil.copy2(jsonl_file, target_path)

                # Contar ejemplos
                with open(target_path, "r", encoding="utf-8") as f:
                    examples = sum(1 for _ in f)

                migrated["corpus_files"].append(target_name)
                migrated["total_examples"] += examples

                logger.info(f"  ✅ {target_name}: {examples} documentos")

    # Migrar training datasets
    if include_training:
        training_path = branch_path / "training"

        if training_path.exists():
            logger.info("🎓 Migrando datasets de training...")

            for jsonl_file in training_path.glob("*.jsonl"):
                target_name = f"{branch_name}_training_{jsonl_file.name}"
                target_path = manager.corpus_path / target_name

                # Copiar archivo
                shutil.copy2(jsonl_file, target_path)

                # Contar ejemplos
                with open(target_path, "r", encoding="utf-8") as f:
                    examples = sum(1 for _ in f)

                migrated["training_files"].append(target_name)
                migrated["total_examples"] += examples

                logger.info(f"  ✅ {target_name}: {examples} ejemplos")

    # Guardar reporte de migración
    report_path = manager.base_path / f"migration_{branch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(migrated, indent=2, ensure_ascii=False, fp=f)

    logger.info(f"📝 Reporte guardado: {report_path.name}")

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migra datos de una rama al Sistema Universal")
    parser.add_argument("branch", type=str, help="Nombre de la rama a migrar (ej: antropologia)")
    parser.add_argument("--skip-training", action="store_true", help="No migrar datasets de training/")
    parser.add_argument("--skip-corpus", action="store_true", help="No migrar corpus/spanish/")

    args = parser.parse_args()

    # Inicializar sistema universal
    logger.info("🚀 Inicializando Sistema Universal...")
    manager = UniversalManager()

    # Migrar datos
    result = migrate_branch_data(
        args.branch, manager, include_training=not args.skip_training, include_corpus=not args.skip_corpus
    )

    # Mostrar resultado
    print("\n" + "=" * 70)
    print(f"MIGRACIÓN COMPLETADA: {args.branch.upper()}")
    print("=" * 70)
    print(f"Archivos de corpus migrados: {len(result['corpus_files'])}")
    for file in result["corpus_files"]:
        print(f"  - {file}")

    print(f"\nArchivos de training migrados: {len(result['training_files'])}")
    for file in result["training_files"]:
        print(f"  - {file}")

    print(f"\nTotal de ejemplos migrados: {result['total_examples']}")
    print("=" * 70)

    # Estado del sistema
    status = manager.get_status()
    print(f"\nESTADO DEL SISTEMA UNIVERSAL:")
    print(f"  Total documentos: {status['corpus']['total_documents']}")
    print(f"  Archivos en corpus: {status['corpus']['files']}")
    print("=" * 70)

    logger.info("✅ Migración completada exitosamente")
    logger.info("💡 Ahora puedes entrenar el adaptador universal con:")
    logger.info("   python scripts/train_universal.py")


if __name__ == "__main__":
    main()
