"""
Add Knowledge - Añade cualquier dataset al sistema universal
=============================================================

Script para integrar nuevos datos al corpus global.
Acepta cualquier tipo de dataset en formato JSONL.

Uso:
    python add_knowledge.py <ruta_dataset> [--auto-train]

Ejemplos:
    python add_knowledge.py ../../antropologia/training/premium_dataset.jsonl
    python add_knowledge.py nuevos_datos.jsonl --auto-train
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Añadir path del sistema universal
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_manager import UniversalManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Añade conocimiento al Sistema Universal Sheily")
    parser.add_argument("dataset", type=str, help="Ruta al archivo JSONL con los datos")
    parser.add_argument("--auto-train", action="store_true", help="Entrenar automáticamente después de añadir")
    parser.add_argument("--domain", type=str, default="general", help="Dominio de origen (solo informativo, opcional)")

    args = parser.parse_args()

    # Validar dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"❌ Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    if dataset_path.suffix != ".jsonl":
        logger.error(f"❌ El dataset debe ser formato JSONL")
        sys.exit(1)

    # Inicializar sistema universal
    logger.info("🚀 Inicializando Sistema Universal...")
    manager = UniversalManager()

    # Añadir conocimiento
    logger.info(f"📖 Añadiendo dataset: {dataset_path.name}")
    logger.info(f"   Dominio de origen: {args.domain}")

    result = manager.add_knowledge(dataset_path, auto_train=args.auto_train)

    # Mostrar resultado
    print("\n" + "=" * 70)
    print("CONOCIMIENTO AÑADIDO AL SISTEMA UNIVERSAL")
    print("=" * 70)
    print(f"Fuente: {result['source']}")
    print(f"Ejemplos añadidos: {result['examples']}")
    print(f"Ubicación: {result['target']}")

    if args.auto_train:
        print(f"Auto-entrenamiento: {result.get('auto_train', 'No disponible')}")

    print("=" * 70)

    # Mostrar estado actualizado
    status = manager.get_status()
    print("\nESTADO DEL SISTEMA:")
    print(f"  Total documentos en corpus: {status['corpus']['total_documents']}")
    print(f"  Archivos en corpus: {status['corpus']['files']}")

    if status["adapter"]["status"] == "trained":
        print(f"  Adaptador: {status['adapter']['total_examples']} ejemplos entrenados")
        print(f"  Última actualización: {status['adapter']['last_update']}")
    else:
        print(f"  Adaptador: {status['adapter']['status']}")

    print("=" * 70)

    logger.info("✅ Operación completada exitosamente")


if __name__ == "__main__":
    main()
