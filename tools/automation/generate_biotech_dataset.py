#!/usr/bin/env python3
"""
GENERATE BIOTECH DATASET - Automation Tool
===========================================
Specialized biotechnology dataset generation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "all-Branches" / "biotechnology" / "training" / "data"

BIOTECH_EXAMPLES = [
    {
        "instruction": "¿Qué es CRISPR-Cas9 y cómo funciona?",
        "output": "CRISPR-Cas9 es una herramienta de edición genética que permite modificar ADN con precisión. Sistema: 1) ARN guía lleva proteína Cas9 al sitio objetivo, 2) Cas9 corta ADN en ubicación específica, 3) Célula repara corte insertando/eliminando genes. Aplicaciones: Corrección enfermedades genéticas, cultivos resistentes, investigación biomédica.",
    },
    {
        "instruction": "Explica el proceso de clonación molecular.",
        "output": "Clonación molecular: Copiar fragmento ADN en vector. Pasos: 1) Cortar ADN con enzimas restricción, 2) Insertar en vector (plásmido), 3) Transformar bacteria, 4) Seleccionar colonias con gen insertado, 5) Amplificar y purificar. Usos: Producción proteínas recombinantes, terapia génica.",
    },
    {
        "instruction": "¿Qué son las células madre?",
        "output": "Células madre: Capacidad de autorrenovación y diferenciación. Tipos: 1) Embrionarias (pluripotentes, cualquier célula), 2) Adultas (multipotentes, limitadas), 3) iPSC (reprogramadas, pluripotentes). Aplicaciones: Medicina regenerativa, modelado enfermedades, desarrollo fármacos.",
    },
]


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, example in enumerate(BIOTECH_EXAMPLES):
        entry = {
            "branch": "biotechnology",
            "language": "es",
            "task_type": "instruction",
            "instruction": example["instruction"],
            "output": example["output"],
            "id": f"biotechnology_{i:04d}",
            "quality_score": 1.0,
            "created_at": datetime.now().isoformat(),
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    for filename in ["train.jsonl", "premium_training_dataset.jsonl", "complete_optimized_premium_dataset.jsonl"]:
        file_path = OUTPUT_PATH / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Created: {filename}")

    print(f"\n✅ Biotechnology: 3 files with {len(BIOTECH_EXAMPLES)} examples")


if __name__ == "__main__":
    main()
