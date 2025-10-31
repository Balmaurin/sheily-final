#!/usr/bin/env python3
"""
GENERATE DOMAIN DATASETS - Automation Tool
===========================================
Domain-specific dataset generation for 19+ branches.
Part of Sheily AI Enterprise Automation System.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_PATH = PROJECT_ROOT / "all-Branches"

# Domain-specific content (3 examples per domain for initial setup)
DOMAIN_DATA = {
    "sports": {
        "examples": [
            {
                "instruction": "¿Cuál es el mejor plan de entrenamiento para un maratón?",
                "output": "Un plan de maratón efectivo incluye: 1) 16-20 semanas de preparación, 2) 4-5 días de entrenamiento semanal con carrera larga dominical hasta 32km, 3) Entrenamientos de tempo, intervalos y recuperación activa, 4) Aumento gradual de volumen (10% semanal máximo), 5) Descanso adecuado y nutrición rica en carbohidratos complejos.",
            },
            {
                "instruction": "Explica qué es el entrenamiento HIIT y sus beneficios.",
                "output": "HIIT (High-Intensity Interval Training) alterna ejercicios de alta intensidad con periodos de recuperación. Beneficios: 1) Mayor quema calórica post-ejercicio (EPOC), 2) Mejora VO2 max, 3) Preservación de masa muscular, 4) Eficiencia temporal (20-30min), 5) Mejora sensibilidad insulina. Estructura típica: 20-30seg máximo esfuerzo + 60-90seg recuperación activa x 6-10 repeticiones.",
            },
            {
                "instruction": "¿Cuáles son las reglas fundamentales del fútbol?",
                "output": "Reglas básicas: 1) 11 jugadores por equipo, 2) Objetivo: Meter gol en portería contraria, 3) No manos excepto portero, 4) Fuera de juego: Atacante adelantado al último defensa, 5) Faltas: Amarilla (advertencia), roja (expulsión), 6) Tiempo: 2x45 minutos. Posiciones: defensas, centrocampistas, delanteros. Formaciones comunes: 4-4-2, 4-3-3, 3-5-2.",
            },
        ]
    },
    # Add more domains as needed
}


def generate_dataset_jsonl(branch_name: str, examples: list) -> list:
    """Generate JSONL lines for dataset"""
    lines = []
    for i, example in enumerate(examples):
        entry = {
            "branch": branch_name,
            "language": "es",
            "task_type": "instruction",
            "instruction": example["instruction"],
            "output": example["output"],
            "id": f"{branch_name}_{i:04d}",
            "quality_score": 1.0,
            "created_at": datetime.now().isoformat(),
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return lines


def create_training_dataset(branch_name: str, examples: list):
    """Create training dataset files for a branch"""
    branch_path = BASE_PATH / branch_name / "training" / "data"
    branch_path.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train.jsonl": examples,
        "premium_training_dataset.jsonl": examples,
        "complete_optimized_premium_dataset.jsonl": examples,
    }

    created_files = 0
    for filename, data in datasets.items():
        file_path = branch_path / filename
        lines = generate_dataset_jsonl(branch_name, data)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        created_files += 1

    return created_files


def main():
    print("\n" + "=" * 70)
    print("  DOMAIN DATASET GENERATOR")
    print("=" * 70 + "\n")

    total_branches = 0
    total_files = 0

    for branch_name, content in DOMAIN_DATA.items():
        branch_path = BASE_PATH / branch_name

        if not branch_path.exists():
            print(f"⚠️  Rama {branch_name} no existe, saltando...")
            continue

        print(f"📚 Generando dataset para: {branch_name}")
        examples = content["examples"]

        files_created = create_training_dataset(branch_name, examples)

        print(f"   ✅ Creados {files_created} archivos con {len(examples)} ejemplos")

        total_branches += 1
        total_files += files_created

    print("\n" + "=" * 70)
    print(f"Ramas procesadas: {total_branches}")
    print(f"Archivos creados: {total_files}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
