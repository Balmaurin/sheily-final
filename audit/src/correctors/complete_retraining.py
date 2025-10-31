#!/usr/bin/env python3
"""
RETRENAMIENTO COMPLETO DE TODAS LAS RAMAS
Script maestro para corrección completa del proyecto
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Agregar directorio padre al path para importar módulos locales
sys.path.append(str(Path(__file__).parent.parent))

from sheily_train.core.validation.advanced_validation import (
    generate_validation_report,
    validate_adapter_comprehensive,
    validate_all_adapters,
)
from sheily_train.core.validation.validate_adapter import validate_adapter

# Todas las ramas a procesar
ALL_BRANCHES = [
    "antropologia",
    "economia",
    "psicologia",
    "historia",
    "quimica",
    "biologia",
    "filosofia",
    "sociologia",
    "politica",
    "ecologia",
    "educacion",
    "arte",
    "informatica",
    "ciberseguridad",
    "linguistica",
    "tecnologia",
    "derecho",
    "musica",
    "cine",
    "literatura",
    "ingenieria",
    "antropologia_digital",
    "economia_global",
    "filosofia_moderna",
    "marketing",
    "derecho_internacional",
    "psicologia_social",
    "fisica_cuantica",
    "astronomia",
    "IA_multimodal",
    "voz_emocional",
    "metacognicion",
]


def log_session(session_data):
    """Registrar sesión de entrenamiento"""
    log_file = Path("models/lora_adapters/logs/master_training_log.jsonl")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(session_data) + "\n")


def validate_environment():
    """Validar entorno completo"""
    print("🔍 VALIDANDO ENTORNO...")

    checks = [
        ("Modelo base", Path("models/gguf/llama-3.2.gguf")),
        ("Corpus", Path("corpus_ES")),
        ("Script entrenamiento", Path("sheily_train/core/training/training_pipeline.py")),
        ("Script validación", Path("sheily_train/core/validation/validate_adapter.py")),
    ]

    for name, path in checks:
        if path.exists():
            print(f"✅ {name}: OK")
        else:
            print(f"❌ {name}: FALTA")
            return False

    print("✅ Entorno validado completamente")
    return True


def process_all_branches():
    """Procesar todas las ramas"""
    if not validate_environment():
        print("❌ Entorno inválido")
        return False

    results = {"total": len(ALL_BRANCHES), "successful": 0, "failed": 0, "sessions": []}

    print(f"🚀 INICIANDO RETRENAMIENTO COMPLETO")
    print(f"📋 Ramas a procesar: {results['total']}")

    for i, branch in enumerate(ALL_BRANCHES, 1):
        print(f"\n📍 Progreso: {i}/{results['total']}")
        print(f"🎯 Procesando: {branch}")

        start_time = time.time()

        # Ejecutar pipeline
        cmd = [
            sys.executable,
            "sheily_train/core/training/training_pipeline.py",
            branch,
            "corpus_ES",
            f"models/lora_adapters/retraining/{branch}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            processing_time = time.time() - start_time

            if result.returncode == 0:
                results["successful"] += 1
                status = "SUCCESS"
                print("✅ Completado exitosamente")
            else:
                results["failed"] += 1
                status = "FAILED"
                print(f"❌ Fallido: {result.stderr.strip()}")

            # Registrar sesión
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "branch": branch,
                "status": status,
                "processing_time": processing_time,
                "return_code": result.returncode,
            }
            results["sessions"].append(session_data)
            log_session(session_data)

        except subprocess.TimeoutExpired:
            results["failed"] += 1
            print("⏰ Timeout")
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "branch": branch,
                "status": "TIMEOUT",
                "processing_time": time.time() - start_time,
            }
            log_session(session_data)

        except Exception as e:
            results["failed"] += 1
            print(f"❌ Error: {e}")
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "branch": branch,
                "status": "ERROR",
                "error": str(e),
            }
            log_session(session_data)

        # Pausa entre ramas
        if i < results["total"]:
            print("⏳ Pausa breve...")
            time.sleep(5)

    return results


def generate_final_report(results):
    """Generar reporte final"""
    print("\n" + "=" * 60)
    print("📊 REPORTE FINAL DE CORRECCIÓN")
    print("=" * 60)

    print(f"✅ Exitosos: {results['successful']}")
    print(f"❌ Fallidos: {results['failed']}")
    print(f"📈 Tasa de éxito: {results['successful']/results['total']*100:.1f}%")

    # Guardar reporte detallado
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_branches": results["total"],
            "successful": results["successful"],
            "failed": results["failed"],
            "success_rate": results["successful"] / results["total"] * 100,
        },
        "sessions": results["sessions"],
    }

    with open("audit_2024/reports/final_correction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n✅ Reporte guardado: audit_2024/reports/final_correction_report.json")

    if results["successful"] / results["total"] >= 0.95:
        print("🎉 CORRECCIÓN COMPLETA EXITOSA")
        return True
    else:
        print("⚠️ CORRECCIÓN PARCIAL - REQUIERE REVISIÓN")
        return False


def main():
    """Función principal"""
    print("🔥 CORRECCIÓN COMPLETA DEL PROYECTO SHEILY")
    print("=" * 60)

    # Procesar todas las ramas
    results = process_all_branches()

    # Si el entorno es inválido, no generar reporte
    if results is False:
        print("❌ No se puede continuar con entorno inválido")
        return 1

    # Generar reporte final
    success = generate_final_report(results)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
