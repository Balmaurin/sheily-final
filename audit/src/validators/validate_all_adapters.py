#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDADOR COMPLETO DE ADAPTADORES LoRA
====================================
Valida todos los adaptadores LoRA generados y verifica su integridad.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def validate_single_adapter(adapter_path):
    """Valida un adaptador individual"""
    results = {"valid": False, "score": 0, "max_score": 100, "files": {}, "issues": []}

    adapter_path = Path(adapter_path)

    # 1. Verificar archivos requeridos
    required_files = {
        "adapter_config.json": 20,
        "adapter_model.safetensors": 40,
        "metadata.json": 10,
    }

    total_file_score = 0
    for file_name, file_score in required_files.items():
        file_path = adapter_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            results["files"][file_name] = {
                "exists": True,
                "size": size,
                "size_mb": size / (1024 * 1024),
            }
            total_file_score += file_score

            # Bonus por tamaño adecuado
            if file_name == "adapter_model.safetensors" and size > 1000000:  # > 1MB
                total_file_score += 10
            elif file_name == "adapter_config.json" and size > 100:  # > 100 bytes
                total_file_score += 5
        else:
            results["files"][file_name] = {"exists": False, "size": 0}
            results["issues"].append(f"Falta archivo: {file_name}")

    results["score"] += min(total_file_score, sum(required_files.values()) + 10)

    # 2. Validar configuración JSON
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Verificar campos críticos
            critical_fields = ["base_model_name", "branch", "peft_type", "r", "lora_alpha"]
            config_score = 0

            for field in critical_fields:
                if field in config:
                    config_score += 4  # 4 puntos por campo crítico

                    # Bonus por valores válidos
                    if field == "peft_type" and config[field] == "LORA":
                        config_score += 2
                    elif field == "r" and isinstance(config[field], int) and config[field] > 0:
                        config_score += 2
                    elif field == "lora_alpha" and isinstance(config[field], int) and config[field] > 0:
                        config_score += 2
                else:
                    results["issues"].append(f"Campo crítico faltante: {field}")

            results["score"] += config_score

        except json.JSONDecodeError as e:
            results["issues"].append(f"Config JSON inválido: {e}")
        except Exception as e:
            results["issues"].append(f"Error leyendo config: {e}")

    # 3. Validar modelo
    model_file = adapter_path / "adapter_model.safetensors"
    if model_file.exists():
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)

            # Verificar estructura básica del modelo
            if "lora_branch" in model_data and "rank" in model_data:
                results["score"] += 15
            else:
                results["issues"].append("Estructura de modelo incompleta")

        except Exception as e:
            results["issues"].append(f"Error leyendo modelo: {e}")

    # 4. Validar metadatos
    metadata_file = adapter_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if "branch" in metadata and "status" in metadata:
                results["score"] += 10
            else:
                results["issues"].append("Metadatos incompletos")

        except Exception as e:
            results["issues"].append(f"Error leyendo metadatos: {e}")

    # Determinar validez
    results["valid"] = results["score"] >= 70 and len(results["issues"]) == 0

    return results


def validate_all_adapters():
    """Valida todos los adaptadores generados"""
    print("🔍 VALIDACIÓN COMPLETA DE ADAPTADORES LoRA")
    print("=" * 60)

    base_path = Path("models/lora_adapters/retraining")
    if not base_path.exists():
        print("❌ Directorio de adaptadores no encontrado")
        return

    # Encontrar todas las ramas
    all_results = {}
    total_adapters = 0
    valid_adapters = 0

    print(f"📁 Buscando adaptadores en: {base_path}")
    print()

    for adapter_dir in base_path.iterdir():
        if adapter_dir.is_dir() and not adapter_dir.name.startswith("validation_report"):
            branch_name = adapter_dir.name
            total_adapters += 1

            print(f"🔍 Validando: {branch_name}")

            # Validar adaptador
            validation = validate_single_adapter(adapter_dir)
            all_results[branch_name] = validation

            # Mostrar resultado
            status = "✅" if validation["valid"] else "❌"
            score = validation["score"]
            print(f"   {status} Puntuación: {score}/{validation['max_score']}")

            if validation["issues"]:
                for issue in validation["issues"][:2]:  # Mostrar máximo 2 issues
                    print(f"      💡 {issue}")

            if validation["valid"]:
                valid_adapters += 1

            print()

    # Generar reporte final
    print("=" * 60)
    print("📊 REPORTE FINAL DE VALIDACIÓN")
    print("=" * 60)

    print(f"📋 Total de adaptadores: {total_adapters}")
    print(f"✅ Adaptadores válidos: {valid_adapters}")
    print(f"❌ Adaptadores inválidos: {total_adapters - valid_adapters}")
    print(f"📈 Tasa de éxito: {valid_adapters/total_adapters*100:.1f}%")

    # Calcular puntuación promedio
    scores = [r["score"] for r in all_results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"🎯 Puntuación promedio: {avg_score:.1f}/{100}")

    # Crear reporte detallado
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_adapters": total_adapters,
            "valid_adapters": valid_adapters,
            "invalid_adapters": total_adapters - valid_adapters,
            "success_rate": valid_adapters / total_adapters * 100 if total_adapters > 0 else 0,
            "average_score": avg_score,
        },
        "details": all_results,
    }

    # Guardar reporte
    report_file = base_path / f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Reporte guardado: {report_file}")

    # Evaluar resultado general
    if valid_adapters / total_adapters >= 0.95:
        print("🎉 VALIDACIÓN COMPLETA EXITOSA")
        print("✅ Todos los adaptadores están correctamente estructurados")
        return True
    elif valid_adapters / total_adapters >= 0.80:
        print("✅ VALIDACIÓN PARCIALMENTE EXITOSA")
        print("⚠️ Algunos adaptadores necesitan revisión")
        return True
    else:
        print("❌ VALIDACIÓN FALLIDA")
        print("🔧 Muchos adaptadores necesitan corrección")
        return False


def main():
    """Función principal"""
    success = validate_all_adapters()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
