#!/usr/bin/env python3
"""
AUDITORÍA DE ADAPTADORES LoRA - FASE 1
Script para identificar adaptadores funcionales vs corruptos
"""

import json
import os
import sys
from pathlib import Path


def analyze_adapter(adapter_path):
    """Analizar un adaptador individual"""
    result = {"path": str(adapter_path), "functional": False, "issues": [], "size": 0, "files": []}

    try:
        # Verificar archivos presentes
        config_file = adapter_path / "adapter_config.json"
        model_files = list(adapter_path.glob("adapter_model*.safetensors")) + list(
            adapter_path.glob("adapter_model*.bin")
        )

        result["files"] = [f.name for f in model_files]
        if config_file.exists():
            result["files"].append("adapter_config.json")

        # Verificar config
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                result["config_valid"] = True
                result["config_size"] = len(json.dumps(config))
            except:
                result["config_valid"] = False
                result["issues"].append("Config JSON inválido")
        else:
            result["config_valid"] = False
            result["issues"].append("Falta adapter_config.json")

        # Verificar modelos
        model_issues = []
        for model_file in model_files:
            size = model_file.stat().st_size
            result["size"] += size

            if size < 1000:  # Menos de 1KB = sospechoso
                model_issues.append(f"{model_file.name}: {size} bytes (muy pequeño)")
            elif size < 100000:  # Menos de 100KB = posiblemente corrupto
                model_issues.append(f"{model_file.name}: {size} bytes (pequeño)")

        if model_issues:
            result["issues"].extend(model_issues)

        # Determinar si es funcional
        if (
            config_file.exists()
            and result.get("config_valid", False)
            and len(model_files) > 0
            and result["size"] > 100000
        ):
            result["functional"] = True

    except Exception as e:
        result["issues"].append(f"Error analizando: {str(e)}")

    return result


def main():
    """Función principal de auditoría"""
    base_path = Path("models/lora_adapters")
    branches_file = Path("BRANCHES.txt")

    # Leer ramas definidas
    defined_branches = []
    if branches_file.exists():
        with open(branches_file, "r", encoding="utf-8") as f:
            defined_branches = [line.strip() for line in f if line.strip()]

    print("🔍 AUDITORÍA DE ADAPTADORES LoRA")
    print("=" * 50)
    print(f"Ramas definidas: {len(defined_branches)}")
    print(f"Directorios encontrados: {len(list(base_path.iterdir()))}")

    # Análisis por rama
    results = {"functional": [], "corrupted": [], "missing": [], "total_size": 0, "total_files": 0}

    for branch in defined_branches:
        branch_path = base_path / branch
        if branch_path.exists():
            analysis = analyze_adapter(branch_path)
            results["total_size"] += analysis["size"]
            results["total_files"] += len(analysis["files"])

            if analysis["functional"]:
                results["functional"].append((branch, analysis))
                status = "✅ FUNCIONAL"
            else:
                results["corrupted"].append((branch, analysis))
                status = "❌ CORRUPTO"

            print(f"\n{branch}: {status}")
            print(f"  Tamaño total: {analysis['size']/1024:.1f}KB")
            print(f"  Archivos: {len(analysis['files'])}")
            if analysis["issues"]:
                for issue in analysis["issues"][:3]:  # Solo primeros 3 issues
                    print(f"  ⚠️  {issue}")
        else:
            results["missing"].append(branch)
            print(f"\n{branch}: ❌ FALTA DIRECTORIO")

    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE AUDITORÍA")
    print("=" * 50)
    print(f"✅ Adaptadores funcionales: {len(results['functional'])}")
    print(f"❌ Adaptadores corruptos: {len(results['corrupted'])}")
    print(f"❌ Ramas faltantes: {len(results['missing'])}")
    print(f"📁 Tamaño total: {results['total_size']/1024/1024:.1f}MB")
    print(f"📄 Total archivos: {results['total_files']}")

    # Guardar resultados
    with open("audit_2024/reports/lora_audit.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": str(Path.cwd().resolve()),
                "summary": {
                    "functional": len(results["functional"]),
                    "corrupted": len(results["corrupted"]),
                    "missing": len(results["missing"]),
                    "total_size_mb": results["total_size"] / 1024 / 1024,
                    "total_files": results["total_files"],
                },
                "details": {
                    "functional": {k: v for k, v in results["functional"]},
                    "corrupted": {k: v for k, v in results["corrupted"]},
                    "missing": results["missing"],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n✅ Auditoría guardada en: audit_2024/reports/lora_audit.json")
    return results


if __name__ == "__main__":
    main()
