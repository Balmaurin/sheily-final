#!/usr/bin/env python3
"""
AUDITORÍA DEL CORPUS - FASE 1
Script para catalogar datos de entrenamiento válidos por rama
"""

import json
import os
import sys
from pathlib import Path


def analyze_corpus_data(corpus_path):
    """Analizar datos del corpus por rama"""
    results = []

    # Leer ramas definidas
    branches_file = Path("BRANCHES.txt")
    if not branches_file.exists():
        print("❌ Archivo BRANCHES.txt no encontrado")
        return results

    with open(branches_file, "r", encoding="utf-8") as f:
        defined_branches = [line.strip() for line in f if line.strip()]

    print("📚 AUDITORÍA DEL CORPUS")
    print("=" * 50)
    print(f"Ramas definidas: {len(defined_branches)}")

    for branch in defined_branches:
        branch_path = corpus_path / branch
        branch_analysis = {
            "branch": branch,
            "path": str(branch_path),
            "exists": branch_path.exists(),
            "total_files": 0,
            "jsonl_files": 0,
            "total_size": 0,
            "subdirs": [],
            "sample_content": None,
            "data_quality": "unknown",
        }

        if branch_path.exists():
            # Analizar estructura
            for item in branch_path.iterdir():
                if item.is_dir():
                    branch_analysis["subdirs"].append(item.name)
                    subdir_path = branch_path / item.name

                    # Buscar archivos JSONL
                    jsonl_files = list(subdir_path.glob("*.jsonl"))
                    branch_analysis["jsonl_files"] += len(jsonl_files)
                    branch_analysis["total_files"] += len(list(subdir_path.glob("*")))

                    for jsonl_file in jsonl_files:
                        size = jsonl_file.stat().st_size
                        branch_analysis["total_size"] += size

                        # Verificar contenido si es pequeño
                        if size < 10000 and size > 0:  # Menos de 10KB pero no vacío
                            try:
                                with open(jsonl_file, "r", encoding="utf-8") as f:
                                    first_line = f.readline().strip()
                                    if first_line:
                                        branch_analysis["sample_content"] = first_line[:200] + "..."
                                        # Verificar si parece JSON válido
                                        try:
                                            json.loads(first_line)
                                            branch_analysis["data_quality"] = "good"
                                        except:
                                            branch_analysis["data_quality"] = "invalid_json"
                            except:
                                branch_analysis["data_quality"] = "unreadable"

            # Determinar calidad general
            if branch_analysis["jsonl_files"] > 0:
                if branch_analysis["total_size"] > 1000000:  # Más de 1MB
                    quality = "excellent"
                elif branch_analysis["total_size"] > 100000:  # Más de 100KB
                    quality = "good"
                else:
                    quality = "poor"
                branch_analysis["data_quality"] = quality

        results.append(branch_analysis)

        # Mostrar progreso
        if branch_analysis["exists"]:
            status = "✅" if branch_analysis["data_quality"] in ["excellent", "good"] else "⚠️"
            print(f"\n{branch}: {status} {branch_analysis['data_quality']}")
            print(f"  Tamaño: {branch_analysis['total_size']/1024:.1f}KB")
            print(f"  JSONL: {branch_analysis['jsonl_files']} archivos")
            print(f"  Subdirs: {', '.join(branch_analysis['subdirs'])}")
        else:
            print(f"\n{branch}: ❌ FALTA DIRECTORIO")

    return results


def generate_corpus_report(results):
    """Generar reporte detallado del corpus"""
    report = {
        "timestamp": str(Path.cwd().resolve()),
        "summary": {
            "total_branches": len(results),
            "branches_with_data": sum(1 for r in results if r["exists"]),
            "excellent_quality": sum(1 for r in results if r["data_quality"] == "excellent"),
            "good_quality": sum(1 for r in results if r["data_quality"] == "good"),
            "poor_quality": sum(1 for r in results if r["data_quality"] == "poor"),
            "missing_branches": sum(1 for r in results if not r["exists"]),
        },
        "details": results,
    }

    # Guardar reporte
    with open("audit_2024/reports/corpus_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    """Función principal"""
    corpus_path = Path("corpus_ES")

    if not corpus_path.exists():
        print(f"❌ Directorio de corpus no encontrado: {corpus_path}")
        return

    # Analizar corpus
    results = analyze_corpus_data(corpus_path)

    # Generar reporte
    report = generate_corpus_report(results)

    # Mostrar resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DEL CORPUS")
    print("=" * 50)
    print(f"✅ Ramas con datos: {report['summary']['branches_with_data']}")
    print(f"🏆 Calidad excelente: {report['summary']['excellent_quality']}")
    print(f"✅ Calidad buena: {report['summary']['good_quality']}")
    print(f"⚠️ Calidad pobre: {report['summary']['poor_quality']}")
    print(f"❌ Ramas faltantes: {report['summary']['missing_branches']}")

    print("\n✅ Auditoría del corpus guardada en: audit_2024/reports/corpus_audit.json")

    return results


if __name__ == "__main__":
    main()
