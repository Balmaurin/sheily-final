#!/usr/bin/env python3
"""
GENERADOR DE REPORTE CONSOLIDADO - FASE 1
Script para generar reporte completo del estado actual del proyecto
"""

import json
import sys
from pathlib import Path


def load_audit_data():
    """Cargar datos de auditorías previas"""
    reports_dir = Path("audit_2024/reports")

    # Cargar auditoría LoRA
    lora_file = reports_dir / "lora_audit.json"
    if lora_file.exists():
        try:
            with open(lora_file, "r", encoding="utf-8") as f:
                lora_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error al cargar archivo de auditoría LoRA: {e}")
            return None
    else:
        print("❌ Archivo de auditoría LoRA no encontrado")
        return None

    # Cargar auditoría corpus
    corpus_file = reports_dir / "corpus_audit.json"
    if corpus_file.exists():
        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Error al cargar archivo de auditoría corpus: {e}")
            return None
    else:
        print("❌ Archivo de auditoría corpus no encontrado")
        return None

    return lora_data, corpus_data


def analyze_cross_references(lora_data, corpus_data):
    """Analizar correlaciones entre LoRA y corpus"""
    analysis = {
        "functional_with_good_data": [],
        "functional_with_poor_data": [],
        "corrupted_with_good_data": [],
        "corrupted_with_poor_data": [],
        "priority_retraining": [],
    }

    # Crear mapas por rama
    corpus_details = corpus_data.get("details", [])
    lora_details = lora_data.get("details", {})
    corpus_by_branch = {item["branch"]: item for item in corpus_details if "branch" in item}
    lora_by_branch = {}

    # Procesar datos LoRA funcionales
    functional_lora = lora_details.get("functional", {})
    for branch, details in functional_lora.items():
        lora_by_branch[branch] = {"functional": True, "details": details}

    # Procesar datos LoRA corruptos
    corrupted_lora = lora_details.get("corrupted", {})
    for branch, details in corrupted_lora.items():
        lora_by_branch[branch] = {"functional": False, "details": details}

    # Cruzar datos
    for branch in lora_by_branch:
        if branch in corpus_by_branch:
            corpus_info = corpus_by_branch[branch]
            lora_info = lora_by_branch[branch]

            corpus_quality = corpus_info.get("data_quality", "unknown")

            if lora_info["functional"]:
                if corpus_quality in ["excellent", "good"]:
                    analysis["functional_with_good_data"].append(branch)
                else:
                    analysis["functional_with_poor_data"].append(branch)
            else:
                if corpus_quality in ["excellent", "good"]:
                    analysis["corrupted_with_good_data"].append(branch)
                    analysis["priority_retraining"].append(branch)
                else:
                    analysis["corrupted_with_poor_data"].append(branch)

    return analysis


def generate_recommendations(lora_data, corpus_data, cross_analysis):
    """Generar recomendaciones basadas en los hallazgos"""
    recommendations = {
        "immediate_actions": [],
        "retraining_priority": [],
        "data_improvements": [],
        "structural_changes": [],
    }

    # Acciones inmediatas
    if cross_analysis["corrupted_with_good_data"]:
        recommendations["immediate_actions"].append(
            f"🚨 RETRENAMIENTO URGENTE: {len(cross_analysis['corrupted_with_good_data'])} ramas tienen datos excelentes pero adaptadores corruptos"
        )

    if cross_analysis["functional_with_poor_data"]:
        recommendations["immediate_actions"].append(
            f"⚠️ MEJORA DE DATOS: {len(cross_analysis['functional_with_poor_data'])} ramas tienen adaptadores funcionales pero datos pobres"
        )

    # Prioridad de reentrenamiento
    recommendations["retraining_priority"] = cross_analysis["priority_retraining"]

    # Mejoras de datos
    corpus_details = corpus_data.get("details", [])
    poor_data_branches = [item for item in corpus_details if item.get("data_quality") == "poor"]
    if poor_data_branches:
        recommendations["data_improvements"].append(
            f"📚 MEJORAR DATOS: {len(poor_data_branches)} ramas necesitan más datos de entrenamiento"
        )

    # Cambios estructurales
    recommendations["structural_changes"].extend(
        [
            "🔄 Crear estructura organizada de directorios para adaptadores",
            "📁 Separar versiones experimentales de producción",
            "📋 Crear documentación clara de procesos",
            "🔍 Implementar validación automática de adaptadores",
        ]
    )

    return recommendations


def main():
    """Función principal"""
    print("📋 GENERANDO REPORTE CONSOLIDADO")
    print("=" * 60)

    # Cargar datos
    data = load_audit_data()
    if not data:
        return

    lora_data, corpus_data = data

    # Análisis cruzado
    cross_analysis = analyze_cross_references(lora_data, corpus_data)

    # Generar recomendaciones
    recommendations = generate_recommendations(lora_data, corpus_data, cross_analysis)

    # Crear reporte consolidado
    consolidated_report = {
        "timestamp": str(Path.cwd().resolve()),
        "executive_summary": {
            "lora_functional": lora_data.get("summary", {}).get("functional", 0),
            "lora_corrupted": lora_data.get("summary", {}).get("corrupted", 0),
            "corpus_excellent": corpus_data.get("summary", {}).get("excellent_quality", 0),
            "corpus_good": corpus_data.get("summary", {}).get("good_quality", 0),
            "corpus_poor": corpus_data.get("summary", {}).get("poor_quality", 0),
            "ready_for_production": len(cross_analysis.get("functional_with_good_data", [])),
            "needs_retraining": len(cross_analysis.get("corrupted_with_good_data", [])),
            "needs_data_improvement": len(cross_analysis.get("functional_with_poor_data", [])),
        },
        "cross_analysis": cross_analysis,
        "recommendations": recommendations,
        "next_steps": [
            "✅ Completar análisis de scripts de entrenamiento",
            "📋 Crear plan detallado de reentrenamiento",
            "🔧 Implementar estructura de directorios organizada",
            "📚 Mejorar datos de entrenamiento para ramas pobres",
            "🔄 Crear pipeline de entrenamiento automatizado",
        ],
    }

    # Guardar reporte
    try:
        with open("audit_2024/reports/consolidated_report.json", "w", encoding="utf-8") as f:
            json.dump(consolidated_report, f, indent=2, ensure_ascii=False)
    except (IOError, OSError) as e:
        print(f"❌ Error al guardar reporte consolidado: {e}")
        return None

    # Mostrar resumen ejecutivo
    print("🏆 RAMAS LISTAS PARA PRODUCCIÓN:")
    for branch in cross_analysis["functional_with_good_data"]:
        print(f"  ✅ {branch}")

    print(
        f"\n🔧 RAMAS QUE NECESITAN RETRENAMIENTO URGENTE ({len(cross_analysis['priority_retraining'])}):"
    )
    for branch in cross_analysis["priority_retraining"][:5]:  # Mostrar primeras 5
        print(f"  🔄 {branch}")
    if len(cross_analysis["priority_retraining"]) > 5:
        print(f"  ... y {len(cross_analysis['priority_retraining']) - 5} más")

    print("\n📊 ESTADÍSTICAS CLAVE:")
    total_branches = 39
    lora_functional = lora_data.get("summary", {}).get("functional", 0)
    corpus_excellent = corpus_data.get("summary", {}).get("excellent_quality", 0)
    ready_branches = len(cross_analysis.get("functional_with_good_data", []))

    print(
        f"  • Adaptadores funcionales: {lora_functional}/{total_branches} ({lora_functional/total_branches*100:.1f}%)"
        if total_branches > 0
        else f"  • Adaptadores funcionales: {lora_functional}/{total_branches} (N/A%)"
    )
    print(
        f"  • Datos de calidad excelente: {corpus_excellent}/{total_branches} ({corpus_excellent/total_branches*100:.1f}%)"
        if total_branches > 0
        else f"  • Datos de calidad excelente: {corpus_excellent}/{total_branches} (N/A%)"
    )
    print(
        f"  • Ramas listas completamente: {ready_branches}/{total_branches} ({ready_branches/total_branches*100:.1f}%)"
        if total_branches > 0
        else f"  • Ramas listas completamente: {ready_branches}/{total_branches} (N/A%)"
    )

    print("\n✅ Reporte consolidado guardado en: audit_2024/reports/consolidated_report.json")
    return consolidated_report


if __name__ == "__main__":
    main()
