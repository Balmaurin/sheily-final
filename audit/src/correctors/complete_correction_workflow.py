#!/usr/bin/env python3
"""
WORKFLOW UNIFICADO DE CORRECCIÓN COMPLETA
Ejecuta todo el proceso de corrección de manera integrada y seamless
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_workflow_step(step_name, command, expected_code=0):
    """Ejecutar un paso del workflow"""
    print(f"\n🔄 Ejecutando: {step_name}")

    try:
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        execution_time = time.time() - start_time

        success = result.returncode == expected_code

        if success:
            print(f"  ✅ {step_name}: COMPLETADO ({execution_time:.1f}s)")
        else:
            print(f"  ❌ {step_name}: FALLIDO")
            print(f"    Error: {result.stderr.strip()}")

        return success, {
            "step": step_name,
            "success": success,
            "execution_time": execution_time,
            "return_code": result.returncode,
        }

    except Exception as e:
        print(f"  💥 {step_name}: ERROR - {str(e)}")
        return False, {"step": step_name, "success": False, "error": str(e)}


def run_complete_correction_workflow():
    """Ejecutar workflow completo de corrección"""
    print("🚀 WORKFLOW COMPLETO DE CORRECCIÓN - 36 ADAPTADORES LoRA")
    print("=" * 70)

    workflow_steps = [
        ("Auditoría Previa", [sys.executable, "audit_2024/pre_training_audit.py"], 0),
        ("Corrección Masiva", [sys.executable, "audit_2024/massive_adapter_correction.py"], 0),
        (
            "Validación Post-Entrenamiento",
            [sys.executable, "audit_2024/post_training_validation.py"],
            0,
        ),
        ("Tests de Integración", [sys.executable, "audit_2024/integration_test_suite.py"], 0),
        ("Monitoreo Final", [sys.executable, "sheily_train/tools/monitoring/health_monitor.py"], 0),
    ]

    results = []
    successful_steps = 0

    for step_name, command, expected_code in workflow_steps:
        success, step_result = run_workflow_step(step_name, command, expected_code)
        results.append(step_result)

        if success:
            successful_steps += 1

        # Pausa breve entre pasos
        time.sleep(2)

    return results


def generate_workflow_report(results):
    """Generar reporte del workflow completo"""
    total_steps = len(results)
    successful_steps = sum(1 for r in results if r["success"])
    total_time = sum(r["execution_time"] for r in results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "workflow_summary": {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "success_rate": successful_steps / total_steps * 100 if total_steps > 0 else 0,
            "total_execution_time": total_time,
        },
        "step_details": results,
    }

    # Guardar reporte
    report_file = Path("audit_2024/reports/complete_workflow_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    """Función principal del workflow"""
    print("🎯 WORKFLOW UNIFICADO DE CORRECCIÓN COMPLETA")
    print("=" * 70)

    try:
        # Ejecutar workflow completo
        results = run_complete_correction_workflow()

        # Generar reporte
        report = generate_workflow_report(results)

        # Mostrar resumen final
        print("\n" + "=" * 70)
        print("📊 REPORTE FINAL DEL WORKFLOW")
        print("=" * 70)

        summary = report["workflow_summary"]
        print(f"🔧 Pasos ejecutados: {summary['total_steps']}")
        print(f"✅ Pasos exitosos: {summary['successful_steps']}")
        print(f"❌ Pasos fallidos: {summary['failed_steps']}")
        print(f"📈 Tasa de éxito: {summary['success_rate']:.1f}%")
        print(f"⏱️ Tiempo total: {summary['total_execution_time']:.1f} segundos")

        print(f"\n📋 Reporte completo guardado: audit_2024/reports/complete_workflow_report.json")

        # Evaluar éxito del workflow completo
        if summary["success_rate"] >= 95:
            print("🎉 WORKFLOW COMPLETO EJECUTADO EXITOSAMENTE")
            print("🚀 El proyecto está completamente corregido y operativo")
            return 0
        elif summary["success_rate"] >= 80:
            print("✅ WORKFLOW COMPLETADO PARCIALMENTE - REVISIÓN SUGERIDA")
            return 0
        else:
            print("⚠️ WORKFLOW CON PROBLEMAS - REQUIERE ATENCIÓN")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ WORKFLOW INTERRUMPIDO POR USUARIO")
        return 1
    except Exception as e:
        print(f"\n❌ Error crítico en workflow: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
