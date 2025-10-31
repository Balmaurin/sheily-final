#!/usr/bin/env python3
"""
SUITE DE TESTS DE INTEGRACIÓN COMPLETA
Verifica que todos los componentes funcionen perfectamente juntos
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_integration_test(test_name, command, expected_return_code=0, timeout=60):
    """Ejecutar un test de integración"""
    print(f"🧪 Ejecutando: {test_name}")

    try:
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        execution_time = time.time() - start_time

        success = result.returncode == expected_return_code

        test_result = {
            "test_name": test_name,
            "success": success,
            "return_code": result.returncode,
            "execution_time": execution_time,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr),
        }

        if success:
            print(f"  ✅ {test_name}: PASSED ({execution_time:.1f}s)")
        else:
            print(f"  ❌ {test_name}: FAILED")
            print(f"    Código de retorno: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")

        return test_result

    except subprocess.TimeoutExpired:
        print(f"  ⏰ {test_name}: TIMEOUT")
        return {
            "test_name": test_name,
            "success": False,
            "return_code": -1,
            "execution_time": timeout,
            "error": "Timeout",
        }
    except Exception as e:
        print(f"  💥 {test_name}: ERROR - {str(e)}")
        return {"test_name": test_name, "success": False, "return_code": -2, "error": str(e)}


def run_complete_integration_suite():
    """Ejecutar suite completa de integración"""
    print("🚀 SUITE COMPLETA DE TESTS DE INTEGRACIÓN")
    print("=" * 60)

    base_path = Path(".")

    test_results = []

    # Test 1: Auditoría previa
    print("\n📋 FASE 1: AUDITORÍA PREVIA")
    test_results.append(
        run_integration_test(
            "Pre-training Audit",
            [sys.executable, "audit_2024/pre_training_audit.py"],
            expected_return_code=0,
            timeout=60,
        )
    )

    # Test 2: Scripts principales
    print("\n🔧 FASE 2: SCRIPTS PRINCIPALES")
    test_results.append(
        run_integration_test(
            "Complete Correction",
            [sys.executable, "audit_2024/complete_correction.py"],
            expected_return_code=0,
            timeout=120,
        )
    )

    # Test 3: Validación avanzada
    print("\n🔍 FASE 3: VALIDACIÓN AVANZADA")
    test_results.append(
        run_integration_test(
            "Advanced Validation",
            [
                sys.executable,
                "sheily_train/core/validation/advanced_validation.py",
                "models/lora_adapters/retraining/",
            ],
            expected_return_code=0,
            timeout=60,
        )
    )

    # Test 4: Monitoreo de salud
    print("\n💻 FASE 4: MONITOREO DE SALUD")
    test_results.append(
        run_integration_test(
            "Health Monitor",
            [sys.executable, "sheily_train/tools/monitoring/health_monitor.py"],
            expected_return_code=0,
            timeout=30,
        )
    )

    return test_results


def generate_integration_report(test_results):
    """Generar reporte de integración"""
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r["success"])
    failed_tests = total_tests - passed_tests

    # Calcular tiempo total
    total_time = sum(r["execution_time"] for r in test_results)

    # Crear reporte
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests * 100 if total_tests > 0 else 0,
            "total_execution_time": total_time,
        },
        "test_details": test_results,
    }

    # Guardar reporte
    report_file = Path("audit_2024/reports/integration_test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    """Función principal de integración"""
    print("🔗 VERIFICACIÓN DE INTEGRACIÓN SEAMLESS")
    print("=" * 60)

    try:
        # Ejecutar suite completa de tests
        test_results = run_complete_integration_suite()

        # Generar reporte
        report = generate_integration_report(test_results)

        # Mostrar resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE INTEGRACIÓN")
        print("=" * 60)

        summary = report["summary"]
        print(f"🧪 Tests ejecutados: {summary['total_tests']}")
        print(f"✅ Tests aprobados: {summary['passed_tests']}")
        print(f"❌ Tests fallidos: {summary['failed_tests']}")
        print(f"📈 Tasa de éxito: {summary['success_rate']:.1f}%")
        print(f"⏱️ Tiempo total: {summary['total_execution_time']:.1f} segundos")

        print(f"\n📋 Reporte guardado: audit_2024/reports/integration_test_report.json")

        # Evaluar integración general
        if summary["success_rate"] >= 95:
            print("🎉 INTEGRACIÓN SEAMLESS COMPLETAMENTE EXITOSA")
            return 0
        elif summary["success_rate"] >= 80:
            print("✅ INTEGRACIÓN SEAMLESS PARCIALMENTE EXITOSA")
            return 0
        else:
            print("⚠️ INTEGRACIÓN SEAMLESS CON PROBLEMAS")
            return 1

    except Exception as e:
        print(f"❌ Error crítico en integración: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
