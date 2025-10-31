#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🏆 SHEILY TEST SUITE MAESTRO - RUNNER UNIFICADO

Sistema completo de tests para Sheily-AI organizado profesionalmente:

Estructura:
├── unit/         → Tests unitarios (branches, config, datasets, etc.)
├── integration/  → Tests de integración (API, E2E, etc.)  
├── evaluation/   → Tests de evaluación RAG (métricas, corpus, etc.)
├── core/         → Tests core del sistema (memoria, RAG core, etc.)
└── config/       → Configuración centralizada

Características:
✅ Ejecución selectiva por categoría
✅ Reportes unificados y detallados
✅ Configuración centralizada
✅ Paralelización inteligente
✅ Sistema de umbrales de calidad
✅ Integración con CI/CD
"""

import argparse
import concurrent.futures
import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuración
TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
REPORTS_ROOT = PROJECT_ROOT / "reports"

# Importar configuración
sys.path.insert(0, str(TESTS_ROOT))
from config.test_config import *


class TestResult:
    """Resultado de ejecución de un test"""

    def __init__(
        self,
        category: str,
        test_name: str,
        passed: bool,
        execution_time: float,
        details: Optional[Dict] = None,
    ):
        self.category = category
        self.test_name = test_name
        self.passed = passed
        self.execution_time = execution_time
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class SheilyTestSuite:
    """Suite maestra de tests de Sheily"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.categories = {
            "unit": TESTS_ROOT / "unit",
            "integration": TESTS_ROOT / "integration",
            "evaluation": TESTS_ROOT / "evaluation",
            "core": TESTS_ROOT / "core",
        }

    def setup_environment(self) -> bool:
        """Configurar entorno para tests"""
        print("🔧 Configurando entorno de tests...")

        # Crear directorios de reportes
        for category in self.categories.keys():
            (REPORTS_ROOT / category).mkdir(parents=True, exist_ok=True)

        # Verificar dependencias críticas
        try:
            if not SHEILY_GGUF_PATH.exists():
                print(f"⚠️  Modelo GGUF no encontrado: {SHEILY_GGUF_PATH}")
                print("   Algunos tests RAG se saltarán")
            else:
                print(f"✅ Modelo GGUF: {SHEILY_GGUF_PATH}")

            if not CORPUS_ROOT.exists():
                print(f"⚠️  Corpus no encontrado: {CORPUS_ROOT}")
                print("   Tests dinámicos se saltarán")
            else:
                print(f"✅ Corpus: {CORPUS_ROOT}")

            print(f"✅ Reportes: {REPORTS_ROOT}")
            return True

        except Exception as e:
            print(f"❌ Error configurando entorno: {e}")
            return False

    def discover_tests(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Descubrir tests por categoría"""
        discovered = {}

        categories_to_scan = [category] if category else self.categories.keys()

        for cat in categories_to_scan:
            if cat not in self.categories:
                continue

            cat_path = self.categories[cat]
            if not cat_path.exists():
                discovered[cat] = []
                continue

            test_files = []
            for test_file in cat_path.glob("test_*.py"):
                test_files.append(str(test_file.relative_to(TESTS_ROOT)))

            discovered[cat] = sorted(test_files)

        return discovered

    def run_category(self, category: str, parallel: bool = False) -> List[TestResult]:
        """Ejecutar tests de una categoría"""
        category_path = self.categories.get(category)
        if not category_path or not category_path.exists():
            print(f"⚠️  Categoría '{category}' no encontrada")
            return []

        print(f"\n📂 Ejecutando tests de categoría: {category.upper()}")
        print("-" * 50)

        # Casos especiales por categoría
        if category == "evaluation":
            return self._run_evaluation_tests()
        elif category == "unit":
            return self._run_unit_tests()
        elif category == "integration":
            return self._run_integration_tests()
        elif category == "core":
            return self._run_core_tests()
        else:
            return self._run_generic_tests(category)

    def _run_evaluation_tests(self) -> List[TestResult]:
        """Ejecutar tests de evaluación RAG"""
        results = []

        # Test completo RAG suite
        suite_path = TESTS_ROOT / "evaluation" / "test_complete_rag_suite.py"
        if suite_path.exists():
            print("🎯 Ejecutando suite completa RAG...")

            start_time = time.time()
            try:
                # Usar el runner especializado
                from run_complete_rag_evaluation import run_complete_suite

                summary = run_complete_suite(verbose=False, quick_mode=False)

                execution_time = time.time() - start_time
                passed = summary.get("success", False)

                result = TestResult(
                    category="evaluation",
                    test_name="rag_complete_suite",
                    passed=passed,
                    execution_time=execution_time,
                    details=summary,
                )
                results.append(result)

                status = "✅" if passed else "❌"
                print(f"   {status} Suite RAG completa: {execution_time:.1f}s")

            except Exception as e:
                execution_time = time.time() - start_time
                result = TestResult(
                    category="evaluation",
                    test_name="rag_complete_suite",
                    passed=False,
                    execution_time=execution_time,
                    details={"error": str(e)},
                )
                results.append(result)
                print(f"   ❌ Error en suite RAG: {e}")

        return results

    def _run_unit_tests(self) -> List[TestResult]:
        """Ejecutar tests unitarios"""
        return self._run_unittest_discovery("unit")

    def _run_integration_tests(self) -> List[TestResult]:
        """Ejecutar tests de integración"""
        return self._run_unittest_discovery("integration")

    def _run_core_tests(self) -> List[TestResult]:
        """Ejecutar tests core"""
        return self._run_unittest_discovery("core")

    def _run_generic_tests(self, category: str) -> List[TestResult]:
        """Ejecutar tests genéricos usando unittest"""
        return self._run_unittest_discovery(category)

    def _run_unittest_discovery(self, category: str) -> List[TestResult]:
        """Ejecutar tests usando unittest discovery"""
        results = []
        category_path = self.categories[category]

        if not category_path.exists():
            return results

        # Cambiar directorio de trabajo al proyecto para que los tests funcionen
        import os

        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)

        try:
            # Descubrir y ejecutar tests
            loader = unittest.TestLoader()
            suite = loader.discover(str(category_path), pattern="test_*.py")

            # Contar tests
            test_count = suite.countTestCases()
            if test_count == 0:
                print(f"   ℹ️  No hay tests en {category}")
                return results

            print(f"   🔬 Ejecutando {test_count} tests unitarios...")

            start_time = time.time()
            # Usar StringIO para capturar la salida en lugar de /dev/null
            import io

            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2)
            test_result = runner.run(suite)
            execution_time = time.time() - start_time
        finally:
            # Restaurar directorio original
            os.chdir(original_cwd)

        # Procesar resultados
        passed = test_result.wasSuccessful()

        # Capturar detalles de fallos
        failure_details = []
        for test, traceback in test_result.failures:
            failure_details.append({"test": str(test), "error": traceback})

        error_details = []
        for test, traceback in test_result.errors:
            error_details.append({"test": str(test), "error": traceback})

        result = TestResult(
            category=category,
            test_name=f"{category}_unittest_suite",
            passed=passed,
            execution_time=execution_time,
            details={
                "tests_run": test_result.testsRun,
                "failures": len(test_result.failures),
                "errors": len(test_result.errors),
                "skipped": len(test_result.skipped),
                "failure_details": failure_details,
                "error_details": error_details,
            },
        )
        results.append(result)

        status = "✅" if passed else "❌"
        print(f"   {status} {test_count} tests: {execution_time:.1f}s")

        return results

    def generate_report(self, results: List[TestResult]) -> Dict:
        """Generar reporte consolidado"""

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        total_time = sum(r.execution_time for r in results)

        # Agrupar por categoría
        by_category = {}
        for result in results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        report = {
            "execution_info": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": round(total_time, 2),
                "categories_tested": list(by_category.keys()),
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": round(passed_tests / total_tests * 100, 1)
                if total_tests > 0
                else 0,
            },
            "by_category": {},
            "detailed_results": [],
        }

        # Detalles por categoría
        for category, cat_results in by_category.items():
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_total = len(cat_results)
            cat_time = sum(r.execution_time for r in cat_results)

            report["by_category"][category] = {
                "tests": cat_total,
                "passed": cat_passed,
                "failed": cat_total - cat_passed,
                "execution_time": round(cat_time, 2),
                "success_rate": round(cat_passed / cat_total * 100, 1) if cat_total > 0 else 0,
            }

        # Detalles individuales
        for result in results:
            report["detailed_results"].append(
                {
                    "category": result.category,
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp,
                    "details": result.details,
                }
            )

        return report

    def run_all(self, categories: Optional[List[str]] = None, parallel: bool = False) -> Dict:
        """Ejecutar todos los tests"""

        print("\n" + "=" * 80)
        print("🏆 SHEILY TEST SUITE MAESTRO")
        print("=" * 80)
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.setup_environment():
            return {"success": False, "error": "Falló configuración del entorno"}

        # Determinar categorías a ejecutar
        cats_to_run = categories or list(self.categories.keys())
        print(f"📂 Categorías: {', '.join(cats_to_run)}")
        print("=" * 80)

        # Descubrir tests
        discovered = self.discover_tests()
        total_discovered = sum(len(tests) for tests in discovered.values())
        print(f"🔍 Tests descubiertos: {total_discovered}")

        for category, tests in discovered.items():
            if category in cats_to_run:
                print(f"   • {category}: {len(tests)} tests")

        # Ejecutar tests
        all_results = []
        start_time = time.time()

        for category in cats_to_run:
            if category in discovered:
                category_results = self.run_category(category, parallel)
                all_results.extend(category_results)

        total_execution_time = time.time() - start_time

        # Generar reporte
        report = self.generate_report(all_results)
        report["execution_info"]["total_execution_time"] = round(total_execution_time, 2)

        # Mostrar resumen
        self._print_summary(report)

        # Guardar reporte
        report_path = REPORTS_ROOT / "sheily_test_suite_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📋 Reporte completo: {report_path}")

        # Validar Quality Gates
        try:
            from core.quality_gates import QualityGateValidator

            validator = QualityGateValidator()
            quality_report = validator.validate_report(report)

            # Guardar reporte de calidad
            quality_path = REPORTS_ROOT / "quality_gates_report.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)

            print(f"🎯 Quality Gates: {quality_path}")

            # Actualizar reporte principal con información de calidad
            report["quality_validation"] = {
                "all_gates_passed": quality_report["summary"]["all_gates_passed"],
                "quality_score": quality_report["summary"]["quality_score"],
                "violations_count": quality_report["summary"]["violations_count"],
            }

        except Exception as e:
            print(f"⚠️  Error en Quality Gates: {e}")
            report["quality_validation"] = {"error": str(e)}

        return report

    def _print_summary(self, report: Dict):
        """Imprimir resumen de resultados"""
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE EJECUCIÓN")
        print("=" * 80)

        summary = report["summary"]
        print(f"⏱️  Tiempo total: {report['execution_info']['total_execution_time']}s")
        print(f"✅ Tests pasados: {summary['passed']}")
        print(f"❌ Tests fallados: {summary['failed']}")
        print(f"📈 Total: {summary['total_tests']}")
        print(f"🎯 Tasa de éxito: {summary['success_rate']}%")

        print(f"\n📂 POR CATEGORÍA:")
        for category, data in report["by_category"].items():
            status = "✅" if data["failed"] == 0 else "❌"
            print(
                f"   {status} {category}: {data['passed']}/{data['tests']} ({data['success_rate']}%)"
            )

        # Calificación general
        if summary["success_rate"] >= 95:
            grade = "🏆 EXCELENTE"
        elif summary["success_rate"] >= 85:
            grade = "🥇 MUY BUENO"
        elif summary["success_rate"] >= 75:
            grade = "🥈 BUENO"
        elif summary["success_rate"] >= 65:
            grade = "🥉 ACEPTABLE"
        else:
            grade = "⚠️  NECESITA MEJORA"

        print(f"\n🏅 Calificación: {grade}")

        # Mostrar información de Quality Gates si está disponible
        if "quality_validation" in report and "error" not in report["quality_validation"]:
            qv = report["quality_validation"]
            gates_status = "✅ PASADO" if qv["all_gates_passed"] else "❌ FALLIDO"
            print(f"🎯 Quality Gates: {gates_status} ({qv['quality_score']:.1f}/100)")
            if qv["violations_count"] > 0:
                print(f"⚠️  Violaciones: {qv['violations_count']}")

        print("=" * 80)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="🏆 Suite Maestro de Tests de Sheily-AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_sheily_tests.py                    # Todos los tests
  python run_sheily_tests.py --category unit    # Solo tests unitarios
  python run_sheily_tests.py --category evaluation --quick  # RAG rápido
  python run_sheily_tests.py --list             # Listar tests disponibles
  python run_sheily_tests.py --parallel         # Ejecución paralela
        """,
    )

    parser.add_argument(
        "-c",
        "--category",
        choices=["unit", "integration", "evaluation", "core"],
        help="Ejecutar solo una categoría específica",
    )
    parser.add_argument("--list", action="store_true", help="Listar tests disponibles")
    parser.add_argument(
        "--parallel", action="store_true", help="Ejecución paralela (donde sea posible)"
    )
    parser.add_argument("--quick", action="store_true", help="Modo rápido para tests RAG")

    args = parser.parse_args()

    suite = SheilyTestSuite()

    # Listar tests
    if args.list:
        print("📋 TESTS DISPONIBLES POR CATEGORÍA:")
        print("=" * 50)

        discovered = suite.discover_tests()
        for category, tests in discovered.items():
            print(f"\n📂 {category.upper()} ({len(tests)} tests):")
            for test in tests:
                print(f"   • {test}")

        return 0

    # Ejecutar tests
    try:
        categories = [args.category] if args.category else None
        report = suite.run_all(categories=categories, parallel=args.parallel)

        if report.get("success", True) and report["summary"]["failed"] == 0:
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Ejecución interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
