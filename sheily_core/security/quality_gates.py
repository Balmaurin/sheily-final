#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 SISTEMA DE QUALITY GATES - SHEILY AI

Sistema profesional de umbrales de calidad que evalúa si los resultados
de los tests cumplen con los estándares mínimos establecidos.

Características:
✅ Evaluación automática de umbrales
✅ Reportes detallados de calidad
✅ Integración con CI/CD
✅ Alertas y recomendaciones
✅ Métricas de tendencia

Uso:
    from tests.core.quality_gates import QualityGateValidator
    validator = QualityGateValidator()
    result = validator.validate_report(report_data)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class QualityGateValidator:
    """Validador de umbrales de calidad para Sheily-AI"""

    def __init__(self, thresholds_file: Optional[Path] = None):
        self.config_dir = Path(__file__).resolve().parent.parent / "config"
        self.thresholds_file = (
            thresholds_file
            or Path(__file__).resolve().parent.parent.parent
            / "config"
            / "test"
            / "config"
            / "quality_thresholds.yaml"
        )
        self.thresholds = self._load_thresholds()
        self.violations: List[Dict] = []

    def _load_thresholds(self) -> Dict:
        """Cargar umbrales de configuración"""
        if not self.thresholds_file.exists():
            raise FileNotFoundError(f"Archivo de umbrales no encontrado: {self.thresholds_file}")

        with open(self.thresholds_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def validate_report(self, report_data: Dict) -> Dict:
        """
        Validar un reporte completo contra los umbrales de calidad

        Args:
            report_data: Datos del reporte de tests

        Returns:
            Dict con resultados de validación
        """
        print("\n🎯 VALIDANDO QUALITY GATES...")
        print("=" * 50)

        validation_start = time.time()
        self.violations.clear()

        # Validar por categorías
        category_results = {}

        for category, data in report_data.get("by_category", {}).items():
            category_results[category] = self._validate_category(category, data)

        # Validar métricas generales
        general_results = self._validate_general_metrics(report_data)

        # Validar métricas RAG específicas si existen
        rag_results = self._validate_rag_metrics(report_data)

        validation_time = time.time() - validation_start

        # Calcular resultado final
        all_passed = len(self.violations) == 0
        total_validations = (
            len(category_results) + (1 if general_results else 0) + (1 if rag_results else 0)
        )

        # Generar reporte de calidad
        quality_report = {
            "validation_info": {
                "timestamp": datetime.now().isoformat(),
                "validation_time": round(validation_time, 3),
                "thresholds_file": str(self.thresholds_file),
            },
            "summary": {
                "all_gates_passed": all_passed,
                "total_validations": total_validations,
                "violations_count": len(self.violations),
                "quality_score": self._calculate_quality_score(report_data),
            },
            "category_results": category_results,
            "general_validation": general_results,
            "rag_validation": rag_results,
            "violations": self.violations,
            "recommendations": self._generate_recommendations(),
        }

        self._print_validation_summary(quality_report)
        return quality_report

    def _validate_category(self, category: str, data: Dict) -> Dict:
        """Validar métricas de una categoría específica"""
        category_thresholds = self.thresholds.get(f"{category}_tests", {})
        if not category_thresholds:
            return {"status": "no_thresholds", "validated": False}

        results = {"status": "validated", "validated": True, "checks": {}}

        # Validar tasa de éxito
        if "success_rate" in category_thresholds:
            threshold = category_thresholds["success_rate"]
            actual = data.get("success_rate", 0)
            passed = actual >= threshold

            results["checks"]["success_rate"] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }

            if not passed:
                self.violations.append(
                    {
                        "category": category,
                        "metric": "success_rate",
                        "threshold": threshold,
                        "actual": actual,
                        "severity": "high",
                    }
                )

        # Validar tiempo de ejecución
        if "execution_time" in category_thresholds:
            threshold = category_thresholds["execution_time"]
            actual = data.get("execution_time", 0)
            passed = actual <= threshold

            results["checks"]["execution_time"] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }

            if not passed:
                self.violations.append(
                    {
                        "category": category,
                        "metric": "execution_time",
                        "threshold": threshold,
                        "actual": actual,
                        "severity": "medium",
                    }
                )

        return results

    def _validate_general_metrics(self, report_data: Dict) -> Dict:
        """Validar métricas generales del sistema"""
        general_thresholds = self.thresholds.get("general_quality", {})
        if not general_thresholds:
            return {"status": "no_thresholds", "validated": False}

        results = {"status": "validated", "validated": True, "checks": {}}
        summary = report_data.get("summary", {})

        # Validar tasa de éxito general
        if "overall_success_rate" in general_thresholds:
            threshold = general_thresholds["overall_success_rate"]
            actual = summary.get("success_rate", 0)
            passed = actual >= threshold

            results["checks"]["overall_success_rate"] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }

            if not passed:
                self.violations.append(
                    {
                        "category": "general",
                        "metric": "overall_success_rate",
                        "threshold": threshold,
                        "actual": actual,
                        "severity": "critical",
                    }
                )

        # Validar tiempo total
        if "total_execution_time" in general_thresholds:
            threshold = general_thresholds["total_execution_time"]
            actual = report_data.get("execution_info", {}).get("total_execution_time", 0)
            passed = actual <= threshold

            results["checks"]["total_execution_time"] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }

            if not passed:
                self.violations.append(
                    {
                        "category": "general",
                        "metric": "total_execution_time",
                        "threshold": threshold,
                        "actual": actual,
                        "severity": "low",
                    }
                )

        return results

    def _validate_rag_metrics(self, report_data: Dict) -> Optional[Dict]:
        """Validar métricas específicas de evaluación RAG"""
        # Buscar resultados RAG en los detalles
        rag_data = None
        for result in report_data.get("detailed_results", []):
            if result.get("test_name") == "rag_complete_suite" and result.get("details"):
                # Aquí necesitaríamos acceso a métricas RAG específicas
                # Por ahora, validamos la existencia y éxito del test
                break

        rag_thresholds = self.thresholds.get("rag_evaluation", {})
        if not rag_thresholds:
            return None

        # Validación básica de que el test RAG pasó
        results = {"status": "basic_validation", "validated": True, "checks": {}}

        # Buscar test RAG en los resultados
        rag_test_found = False
        rag_test_passed = False

        for result in report_data.get("detailed_results", []):
            if "rag" in result.get("test_name", "").lower():
                rag_test_found = True
                rag_test_passed = result.get("passed", False)
                break

        results["checks"]["rag_test_execution"] = {
            "threshold": "required",
            "actual": "found" if rag_test_found else "not_found",
            "passed": rag_test_found and rag_test_passed,
        }

        if not (rag_test_found and rag_test_passed):
            self.violations.append(
                {
                    "category": "rag_evaluation",
                    "metric": "rag_test_execution",
                    "threshold": "required",
                    "actual": "failed" if rag_test_found else "not_found",
                    "severity": "high",
                }
            )

        return results

    def _calculate_quality_score(self, report_data: Dict) -> float:
        """Calcular puntuación general de calidad (0-100)"""
        base_score = 100.0

        # Penalizaciones por violaciones
        for violation in self.violations:
            severity = violation.get("severity", "medium")
            if severity == "critical":
                base_score -= 25
            elif severity == "high":
                base_score -= 15
            elif severity == "medium":
                base_score -= 10
            elif severity == "low":
                base_score -= 5

        # Bonus por alta tasa de éxito
        success_rate = report_data.get("summary", {}).get("success_rate", 0)
        if success_rate >= 95:
            base_score += 5
        elif success_rate >= 90:
            base_score += 2

        return max(0.0, min(100.0, base_score))

    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en violaciones"""
        recommendations = []

        if not self.violations:
            recommendations.append("✅ Todos los quality gates pasaron exitosamente")
            recommendations.append("🎯 El sistema cumple con los estándares de calidad establecidos")
            return recommendations

        # Agrupar por severidad
        critical_violations = [v for v in self.violations if v.get("severity") == "critical"]
        high_violations = [v for v in self.violations if v.get("severity") == "high"]
        medium_violations = [v for v in self.violations if v.get("severity") == "medium"]

        if critical_violations:
            recommendations.append(
                "🚨 CRÍTICO: Hay violaciones críticas que deben resolverse inmediatamente"
            )
            for v in critical_violations:
                recommendations.append(
                    f"   • {v['category']}.{v['metric']}: {v['actual']} < {v['threshold']}"
                )

        if high_violations:
            recommendations.append("⚠️  ALTO: Violaciones importantes que requieren atención")
            for v in high_violations:
                recommendations.append(
                    f"   • {v['category']}.{v['metric']}: {v['actual']} vs {v['threshold']}"
                )

        if medium_violations:
            recommendations.append("📋 MEDIO: Mejoras recomendadas para optimizar calidad")

        # Recomendaciones específicas
        if any(v.get("metric") == "success_rate" for v in self.violations):
            recommendations.append("🔧 Revisar tests fallidos y corregir problemas de código")

        if any(v.get("metric") == "execution_time" for v in self.violations):
            recommendations.append("⚡ Optimizar rendimiento de tests lentos")

        if any("rag" in v.get("category", "") for v in self.violations):
            recommendations.append("🤖 Revisar configuración y parámetros del sistema RAG")

        return recommendations

    def _print_validation_summary(self, quality_report: Dict):
        """Imprimir resumen de validación"""
        summary = quality_report["summary"]

        print(f"\n🎯 RESULTADOS QUALITY GATES:")
        print(f"   Estado: {'✅ PASADO' if summary['all_gates_passed'] else '❌ FALLIDO'}")
        print(f"   Puntuación: {summary['quality_score']:.1f}/100")
        print(f"   Violaciones: {summary['violations_count']}")

        if self.violations:
            print(f"\n⚠️  VIOLACIONES DETECTADAS:")
            for v in self.violations[:5]:  # Mostrar solo las primeras 5
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "ℹ️"}
                emoji = severity_emoji.get(v.get("severity", "medium"), "📋")
                print(
                    f"   {emoji} {v['category']}.{v['metric']}: {v['actual']} vs {v['threshold']}"
                )

            if len(self.violations) > 5:
                print(f"   ... y {len(self.violations) - 5} violaciones más")

        print(f"\n📊 RECOMENDACIONES:")
        for rec in quality_report["recommendations"][:3]:  # Mostrar solo las primeras 3
            print(f"   {rec}")


def validate_quality_gates(report_path: Path) -> bool:
    """
    Función de conveniencia para validar quality gates desde un archivo de reporte

    Args:
        report_path: Ruta al archivo JSON del reporte

    Returns:
        True si todos los gates pasan, False en caso contrario
    """
    if not report_path.exists():
        print(f"❌ Archivo de reporte no encontrado: {report_path}")
        return False

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        validator = QualityGateValidator()
        quality_report = validator.validate_report(report_data)

        return quality_report["summary"]["all_gates_passed"]

    except Exception as e:
        print(f"❌ Error validando quality gates: {e}")
        return False


if __name__ == "__main__":
    # Ejemplo de uso
    import sys

    if len(sys.argv) > 1:
        report_file = Path(sys.argv[1])
        result = validate_quality_gates(report_file)
        sys.exit(0 if result else 1)
    else:
        print("Uso: python quality_gates.py <ruta_reporte.json>")
        sys.exit(1)
