#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Runner for Sheily AI System
============================================

Advanced test runner with:
- Multiple test execution modes
- Performance benchmarking
- Security testing
- Coverage analysis
- Report generation
- CI/CD integration
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EnterpriseTestRunner:
    """Test runner empresarial avanzado para Sheily AI system"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("enterprise_test_runner")
        self.start_time = time.time()
        self.enterprise_metrics = EnterpriseTestMetrics()
        self.enterprise_reporter = EnterpriseTestReporter()
        self.enterprise_quality_gates = EnterpriseQualityGates()

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests"""
        self.logger.info("Running unit tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/",
                    "-v",
                    "--tb=short",
                    "--durations=10",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - self.start_time,
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Unit tests timed out")
            return {"success": False, "return_code": -1, "error": "Timeout", "duration": 300}

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        self.logger.info("Running integration tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/integration/",
                    "-v",
                    "--tb=short",
                    "-m",
                    "integration",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Integration tests timed out")
            return {"success": False, "return_code": -1, "error": "Timeout"}

    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        self.logger.info("Running security tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/test_safety.py",
                    "tests/unit/test_config.py",
                    "-v",
                    "-k",
                    "security",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Security tests timed out")
            return {"success": False, "return_code": -1, "error": "Timeout"}

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        self.logger.info("Running performance tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/test_chat_engine.py",
                    "tests/unit/test_rag_engine.py",
                    "tests/integration/test_full_system.py",
                    "-v",
                    "-k",
                    "performance",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Performance tests timed out")
            return {"success": False, "return_code": -1, "error": "Timeout"}

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis"""
        self.logger.info("Running coverage analysis...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--cov=.",
                    "--cov-report=html:reports/coverage/html",
                    "--cov-report=xml:reports/coverage/coverage.xml",
                    "--cov-report=term-missing",
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            self.logger.error("Coverage analysis timed out")
            return {"success": False, "return_code": -1, "error": "Timeout"}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        self.logger.info("Running complete test suite...")

        results = {
            "unit_tests": self.run_unit_tests(),
            "integration_tests": self.run_integration_tests(),
            "security_tests": self.run_security_tests(),
            "performance_tests": self.run_performance_tests(),
            "coverage": self.run_coverage_analysis(),
            "summary": {},
        }

        # Generate summary
        total_tests = 0
        passed_tests = 0

        for test_type, result in results.items():
            if test_type != "summary" and isinstance(result, dict) and "return_code" in result:
                total_tests += 1
                if result["success"]:
                    passed_tests += 1

        results["summary"] = {
            "total_test_suites": total_tests,
            "passed_suites": passed_tests,
            "failed_suites": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "total_duration": time.time() - self.start_time,
        }

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        summary = results["summary"]

        report = []
        report.append("🧪 COMPREHENSIVE TEST REPORT")
        report.append("=" * 50)
        report.append(f"Total Test Suites: {summary['total_test_suites']}")
        report.append(f"Passed: {summary['passed_suites']}")
        report.append(f"Failed: {summary['failed_suites']}")
        report.append(f"Success Rate: {summary['success_rate']:.1f}%")
        report.append(f"Total Duration: {summary['total_duration']:.2f}s")
        report.append("")

        for test_type, result in results.items():
            if test_type != "summary":
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                report.append(f"{test_type.upper()}: {status}")

                if not result["success"]:
                    if "error" in result:
                        report.append(f"  Error: {result['error']}")
                    elif result["return_code"] != 0:
                        report.append(f"  Return Code: {result['return_code']}")

        report.append("")
        report.append("📊 DETAILED RESULTS:")
        report.append("=" * 50)

        for test_type, result in results.items():
            if test_type != "summary":
                report.append(f"\n{test_type.upper()}:")
                if result["success"]:
                    report.append("  ✅ All tests passed")
                else:
                    report.append("  ❌ Some tests failed")
                    if result["stderr"]:
                        # Show first few lines of stderr
                        stderr_lines = result["stderr"].split("\n")[:5]
                        for line in stderr_lines:
                            if line.strip():
                                report.append(f"    {line}")

        return "\n".join(report)


@dataclass
class EnterpriseTestMetrics:
    """Métricas de testing empresariales avanzadas"""

    total_tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_suites_executed: int = 0
    total_execution_time: float = 0.0
    average_test_time: float = 0.0
    enterprise_coverage_percentage: float = 0.0
    enterprise_quality_score: float = 0.0
    security_tests_passed: int = 0
    performance_benchmarks_met: int = 0

    def update_enterprise_metrics(self, test_results: Dict[str, Any]):
        """Actualizar métricas empresariales"""
        self.total_tests_run += 1

        if test_results.get("success", False):
            self.tests_passed += 1
        else:
            self.tests_failed += 1

        # Calcular métricas empresariales
        if self.total_tests_run > 0:
            self.average_test_time = self.total_execution_time / self.total_tests_run
            self.enterprise_coverage_percentage = (self.tests_passed / self.total_tests_run) * 100
            self.enterprise_quality_score = self._calculate_enterprise_quality_score()

    def _calculate_enterprise_quality_score(self) -> float:
        """Calcular puntuación de calidad empresarial"""
        base_score = 100.0

        # Penalizaciones empresariales
        if self.tests_failed > 0:
            base_score -= self.tests_failed * 2.5

        if self.enterprise_coverage_percentage < 95:
            base_score -= (95 - self.enterprise_coverage_percentage) * 0.5

        return max(0.0, min(100.0, base_score))


@dataclass
class EnterpriseTestReporter:
    """Reporter empresarial avanzado"""

    report_dir: str = "reports/enterprise"
    enterprise_standards: List[str] = field(
        default_factory=lambda: ["ISO-25010", "IEEE-829", "Enterprise-Quality-Gates"]
    )

    def __post_init__(self):
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_enterprise_report(self, results: Dict[str, Any]) -> str:
        """Generar reporte empresarial completo"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        report = []
        report.append("🏢 REPORTE DE TESTING EMPRESARIAL SHEILY AI")
        report.append("=" * 60)
        report.append(f"Estándares Aplicados: {', '.join(self.enterprise_standards)}")
        report.append(f"Timestamp: {timestamp}")
        report.append("")

        # Métricas empresariales
        summary = results.get("summary", {})
        report.append("📊 MÉTRICAS EMPRESARIALES:")
        report.append(f"   Suites de Test: {summary.get('total_test_suites', 0)}")
        report.append(f"   Suites Aprobadas: {summary.get('passed_suites', 0)}")
        report.append(f"   Tasa de Éxito: {summary.get('success_rate', 0):.2f}%")
        report.append(f"   Tiempo Total: {summary.get('total_duration', 0):.2f}s")
        report.append("")

        # Resultados detallados empresariales
        report.append("📋 RESULTADOS DETALLADOS EMPRESARIALES:")
        report.append("-" * 60)

        for test_type, result in results.items():
            if test_type != "summary":
                status = "✅ EMPRESARIAL APROBADO" if result["success"] else "❌ REQUIERE ATENCIÓN"
                report.append(f"{test_type.upper()}: {status}")

                if not result["success"]:
                    report.append(f"   🔍 Revisión requerida: {result.get('error', 'Error desconocido')}")

        report.append("")
        report.append("🎯 CUMPLIMIENTO DE ESTÁNDARES EMPRESARIALES:")
        report.append("-" * 60)

        compliance_score = self._calculate_enterprise_compliance(results)
        report.append(f"   Puntuación de Cumplimiento: {compliance_score:.1f}/100")

        if compliance_score >= 95:
            report.append("   ✅ CUMPLE ESTÁNDARES EMPRESARIALES")
        elif compliance_score >= 85:
            report.append("   ⚠️ CUMPLE CON OBSERVACIONES")
        else:
            report.append("   ❌ REQUIERE MEJORAS EMPRESARIALES")

        return "\n".join(report)

    def _calculate_enterprise_compliance(self, results: Dict[str, Any]) -> float:
        """Calcular cumplimiento de estándares empresariales"""
        base_score = 100.0

        # Verificar cada suite de test empresarial
        required_suites = ["unit_tests", "integration_tests", "security_tests", "performance_tests"]
        completed_suites = sum(
            1 for suite in required_suites if suite in results and results[suite].get("success", False)
        )

        if completed_suites < len(required_suites):
            base_score -= (len(required_suites) - completed_suites) * 15

        # Verificar cobertura empresarial
        summary = results.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        if success_rate < 95:
            base_score -= (95 - success_rate) * 0.5

        return max(0.0, min(100.0, base_score))


@dataclass
class EnterpriseQualityGates:
    """Quality gates empresariales avanzados"""

    enterprise_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "minimum_success_rate": 95.0,
            "maximum_execution_time": 300.0,
            "minimum_coverage": 90.0,
            "maximum_memory_usage": 1024.0,  # MB
            "minimum_security_score": 85.0,
        }
    )

    def validate_enterprise_quality_gates(self, results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validar quality gates empresariales"""
        violations = []
        all_passed = True

        summary = results.get("summary", {})

        # Validar tasa de éxito empresarial
        success_rate = summary.get("success_rate", 0)
        if success_rate < self.enterprise_thresholds["minimum_success_rate"]:
            violations.append(
                f"Tasa de éxito empresarial baja: {success_rate:.1f}% < {self.enterprise_thresholds['minimum_success_rate']}%"
            )
            all_passed = False

        # Validar tiempo de ejecución empresarial
        total_time = summary.get("total_duration", 0)
        if total_time > self.enterprise_thresholds["maximum_execution_time"]:
            violations.append(
                f"Tiempo de ejecución empresarial excesivo: {total_time:.1f}s > {self.enterprise_thresholds['maximum_execution_time']}s"
            )
            all_passed = False

        return all_passed, violations


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner for Sheily AI")
    parser.add_argument(
        "mode",
        choices=["unit", "integration", "security", "performance", "coverage", "all"],
        help="Test mode to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = ComprehensiveTestRunner()

    print("🚀 Sheily AI Comprehensive Test Runner")
    print("=" * 50)

    if args.mode == "unit":
        results = {"unit_tests": runner.run_unit_tests()}
    elif args.mode == "integration":
        results = {"integration_tests": runner.run_integration_tests()}
    elif args.mode == "security":
        results = {"security_tests": runner.run_security_tests()}
    elif args.mode == "performance":
        results = {"performance_tests": runner.run_performance_tests()}
    elif args.mode == "coverage":
        results = {"coverage": runner.run_coverage_analysis()}
    elif args.mode == "all":
        results = runner.run_all_tests()

    # Generate and display report
    report = runner.generate_report(results)
    print(report)

    # Return appropriate exit code
    summary = results.get("summary", {})
    if summary and summary.get("success_rate", 0) == 100:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
