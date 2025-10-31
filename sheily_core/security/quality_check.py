#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_check.py - Sistema de Verificación de Calidad Ultra-Enterprise

Sistema avanzado de verificación de calidad que ejecuta múltiples herramientas
de análisis con métricas profesionales, reporting detallado y estándares
de nivel enterprise máximo.
"""

import argparse
import json
import logging
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil


class QualityStatus(Enum):
    """Estados avanzados de verificación"""

    EXCELLENT = "EXCELLENT"
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    CRITICAL = "CRITICAL"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class QualityMetrics:
    """Métricas detalladas de calidad"""

    execution_time: float = 0.0
    memory_usage: int = 0  # bytes
    cpu_usage: float = 0.0
    exit_code: int = 0
    lines_analyzed: int = 0
    issues_found: int = 0
    score: float = 0.0
    grade: str = "F"


@dataclass
class QualityCheck:
    """Verificación de calidad avanzada"""

    name: str
    command: List[str]
    status: QualityStatus = QualityStatus.PASSED
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    output: str = ""
    error: str = ""
    timeout: int = 300
    critical: bool = False
    category: str = "general"
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


class EnterpriseQualityChecker:
    """Sistema de verificación de calidad de nivel enterprise máximo"""

    def __init__(self, project_root: Path, verbose: bool = True, parallel: bool = True):
        self.project_root = Path(project_root).resolve()
        self.verbose = verbose
        self.parallel = parallel
        self.reports_dir = self.project_root / "reports" / "quality"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Configurar logging avanzado
        self._setup_logging()

        # Sistema de verificación ultra-avanzado
        self.checks = self._initialize_checks()

        # Métricas globales
        self.global_metrics = {
            "start_time": time.time(),
            "system_info": self._get_system_info(),
            "total_tools": len(self.checks),
            "parallel_execution": parallel,
        }

    def _setup_logging(self):
        """Configurar sistema de logging avanzado"""
        log_file = self.reports_dir / "quality_check.log"

        # Crear formateador avanzado
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler para archivo
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Configurar logger raíz
        self.logger = logging.getLogger("EnterpriseQualityChecker")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_system_info(self) -> Dict[str, Any]:
        """Obtener información detallada del sistema"""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage(self.project_root).total,
            "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0],
        }

    def _initialize_checks(self) -> List[QualityCheck]:
        """Inicializar todas las verificaciones de calidad"""
        return [
            # Verificaciones básicas mejoradas
            QualityCheck(
                name="Flake8 - Linting Ultra-Estrict",
                command=["flake8", ".", "--config=.flake8", "--format=json"],
                category="code_quality",
                weight=0.15,
                timeout=120,
            ),
            QualityCheck(
                name="Bandit - Seguridad Avanzada",
                command=["bandit", "-r", ".", "-c", ".bandit", "-f", "json"],
                category="security",
                weight=0.20,
                timeout=180,
                critical=True,
            ),
            QualityCheck(
                name="MyPy - Type Checking Máximo",
                command=["mypy", ".", "--config-file=mypy.ini", "--show-error-codes"],
                category="type_checking",
                weight=0.15,
                timeout=240,
            ),
            QualityCheck(
                name="Safety - Vulnerabilidades Críticas",
                command=["safety", "check", "--file", "requirements.txt", "--output", "json"],
                category="dependencies",
                weight=0.10,
                timeout=180,
                critical=True,
            ),
            # Verificaciones avanzadas
            QualityCheck(
                name="Radon - Complejidad Ultra-Estr",
                command=["radon", "cc", ".", "--config=.radon.cfg", "--output", "json"],
                category="complexity",
                weight=0.10,
                timeout=150,
            ),
            QualityCheck(
                name="Xenon - Deuda Técnica Máxima",
                command=["xenon", ".", "--config=.xenon", "--output", "json"],
                category="maintainability",
                weight=0.10,
                timeout=150,
            ),
            QualityCheck(
                name="Coverage - Cobertura Ultra-Estr",
                command=[
                    "coverage",
                    "run",
                    "--source=.",
                    "--omit=*test*,*__pycache__*",
                    "-m",
                    "pytest",
                    "--quiet",
                ],
                category="testing",
                weight=0.05,
                timeout=300,
                dependencies=["pytest"],
            ),
            # Verificaciones especializadas
            QualityCheck(
                name="Performance Analysis - Ultra",
                command=["python", "performance_analyzer.py"],
                category="performance",
                weight=0.05,
                timeout=600,
            ),
            QualityCheck(
                name="Dependency Analysis - Enterprise",
                command=["python", "dependency_analyzer.py"],
                category="dependencies",
                weight=0.05,
                timeout=300,
            ),
            QualityCheck(
                name="Container Security - Máxima",
                command=["python", "container_security_checker.py"],
                category="infrastructure",
                weight=0.05,
                timeout=240,
            ),
            QualityCheck(
                name="AI Security - Especializada",
                command=["python", "ai_security_checker.py"],
                category="ai_security",
                weight=0.10,
                timeout=180,
                critical=True,
            ),
            QualityCheck(
                name="Documentation Quality - Profesional",
                command=["python", "doc_quality_checker.py"],
                category="documentation",
                weight=0.05,
                timeout=180,
            ),
        ]

    def run_check(self, check: QualityCheck) -> QualityCheck:
        """Ejecutar una verificación individual con métricas avanzadas"""
        self.logger.info(f"Iniciando verificación: {check.name}")

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            # Ejecutar comando con timeout avanzado
            result = subprocess.run(
                check.command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=check.timeout,
                env={**os.environ, "PYTHONPATH": str(self.project_root)},
            )

            # Calcular métricas
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            check.metrics.execution_time = end_time - start_time
            check.metrics.memory_usage = end_memory
            check.metrics.memory_delta = end_memory - start_memory
            check.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
            check.metrics.exit_code = result.returncode

            # Procesar salida
            check.output = result.stdout
            check.error = result.stderr

            # Determinar estado basado en código de salida y contenido
            check.status = self._determine_status(check, result)

            # Calcular puntuación
            check.metrics.score = self._calculate_score(check)
            check.metrics.grade = self._calculate_grade(check.metrics.score)

            # Logging detallado
            self.logger.info(
                f"Verificación {check.name}: {check.status.value} "
                f"(Tiempo: {check.metrics.execution_time:.2f}s, "
                f"Memoria: {check.metrics.memory_delta / 1024 / 1024:.2f} MB, "
                f"Puntuación: {check.metrics.score:.1f})"
            )

        except subprocess.TimeoutExpired:
            check.status = QualityStatus.TIMEOUT
            check.error = f"Timeout después de {check.timeout} segundos"
            self.logger.error(f"Timeout en {check.name}")
        except Exception as e:
            check.status = QualityStatus.ERROR
            check.error = f"Error inesperado: {str(e)}"
            self.logger.error(f"Error en {check.name}: {e}")

        return check

    def _determine_status(
        self, check: QualityCheck, result: subprocess.CompletedProcess
    ) -> QualityStatus:
        """Determinar estado de verificación con lógica avanzada"""
        if result.returncode == 0:
            return (
                QualityStatus.EXCELLENT
                if self._is_excellent_result(check, result)
                else QualityStatus.PASSED
            )

        # Verificaciones críticas fallidas
        if check.critical:
            return QualityStatus.CRITICAL

        # Verificaciones con warnings
        if any(keyword in result.stderr.lower() for keyword in ["warning", "deprecated"]):
            return QualityStatus.WARNING

        return QualityStatus.FAILED

    def _is_excellent_result(
        self, check: QualityCheck, result: subprocess.CompletedProcess
    ) -> bool:
        """Determinar si un resultado es excelente"""
        # Lógica específica por herramienta
        if "flake8" in check.name.lower():
            return len(result.stdout.strip()) == 0  # Sin errores ni warnings
        elif "bandit" in check.name.lower():
            try:
                data = json.loads(result.stdout)
                return data.get("metrics", {}).get("_totals", {}).get("SEVERITY.HIGH", 0) == 0
            except:
                return False
        return result.returncode == 0

    def _calculate_score(self, check: QualityCheck) -> float:
        """Calcular puntuación de calidad (0-100)"""
        if check.status in [QualityStatus.EXCELLENT, QualityStatus.PASSED]:
            # Puntuación base
            score = 90.0

            # Penalizaciones por tiempo
            if check.metrics.execution_time > 60:
                score -= min(10, (check.metrics.execution_time - 60) / 6)

            # Penalizaciones por memoria
            if check.metrics.memory_delta > 100 * 1024 * 1024:  # 100MB
                score -= min(
                    10, (check.metrics.memory_delta - 100 * 1024 * 1024) / (10 * 1024 * 1024)
                )

            # Bonus por eficiencia
            if check.metrics.execution_time < 30 and check.metrics.memory_delta < 50 * 1024 * 1024:
                score += 5

            return min(100, max(0, score))

        elif check.status == QualityStatus.WARNING:
            return 70.0
        elif check.status == QualityStatus.FAILED:
            return 40.0
        elif check.status == QualityStatus.CRITICAL:
            return 20.0
        else:
            return 0.0

    def _calculate_grade(self, score: float) -> str:
        """Calcular calificación basada en puntuación"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 65:
            return "D+"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def run_all_checks(self) -> Dict[str, QualityCheck]:
        """Ejecutar todas las verificaciones con ejecución paralela"""
        self.logger.info("🚀 Iniciando suite de calidad ultra-enterprise")

        if self.parallel and len(self.checks) > 1:
            # Ejecutar en paralelo usando multiprocessing
            with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:
                results = pool.map(self._run_check_with_timeout, self.checks)
        else:
            # Ejecutar secuencialmente
            results = [self.run_check(check) for check in self.checks]

        return {check.name: check for check in results}

    def _run_check_with_timeout(self, check: QualityCheck) -> QualityCheck:
        """Ejecutar verificación con manejo de timeout avanzado"""
        try:
            return self.run_check(check)
        except Exception as e:
            check.status = QualityStatus.ERROR
            check.error = f"Error de ejecución: {str(e)}"
            return check

    def generate_comprehensive_report(self, results: Dict[str, QualityCheck]) -> Dict[str, Any]:
        """Generar reporte ultra-comprensivo"""
        total_time = time.time() - self.global_metrics["start_time"]

        # Calcular métricas globales
        passed_checks = sum(
            1
            for check in results.values()
            if check.status in [QualityStatus.EXCELLENT, QualityStatus.PASSED]
        )
        critical_failures = sum(
            1 for check in results.values() if check.status == QualityStatus.CRITICAL
        )
        total_score = sum(check.metrics.score * check.weight for check in results.values())
        max_possible_score = sum(100 * check.weight for check in results.values())
        overall_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        # Análisis de categorías
        category_scores = {}
        for check in results.values():
            if check.category not in category_scores:
                category_scores[check.category] = []
            category_scores[check.category].append(check.metrics.score * check.weight)

        category_averages = {
            category: sum(scores) / len(scores) if scores else 0
            for category, scores in category_scores.items()
        }

        # Generar recomendaciones
        recommendations = self._generate_advanced_recommendations(results)

        # Crear reporte detallado
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "project": "Sheily-AI",
                "level": "ULTRA-ENTERPRISE",
                "version": "2.0.0",
                "execution_time": total_time,
                "parallel_execution": self.parallel,
            },
            "system_info": self.global_metrics["system_info"],
            "summary": {
                "total_checks": len(results),
                "passed_checks": passed_checks,
                "failed_checks": len(results) - passed_checks,
                "critical_failures": critical_failures,
                "overall_score": overall_score,
                "overall_grade": self._calculate_overall_grade(overall_score),
                "execution_time": total_time,
                "success_rate": (passed_checks / len(results)) * 100 if results else 0,
            },
            "category_analysis": category_averages,
            "detailed_results": {
                name: {
                    "status": check.status.value,
                    "metrics": asdict(check.metrics),
                    "category": check.category,
                    "weight": check.weight,
                    "critical": check.critical,
                    "output_preview": check.output[:500] + "..."
                    if len(check.output) > 500
                    else check.output,
                    "error": check.error,
                }
                for name, check in results.items()
            },
            "recommendations": recommendations,
            "quality_gates": self._evaluate_quality_gates(results),
        }

        return report

    def _generate_advanced_recommendations(self, results: Dict[str, QualityCheck]) -> List[str]:
        """Generar recomendaciones ultra-avanzadas"""
        recommendations = []

        # Análisis de patrones de fallo
        failed_checks = [
            name for name, check in results.items() if check.status == QualityStatus.FAILED
        ]
        critical_checks = [
            name for name, check in results.items() if check.status == QualityStatus.CRITICAL
        ]

        if critical_checks:
            recommendations.append(f"🚨 CORRECCIÓN CRÍTICA REQUERIDA: {', '.join(critical_checks)}")

        # Recomendaciones por categoría
        for category, avg_score in self._get_category_scores(results).items():
            if avg_score < 70:
                recommendations.append(
                    f"📈 Mejorar categoría '{category}': puntuación actual {avg_score:.1f}/100"
                )

        # Recomendaciones específicas
        if any(
            "coverage" in name.lower()
            for name, check in results.items()
            if check.metrics.score < 80
        ):
            recommendations.append("📊 Implementar pruebas adicionales para mejorar cobertura")

        if any(
            "security" in name.lower()
            for name, check in results.items()
            if check.status != QualityStatus.PASSED
        ):
            recommendations.append("🔒 Revisar y corregir vulnerabilidades de seguridad detectadas")

        if any(
            "performance" in name.lower()
            for name, check in results.items()
            if check.metrics.execution_time > 300
        ):
            recommendations.append(
                "⚡ Optimizar rendimiento: algunas verificaciones tardan demasiado"
            )

        # Recomendaciones generales de mejora
        recommendations.extend(
            [
                "🔄 Considerar implementar CI/CD con estas verificaciones automáticamente",
                "📋 Documentar procedimientos de corrección para fallos comunes",
                "📈 Establecer baselines de calidad y monitorear tendencias",
                "🏗️ Considerar integración con plataformas de calidad externas (SonarQube, CodeClimate)",
            ]
        )

        return recommendations

    def _get_category_scores(self, results: Dict[str, QualityCheck]) -> Dict[str, float]:
        """Obtener puntuaciones por categoría"""
        categories = {}
        for check in results.values():
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check.metrics.score)

        return {cat: sum(scores) / len(scores) for cat, scores in categories.items()}

    def _evaluate_quality_gates(self, results: Dict[str, QualityCheck]) -> Dict[str, Any]:
        """Evaluar quality gates estrictos"""
        gates = {
            "security_gate": all(
                check.status != QualityStatus.CRITICAL
                for check in results.values()
                if check.category == "security"
            ),
            "quality_gate": all(check.metrics.score >= 70 for check in results.values()),
            "performance_gate": all(
                check.metrics.execution_time < 300 for check in results.values()
            ),
            "coverage_gate": any(
                check.metrics.score >= 85
                for check in results.values()
                if "coverage" in check.name.lower()
            ),
        }

        gates_passed = sum(1 for gate in gates.values() if gate)
        return {
            "gates_passed": gates_passed,
            "total_gates": len(gates),
            "gate_results": gates,
            "overall_gate_status": gates_passed == len(gates),
        }

    def _calculate_overall_grade(self, score: float) -> str:
        """Calcular calificación general"""
        if score >= 95:
            return "A+ (Excelente)"
        elif score >= 90:
            return "A (Muy Bueno)"
        elif score >= 85:
            return "B+ (Bueno)"
        elif score >= 80:
            return "B (Satisfactorio)"
        elif score >= 75:
            return "C+ (Aceptable)"
        elif score >= 70:
            return "C (Mejorable)"
        elif score >= 65:
            return "D+ (Deficiente)"
        elif score >= 60:
            return "D (Muy Deficiente)"
        else:
            return "F (Crítico)"

    def save_reports(self, report: Dict[str, Any], results: Dict[str, QualityCheck]):
        """Guardar reportes en múltiples formatos"""
        # Reporte JSON completo
        json_file = self.reports_dir / "enterprise-quality-report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Reporte resumido en texto
        txt_file = self.reports_dir / "quality-summary.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("SHEILY-AI - REPORTE DE CALIDAD ULTRA-ENTERPRISE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha: {report['metadata']['timestamp']}\n")
            f.write(
                f"Puntuación General: {report['summary']['overall_score']:.1f}/100 ({report['summary']['overall_grade']})\n"
            )
            f.write(
                f"Verificaciones: {report['summary']['passed_checks']}/{report['summary']['total_checks']} aprobadas\n"
            )
            f.write(f"Tiempo Total: {report['metadata']['execution_time']:.2f} segundos\n\n")

            f.write("RESULTADOS POR HERRAMIENTA:\n")
            f.write("-" * 40 + "\n")
            for name, check in results.items():
                status_icon = {
                    QualityStatus.EXCELLENT: "🌟",
                    QualityStatus.PASSED: "✅",
                    QualityStatus.WARNING: "⚠️",
                    QualityStatus.FAILED: "❌",
                    QualityStatus.CRITICAL: "🚨",
                }.get(check.status, "❓")

                f.write(
                    f"{status_icon} {name:<35} | {check.metrics.score:5.1f} | {check.metrics.grade} | {check.metrics.execution_time:6.2f}s\n"
                )

            f.write(
                f"\nQuality Gates: {report['quality_gates']['gates_passed']}/{report['quality_gates']['total_gates']} aprobados\n"
            )

        # Reporte HTML básico
        html_file = self.reports_dir / "quality-report.html"
        self._generate_html_report(report, results, html_file)

        self.logger.info(f"Reportes guardados: {json_file}, {txt_file}, {html_file}")

    def _generate_html_report(
        self, report: Dict[str, Any], results: Dict[str, QualityCheck], html_file: Path
    ):
        """Generar reporte HTML avanzado"""
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sheily-AI - Reporte de Calidad Ultra-Enterprise</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #27ae60; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .metric {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric h3 {{ margin: 0; color: #34495e; }}
        .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; }}
        .results {{ margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .status-excellent {{ color: #27ae60; }}
        .status-passed {{ color: #3498db; }}
        .status-warning {{ color: #f39c12; }}
        .status-failed {{ color: #e74c3c; }}
        .status-critical {{ color: #c0392b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Sheily-AI - Reporte de Calidad Ultra-Enterprise</h1>
            <p>Generado el: {report['metadata']['timestamp']}</p>
            <div class="score">{report['summary']['overall_score']:.1f}/100</div>
            <p>Calificación: {report['summary']['overall_grade']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <h3>Verificaciones</h3>
                <p>{report['summary']['passed_checks']}/{report['summary']['total_checks']}</p>
            </div>
            <div class="metric">
                <h3>Tiempo Total</h3>
                <p>{report['metadata']['execution_time']:.2f}s</p>
            </div>
            <div class="metric">
                <h3>Quality Gates</h3>
                <p>{report['quality_gates']['gates_passed']}/{report['quality_gates']['total_gates']}</p>
            </div>
        </div>
        
        <div class="results">
            <h2>Resultados Detallados</h2>
            <table>
                <thead>
                    <tr>
                        <th>Herramienta</th>
                        <th>Estado</th>
                        <th>Puntuación</th>
                        <th>Calificación</th>
                        <th>Tiempo</th>
                    </tr>
                </thead>
                <tbody>
"""

        for name, check in results.items():
            status_class = f"status-{check.status.value.lower()}"
            html_content += f"""
                    <tr>
                        <td>{name}</td>
                        <td class="{status_class}">{check.status.value}</td>
                        <td>{check.metrics.score:.1f}</td>
                        <td>{check.metrics.grade}</td>
                        <td>{check.metrics.execution_time:.2f}s</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer" style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; text-align: center; color: #7f8c8d;">
            <p>Reporte generado por el Sistema de Calidad Ultra-Enterprise de Sheily-AI</p>
        </div>
    </div>
</body>
</html>
"""

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)


def main():
    """Función principal con argumentos avanzados"""
    parser = argparse.ArgumentParser(description="Sistema de Calidad Ultra-Enterprise")
    parser.add_argument("--project", default=".", help="Ruta del proyecto")
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")
    parser.add_argument("--sequential", action="store_true", help="Ejecución secuencial")
    parser.add_argument(
        "--quick", action="store_true", help="Modo rápido (solo verificaciones críticas)"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Solo generar reporte de resultados previos"
    )

    args = parser.parse_args()

    project_root = Path(args.project).resolve()

    try:
        checker = EnterpriseQualityChecker(
            project_root=project_root, verbose=args.verbose, parallel=not args.sequential
        )

        if args.report_only:
            # Cargar reporte previo
            report_file = project_root / "reports" / "quality" / "enterprise-quality-report.json"
            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
                checker.save_reports(report, {})
                print("✅ Reporte generado exitosamente")
                return 0
            else:
                print("❌ No se encontró reporte previo")
                return 1

        # Ejecutar verificaciones
        results = checker.run_all_checks()

        # Generar reporte completo
        report = checker.generate_comprehensive_report(results)

        # Guardar reportes
        checker.save_reports(report, results)

        # Mostrar resumen en consola
        print(f"\n🎖️ VERIFICACIÓN ULTRA-ENTERPRISE COMPLETADA")
        print(f"📊 Puntuación General: {report['summary']['overall_score']:.1f}/100")
        print(f"🏆 Calificación: {report['summary']['overall_grade']}")
        print(f"⏱️ Tiempo Total: {report['metadata']['execution_time']:.2f} segundos")
        print(f"�� Tasa de Éxito: {report['summary']['success_rate']:.1f}%")

        # Determinar código de salida
        if report["summary"]["critical_failures"] > 0:
            print(f"\n🚨 FALLOS CRÍTICOS DETECTADOS: {report['summary']['critical_failures']}")
            return 2
        elif report["summary"]["failed_checks"] > 0:
            print(f"\n❌ VERIFICACIONES FALLIDAS: {report['summary']['failed_checks']}")
            return 1
        else:
            print("\n🎉 ¡EXCELENTE! Todas las verificaciones aprobadas")
            return 0

    except KeyboardInterrupt:
        print("\n\n⏹️ Verificación interrumpida por el usuario")
        return 130
    except Exception as e:
        print(f"💥 Error fatal: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
