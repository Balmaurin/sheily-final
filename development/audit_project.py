#!/usr/bin/env python3
# ==============================================================================
# AUDITORÍA COMPLETA DEL PROYECTO SHEILY AI
# ==============================================================================
# Script profesional de auditoría que analiza todos los aspectos del proyecto

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil


@dataclass
class AuditResult:
    """Resultado de una auditoría específica"""

    component: str
    status: str  # "PASS", "WARN", "FAIL", "INFO"
    score: float  # 0-100
    findings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class ProjectAudit:
    """Auditoría completa del proyecto"""

    timestamp: str
    project_info: Dict[str, Any]
    overall_score: float
    component_audits: List[AuditResult]
    summary: Dict[str, Any]


class SheilyAuditor:
    """Auditor completo del proyecto Sheily AI"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.audit_results: List[AuditResult] = []
        self.start_time = datetime.now()

    def log_audit(
        self,
        component: str,
        status: str,
        score: float,
        findings: List[str],
        recommendations: List[str],
        **metadata,
    ):
        """Registrar resultado de auditoría"""
        result = AuditResult(
            component=component,
            status=status,
            score=score,
            findings=findings,
            recommendations=recommendations,
            metadata=metadata,
        )
        self.audit_results.append(result)
        return result

    def audit_project_structure(self) -> AuditResult:
        """Auditar estructura general del proyecto"""
        findings = []
        recommendations = []

        # Verificar archivos críticos
        critical_files = [
            "requirements.txt",
            "README.md",
            ".gitignore",
            "Dockerfile",
            "docker-compose.yml",
        ]

        missing_files = []
        for file in critical_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)

        if missing_files:
            findings.append(f"Archivos críticos faltantes: {', '.join(missing_files)}")
            recommendations.append("Crear archivos críticos faltantes para proyecto profesional")
            return self.log_audit(
                "Estructura del Proyecto", "FAIL", 30.0, findings, recommendations
            )

        # Verificar estructura de directorios
        expected_dirs = ["audit_2024", "models", "sheily_core", "data", "logs", "reports", "docs"]

        missing_dirs = []
        for dir_name in expected_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            findings.append(f"Directorios esperados faltantes: {', '.join(missing_dirs)}")

        # Calcular puntuación
        total_checks = len(critical_files) + len(expected_dirs)
        passed_checks = total_checks - len(missing_files) - len(missing_dirs)
        score = (passed_checks / total_checks) * 100

        if score >= 90:
            status = "PASS"
            findings.append("Estructura del proyecto excelente")
        elif score >= 70:
            status = "WARN"
            recommendations.append("Completar estructura con directorios faltantes")
        else:
            status = "FAIL"
            recommendations.append("Reestructurar proyecto siguiendo mejores prácticas")

        return self.log_audit("Estructura del Proyecto", status, score, findings, recommendations)

    def audit_dependencies(self) -> AuditResult:
        """Auditar dependencias del proyecto"""
        findings = []
        recommendations = []

        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return self.log_audit(
                "Dependencias",
                "FAIL",
                0.0,
                ["Archivo requirements.txt no encontrado"],
                ["Crear archivo requirements.txt consolidado"],
            )

        try:
            with open(req_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("#")
            ]

            # Verificar dependencias críticas
            critical_deps = ["torch", "transformers", "fastapi", "pytest"]
            missing_critical = []

            for dep in critical_deps:
                found = False
                for line in lines:
                    if line.lower().startswith(dep.lower()):
                        found = True
                        break
                if not found:
                    missing_critical.append(dep)

            if missing_critical:
                findings.append(f"Dependencias críticas faltantes: {', '.join(missing_critical)}")

            # Verificar formato de versiones
            unversioned = [line for line in lines if not any(char in line for char in ">=<=")]
            if unversioned:
                findings.append(f"Dependencias sin versión: {len(unversioned)}")

            # Calcular puntuación
            total_deps = len(lines)
            issues = len(missing_critical) + len(unversioned)

            if issues == 0:
                score = 100.0
                status = "PASS"
                findings.append(f"Dependencias bien organizadas: {total_deps} paquetes")
            elif issues <= 2:
                score = 80.0
                status = "WARN"
                recommendations.append("Agregar versiones específicas a dependencias")
            else:
                score = 50.0
                status = "FAIL"
                recommendations.append("Revisar y completar archivo requirements.txt")

        except Exception as e:
            return self.log_audit(
                "Dependencias",
                "FAIL",
                0.0,
                [f"Error al leer requirements.txt: {e}"],
                ["Verificar formato del archivo requirements.txt"],
            )

        return self.log_audit(
            "Dependencias",
            status,
            score,
            findings,
            recommendations,
            total_dependencies=total_deps,
            issues_found=issues,
        )

    def audit_docker_configuration(self) -> AuditResult:
        """Auditar configuración Docker"""
        findings = []
        recommendations = []

        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"

        # Verificar Dockerfile
        if not dockerfile.exists():
            return self.log_audit(
                "Docker",
                "FAIL",
                0.0,
                ["Archivo Dockerfile no encontrado"],
                ["Crear Dockerfile profesional"],
            )

        try:
            with open(dockerfile, "r", encoding="utf-8") as f:
                dockerfile_content = f.read()

            # Verificaciones básicas
            if "FROM python:" not in dockerfile_content:
                findings.append("Dockerfile no usa imagen Python oficial")

            if "USER " not in dockerfile_content:
                findings.append("No se configura usuario no-root")

            if "VOLUME" not in dockerfile_content:
                findings.append("No se configuran volúmenes persistentes")

            if "HEALTHCHECK" not in dockerfile_content:
                findings.append("No se configura health check")

            # Verificar tamaño
            lines = len(dockerfile_content.split("\n"))
            if lines < 20:
                findings.append("Dockerfile muy básico, podría necesitar mejoras")

        except Exception as e:
            return self.log_audit(
                "Docker",
                "FAIL",
                0.0,
                [f"Error al leer Dockerfile: {e}"],
                ["Verificar formato del Dockerfile"],
            )

        # Verificar docker-compose
        if docker_compose.exists():
            try:
                with open(docker_compose, "r", encoding="utf-8") as f:
                    compose_content = f.read()

                if "services:" not in compose_content:
                    findings.append("docker-compose.yml mal formateado")

                if "version:" not in compose_content:
                    findings.append("docker-compose sin versión especificada")

                findings.append("Docker Compose configurado correctamente")

            except Exception as e:
                findings.append(f"Error en docker-compose.yml: {e}")

        # Calcular puntuación
        issues = len(
            [f for f in findings if "Error" in f or "mal formateado" in f or "no encontrado" in f]
        )

        if issues == 0:
            score = 100.0
            status = "PASS"
            recommendations.append("Configuración Docker excelente")
        elif issues <= 2:
            score = 75.0
            status = "WARN"
            recommendations.append("Mejorar configuración Docker con mejores prácticas")
        else:
            score = 40.0
            status = "FAIL"
            recommendations.append("Reconfigurar Docker completamente")

        return self.log_audit("Docker", status, score, findings, recommendations)

    def audit_security(self) -> AuditResult:
        """Auditar aspectos de seguridad"""
        findings = []
        recommendations = []

        # Verificar .gitignore
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            with open(gitignore, "r", encoding="utf-8") as f:
                gitignore_content = f.read()

            if "__pycache__" not in gitignore_content:
                findings.append(".gitignore no excluye __pycache__")

            if ".env" not in gitignore_content:
                findings.append(".gitignore no excluye archivos .env")

            if "node_modules" not in gitignore_content:
                findings.append(".gitignore no excluye node_modules")

        # Verificar archivos sensibles
        sensitive_files = [".env", "config.json", "secrets.json", ".aws/credentials"]

        found_sensitive = []
        for file in sensitive_files:
            if (self.project_root / file).exists():
                found_sensitive.append(file)

        if found_sensitive:
            findings.append(f"Archivos sensibles encontrados: {', '.join(found_sensitive)}")

        # Verificar permisos de archivos
        try:
            # Verificar si hay archivos con permisos demasiado abiertos
            result = subprocess.run(
                ["find", str(self.project_root), "-type", "f", "-perm", "/o+w"],
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                findings.append("Algunos archivos tienen permisos de escritura para otros")
        except Exception:
            pass

        # Calcular puntuación
        issues = len([f for f in findings if "no excluye" in f or "sensibles" in f])

        if issues == 0:
            score = 100.0
            status = "PASS"
            findings.append("Configuración de seguridad buena")
        elif issues <= 2:
            score = 70.0
            status = "WARN"
            recommendations.append("Mejorar configuración de seguridad")
        else:
            score = 30.0
            status = "FAIL"
            recommendations.append("Revisar seguridad del proyecto completamente")

        return self.log_audit("Seguridad", status, score, findings, recommendations)

    def audit_performance(self) -> AuditResult:
        """Auditar aspectos de rendimiento"""
        findings = []
        recommendations = []

        # Verificar tamaño del proyecto
        try:
            result = subprocess.run(
                ["du", "-sh", str(self.project_root)], capture_output=True, text=True
            )
            project_size = result.stdout.strip()
            findings.append(f"Tamaño total del proyecto: {project_size}")
        except Exception:
            project_size = "No calculable"

        # Verificar archivos grandes
        large_files = []
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
                large_files.append(
                    f"{file_path.relative_to(self.project_root)} ({file_path.stat().st_size / (1024*1024):.1f}MB)"
                )

        if large_files:
            findings.append(f"Archivos grandes encontrados: {len(large_files)}")

        # Verificar estructura de módulos Python
        python_files = list(self.project_root.rglob("*.py"))
        modules_without_init = []

        for py_file in python_files:
            py_dir = py_file.parent
            init_file = py_dir / "__init__.py"
            if py_dir != self.project_root and not init_file.exists():
                modules_without_init.append(str(py_dir.relative_to(self.project_root)))

        if modules_without_init:
            findings.append(f"Módulos Python sin __init__.py: {len(modules_without_init)}")

        # Calcular puntuación
        issues = len(large_files) + len(modules_without_init)

        if issues == 0:
            score = 100.0
            status = "PASS"
            findings.append("Rendimiento y estructura excelentes")
        elif issues <= 3:
            score = 80.0
            status = "WARN"
            recommendations.append("Optimizar algunos aspectos de rendimiento")
        else:
            score = 60.0
            status = "FAIL"
            recommendations.append("Revisar rendimiento y estructura del proyecto")

        return self.log_audit(
            "Rendimiento",
            status,
            score,
            findings,
            recommendations,
            project_size=project_size,
            large_files=len(large_files),
        )

    def audit_documentation(self) -> AuditResult:
        """Auditar documentación del proyecto"""
        findings = []
        recommendations = []

        # Verificar archivos de documentación
        docs_structure = [
            "README.md",
            "docs/",
            "DOCKER_DEPLOYMENT.md",
            "ENVIRONMENT_CONSOLIDATION.md",
            "REQUIREMENTS_CONSOLIDATION.md",
        ]

        missing_docs = []
        for doc in docs_structure:
            doc_path = self.project_root / doc
            if not doc_path.exists():
                missing_docs.append(doc)

        if missing_docs:
            findings.append(f"Documentación faltante: {', '.join(missing_docs)}")

        # Verificar README principal
        readme = self.project_root / "README.md"
        if readme.exists():
            with open(readme, "r", encoding="utf-8") as f:
                readme_content = f.read()

            if len(readme_content) < 100:
                findings.append("README muy corto, podría necesitar más detalles")

            if "instalación" not in readme_content.lower():
                findings.append("README no incluye sección de instalación")

            if "uso" not in readme_content.lower():
                findings.append("README no incluye sección de uso")

        # Calcular puntuación
        issues = len(missing_docs) + sum(1 for f in findings if "corto" in f or "no incluye" in f)

        if issues == 0:
            score = 100.0
            status = "PASS"
            findings.append("Documentación completa y profesional")
        elif issues <= 2:
            score = 75.0
            status = "WARN"
            recommendations.append("Completar documentación faltante")
        else:
            score = 40.0
            status = "FAIL"
            recommendations.append("Crear documentación completa del proyecto")

        return self.log_audit("Documentación", status, score, findings, recommendations)

    def audit_code_quality(self) -> AuditResult:
        """Auditar calidad del código"""
        findings = []
        recommendations = []

        # Contar archivos Python
        python_files = list(self.project_root.rglob("*.py"))
        total_python_files = len(python_files)

        findings.append(f"Total de archivos Python: {total_python_files}")

        # Verificar módulos con __init__.py
        modules_with_init = 0
        for py_file in python_files:
            py_dir = py_file.parent
            if (py_dir / "__init__.py").exists():
                modules_with_init += 1

        init_percentage = (
            (modules_with_init / total_python_files * 100) if total_python_files > 0 else 0
        )
        findings.append(f"Módulos con __init__.py: {init_percentage:.1f}%")

        # Verificar archivos muy grandes
        large_python_files = []
        for py_file in python_files:
            if py_file.stat().st_size > 10 * 1024:  # 10KB
                large_python_files.append(py_file.name)

        if large_python_files:
            findings.append(f"Archivos Python grandes (>10KB): {len(large_python_files)}")

        # Calcular puntuación
        issues = len(large_python_files)

        if issues == 0 and init_percentage >= 80:
            score = 100.0
            status = "PASS"
            findings.append("Calidad de código excelente")
        elif issues <= 5 and init_percentage >= 60:
            score = 80.0
            status = "WARN"
            recommendations.append("Mejorar organización de módulos grandes")
        else:
            score = 60.0
            status = "FAIL"
            recommendations.append("Reorganizar código y mejorar estructura")

        return self.log_audit(
            "Calidad de Código",
            status,
            score,
            findings,
            recommendations,
            total_python_files=total_python_files,
            init_coverage=init_percentage,
        )

    def run_complete_audit(self) -> ProjectAudit:
        """Ejecutar auditoría completa del proyecto"""
        print("🔍 INICIANDO AUDITORÍA COMPLETA DEL PROYECTO SHEILY AI")
        print("=" * 70)

        # Ejecutar todas las auditorías
        audits = [
            self.audit_project_structure(),
            self.audit_dependencies(),
            self.audit_docker_configuration(),
            self.audit_security(),
            self.audit_performance(),
            self.audit_documentation(),
            self.audit_code_quality(),
        ]

        # Calcular puntuación general
        total_score = sum(audit.score for audit in audits)
        overall_score = total_score / len(audits)

        # Información del proyecto
        project_info = {
            "name": "Sheily AI",
            "root_path": str(self.project_root),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "total_files": len(list(self.project_root.rglob("*"))),
            "total_size_mb": sum(
                f.stat().st_size for f in self.project_root.rglob("*") if f.is_file()
            )
            / (1024 * 1024),
        }

        # Generar resumen
        status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
        for audit in audits:
            status_counts[audit.status] += 1

        summary = {
            "total_components": len(audits),
            "status_distribution": status_counts,
            "audit_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "recommendations_count": sum(len(audit.recommendations) for audit in audits),
        }

        return ProjectAudit(
            timestamp=datetime.now().isoformat(),
            project_info=project_info,
            overall_score=overall_score,
            component_audits=audits,
            summary=summary,
        )

    def generate_report(self, audit: ProjectAudit) -> str:
        """Generar reporte de auditoría en formato legible"""
        report_lines = []

        # Encabezado
        report_lines.append("🔍 REPORTE DE AUDITORÍA COMPLETA - PROYECTO SHEILY AI")
        report_lines.append("=" * 70)
        report_lines.append(f"Fecha de auditoría: {audit.timestamp}")
        report_lines.append(f"Puntuación general: {audit.overall_score:.1f}/100")
        report_lines.append("")

        # Información del proyecto
        report_lines.append("📋 INFORMACIÓN DEL PROYECTO:")
        report_lines.append("-" * 40)
        for key, value in audit.project_info.items():
            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")

        # Resultados por componente
        report_lines.append("📊 RESULTADOS POR COMPONENTE:")
        report_lines.append("-" * 40)

        for component_audit in audit.component_audits:
            status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(
                component_audit.status, "❓"
            )
            report_lines.append(f"{status_icon} {component_audit.component}:")
            report_lines.append(f"   Puntuación: {component_audit.score:.1f}/100")
            report_lines.append(f"   Estado: {component_audit.status}")

            if component_audit.findings:
                report_lines.append("   Hallazgos:")
                for finding in component_audit.findings:
                    report_lines.append(f"     • {finding}")

            if component_audit.recommendations:
                report_lines.append("   Recomendaciones:")
                for rec in component_audit.recommendations:
                    report_lines.append(f"     • {rec}")

            report_lines.append("")

        # Resumen ejecutivo
        report_lines.append("🎯 RESUMEN EJECUTIVO:")
        report_lines.append("-" * 40)
        report_lines.append(f"• Componentes auditados: {audit.summary['total_components']}")
        report_lines.append(f"• Estado PASS: {audit.summary['status_distribution']['PASS']}")
        report_lines.append(f"• Estado WARN: {audit.summary['status_distribution']['WARN']}")
        report_lines.append(f"• Estado FAIL: {audit.summary['status_distribution']['FAIL']}")
        report_lines.append(f"• Recomendaciones totales: {audit.summary['recommendations_count']}")
        report_lines.append(
            f"• Tiempo de auditoría: {audit.summary['audit_duration_seconds']:.1f} segundos"
        )
        report_lines.append("")

        # Conclusión
        if audit.overall_score >= 90:
            conclusion = "🎉 EXCELENTE - El proyecto está en excelentes condiciones"
        elif audit.overall_score >= 75:
            conclusion = "✅ BUENO - El proyecto está bien estructurado con algunas mejoras menores"
        elif audit.overall_score >= 60:
            conclusion = "⚠️ ACEPTABLE - El proyecto necesita mejoras en algunos aspectos"
        else:
            conclusion = "❌ REQUIERE ATENCIÓN - El proyecto necesita revisión significativa"

        report_lines.append(conclusion)
        report_lines.append("=" * 70)

        return "\n".join(report_lines)

    def save_audit_report(self, audit: ProjectAudit, output_file: str = None):
        """Guardar reporte de auditoría en archivo"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"audit_report_{timestamp}.json"

        # Crear directorio de reportes si no existe
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        output_path = reports_dir / output_file

        # Guardar en formato JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(audit), f, indent=2, ensure_ascii=False)

        # También guardar reporte legible
        txt_report = self.generate_report(audit)
        txt_path = reports_dir / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_report)

        print(f"📋 Reporte de auditoría guardado en:")
        print(f"   • JSON: {output_path}")
        print(f"   • Texto: {txt_path}")


def main():
    """Función principal de auditoría"""
    print("🚀 INICIANDO AUDITORÍA COMPLETA DEL PROYECTO SHEILY AI")
    print("=" * 70)

    # Crear auditor
    auditor = SheilyAuditor()

    # Ejecutar auditoría completa
    audit_result = auditor.run_complete_audit()

    # Mostrar reporte en consola
    report = auditor.generate_report(audit_result)
    print(report)

    # Guardar reporte
    auditor.save_audit_report(audit_result)

    # Mostrar recomendaciones finales
    total_recommendations = sum(
        len(audit.recommendations) for audit in audit_result.component_audits
    )
    if total_recommendations == 0:
        print("🎉 ¡AUDITORÍA COMPLETADA! El proyecto está en excelentes condiciones.")
    elif total_recommendations <= 3:
        print(
            f"✅ Auditoría completada. Se encontraron {total_recommendations} recomendaciones menores."
        )
    else:
        print(
            f"⚠️  Auditoría completada. Se encontraron {total_recommendations} recomendaciones para mejorar."
        )

    print(f"\n📊 Puntuación general: {audit_result.overall_score:.1f}/100")


if __name__ == "__main__":
    main()
