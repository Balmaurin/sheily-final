#!/usr/bin/env python3
"""
🏢 AUDITORÍA EMPRESARIAL COMPLETA - SHEILY AI
==============================================

Auditoría profunda a nivel empresarial del proyecto Sheily AI.
Genera reporte completo con análisis de:
- Arquitectura y estructura
- Calidad de código
- Seguridad
- Testing
- Documentación
- Dependencias
- Performance
- DevOps

Autor: Sistema de Auditoría Automatizado
Fecha: 31 de Octubre de 2025
Nivel: Enterprise Grade
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Configuración de colores para terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class EnterpriseAuditor:
    """Auditor empresarial completo para Sheily AI"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.now().isoformat()
        self.results = {
            "timestamp": self.timestamp,
            "project": "Sheily AI",
            "audit_level": "Enterprise Grade",
            "sections": {}
        }
        
    def print_header(self, text: str):
        """Imprimir encabezado con formato"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
        
    def print_section(self, text: str):
        """Imprimir sección"""
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}>> {text}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'-'*78}{Colors.ENDC}")
        
    def print_success(self, text: str):
        """Imprimir mensaje de éxito"""
        print(f"{Colors.OKGREEN}[OK]{Colors.ENDC} {text}")
        
    def print_warning(self, text: str):
        """Imprimir advertencia"""
        print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {text}")
        
    def print_error(self, text: str):
        """Imprimir error"""
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {text}")
        
    def print_info(self, text: str):
        """Imprimir información"""
        print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {text}")
        
    # =========================================================================
    # SECCIÓN 1: ARQUITECTURA Y ESTRUCTURA
    # =========================================================================
    
    def audit_architecture(self) -> Dict[str, Any]:
        """Auditar arquitectura y estructura del proyecto"""
        self.print_section("1. ARQUITECTURA Y ESTRUCTURA DEL PROYECTO")
        
        architecture = {
            "structure": {},
            "modules": {},
            "code_metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        # Analizar estructura de directorios
        core_dirs = ["sheily_core", "sheily_train", "tests", "tools", "all-Branches"]
        for dir_name in core_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                py_files = list(dir_path.rglob("*.py"))
                architecture["structure"][dir_name] = {
                    "exists": True,
                    "python_files": len(py_files),
                    "size_mb": sum(f.stat().st_size for f in py_files) / (1024 * 1024)
                }
                self.print_success(f"{dir_name}: {len(py_files)} archivos Python, "
                                 f"{architecture['structure'][dir_name]['size_mb']:.2f} MB")
            else:
                architecture["structure"][dir_name] = {"exists": False}
                self.print_warning(f"{dir_name}: NO ENCONTRADO")
        
        # Análisis de módulos en sheily_core
        core_path = self.project_root / "sheily_core"
        if core_path.exists():
            modules = [d.name for d in core_path.iterdir() if d.is_dir() and not d.name.startswith("__")]
            architecture["modules"]["sheily_core"] = modules
            self.print_info(f"Módulos en sheily_core: {len(modules)}")
            for module in sorted(modules):
                self.print_info(f"  • {module}")
        
        # Métricas de código
        total_py_files = len(list(self.project_root.rglob("*.py")))
        architecture["code_metrics"]["total_python_files"] = total_py_files
        self.print_info(f"Total archivos Python: {total_py_files}")
        
        # Verificar patrones de arquitectura
        patterns_found = []
        if (core_path / "integration").exists():
            patterns_found.append("Microservicios/Integración")
        if (core_path / "security").exists():
            patterns_found.append("Seguridad por diseño")
        if (core_path / "monitoring").exists():
            patterns_found.append("Observabilidad")
        
        architecture["architectural_patterns"] = patterns_found
        self.print_success(f"Patrones arquitectónicos detectados: {', '.join(patterns_found)}")
        
        # Recomendaciones
        if total_py_files > 1000:
            architecture["recommendations"].append(
                "Considerar modularización adicional debido al alto número de archivos"
            )
        
        self.results["sections"]["architecture"] = architecture
        return architecture
    
    # =========================================================================
    # SECCIÓN 2: CALIDAD DE CÓDIGO
    # =========================================================================
    
    def audit_code_quality(self) -> Dict[str, Any]:
        """Auditar calidad del código"""
        self.print_section("2. CALIDAD DE CÓDIGO")
        
        quality = {
            "pep8_compliance": {},
            "type_hints": {},
            "docstrings": {},
            "complexity": {},
            "issues": [],
            "score": 0
        }
        
        # Verificar imports y estructura básica
        self.print_info("Analizando estructura de imports...")
        core_files = list((self.project_root / "sheily_core").rglob("*.py"))
        
        files_with_docstrings = 0
        files_with_type_hints = 0
        total_analyzed = 0
        
        for py_file in core_files[:50]:  # Muestreo de 50 archivos
            try:
                content = py_file.read_text(encoding='utf-8')
                total_analyzed += 1
                
                # Verificar docstrings
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
                
                # Verificar type hints
                if " -> " in content or ": " in content:
                    files_with_type_hints += 1
                    
            except Exception:
                pass
        
        if total_analyzed > 0:
            docstring_percentage = (files_with_docstrings / total_analyzed) * 100
            type_hint_percentage = (files_with_type_hints / total_analyzed) * 100
            
            quality["docstrings"]["percentage"] = docstring_percentage
            quality["type_hints"]["percentage"] = type_hint_percentage
            
            self.print_info(f"Archivos con docstrings: {docstring_percentage:.1f}%")
            self.print_info(f"Archivos con type hints: {type_hint_percentage:.1f}%")
            
            if docstring_percentage < 50:
                quality["issues"].append("Baja cobertura de docstrings (<50%)")
                self.print_warning("⚠️ Baja cobertura de docstrings")
            else:
                self.print_success("✓ Buena cobertura de docstrings")
        
        # Calcular score general
        score_components = []
        if docstring_percentage >= 70:
            score_components.append(25)
        elif docstring_percentage >= 50:
            score_components.append(15)
        
        if type_hint_percentage >= 60:
            score_components.append(25)
        elif type_hint_percentage >= 40:
            score_components.append(15)
        
        quality["score"] = sum(score_components)
        
        self.results["sections"]["code_quality"] = quality
        return quality
    
    # =========================================================================
    # SECCIÓN 3: SEGURIDAD
    # =========================================================================
    
    def audit_security(self) -> Dict[str, Any]:
        """Auditar seguridad del proyecto"""
        self.print_section("3. SEGURIDAD Y VULNERABILIDADES")
        
        security = {
            "secrets_exposed": [],
            "vulnerable_patterns": [],
            "security_features": [],
            "recommendations": [],
            "severity": "LOW"
        }
        
        # Verificar archivos de seguridad
        security_files = [
            ".secrets.baseline",
            ".pre-commit-config.yaml",
            ".env.example",
            "sheily_core/security"
        ]
        
        for item in security_files:
            path = self.project_root / item
            if path.exists():
                security["security_features"].append(item)
                self.print_success(f"Encontrado: {item}")
            else:
                self.print_warning(f"Falta: {item}")
        
        # Buscar patrones inseguros en código
        self.print_info("Buscando patrones potencialmente inseguros...")
        unsafe_patterns = {
            r"eval\(": "Uso de eval()",
            r"exec\(": "Uso de exec()",
            r"__import__\(": "Uso de __import__()",
            r"subprocess\.call\(": "Uso directo de subprocess.call",
            r"password\s*=\s*['\"][^'\"]+['\"]": "Password hardcodeado",
        }
        
        vulnerable_files = defaultdict(list)
        py_files = list((self.project_root / "sheily_core").rglob("*.py"))
        
        for py_file in py_files[:100]:  # Muestreo
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, description in unsafe_patterns.items():
                    if re.search(pattern, content):
                        vulnerable_files[str(py_file.relative_to(self.project_root))].append(description)
            except Exception:
                pass
        
        if vulnerable_files:
            security["vulnerable_patterns"] = dict(vulnerable_files)
            security["severity"] = "MEDIUM" if len(vulnerable_files) > 5 else "LOW"
            self.print_warning(f"Patrones inseguros encontrados en {len(vulnerable_files)} archivos")
        else:
            self.print_success("No se encontraron patrones inseguros evidentes")
        
        # Verificar .env en .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            if ".env" in gitignore_content:
                self.print_success(".env está en .gitignore")
            else:
                security["recommendations"].append("Agregar .env a .gitignore")
                self.print_warning(".env NO está en .gitignore")
        
        self.results["sections"]["security"] = security
        return security
    
    # =========================================================================
    # SECCIÓN 4: TESTING
    # =========================================================================
    
    def audit_testing(self) -> Dict[str, Any]:
        """Auditar suite de testing"""
        self.print_section("4. TESTING Y COBERTURA")
        
        testing = {
            "test_files": 0,
            "test_structure": {},
            "coverage": {},
            "recommendations": []
        }
        
        tests_path = self.project_root / "tests"
        if tests_path.exists():
            test_files = list(tests_path.rglob("test_*.py"))
            testing["test_files"] = len(test_files)
            self.print_info(f"Archivos de test encontrados: {len(test_files)}")
            
            # Analizar estructura de tests
            test_dirs = ["unit", "integration", "security", "e2e", "performance"]
            for test_dir in test_dirs:
                dir_path = tests_path / test_dir
                if dir_path.exists():
                    tests_in_dir = len(list(dir_path.rglob("test_*.py")))
                    testing["test_structure"][test_dir] = tests_in_dir
                    self.print_success(f"  {test_dir}: {tests_in_dir} tests")
                else:
                    testing["test_structure"][test_dir] = 0
                    self.print_warning(f"  {test_dir}: NO EXISTE")
            
            # Verificar pytest.ini
            if (self.project_root / "pytest.ini").exists():
                self.print_success("pytest.ini configurado")
            else:
                self.print_warning("pytest.ini NO encontrado")
                testing["recommendations"].append("Crear pytest.ini para configuración")
            
        else:
            self.print_error("Directorio tests/ NO ENCONTRADO")
            testing["recommendations"].append("Crear estructura de tests completa")
        
        self.results["sections"]["testing"] = testing
        return testing
    
    # =========================================================================
    # SECCIÓN 5: DOCUMENTACIÓN
    # =========================================================================
    
    def audit_documentation(self) -> Dict[str, Any]:
        """Auditar documentación del proyecto"""
        self.print_section("5. DOCUMENTACIÓN")
        
        documentation = {
            "readme_files": 0,
            "api_docs": False,
            "quality_score": 0,
            "missing": [],
            "found": []
        }
        
        # Verificar READMEs
        readme_files = list(self.project_root.rglob("README.md"))
        documentation["readme_files"] = len(readme_files)
        self.print_info(f"Archivos README.md encontrados: {len(readme_files)}")
        
        # Documentación esencial
        essential_docs = {
            "README.md": "Documentación principal",
            "LICENSE": "Licencia",
            ".env.example": "Ejemplo de configuración",
            "requirements.txt": "Dependencias",
            "docs/": "Directorio de documentación"
        }
        
        for doc, description in essential_docs.items():
            path = self.project_root / doc
            if path.exists():
                documentation["found"].append(doc)
                self.print_success(f"{doc}: {description}")
            else:
                documentation["missing"].append(doc)
                self.print_warning(f"Falta: {doc} ({description})")
        
        # Calcular score
        found_percentage = (len(documentation["found"]) / len(essential_docs)) * 100
        documentation["quality_score"] = found_percentage
        
        if found_percentage >= 80:
            self.print_success(f"Score de documentación: {found_percentage:.0f}% ✓")
        else:
            self.print_warning(f"Score de documentación: {found_percentage:.0f}% (mejorable)")
        
        self.results["sections"]["documentation"] = documentation
        return documentation
    
    # =========================================================================
    # SECCIÓN 6: DEPENDENCIAS
    # =========================================================================
    
    def audit_dependencies(self) -> Dict[str, Any]:
        """Auditar dependencias del proyecto"""
        self.print_section("6. DEPENDENCIAS Y COMPATIBILIDAD")
        
        dependencies = {
            "requirements_files": [],
            "total_dependencies": 0,
            "outdated": [],
            "security_issues": []
        }
        
        # Buscar archivos de requirements
        req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py"]
        for req_file in req_files:
            path = self.project_root / req_file
            if path.exists():
                dependencies["requirements_files"].append(req_file)
                self.print_success(f"Encontrado: {req_file}")
                
                # Contar dependencias en requirements.txt
                if req_file == "requirements.txt":
                    content = path.read_text()
                    deps = [line for line in content.split('\n') 
                           if line.strip() and not line.strip().startswith('#')]
                    dependencies["total_dependencies"] = len(deps)
                    self.print_info(f"Total de dependencias: {len(deps)}")
        
        # Verificar duplicados
        if "requirements.txt" in dependencies["requirements_files"] and \
           "requirements-dev.txt" in dependencies["requirements_files"]:
            self.print_warning("Múltiples archivos de requirements - considerar unificar")
        
        self.results["sections"]["dependencies"] = dependencies
        return dependencies
    
    # =========================================================================
    # SECCIÓN 7: PERFORMANCE Y ESCALABILIDAD
    # =========================================================================
    
    def audit_performance(self) -> Dict[str, Any]:
        """Auditar performance y escalabilidad"""
        self.print_section("7. PERFORMANCE Y ESCALABILIDAD")
        
        performance = {
            "async_usage": False,
            "caching": False,
            "optimization_features": [],
            "recommendations": []
        }
        
        # Buscar uso de async/await
        core_files = list((self.project_root / "sheily_core").rglob("*.py"))
        async_count = 0
        
        for py_file in core_files[:50]:
            try:
                content = py_file.read_text(encoding='utf-8')
                if "async def" in content or "await " in content:
                    async_count += 1
            except Exception:
                pass
        
        if async_count > 0:
            performance["async_usage"] = True
            performance["optimization_features"].append("Async/Await")
            self.print_success(f"Uso de async/await detectado en {async_count} archivos")
        
        # Buscar caching
        cache_patterns = ["@lru_cache", "@cache", "redis", "memcached"]
        for pattern in cache_patterns:
            for py_file in core_files[:50]:
                try:
                    if pattern in py_file.read_text(encoding='utf-8'):
                        performance["caching"] = True
                        performance["optimization_features"].append(f"Caching ({pattern})")
                        break
                except Exception:
                    pass
        
        if performance["optimization_features"]:
            self.print_success(f"Features de optimización: {', '.join(performance['optimization_features'])}")
        else:
            self.print_warning("No se detectaron features de optimización evidentes")
            performance["recommendations"].append("Considerar implementar caching y async operations")
        
        self.results["sections"]["performance"] = performance
        return performance
    
    # =========================================================================
    # SECCIÓN 8: DEVOPS Y DEPLOYMENT
    # =========================================================================
    
    def audit_devops(self) -> Dict[str, Any]:
        """Auditar configuración de DevOps y deployment"""
        self.print_section("8. DEVOPS Y DEPLOYMENT")
        
        devops = {
            "docker": False,
            "ci_cd": False,
            "monitoring": False,
            "features": [],
            "missing": []
        }
        
        # Verificar Docker
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        docker_found = []
        for docker_file in docker_files:
            if (self.project_root / docker_file).exists():
                docker_found.append(docker_file)
                self.print_success(f"Encontrado: {docker_file}")
        
        if docker_found:
            devops["docker"] = True
            devops["features"].append(f"Docker ({len(docker_found)} archivos)")
        else:
            devops["missing"].append("Docker")
            self.print_warning("Configuración Docker no encontrada")
        
        # Verificar CI/CD
        ci_paths = [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"]
        for ci_path in ci_paths:
            if (self.project_root / ci_path).exists():
                devops["ci_cd"] = True
                devops["features"].append(f"CI/CD ({ci_path})")
                self.print_success(f"CI/CD encontrado: {ci_path}")
                break
        
        if not devops["ci_cd"]:
            devops["missing"].append("CI/CD")
            self.print_warning("No se encontró configuración CI/CD")
        
        # Verificar monitoring
        monitoring_files = ["monitoring/prometheus.yml", "docker-compose.yml"]
        for mon_file in monitoring_files:
            path = self.project_root / mon_file
            if path.exists():
                try:
                    content = path.read_text()
                    if "prometheus" in content.lower() or "grafana" in content.lower():
                        devops["monitoring"] = True
                        devops["features"].append("Monitoring (Prometheus/Grafana)")
                        self.print_success("Monitoring configurado")
                        break
                except Exception:
                    pass
        
        if not devops["monitoring"]:
            devops["missing"].append("Monitoring")
        
        self.results["sections"]["devops"] = devops
        return devops
    
    # =========================================================================
    # MÉTODO PRINCIPAL Y REPORTE
    # =========================================================================
    
    def run_complete_audit(self) -> Dict[str, Any]:
        """Ejecutar auditoría completa"""
        self.print_header("AUDITORÍA EMPRESARIAL COMPLETA - SHEILY AI")
        
        print(f"{Colors.OKBLUE}Proyecto:{Colors.ENDC} Sheily AI")
        print(f"{Colors.OKBLUE}Nivel de Auditoría:{Colors.ENDC} Enterprise Grade")
        print(f"{Colors.OKBLUE}Fecha:{Colors.ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.OKBLUE}Auditor:{Colors.ENDC} Sistema Automatizado de Auditoría")
        
        # Ejecutar todas las secciones
        try:
            self.audit_architecture()
            self.audit_code_quality()
            self.audit_security()
            self.audit_testing()
            self.audit_documentation()
            self.audit_dependencies()
            self.audit_performance()
            self.audit_devops()
        except Exception as e:
            self.print_error(f"Error durante auditoría: {e}")
            self.results["errors"] = str(e)
        
        # Generar resumen ejecutivo
        self.generate_executive_summary()
        
        return self.results
    
    def generate_executive_summary(self):
        """Generar resumen ejecutivo de la auditoría"""
        self.print_header("RESUMEN EJECUTIVO")
        
        sections = self.results.get("sections", {})
        
        # Calcular score general
        scores = {
            "Arquitectura": 85 if sections.get("architecture", {}).get("structure") else 50,
            "Calidad de Código": sections.get("code_quality", {}).get("score", 0),
            "Seguridad": 80 if sections.get("security", {}).get("severity") == "LOW" else 50,
            "Testing": 70 if sections.get("testing", {}).get("test_files", 0) > 10 else 30,
            "Documentación": sections.get("documentation", {}).get("quality_score", 0),
            "Dependencias": 75 if sections.get("dependencies", {}).get("total_dependencies", 0) > 30 else 50,
            "Performance": 70 if sections.get("performance", {}).get("async_usage") else 40,
            "DevOps": 80 if sections.get("devops", {}).get("docker") else 40
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        self.results["executive_summary"] = {
            "overall_score": overall_score,
            "section_scores": scores,
            "status": self.get_status_from_score(overall_score),
            "timestamp": self.timestamp
        }
        
        # Mostrar scores por sección
        print(f"\n{Colors.BOLD}Scores por Sección:{Colors.ENDC}\n")
        for section, score in scores.items():
            color = Colors.OKGREEN if score >= 70 else (Colors.WARNING if score >= 50 else Colors.FAIL)
            bar = "█" * int(score / 5)
            print(f"{section:.<25} {color}{score:>3.0f}/100{Colors.ENDC} {bar}")
        
        # Score general
        print(f"\n{Colors.BOLD}{'─'*78}{Colors.ENDC}")
        overall_color = Colors.OKGREEN if overall_score >= 70 else (Colors.WARNING if overall_score >= 50 else Colors.FAIL)
        print(f"{Colors.BOLD}SCORE GENERAL:{Colors.ENDC} {overall_color}{overall_score:.1f}/100{Colors.ENDC}")
        print(f"{Colors.BOLD}ESTADO:{Colors.ENDC} {overall_color}{self.get_status_from_score(overall_score)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'─'*78}{Colors.ENDC}\n")
        
        # Recomendaciones prioritarias
        self.print_section("RECOMENDACIONES PRIORITARIAS")
        
        recommendations = []
        
        if scores["Calidad de Código"] < 60:
            recommendations.append("[ALTA] Mejorar docstrings y type hints en el codigo")
        
        if scores["Testing"] < 60:
            recommendations.append("[ALTA] Ampliar suite de tests y mejorar cobertura")
        
        if scores["Seguridad"] < 70:
            recommendations.append("[MEDIA] Revisar y corregir patrones de seguridad")
        
        if scores["Documentación"] < 70:
            recommendations.append("[MEDIA] Completar documentacion esencial")
        
        if scores["DevOps"] < 60:
            recommendations.append("[BAJA] Implementar CI/CD y monitoring")
        
        if not recommendations:
            recommendations.append("[OK] Proyecto en buen estado general")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        self.results["executive_summary"]["prioritized_recommendations"] = recommendations
    
    def get_status_from_score(self, score: float) -> str:
        """Obtener estado basado en score"""
        if score >= 85:
            return "EXCELENTE (*****)"
        elif score >= 70:
            return "BUENO (****)"
        elif score >= 50:
            return "ACEPTABLE (***)"
        elif score >= 30:
            return "NECESITA MEJORAS (**)"
        else:
            return "CRITICO (*)"
    
    def save_report(self, output_file: str = "enterprise_audit_report.json"):
        """Guardar reporte en archivo JSON"""
        report_path = self.project_root / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.print_success(f"\n📄 Reporte guardado en: {output_file}")
        
        # También crear un resumen en markdown
        md_file = output_file.replace('.json', '.md')
        self.save_markdown_report(md_file)
    
    def save_markdown_report(self, output_file: str):
        """Guardar resumen en formato Markdown"""
        report_path = self.project_root / output_file
        
        summary = self.results.get("executive_summary", {})
        overall_score = summary.get("overall_score", 0)
        
        md_content = f"""# 🏢 Reporte de Auditoría Empresarial - Sheily AI

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Nivel:** Enterprise Grade  
**Score General:** {overall_score:.1f}/100  
**Estado:** {summary.get('status', 'N/A')}

---

## 📊 Scores por Sección

"""
        
        for section, score in summary.get("section_scores", {}).items():
            stars = "*" * (int(score) // 20)
            md_content += f"- **{section}:** {score:.0f}/100 {stars}\n"
        
        md_content += "\n---\n\n## 🎯 Recomendaciones Prioritarias\n\n"
        
        for rec in summary.get("prioritized_recommendations", []):
            md_content += f"{rec}\n"
        
        md_content += "\n---\n\n## 📋 Detalles por Sección\n\n"
        
        for section_name, section_data in self.results.get("sections", {}).items():
            md_content += f"### {section_name.replace('_', ' ').title()}\n\n"
            md_content += f"```json\n{json.dumps(section_data, indent=2, ensure_ascii=False)}\n```\n\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.print_success(f"📄 Resumen Markdown guardado en: {output_file}")


def main():
    """Función principal"""
    project_root = Path(__file__).parent
    
    auditor = EnterpriseAuditor(project_root)
    results = auditor.run_complete_audit()
    auditor.save_report()
    
    # Retornar código de salida basado en score
    overall_score = results.get("executive_summary", {}).get("overall_score", 0)
    if overall_score >= 70:
        sys.exit(0)
    elif overall_score >= 50:
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
