#!/usr/bin/env python3
"""
Tests para el Sistema de Auditoría Sheily AI
Validación completa de todas las funcionalidades de auditoría
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Importar sistema de auditoría
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_audit_system import AdvancedAuditSystem
from realtime_audit_dashboard import RealTimeAuditDashboard, ComplianceFramework
from monitoring_system import MonitoringService, MetricsCollector
from utils.audit_utils import AuditUtils


class TestAdvancedAuditSystem:
    """Tests para el sistema de auditoría avanzada"""

    def setup_method(self):
        """Configuración para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_system = AdvancedAuditSystem(self.temp_dir)

    def teardown_method(self):
        """Limpieza después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_system_initialization(self):
        """Test inicialización del sistema de auditoría"""
        assert self.audit_system.project_root == Path(self.temp_dir)
        assert self.audit_system.audit_dir == Path(self.temp_dir) / "audit_2025"
        assert self.audit_system.reports_dir == Path(self.temp_dir) / "audit_2025" / "reports"

    def test_file_system_analysis(self):
        """Test análisis del sistema de archivos"""
        # Crear archivos de prueba
        test_files = [
            "test1.py",
            "test2.py",
            "config.json",
            "README.md",
            "requirements.txt"
        ]

        for file in test_files:
            (Path(self.temp_dir) / file).touch()

        self.audit_system.analyze_file_system()

        # Verificar que se detectaron los archivos
        assert self.audit_system.metrics["files"]["total"] == 5
        assert self.audit_system.metrics["files"]["python"] == 2
        assert self.audit_system.metrics["files"]["json"] == 1
        assert self.audit_system.metrics["files"]["md"] == 1

    def test_code_complexity_analysis(self):
        """Test análisis de complejidad de código"""
        # Crear archivo Python de prueba
        test_code = '''
def simple_function():
    return "hello"

def complex_function(x, y, z):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(f"Even: {i}")
                if y > i:
                    return z + i
    return 0

class TestClass:
    def method_one(self):
        pass

    def method_two(self, a, b):
        try:
            return a / b
        except ZeroDivisionError:
            return None
'''

        py_file = Path(self.temp_dir) / "test_complexity.py"
        py_file.write_text(test_code)

        self.audit_system.analyze_code_complexity()

        # Verificar análisis de complejidad
        assert "complexity" in self.audit_system.metrics
        assert "cyclomatic" in self.audit_system.metrics["complexity"]
        assert str(py_file) in self.audit_system.metrics["complexity"]["cyclomatic"]

    def test_security_scanning(self):
        """Test escaneo de seguridad"""
        # Crear archivo con posibles problemas de seguridad
        test_code = '''
import os
password = "hardcoded_password_123"
api_key = "sk-1234567890abcdef"

def execute_command(cmd):
    os.system(cmd)  # Potential security issue

def eval_code(code):
    eval(code)  # Security risk
'''

        py_file = Path(self.temp_dir) / "test_security.py"
        py_file.write_text(test_code)

        self.audit_system.scan_security()

        # Verificar detección de problemas de seguridad
        assert "security" in self.audit_system.metrics
        assert self.audit_system.metrics["security"]["issues_found"] >= 2

    def test_quality_gates(self):
        """Test puertas de calidad"""
        # Configurar métricas de prueba
        self.audit_system.metrics = {
            "testing": {"estimated_coverage": 75},
            "security": {"issues_found": 2},
            "files": {"python": 100},
            "statistics": {"total_lines": 5000},
            "dependencies": {"total": 50, "outdated_count": 0}
        }

        result = self.audit_system.check_quality_gates()

        # Verificar que las puertas de calidad pasan
        assert result["passed"] == True
        assert result["details"]["code_coverage"]["passed"] == True
        assert result["details"]["security_issues"]["passed"] == True

    def test_audit_report_generation(self):
        """Test generación de reportes de auditoría"""
        # Configurar métricas mínimas
        self.audit_system.metrics = {
            "files": {"python": 10, "total": 20},
            "testing": {"estimated_coverage": 75, "total_tests": 50},
            "security": {"issues_found": 1, "issues": []},
            "statistics": {"total_lines": 1000},
            "dependencies": {"total": 20, "outdated_count": 0}
        }

        self.audit_system.create_audit_reports()

        # Verificar que se crearon los reportes
        reports_dir = Path(self.temp_dir) / "audit_2025" / "reports"
        assert reports_dir.exists()

        # Verificar archivos de reporte
        json_reports = list(reports_dir.glob("*.json"))
        html_reports = list(reports_dir.glob("*.html"))
        txt_reports = list(reports_dir.glob("*.txt"))

        assert len(json_reports) >= 1
        assert len(html_reports) >= 1
        assert len(txt_reports) >= 1


class TestRealTimeDashboard:
    """Tests para el dashboard en tiempo real"""

    def setup_method(self):
        """Configuración para tests de dashboard"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_dir = Path(self.temp_dir) / "audit_2025"
        self.dashboard = RealTimeAuditDashboard(self.audit_dir)

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dashboard_initialization(self):
        """Test inicialización del dashboard"""
        assert self.dashboard.audit_dir == self.audit_dir
        assert isinstance(self.dashboard.metrics_history, dict)
        assert isinstance(self.dashboard.alerts, list)

    def test_compliance_report_generation(self):
        """Test generación de reporte de cumplimiento"""
        test_metrics = {
            "testing": {"estimated_coverage": 75},
            "security": {"issues_found": 2},
            "files": {"python": 100}
        }

        report = self.dashboard.generate_compliance_report(test_metrics)

        # Verificar que el reporte contiene información de cumplimiento
        assert "COMPLIANCE REPORT" in report
        assert "SOC2" in report
        assert "ISO_27001" in report
        assert "OWASP" in report

    def test_compliance_certificate_generation(self):
        """Test generación de certificado de cumplimiento"""
        certificate = self.dashboard.compliance_framework.generate_compliance_certificate()

        # Verificar contenido del certificado
        assert "COMPLIANCE CERTIFICATE" in certificate
        assert "SOC2 Type II" in certificate
        assert "ISO 27001" in certificate
        assert "APPROVED FOR ENTERPRISE DEPLOYMENT" in certificate


class TestMonitoringService:
    """Tests para el servicio de monitoreo"""

    def setup_method(self):
        """Configuración para tests de monitoreo"""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = MonitoringService(Path(self.temp_dir))

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_monitoring_initialization(self):
        """Test inicialización del servicio de monitoreo"""
        assert self.monitor.running == False
        assert self.monitor.collector is not None
        assert self.monitor.detector is not None
        assert self.monitor.alerts is not None

    def test_metrics_collection(self):
        """Test recolección de métricas"""
        result = self.monitor.collect_and_analyze()

        # Verificar estructura del resultado
        assert "metrics" in result
        assert "anomalies" in result
        assert "alerts" in result
        assert "health" in result

        # Verificar métricas del sistema
        metrics = result["metrics"]
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "timestamp" in metrics

    def test_health_check(self):
        """Test verificación de salud"""
        health = self.monitor.health.run_full_health_check()

        # Verificar estructura del reporte de salud
        assert "timestamp" in health
        assert "overall_status" in health
        assert "disk" in health
        assert "memory" in health
        assert "processes" in health

        # Verificar que el estado general es válido
        assert health["overall_status"] in ["HEALTHY", "UNHEALTHY"]


class TestAuditUtils:
    """Tests para utilidades de auditoría"""

    def setup_method(self):
        """Configuración para tests de utilidades"""
        self.temp_dir = tempfile.mkdtemp()
        self.utils = AuditUtils(Path(self.temp_dir))

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_utils_initialization(self):
        """Test inicialización de utilidades"""
        assert self.utils.audit_dir == Path(self.temp_dir)
        assert self.utils.reports_dir == Path(self.temp_dir) / "reports"

    def test_system_validation(self):
        """Test validación del sistema"""
        validation = self.utils.validate_audit_system()

        # Verificar estructura de validación
        assert "timestamp" in validation
        assert "system_integrity" in validation
        assert "missing_components" in validation
        assert "recommendations" in validation

    def test_audit_summary_generation(self):
        """Test generación de resumen de auditoría"""
        summary = self.utils.generate_audit_summary()

        # Verificar contenido del resumen
        assert "timestamp" in summary
        assert "audit_system_version" in summary
        assert "project_status" in summary
        assert "system_health" in summary

    def test_data_export(self):
        """Test exportación de datos"""
        export_path = self.utils.export_audit_data("json")

        # Verificar que se creó el archivo de exportación
        assert Path(export_path).exists()

        # Verificar contenido del archivo exportado
        with open(export_path, "r", encoding="utf-8") as f:
            export_data = json.load(f)

        assert "export_timestamp" in export_data
        assert "audit_config" in export_data
        assert "system_summary" in export_data


class TestIntegration:
    """Tests de integración del sistema completo"""

    def setup_method(self):
        """Configuración para tests de integración"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Limpieza después de tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_audit_workflow(self):
        """Test flujo completo de auditoría"""
        # Crear sistema de auditoría
        audit_system = AdvancedAuditSystem(self.temp_dir)

        # Configurar métricas de prueba
        audit_system.metrics = {
            "files": {"python": 50, "total": 100},
            "testing": {"estimated_coverage": 75, "total_tests": 100},
            "security": {"issues_found": 2, "issues": []},
            "statistics": {"total_lines": 2500},
            "dependencies": {"total": 30, "outdated_count": 1}
        }

        # Ejecutar flujo completo
        result = audit_system.run_complete_audit()

        # Verificar resultado completo
        assert "timestamp" in result
        assert "duration_seconds" in result
        assert "status" in result
        assert "metrics" in result
        assert "quality_passed" in result

        # Verificar que se crearon reportes
        reports_dir = Path(self.temp_dir) / "audit_2025" / "reports"
        assert reports_dir.exists()

        json_reports = list(reports_dir.glob("*.json"))
        assert len(json_reports) >= 1

    def test_dashboard_integration(self):
        """Test integración del dashboard"""
        audit_dir = Path(self.temp_dir) / "audit_2025"
        dashboard = RealTimeAuditDashboard(audit_dir)

        # Datos de prueba
        test_metrics = {
            "testing": {"estimated_coverage": 75, "total_tests": 100},
            "security": {"issues_found": 1, "issues": []},
            "files": {"python": 50},
            "statistics": {"total_lines": 2000},
            "dependencies": {"total": 25, "outdated_count": 0}
        }

        # Generar reporte de cumplimiento
        compliance_report = dashboard.generate_compliance_report(test_metrics)
        assert "COMPLIANCE REPORT" in compliance_report

        # Generar certificado
        certificate = dashboard.compliance_framework.generate_compliance_certificate()
        assert "COMPLIANCE CERTIFICATE" in certificate

    def test_monitoring_integration(self):
        """Test integración del sistema de monitoreo"""
        monitor = MonitoringService(Path(self.temp_dir))

        # Recolectar métricas
        result = monitor.collect_and_analyze()

        # Verificar estructura completa
        assert "metrics" in result
        assert "anomalies" in result
        assert "alerts" in result
        assert "health" in result

        # Verificar métricas del sistema
        system_metrics = result["metrics"]
        assert "cpu_percent" in system_metrics
        assert "memory_percent" in system_metrics
        assert "timestamp" in system_metrics


# Fixtures para tests
@pytest.fixture
def sample_project_structure(tmp_path):
    """Crear estructura de proyecto de ejemplo para tests"""
    # Crear directorios
    (tmp_path / "models").mkdir()
    (tmp_path / "sheily_core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "config").mkdir()

    # Crear archivos de ejemplo
    (tmp_path / "main.py").write_text("print('Hello World')")
    (tmp_path / "models" / "model.py").write_text("class Model:\n    pass")
    (tmp_path / "sheily_core" / "core.py").write_text("def core_function():\n    return True")
    (tmp_path / "tests" / "test_main.py").write_text("def test_example():\n    assert True")
    (tmp_path / "config" / "settings.json").write_text('{"debug": true}')

    return tmp_path


# Tests con fixtures
class TestWithFixtures:
    """Tests que usan fixtures"""

    def test_audit_with_sample_project(self, sample_project_structure):
        """Test auditoría con proyecto de ejemplo"""
        audit_system = AdvancedAuditSystem(str(sample_project_structure))

        # Ejecutar análisis de archivos
        audit_system.analyze_file_system()

        # Verificar detección de archivos
        assert audit_system.metrics["files"]["python"] >= 3
        assert audit_system.metrics["files"]["json"] >= 1
        assert audit_system.metrics["files"]["total"] >= 5

    def test_complexity_analysis_with_sample(self, sample_project_structure):
        """Test análisis de complejidad con proyecto de ejemplo"""
        audit_system = AdvancedAuditSystem(str(sample_project_structure))

        # Ejecutar análisis de complejidad
        audit_system.analyze_code_complexity()

        # Verificar que se analizaron los archivos
        assert "complexity" in audit_system.metrics
        assert "cyclomatic" in audit_system.metrics["complexity"]
        assert len(audit_system.metrics["complexity"]["cyclomatic"]) >= 3


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v", "--tb=short"])
