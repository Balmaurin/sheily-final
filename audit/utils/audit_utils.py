#!/usr/bin/env python3
"""
Utilidades para el Sistema de Auditoría Sheily AI
Herramientas y funciones de soporte para el sistema de auditoría
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class AuditUtils:
    """Utilidades para el sistema de auditoría"""

    def __init__(self, audit_dir: Path = None):
        """Inicializar utilidades de auditoría"""
        self.audit_dir = audit_dir or Path(__file__).parent.parent
        self.reports_dir = self.audit_dir / "reports"
        self.logs_dir = self.audit_dir / "logs"

    def cleanup_old_reports(self, days: int = 30) -> Dict[str, int]:
        """Limpiar reportes antiguos"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned = {"files": 0, "size_mb": 0}

        if not self.reports_dir.exists():
            return cleaned

        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        size_mb = file_path.stat().st_size / 1024 / 1024
                        file_path.unlink()
                        cleaned["files"] += 1
                        cleaned["size_mb"] += size_mb
                except Exception:
                    continue

        return cleaned

    def backup_audit_data(self, backup_dir: Path = None) -> str:
        """Crear backup de datos de auditoría"""
        if backup_dir is None:
            backup_dir = self.audit_dir / "backups"

        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"audit_backup_{timestamp}.tar.gz"

        try:
            # Crear backup de reportes y logs
            subprocess.run([
                "tar", "-czf", str(backup_dir / backup_name),
                "-C", str(self.audit_dir),
                "reports", "logs", "config"
            ], check=True)

            return str(backup_dir / backup_name)
        except subprocess.CalledProcessError:
            return "Backup failed"

    def generate_audit_summary(self) -> Dict[str, any]:
        """Generar resumen de auditoría"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "audit_system_version": "2.0.0",
            "project_status": "ACTIVE",
            "last_audit": self._get_last_audit_date(),
            "total_reports": self._count_reports(),
            "system_health": self._check_system_health(),
        }

        return summary

    def _get_last_audit_date(self) -> Optional[str]:
        """Obtener fecha de última auditoría"""
        if not self.reports_dir.exists():
            return None

        latest_file = None
        latest_time = 0

        for file_path in self.reports_dir.rglob("*.json"):
            file_time = file_path.stat().st_mtime
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file_path

        if latest_file:
            return datetime.fromtimestamp(latest_time).isoformat()

        return None

    def _count_reports(self) -> int:
        """Contar reportes totales"""
        if not self.reports_dir.exists():
            return 0
        return len(list(self.reports_dir.rglob("*.json")))

    def _check_system_health(self) -> str:
        """Verificar salud del sistema de auditoría"""
        # Verificación básica de salud
        required_files = [
            "advanced_audit_system.py",
            "run_integrated_audit.py",
            "config/audit_config.json"
        ]

        missing = []
        for file in required_files:
            if not (self.audit_dir / file).exists():
                missing.append(file)

        if not missing:
            return "HEALTHY"
        elif len(missing) <= 1:
            return "WARNING"
        else:
            return "CRITICAL"

    def export_audit_data(self, format: str = "json", output_path: Path = None) -> str:
        """Exportar datos de auditoría"""
        if output_path is None:
            output_path = self.audit_dir / f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

        if format.lower() == "json":
            # Exportar configuración y estadísticas
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "audit_config": self._load_config(),
                "system_summary": self.generate_audit_summary(),
                "reports_count": self._count_reports(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def _load_config(self) -> Dict:
        """Cargar configuración de auditoría"""
        config_file = self.audit_dir / "config" / "audit_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def validate_audit_system(self) -> Dict[str, any]:
        """Validar integridad del sistema de auditoría"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "system_integrity": "VALID",
            "missing_components": [],
            "corrupted_files": [],
            "configuration_valid": True,
            "recommendations": []
        }

        # Verificar componentes requeridos
        required_components = [
            "advanced_audit_system.py",
            "run_integrated_audit.py",
            "realtime_audit_dashboard.py",
            "monitoring_system.py",
            "config/audit_config.json"
        ]

        for component in required_components:
            component_path = self.audit_dir / component
            if not component_path.exists():
                validation["missing_components"].append(component)
                validation["system_integrity"] = "INVALID"

        # Verificar archivos de configuración
        config_file = self.audit_dir / "config" / "audit_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError:
                validation["corrupted_files"].append("config/audit_config.json")
                validation["configuration_valid"] = False

        # Generar recomendaciones
        if validation["missing_components"]:
            validation["recommendations"].append(
                f"Instalar componentes faltantes: {validation['missing_components']}"
            )

        if validation["corrupted_files"]:
            validation["recommendations"].append(
                f"Corregir archivos corruptos: {validation['corrupted_files']}"
            )

        return validation

    def optimize_audit_performance(self) -> Dict[str, any]:
        """Optimizar rendimiento del sistema de auditoría"""
        optimization = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_improvements": [],
            "space_saved_mb": 0
        }

        # Limpiar reportes antiguos
        cleaned = self.cleanup_old_reports(days=7)
        if cleaned["files"] > 0:
            optimization["optimizations_applied"].append(
                f"Limpieza de {cleaned['files']} reportes antiguos ({cleaned['size_mb']:.1f}MB)"
            )
            optimization["space_saved_mb"] += cleaned["size_mb"]

        # Optimizar configuración
        config_file = self.audit_dir / "config" / "audit_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Optimizar intervalos de monitoreo
                if config.get("monitoring", {}).get("metrics_collection_interval", 60) < 30:
                    config["monitoring"]["metrics_collection_interval"] = 60
                    optimization["optimizations_applied"].append(
                        "Optimización de intervalo de monitoreo"
                    )

                # Guardar configuración optimizada
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            except Exception as e:
                optimization["optimizations_applied"].append(f"Error en optimización: {e}")

        return optimization

    def generate_maintenance_report(self) -> str:
        """Generar reporte de mantenimiento"""
        report = []
        report.append("AUDIT SYSTEM MAINTENANCE REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Validación del sistema
        validation = self.validate_audit_system()
        report.append("SYSTEM VALIDATION:")
        report.append(f"  Integrity: {validation['system_integrity']}")
        report.append(f"  Missing Components: {len(validation['missing_components'])}")
        report.append(f"  Corrupted Files: {len(validation['corrupted_files'])}")
        report.append("")

        # Estadísticas
        summary = self.generate_audit_summary()
        report.append("SYSTEM STATISTICS:")
        report.append(f"  Last Audit: {summary.get('last_audit', 'Never')}")
        report.append(f"  Total Reports: {summary['total_reports']}")
        report.append(f"  System Health: {summary['system_health']}")
        report.append("")

        # Optimización
        optimization = self.optimize_audit_performance()
        report.append("OPTIMIZATION RESULTS:")
        for opt in optimization["optimizations_applied"]:
            report.append(f"  ✅ {opt}")
        report.append(f"  Space Saved: {optimization['space_saved_mb']:.1f}MB")
        report.append("")

        # Recomendaciones
        if validation["recommendations"]:
            report.append("RECOMMENDATIONS:")
            for rec in validation["recommendations"]:
                report.append(f"  🔧 {rec}")
            report.append("")

        return "\n".join(report)


def main():
    """Función principal de utilidades"""
    utils = AuditUtils()

    print("🛠️ AUDIT SYSTEM UTILITIES")
    print("=" * 40)

    # Generar reporte de mantenimiento
    print("\n📊 GENERATING MAINTENANCE REPORT...")
    maintenance_report = utils.generate_maintenance_report()
    print(maintenance_report)

    # Exportar datos
    print("\n📤 EXPORTING AUDIT DATA...")
    export_path = utils.export_audit_data("json")
    print(f"✅ Audit data exported to: {export_path}")

    # Validar sistema
    print("\n🔍 VALIDATING SYSTEM...")
    validation = utils.validate_audit_system()
    print(f"System Integrity: {validation['system_integrity']}")
    print(f"Missing Components: {len(validation['missing_components'])}")
    print(f"Corrupted Files: {len(validation['corrupted_files'])}")

    if validation["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in validation["recommendations"]:
            print(f"  • {rec}")

    print("\n✅ UTILITIES COMPLETED")


if __name__ == "__main__":
    main()
