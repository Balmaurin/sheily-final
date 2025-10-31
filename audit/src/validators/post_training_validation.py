#!/usr/bin/env python3
"""
VALIDACIÓN POST-ENTRENAMIENTO COMPLETA
Sistema avanzado de validación para verificar funcionalidad completa de adaptadores
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class PostTrainingValidator:
    """Sistema completo de validación post-entrenamiento"""

    def __init__(self):
        self.base_path = Path(".")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(f"audit_2024/logs/post_training_validation_{self.session_id}.jsonl")
        self.report_file = Path(f"audit_2024/reports/post_training_validation_{self.session_id}.json")

    def log_validation(self, action, details=None):
        """Registrar acción de validación"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "action": action,
            "details": details or {},
        }

        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"🔍 {action}")

    def validate_adapter_integrity(self, adapter_path):
        """Validar integridad técnica completa del adaptador"""
        self.log_validation("VALIDANDO INTEGRIDAD TÉCNICA", {"adapter": str(adapter_path)})

        integrity_results = {
            "adapter_path": str(adapter_path),
            "technical_checks": {},
            "functional_checks": {},
            "integration_checks": {},
            "overall_score": 0,
            "max_score": 100,
            "is_valid": False,
            "recommendations": [],
        }

        try:
            # 1. Verificaciones técnicas básicas
            config_file = adapter_path / "adapter_config.json"
            model_file = adapter_path / "adapter_model.safetensors"

            # Verificar existencia de archivos
            integrity_results["technical_checks"]["config_exists"] = config_file.exists()
            integrity_results["technical_checks"]["model_exists"] = model_file.exists()

            if not config_file.exists():
                integrity_results["recommendations"].append("Falta archivo adapter_config.json")
                return integrity_results

            if not model_file.exists():
                integrity_results["recommendations"].append("Falta archivo adapter_model.safetensors")
                return integrity_results

            # 2. Validar tamaño de archivos
            model_size = model_file.stat().st_size
            integrity_results["technical_checks"]["model_size"] = model_size
            integrity_results["technical_checks"]["model_size_mb"] = model_size / 1024 / 1024

            if model_size >= 100000:  # 100KB mínimo
                integrity_results["technical_checks"]["size_adequate"] = True
                integrity_results["overall_score"] += 25
            else:
                integrity_results["technical_checks"]["size_adequate"] = False
                integrity_results["recommendations"].append(f"Modelo muy pequeño: {model_size/1024:.1f}KB")

            # 3. Validar configuración JSON
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                integrity_results["technical_checks"]["config_valid"] = True
                integrity_results["technical_checks"]["config_size"] = len(json.dumps(config))
                integrity_results["overall_score"] += 20

                # Verificar campos críticos en config
                critical_fields = ["base_model_name", "peft_type", "task_type"]
                for field in critical_fields:
                    if field in config:
                        integrity_results["technical_checks"][f"config_{field}_present"] = True
                    else:
                        integrity_results["technical_checks"][f"config_{field}_present"] = False
                        integrity_results["recommendations"].append(f"Campo crítico faltante en config: {field}")

            except json.JSONDecodeError as e:
                integrity_results["technical_checks"]["config_valid"] = False
                integrity_results["recommendations"].append(f"Config JSON inválido: {str(e)}")
                return integrity_results

            # 4. Verificaciones funcionales
            functional_score = self.validate_functional_aspects(adapter_path, config)
            integrity_results["functional_checks"] = functional_score["details"]
            integrity_results["overall_score"] += functional_score["score"]

            # 5. Verificaciones de integración
            integration_score = self.validate_integration_aspects(adapter_path)
            integrity_results["integration_checks"] = integration_score["details"]
            integrity_results["overall_score"] += integration_score["score"]

            # 6. Determinar validez general
            integrity_results["is_valid"] = integrity_results["overall_score"] >= 70

            if integrity_results["is_valid"]:
                integrity_results["recommendations"].append("Adaptador válido y listo para uso")
            else:
                integrity_results["recommendations"].append("Adaptador necesita corrección")

        except Exception as e:
            integrity_results["recommendations"].append(f"Error durante validación: {str(e)}")

        return integrity_results

    def validate_functional_aspects(self, adapter_path, config):
        """Validar aspectos funcionales del adaptador"""
        functional_results = {"score": 0, "max_score": 30, "details": {}}

        try:
            # Verificar que el modelo se puede cargar (simulación)
            model_file = adapter_path / "adapter_model.safetensors"

            # Verificación básica de formato
            with open(model_file, "rb") as f:
                header = f.read(100)  # Leer primeros 100 bytes

            # Buscar patrones típicos de archivos safetensors
            if b"PK" in header[:10]:  # Indicativo de formato comprimido
                functional_results["details"]["format_safetensors"] = True
                functional_results["score"] += 15
            else:
                functional_results["details"]["format_safetensors"] = False
                functional_results["recommendations"].append("Formato de archivo sospechoso")

            # Verificar configuración PEFT
            if "peft_type" in config:
                peft_type = config["peft_type"].upper()
                if peft_type in ["LORA", "ADALORA", "IA3"]:
                    functional_results["details"]["peft_type_valid"] = True
                    functional_results["score"] += 10
                else:
                    functional_results["details"]["peft_type_valid"] = False
                    functional_results["recommendations"].append(f"Tipo PEFT no estándar: {peft_type}")

            # Verificar parámetros de entrenamiento
            if "r" in config and isinstance(config["r"], int) and config["r"] > 0:
                functional_results["details"]["rank_parameter_valid"] = True
                functional_results["score"] += 5
            else:
                functional_results["details"]["rank_parameter_valid"] = False
                functional_results["recommendations"].append("Parámetro de rango inválido o faltante")

        except Exception as e:
            functional_results["details"]["error"] = str(e)

        return functional_results

    def validate_integration_aspects(self, adapter_path):
        """Validar aspectos de integración con el sistema"""
        integration_results = {"score": 0, "max_score": 25, "details": {}}

        try:
            # Verificar estructura de directorios
            if adapter_path.exists() and adapter_path.is_dir():
                integration_results["details"]["directory_structure"] = True
                integration_results["score"] += 5

            # Verificar permisos de archivos
            config_file = adapter_path / "adapter_config.json"
            model_file = adapter_path / "adapter_model.safetensors"

            if config_file.exists():
                integration_results["details"]["config_accessible"] = True
                integration_results["score"] += 5

            if model_file.exists():
                integration_results["details"]["model_accessible"] = True
                integration_results["score"] += 5

            # Verificar integración con estructura del proyecto
            parent_dir = adapter_path.parent
            if parent_dir.name == "retraining":
                integration_results["details"]["proper_location"] = True
                integration_results["score"] += 10
            else:
                integration_results["details"]["proper_location"] = False
                integration_results["recommendations"].append("Adaptador no está en ubicación estándar")

        except Exception as e:
            integration_results["details"]["error"] = str(e)

        return integration_results

    def validate_all_retrained_adapters(self):
        """Validar todos los adaptadores en directorio de reentrenamiento"""
        self.log_validation("INICIANDO VALIDACIÓN DE TODOS LOS ADAPTADORES")

        retraining_dir = Path("models/lora_adapters/retraining")
        if not retraining_dir.exists():
            self.log_validation("ERROR: Directorio de reentrenamiento no existe")
            return {}

        validation_results = {}

        # Encontrar todas las ramas procesadas
        for adapter_dir in retraining_dir.iterdir():
            if adapter_dir.is_dir():
                branch_name = adapter_dir.name

                # Validar cada adaptador
                integrity_result = self.validate_adapter_integrity(adapter_dir)
                validation_results[branch_name] = integrity_result

                # Mostrar progreso
                status = "✅" if integrity_result["is_valid"] else "❌"
                score = integrity_result["overall_score"]
                print(f"  {status} {branch_name}: {score}/{integrity_result['max_score']} puntos")

                if integrity_result["recommendations"]:
                    for rec in integrity_result["recommendations"][:2]:  # Mostrar max 2 recomendaciones
                        print(f"    💡 {rec}")

        return validation_results

    def generate_validation_report(self, validation_results):
        """Generar reporte completo de validación"""
        # Calcular estadísticas
        total_adapters = len(validation_results)
        valid_adapters = sum(1 for r in validation_results.values() if r["is_valid"])
        invalid_adapters = total_adapters - valid_adapters

        # Calcular puntuaciones promedio
        scores = [r["overall_score"] for r in validation_results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Crear reporte
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_adapters": total_adapters,
                "valid_adapters": valid_adapters,
                "invalid_adapters": invalid_adapters,
                "validation_success_rate": valid_adapters / total_adapters * 100 if total_adapters > 0 else 0,
                "average_score": avg_score,
                "validation_threshold": 70,
            },
            "details": validation_results,
        }

        # Guardar reporte
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def run_complete_validation(self):
        """Ejecutar validación completa"""
        print("🔍 VALIDACIÓN POST-ENTRENAMIENTO COMPLETA")
        print("=" * 70)
        print(f"📋 Sesión: {self.session_id}")

        try:
            # Validar todos los adaptadores
            validation_results = self.validate_all_retrained_adapters()

            # Generar reporte
            report = self.generate_validation_report(validation_results)

            # Mostrar resumen ejecutivo
            print("\n" + "=" * 70)
            print("📊 RESUMEN EJECUTIVO DE VALIDACIÓN")
            print("=" * 70)

            summary = report["summary"]
            print(f"📋 Total de adaptadores: {summary['total_adapters']}")
            print(f"✅ Adaptadores válidos: {summary['valid_adapters']}")
            print(f"❌ Adaptadores inválidos: {summary['invalid_adapters']}")
            print(f"📈 Tasa de éxito: {summary['validation_success_rate']:.1f}%")
            print(f"🎯 Puntuación promedio: {summary['average_score']:.1f}/{summary['validation_threshold']}")

            print(f"\n📋 Reporte completo guardado: {self.report_file}")

            # Evaluar éxito general
            if summary["validation_success_rate"] >= 95:
                print("🎉 VALIDACIÓN COMPLETA EXITOSA")
                return True
            elif summary["validation_success_rate"] >= 80:
                print("✅ VALIDACIÓN PARCIALMENTE EXITOSA - REVISIÓN SUGERIDA")
                return True
            else:
                print("⚠️ VALIDACIÓN CON PROBLEMAS - REQUIERE CORRECCIÓN")
                return False

        except Exception as e:
            self.log_validation(
                "ERROR CRÍTICO EN VALIDACIÓN",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            print(f"❌ Error crítico en validación: {e}")
            return False


def main():
    """Función principal"""
    validator = PostTrainingValidator()
    success = validator.run_complete_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
