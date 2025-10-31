#!/usr/bin/env python3
"""
AUDITORÍA PREVIA AL ENTRENAMIENTO - 36 ADAPTADORES LoRA
Sistema completo de auditoría antes del proceso de corrección masiva
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class PreTrainingAuditor:
    """Auditor completo previo al entrenamiento"""

    def __init__(self):
        self.base_path = Path(".")
        self.branches_file = Path("BRANCHES.txt")
        self.corrupted_dir = Path("models/lora_adapters/corrupted")
        self.retraining_dir = Path("models/lora_adapters/retraining")
        self.log_file = Path("audit_2024/logs/pre_training_audit.jsonl")

    def log_audit(self, action, details=None):
        """Registrar acción de auditoría"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {},
        }

        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"🔍 {action}")

    def load_defined_branches(self):
        """Cargar ramas definidas en BRANCHES.txt"""
        if not self.branches_file.exists():
            raise FileNotFoundError(f"Archivo BRANCHES.txt no encontrado: {self.branches_file}")

        with open(self.branches_file, "r", encoding="utf-8") as f:
            branches = [line.strip() for line in f if line.strip()]

        self.log_audit("RAMAS DEFINIDAS CARGADAS", {"count": len(branches), "branches": branches})
        return branches

    def audit_corrupted_adapters(self):
        """Auditoría detallada de adaptadores corruptos"""
        self.log_audit("AUDITORÍA DE ADAPTADORES CORRUPTOS")

        if not self.corrupted_dir.exists():
            self.log_audit("WARNING: Directorio de corruptos no existe")
            return {}

        corrupted_analysis = {}

        for branch_dir in self.corrupted_dir.iterdir():
            if branch_dir.is_dir():
                branch_name = branch_dir.name
                branch_analysis = self.analyze_corrupted_branch(branch_dir)
                corrupted_analysis[branch_name] = branch_analysis

        self.log_audit("AUDITORÍA DE CORRUPTOS COMPLETADA", {"branches_analyzed": len(corrupted_analysis)})
        return corrupted_analysis

    def analyze_corrupted_branch(self, branch_path):
        """Analizar una rama corrupta en detalle"""
        analysis = {
            "path": str(branch_path),
            "files": [],
            "total_size": 0,
            "issues": [],
            "data_quality": "unknown",
        }

        try:
            # Listar archivos
            for file_path in branch_path.iterdir():
                if file_path.is_file():
                    size = file_path.stat().st_size
                    analysis["files"].append({"name": file_path.name, "size": size, "path": str(file_path)})
                    analysis["total_size"] += size

                    # Identificar problemas
                    if size < 1000:
                        analysis["issues"].append(f"Archivo muy pequeño: {file_path.name} ({size} bytes)")
                    if file_path.name == "adapter_model.safetensors" and size < 100000:
                        analysis["issues"].append(f"Modelo principal corrupto: {size} bytes")

            # Verificar datos correspondientes en corpus
            corpus_path = Path(f"corpus_ES/{branch_path.name}")
            if corpus_path.exists():
                corpus_files = list(corpus_path.glob("**/*.jsonl"))
                corpus_size = sum(f.stat().st_size for f in corpus_files)

                if corpus_size > 1000000:  # Más de 1MB
                    analysis["data_quality"] = "excellent"
                elif corpus_size > 100000:  # Más de 100KB
                    analysis["data_quality"] = "good"
                else:
                    analysis["data_quality"] = "poor"

                analysis["corpus_files"] = len(corpus_files)
                analysis["corpus_size"] = corpus_size
            else:
                analysis["data_quality"] = "missing"
                analysis["issues"].append("Datos de corpus faltantes")

        except Exception as e:
            analysis["issues"].append(f"Error analizando: {str(e)}")

        return analysis

    def identify_correction_priorities(self):
        """Identificar prioridades de corrección basadas en calidad de datos"""
        self.log_audit("IDENTIFICANDO PRIORIDADES DE CORRECCIÓN")

        priorities = {
            "group_1_excellent_data": [],  # Datos excelentes + adaptadores corruptos
            "group_2_good_data": [],  # Datos buenos + adaptadores corruptos
            "group_3_poor_data": [],  # Datos pobres + adaptadores corruptos
            "total_branches": 0,
        }

        corrupted_analysis = self.audit_corrupted_adapters()

        for branch_name, analysis in corrupted_analysis.items():
            priorities["total_branches"] += 1

            data_quality = analysis.get("data_quality", "unknown")

            if data_quality == "excellent":
                priorities["group_1_excellent_data"].append(branch_name)
            elif data_quality == "good":
                priorities["group_2_good_data"].append(branch_name)
            else:
                priorities["group_3_poor_data"].append(branch_name)

        # Loguear prioridades identificadas
        self.log_audit(
            "PRIORIDADES IDENTIFICADAS",
            {
                "group_1": len(priorities["group_1_excellent_data"]),
                "group_2": len(priorities["group_2_good_data"]),
                "group_3": len(priorities["group_3_poor_data"]),
                "total": priorities["total_branches"],
            },
        )

        return priorities

    def generate_pre_training_report(self):
        """Generar reporte completo de pre-entrenamiento"""
        self.log_audit("GENERANDO REPORTE DE PRE-ENTRENAMIENTO")

        # Cargar ramas definidas
        defined_branches = self.load_defined_branches()

        # Auditoría de corruptos
        corrupted_analysis = self.audit_corrupted_adapters()

        # Identificar prioridades
        priorities = self.identify_correction_priorities()

        # Crear reporte comprehensivo
        report = {
            "timestamp": datetime.now().isoformat(),
            "audit_summary": {
                "defined_branches": len(defined_branches),
                "corrupted_branches": len(corrupted_analysis),
                "branches_to_correct": priorities["total_branches"],
                "group_1_count": len(priorities["group_1_excellent_data"]),
                "group_2_count": len(priorities["group_2_good_data"]),
                "group_3_count": len(priorities["group_3_poor_data"]),
            },
            "priorities": priorities,
            "corrupted_details": corrupted_analysis,
            "defined_branches": defined_branches,
        }

        # Guardar reporte
        report_file = Path("audit_2024/reports/pre_training_audit_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def validate_training_environment(self):
        """Validar entorno de entrenamiento completo"""
        self.log_audit("VALIDANDO ENTORNO DE ENTRENAMIENTO")

        environment_checks = {
            "modelo_base": Path("models/gguf/llama-3.2.gguf"),
            "corpus": Path("corpus_ES"),
            "script_entrenamiento": Path("sheily_train/core/training/training_pipeline.py"),
            "script_validacion": Path("sheily_train/core/validation/advanced_validation.py"),
            "estructura_corrupta": self.corrupted_dir,
            "estructura_reentrenamiento": self.retraining_dir,
        }

        validation_results = {}

        for check_name, check_path in environment_checks.items():
            if check_path.exists():
                if check_path.is_file():
                    size = check_path.stat().st_size
                    validation_results[check_name] = {
                        "exists": True,
                        "size_mb": size / 1024 / 1024,
                        "status": "OK",
                    }
                else:
                    items = len(list(check_path.iterdir()))
                    validation_results[check_name] = {
                        "exists": True,
                        "items": items,
                        "status": "OK",
                    }
            else:
                validation_results[check_name] = {"exists": False, "status": "MISSING"}

        # Verificar estado general
        critical_missing = [name for name, result in validation_results.items() if not result["exists"]]
        if critical_missing:
            self.log_audit("ERROR: Componentes críticos faltantes", {"missing": critical_missing})
            return False

        self.log_audit(
            "ENTORNO DE ENTRENAMIENTO VALIDADO",
            {
                "model_size_gb": validation_results["modelo_base"]["size_mb"] / 1024,
                "corrupted_branches": validation_results["estructura_corrupta"]["items"],
            },
        )

        return True

    def run_complete_audit(self):
        """Ejecutar auditoría completa"""
        print("🚀 AUDITORÍA PREVIA AL ENTRENAMIENTO - 36 ADAPTADORES")
        print("=" * 70)

        try:
            # Generar reporte completo
            report = self.generate_pre_training_report()

            # Validar entorno
            env_valid = self.validate_training_environment()

            # Mostrar resumen ejecutivo
            print("\n" + "=" * 70)
            print("📊 RESUMEN EJECUTIVO DE AUDITORÍA")
            print("=" * 70)

            summary = report["audit_summary"]
            print(f"📋 Ramas definidas: {summary['defined_branches']}")
            print(f"🔧 Ramas corruptas identificadas: {summary['corrupted_branches']}")
            print(f"🎯 Ramas a corregir: {summary['branches_to_correct']}")
            print(f"🏆 Grupo 1 (datos excelentes): {summary['group_1_count']} ramas")
            print(f"✅ Grupo 2 (datos buenos): {summary['group_2_count']} ramas")
            print(f"⚠️ Grupo 3 (datos pobres): {summary['group_3_count']} ramas")

            if env_valid:
                print("✅ Entorno de entrenamiento: VALIDADO")
            else:
                print("❌ Entorno de entrenamiento: PROBLEMAS DETECTADOS")

            print(f"\n📋 Reporte completo guardado: audit_2024/reports/pre_training_audit_report.json")

            # Determinar si podemos proceder
            if env_valid and summary["branches_to_correct"] > 0:
                print("🎉 LISTO PARA PROCEDER CON CORRECCIÓN MASIVA")
                return True
            else:
                print("⚠️ REVISAR PROBLEMAS ANTES DE CONTINUAR")
                return False

        except Exception as e:
            self.log_audit("ERROR EN AUDITORÍA", {"error": str(e)})
            print(f"❌ Error durante auditoría: {e}")
            return False


def main():
    """Función principal"""
    auditor = PreTrainingAuditor()
    success = auditor.run_complete_audit()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
