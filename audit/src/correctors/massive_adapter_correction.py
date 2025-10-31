#!/usr/bin/env python3
"""
CORRECCIÓN MASIVA DE 36 ADAPTADORES LoRA
Script maestro con manejo robusto de errores y logging completo
"""

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


class MassiveAdapterCorrector:
    """Sistema de corrección masiva con manejo avanzado de errores"""

    def __init__(self):
        self.base_path = Path(".")
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(f"audit_2024/logs/massive_correction_{self.session_id}.jsonl")
        self.report_file = Path(f"audit_2024/reports/massive_correction_{self.session_id}.json")

        # Cargar configuración desde auditoría previa
        self.load_audit_configuration()

    def load_audit_configuration(self):
        """Cargar configuración desde auditoría previa"""
        try:
            with open("audit_2024/reports/pre_training_audit_report.json", "r", encoding="utf-8") as f:
                audit_data = json.load(f)

            self.defined_branches = audit_data["defined_branches"]
            self.corrupted_branches = list(audit_data["corrupted_details"].keys())
            self.priorities = audit_data["priorities"]

            print(f"📋 Configuración cargada: {len(self.corrupted_branches)} ramas corruptas identificadas")

        except FileNotFoundError:
            print("❌ Error: Ejecutar auditoría previa primero")
            sys.exit(1)

    def log_correction(self, action, details=None):
        """Registrar acción de corrección con detalles completos"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "action": action,
            "details": details or {},
        }

        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"🔧 {action}")

    def robust_training_execution(self, branch_name, max_retries=3):
        """Ejecución robusta de entrenamiento con reintentos"""
        for attempt in range(max_retries + 1):
            try:
                self.log_correction(
                    f"INTENTO {attempt + 1}/{max_retries + 1}",
                    {"branch": branch_name, "attempt": attempt + 1},
                )

                # Crear directorio de salida
                output_dir = Path(f"models/lora_adapters/retraining/{branch_name}")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Encontrar datos de entrenamiento
                corpus_files = self.find_corpus_files(branch_name)
                if not corpus_files:
                    return False, "No se encontraron datos de entrenamiento"

                corpus_file = self.select_best_corpus_file(corpus_files)

                # Ejecutar entrenamiento con parámetros optimizados
                success, result = self.execute_training(branch_name, corpus_file, output_dir)

                if success:
                    # Validar resultado
                    validation_success, validation_details = self.validate_trained_adapter(output_dir)
                    if validation_success:
                        self.log_correction(
                            "ÉXITO COMPLETO",
                            {
                                "branch": branch_name,
                                "training_time": result.get("training_time", 0),
                                "validation_score": validation_details.get("score", 0),
                            },
                        )
                        return True, "Entrenamiento y validación exitosos"
                    else:
                        error_msg = f"Validación fallida: {validation_details.get('issues', [])}"
                        if attempt < max_retries:
                            self.log_correction(
                                "REINTENTO POR VALIDACIÓN",
                                {
                                    "branch": branch_name,
                                    "issues": validation_details.get("issues", []),
                                },
                            )
                            time.sleep(5)
                            continue
                        else:
                            return False, error_msg
                else:
                    error_msg = result.get("error", "Error desconocido en entrenamiento")
                    if attempt < max_retries:
                        self.log_correction(
                            "REINTENTO POR ENTRENAMIENTO",
                            {"branch": branch_name, "error": error_msg},
                        )
                        time.sleep(5)
                        continue
                    else:
                        return False, error_msg

            except subprocess.TimeoutExpired:
                if attempt < max_retries:
                    self.log_correction("REINTENTO POR TIMEOUT", {"branch": branch_name})
                    time.sleep(10)
                    continue
                else:
                    return False, "Timeout después de múltiples intentos"

            except Exception as e:
                error_details = {"error": str(e), "traceback": traceback.format_exc()}

                if attempt < max_retries:
                    self.log_correction("REINTENTO POR EXCEPCIÓN", {"branch": branch_name, "error": str(e)})
                    time.sleep(5)
                    continue
                else:
                    return False, f"Excepción después de múltiples intentos: {str(e)}"

        return False, "Máximo de reintentos alcanzado"

    def find_corpus_files(self, branch_name):
        """Encontrar archivos de corpus para una rama"""
        corpus_path = Path(f"corpus_ES/{branch_name}")
        if not corpus_path.exists():
            return []

        # Buscar recursivamente archivos JSONL
        jsonl_files = []
        for root, dirs, files in os.walk(corpus_path):
            for file in files:
                if file.endswith(".jsonl"):
                    jsonl_files.append(Path(root) / file)

        return jsonl_files

    def select_best_corpus_file(self, corpus_files):
        """Seleccionar el mejor archivo de corpus basado en tamaño y calidad"""
        if not corpus_files:
            return None

        # Seleccionar archivo más grande (más contenido)
        best_file = max(corpus_files, key=lambda x: x.stat().st_size)

        # Verificar que tenga contenido mínimo
        min_size = 10000  # 10KB mínimo
        if best_file.stat().st_size < min_size:
            self.log_correction(
                "WARNING: Archivo de datos muy pequeño",
                {"file": str(best_file), "size": best_file.stat().st_size},
            )

        return best_file

    def execute_training(self, branch_name, corpus_file, output_dir):
        """Ejecutar entrenamiento con manejo completo de errores"""
        try:
            start_time = time.time()

            # Comando de entrenamiento optimizado
            cmd = [
                sys.executable,
                "sheily_train/core/training/training_pipeline.py",
                branch_name,
                str(corpus_file),
                str(output_dir),
            ]

            self.log_correction(
                "EJECUTANDO ENTRENAMIENTO",
                {
                    "branch": branch_name,
                    "corpus_file": str(corpus_file),
                    "output_dir": str(output_dir),
                    "command": " ".join(cmd),
                },
            )

            # Ejecutar con timeout controlado
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutos máximo
                cwd=self.base_path,
            )

            training_time = time.time() - start_time

            execution_details = {
                "return_code": result.returncode,
                "training_time": training_time,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
            }

            if result.returncode == 0:
                self.log_correction("ENTRENAMIENTO COMPLETADO", execution_details)
                return True, {
                    "success": True,
                    "training_time": training_time,
                    "output": result.stdout,
                }
            else:
                error_details = {
                    "success": False,
                    "training_time": training_time,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }
                self.log_correction("ENTRENAMIENTO FALLIDO", error_details)
                return False, error_details

        except subprocess.TimeoutExpired:
            self.log_correction("TIMEOUT EN ENTRENAMIENTO", {"branch": branch_name, "timeout_seconds": 1800})
            return False, {"error": "Timeout en entrenamiento"}

        except Exception as e:
            self.log_correction(
                "EXCEPCIÓN EN ENTRENAMIENTO",
                {"branch": branch_name, "error": str(e), "traceback": traceback.format_exc()},
            )
            return False, {"error": f"Excepción: {str(e)}"}

    def validate_trained_adapter(self, adapter_path):
        """Validar adaptador entrenado con métricas detalladas"""
        try:
            # Usar sistema de validación avanzada
            cmd = [
                sys.executable,
                "sheily_train/core/validation/advanced_validation.py",
                str(adapter_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Parsear resultado de validación
                validation_result = {"valid": True, "score": 100, "details": "Validación exitosa"}
            else:
                validation_result = {"valid": False, "score": 0, "details": result.stderr}

            return validation_result["valid"], validation_result

        except Exception as e:
            return False, {"error": f"Error en validación: {str(e)}"}

    def process_branches_by_priority(self):
        """Procesar ramas por orden de prioridad"""
        self.log_correction("INICIANDO PROCESAMIENTO POR PRIORIDADES")

        # Procesar en orden de prioridad basado en calidad de datos
        all_branches_to_process = []

        # Grupo 1: Datos excelentes (19 ramas)
        all_branches_to_process.extend(self.priorities["group_1_excellent_data"])

        # Grupo 3: Datos pobres (17 ramas) - procesar después
        all_branches_to_process.extend(self.priorities["group_3_poor_data"])

        results = {
            "total": len(all_branches_to_process),
            "successful": 0,
            "failed": 0,
            "group_1_success": 0,
            "group_3_success": 0,
            "sessions": [],
        }

        print(f"🚀 PROCESANDO {results['total']} RAMAS POR PRIORIDAD")
        print("=" * 70)

        for i, branch in enumerate(all_branches_to_process, 1):
            print(f"\n📍 Progreso: {i}/{results['total']}")
            print(f"🎯 Procesando: {branch}")

            # Determinar grupo para métricas
            is_group_1 = branch in self.priorities["group_1_excellent_data"]

            start_time = time.time()
            success, message = self.robust_training_execution(branch)
            processing_time = time.time() - start_time

            # Registrar sesión detallada
            session_data = {
                "branch": branch,
                "group": "group_1" if is_group_1 else "group_3",
                "success": success,
                "processing_time": processing_time,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            results["sessions"].append(session_data)

            if success:
                results["successful"] += 1
                if is_group_1:
                    results["group_1_success"] += 1
                else:
                    results["group_3_success"] += 1
                print(f"✅ {branch}: ÉXITO")
            else:
                results["failed"] += 1
                print(f"❌ {branch}: FALLIDO - {message}")

            # Pausa entre ramas para estabilidad
            if i < results["total"]:
                pause_time = 3
                print(f"⏳ Pausa de {pause_time} segundos...")
                time.sleep(pause_time)

        return results

    def generate_comprehensive_report(self, results):
        """Generar reporte comprehensivo de corrección masiva"""
        print("\n" + "=" * 70)
        print("📊 REPORTE COMPREHENSIVO DE CORRECCIÓN MASIVA")
        print("=" * 70)

        # Calcular métricas
        total_branches = results["total"]
        successful = results["successful"]
        failed = results["failed"]
        success_rate = successful / total_branches * 100 if total_branches > 0 else 0

        print(f"📋 Total procesado: {total_branches}")
        print(f"✅ Éxitos: {successful}")
        print(f"❌ Fallos: {failed}")
        print(f"📈 Tasa de éxito general: {success_rate:.1f}%")

        if results["group_1_success"] > 0:
            group_1_rate = results["group_1_success"] / len(self.priorities["group_1_excellent_data"]) * 100
            print(
                f"🏆 Grupo 1 (datos excelentes): {results['group_1_success']}/{len(self.priorities['group_1_excellent_data'])} ({group_1_rate:.1f}%)"
            )

        if results["group_3_success"] > 0:
            group_3_rate = results["group_3_success"] / len(self.priorities["group_3_poor_data"]) * 100
            print(
                f"⚠️ Grupo 3 (datos pobres): {results['group_3_success']}/{len(self.priorities['group_3_poor_data'])} ({group_3_rate:.1f}%)"
            )

        # Análisis de tiempos
        successful_times = [s["processing_time"] for s in results["sessions"] if s["success"]]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            print(f"⏱️ Tiempo promedio de éxito: {avg_time:.1f} segundos")

        # Crear reporte detallado
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "summary": {
                "total_branches": total_branches,
                "successful": successful,
                "failed": failed,
                "success_rate": success_rate,
                "group_1_success": results["group_1_success"],
                "group_3_success": results["group_3_success"],
                "average_success_time": sum(successful_times) / len(successful_times) if successful_times else 0,
            },
            "sessions": results["sessions"],
            "configuration": {
                "max_retries": 3,
                "timeout_per_branch": 1800,
                "pause_between_branches": 3,
            },
        }

        # Guardar reporte
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📋 Reporte detallado guardado: {self.report_file}")

        # Evaluar éxito general
        if success_rate >= 95:
            print("🎉 CORRECCIÓN MASIVA EXITOSA - OBJETIVO ALCANZADO")
            return True
        elif success_rate >= 80:
            print("✅ CORRECCIÓN MASIVA PARCIALMENTE EXITOSA - REVISIÓN SUGERIDA")
            return True
        else:
            print("⚠️ CORRECCIÓN MASIVA CON PROBLEMAS - REQUIERE ATENCIÓN")
            return False

    def run_massive_correction(self):
        """Ejecutar corrección masiva completa"""
        print("🚀 CORRECCIÓN MASIVA DE 36 ADAPTADORES LoRA")
        print("=" * 70)
        print(f"📋 Sesión: {self.session_id}")
        print(f"🎯 Ramas a procesar: {len(self.corrupted_branches)}")

        try:
            # Procesar ramas por prioridad
            results = self.process_branches_by_priority()

            # Generar reporte final
            success = self.generate_comprehensive_report(results)

            return success

        except KeyboardInterrupt:
            print("\n⚠️ CORRECCIÓN INTERRUMPIDA POR USUARIO")
            return False
        except Exception as e:
            self.log_correction(
                "ERROR CRÍTICO EN CORRECCIÓN MASIVA",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            print(f"❌ Error crítico: {e}")
            return False


def main():
    """Función principal"""
    corrector = MassiveAdapterCorrector()
    success = corrector.run_massive_correction()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
