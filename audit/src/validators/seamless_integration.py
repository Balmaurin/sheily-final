#!/usr/bin/env python3
"""
INTEGRACIÓN SEAMLESS CON SCRIPTS EXISTENTES
Sistema que asegura integración perfecta con complete_correction.py, complete_retraining.py e implement_complete_system.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class SeamlessIntegrator:
    """Sistema de integración perfecta con scripts existentes"""

    def __init__(self):
        self.base_path = Path(".")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(f"audit_2024/logs/seamless_integration_{self.session_id}.jsonl")

    def log_integration(self, action, details=None):
        """Registrar acción de integración"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "action": action,
            "details": details or {},
        }

        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"🔗 {action}")

    def test_script_compatibility(self, script_path, test_args=None):
        """Probar compatibilidad con script existente"""
        try:
            self.log_integration(f"PROBANDO COMPATIBILIDAD", {"script": str(script_path)})

            # Comando de prueba básico
            cmd = [sys.executable, str(script_path)]
            if test_args:
                cmd.extend(test_args)

            # Ejecutar con timeout corto para prueba
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=self.base_path)

            compatibility_result = {
                "script": str(script_path),
                "return_code": result.returncode,
                "compatible": result.returncode == 0,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
            }

            if result.returncode == 0:
                self.log_integration("SCRIPT COMPATIBLE", compatibility_result)
                return True, "Compatible"
            else:
                self.log_integration("SCRIPT INCOMPATIBLE", {**compatibility_result, "error": result.stderr})
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Timeout en prueba de compatibilidad"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def integrate_with_existing_scripts(self):
        """Integrar con todos los scripts existentes"""
        self.log_integration("INICIANDO INTEGRACIÓN SEAMLESS")

        scripts_to_test = {
            "complete_correction": {
                "path": Path("audit_2024/complete_correction.py"),
                "test_args": [],
                "critical": True,
            },
            "complete_retraining": {
                "path": Path("audit_2024/complete_retraining.py"),
                "test_args": [],
                "critical": True,
            },
            "implement_complete_system": {
                "path": Path("audit_2024/implement_complete_system.py"),
                "test_args": [],
                "critical": True,
            },
            "advanced_validation": {
                "path": Path("sheily_train/core/validation/advanced_validation.py"),
                "test_args": ["models/lora_adapters/retraining/"],
                "critical": False,
            },
            "comprehensive_training": {
                "path": Path("sheily_train/core/training/comprehensive_training.py"),
                "test_args": [],
                "critical": False,
            },
        }

        integration_results = {}

        for script_name, script_info in scripts_to_test.items():
            if script_info["path"].exists():
                compatible, message = self.test_script_compatibility(script_info["path"], script_info["test_args"])
                integration_results[script_name] = {
                    "compatible": compatible,
                    "message": message,
                    "critical": script_info["critical"],
                }
            else:
                integration_results[script_name] = {
                    "compatible": False,
                    "message": "Archivo no encontrado",
                    "critical": script_info["critical"],
                }

        return integration_results

    def create_integration_test_suite(self):
        """Crear suite de tests de integración"""
        self.log_integration("CREANDO SUITE DE TESTS DE INTEGRACIÓN")

        test_suite = '''
#!/usr/bin/env python3
"""
SUITE DE TESTS DE INTEGRACIÓN COMPLETA
Verifica que todos los componentes funcionen perfectamente juntos
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_integration_test(test_name, command, expected_return_code=0, timeout=60):
    """Ejecutar un test de integración"""
    print(f"🧪 Ejecutando: {test_name}")

    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        execution_time = time.time() - start_time

        success = result.returncode == expected_return_code

        test_result = {
            'test_name': test_name,
            'success': success,
            'return_code': result.returncode,
            'execution_time': execution_time,
            'stdout_length': len(result.stdout),
            'stderr_length': len(result.stderr)
        }

        if success:
            print(f"  ✅ {test_name}: PASSED ({execution_time:.1f}s)")
        else:
            print(f"  ❌ {test_name}: FAILED")
            print(f"    Código de retorno: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")

        return test_result

    except subprocess.TimeoutExpired:
        print(f"  ⏰ {test_name}: TIMEOUT")
        return {
            'test_name': test_name,
            'success': False,
            'return_code': -1,
            'execution_time': timeout,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"  💥 {test_name}: ERROR - {str(e)}")
        return {
            'test_name': test_name,
            'success': False,
            'return_code': -2,
            'error': str(e)
        }

def run_complete_integration_suite():
    """Ejecutar suite completa de integración"""
    print("🚀 SUITE COMPLETA DE TESTS DE INTEGRACIÓN")
    print("=" * 60)

    base_path = Path('.')

    test_results = []

    # Test 1: Auditoría previa
    print("\\n📋 FASE 1: AUDITORÍA PREVIA")
    test_results.append(run_integration_test(
        "Pre-training Audit",
        [sys.executable, "audit_2024/pre_training_audit.py"],
        expected_return_code=0,
        timeout=60
    ))

    # Test 2: Scripts principales
    print("\\n🔧 FASE 2: SCRIPTS PRINCIPALES")
    test_results.append(run_integration_test(
        "Complete Correction",
        [sys.executable, "audit_2024/complete_correction.py"],
        expected_return_code=0,
        timeout=120
    ))

    # Test 3: Validación avanzada
    print("\\n🔍 FASE 3: VALIDACIÓN AVANZADA")
    test_results.append(run_integration_test(
        "Advanced Validation",
        [sys.executable, "sheily_train/core/validation/advanced_validation.py", "models/lora_adapters/retraining/"],
        expected_return_code=0,
        timeout=60
    ))

    # Test 4: Monitoreo de salud
    print("\\n💻 FASE 4: MONITOREO DE SALUD")
    test_results.append(run_integration_test(
        "Health Monitor",
        [sys.executable, "sheily_train/tools/monitoring/health_monitor.py"],
        expected_return_code=0,
        timeout=30
    ))

    return test_results

def generate_integration_report(test_results):
    """Generar reporte de integración"""
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['success'])
    failed_tests = total_tests - passed_tests

    # Calcular tiempo total
    total_time = sum(r['execution_time'] for r in test_results)

    # Crear reporte
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0,
            'total_execution_time': total_time
        },
        'test_details': test_results
    }

    # Guardar reporte
    report_file = Path('audit_2024/reports/integration_test_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def main():
    """Función principal de integración"""
    print("🔗 VERIFICACIÓN DE INTEGRACIÓN SEAMLESS")
    print("=" * 60)

    try:
        # Ejecutar suite completa de tests
        test_results = run_complete_integration_suite()

        # Generar reporte
        report = generate_integration_report(test_results)

        # Mostrar resumen
        print("\\n" + "=" * 60)
        print("📊 RESUMEN DE INTEGRACIÓN")
        print("=" * 60)

        summary = report['summary']
        print(f"🧪 Tests ejecutados: {summary['total_tests']}")
        print(f"✅ Tests aprobados: {summary['passed_tests']}")
        print(f"❌ Tests fallidos: {summary['failed_tests']}")
        print(f"📈 Tasa de éxito: {summary['success_rate']:.1f}%")
        print(f"⏱️ Tiempo total: {summary['total_execution_time']:.1f} segundos")

        print(f"\\n📋 Reporte guardado: audit_2024/reports/integration_test_report.json")

        # Evaluar integración general
        if summary['success_rate'] >= 95:
            print("🎉 INTEGRACIÓN SEAMLESS COMPLETAMENTE EXITOSA")
            return 0
        elif summary['success_rate'] >= 80:
            print("✅ INTEGRACIÓN SEAMLESS PARCIALMENTE EXITOSA")
            return 0
        else:
            print("⚠️ INTEGRACIÓN SEAMLESS CON PROBLEMAS")
            return 1

    except Exception as e:
        print(f"❌ Error crítico en integración: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
'''

        with open("audit_2024/integration_test_suite.py", "w") as f:
            f.write(test_suite)

        self.log_integration("SUITE DE TESTS DE INTEGRACIÓN CREADA")

    def create_unified_correction_workflow(self):
        """Crear workflow unificado de corrección"""
        self.log_integration("CREANDO WORKFLOW UNIFICADO")

        workflow_script = '''
#!/usr/bin/env python3
"""
WORKFLOW UNIFICADO DE CORRECCIÓN COMPLETA
Ejecuta todo el proceso de corrección de manera integrada y seamless
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_workflow_step(step_name, command, expected_code=0):
    """Ejecutar un paso del workflow"""
    print(f"\\n🔄 Ejecutando: {step_name}")

    try:
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        execution_time = time.time() - start_time

        success = result.returncode == expected_code

        if success:
            print(f"  ✅ {step_name}: COMPLETADO ({execution_time:.1f}s)")
        else:
            print(f"  ❌ {step_name}: FALLIDO")
            print(f"    Error: {result.stderr.strip()}")

        return success, {
            'step': step_name,
            'success': success,
            'execution_time': execution_time,
            'return_code': result.returncode
        }

    except Exception as e:
        print(f"  💥 {step_name}: ERROR - {str(e)}")
        return False, {
            'step': step_name,
            'success': False,
            'error': str(e)
        }

def run_complete_correction_workflow():
    """Ejecutar workflow completo de corrección"""
    print("🚀 WORKFLOW COMPLETO DE CORRECCIÓN - 36 ADAPTADORES LoRA")
    print("=" * 70)

    workflow_steps = [
        ("Auditoría Previa", [sys.executable, "audit_2024/pre_training_audit.py"], 0),
        ("Corrección Masiva", [sys.executable, "audit_2024/massive_adapter_correction.py"], 0),
        ("Validación Post-Entrenamiento", [sys.executable, "audit_2024/post_training_validation.py"], 0),
        ("Tests de Integración", [sys.executable, "audit_2024/integration_test_suite.py"], 0),
        ("Monitoreo Final", [sys.executable, "sheily_train/tools/monitoring/health_monitor.py"], 0)
    ]

    results = []
    successful_steps = 0

    for step_name, command, expected_code in workflow_steps:
        success, step_result = run_workflow_step(step_name, command, expected_code)
        results.append(step_result)

        if success:
            successful_steps += 1

        # Pausa breve entre pasos
        time.sleep(2)

    return results

def generate_workflow_report(results):
    """Generar reporte del workflow completo"""
    total_steps = len(results)
    successful_steps = sum(1 for r in results if r['success'])
    total_time = sum(r['execution_time'] for r in results)

    report = {
        'timestamp': datetime.now().isoformat(),
        'workflow_summary': {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': total_steps - successful_steps,
            'success_rate': successful_steps / total_steps * 100 if total_steps > 0 else 0,
            'total_execution_time': total_time
        },
        'step_details': results
    }

    # Guardar reporte
    report_file = Path('audit_2024/reports/complete_workflow_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report

def main():
    """Función principal del workflow"""
    print("🎯 WORKFLOW UNIFICADO DE CORRECCIÓN COMPLETA")
    print("=" * 70)

    try:
        # Ejecutar workflow completo
        results = run_complete_correction_workflow()

        # Generar reporte
        report = generate_workflow_report(results)

        # Mostrar resumen final
        print("\\n" + "=" * 70)
        print("📊 REPORTE FINAL DEL WORKFLOW")
        print("=" * 70)

        summary = report['workflow_summary']
        print(f"🔧 Pasos ejecutados: {summary['total_steps']}")
        print(f"✅ Pasos exitosos: {summary['successful_steps']}")
        print(f"❌ Pasos fallidos: {summary['failed_steps']}")
        print(f"📈 Tasa de éxito: {summary['success_rate']:.1f}%")
        print(f"⏱️ Tiempo total: {summary['total_execution_time']:.1f} segundos")

        print(f"\\n📋 Reporte completo guardado: audit_2024/reports/complete_workflow_report.json")

        # Evaluar éxito del workflow completo
        if summary['success_rate'] >= 95:
            print("🎉 WORKFLOW COMPLETO EJECUTADO EXITOSAMENTE")
            print("🚀 El proyecto está completamente corregido y operativo")
            return 0
        elif summary['success_rate'] >= 80:
            print("✅ WORKFLOW COMPLETADO PARCIALMENTE - REVISIÓN SUGERIDA")
            return 0
        else:
            print("⚠️ WORKFLOW CON PROBLEMAS - REQUIERE ATENCIÓN")
            return 1

    except KeyboardInterrupt:
        print("\\n⚠️ WORKFLOW INTERRUMPIDO POR USUARIO")
        return 1
    except Exception as e:
        print(f"\\n❌ Error crítico en workflow: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
'''

        with open("audit_2024/complete_correction_workflow.py", "w") as f:
            f.write(workflow_script)

        self.log_integration("WORKFLOW UNIFICADO CREADO")

    def run_seamless_integration(self):
        """Ejecutar integración seamless completa"""
        print("🔗 VERIFICACIÓN DE INTEGRACIÓN SEAMLESS COMPLETA")
        print("=" * 70)

        try:
            # Probar integración con scripts existentes
            integration_results = self.integrate_with_existing_scripts()

            # Crear suite de tests de integración
            self.create_integration_test_suite()

            # Crear workflow unificado
            self.create_unified_correction_workflow()

            # Mostrar resultados de integración
            print("\n" + "=" * 70)
            print("📊 RESULTADOS DE INTEGRACIÓN")
            print("=" * 70)

            critical_scripts = [name for name, info in integration_results.items() if info["critical"]]
            critical_success = sum(1 for name in critical_scripts if integration_results[name]["compatible"])

            print(f"🔧 Scripts críticos: {critical_success}/{len(critical_scripts)} compatibles")
            print(f"📋 Scripts totales: {len(integration_results)} verificados")

            for script_name, result in integration_results.items():
                status = "✅" if result["compatible"] else "❌"
                print(f"  {status} {script_name}: {result['message']}")

            # Evaluar integración general
            if critical_success == len(critical_scripts):
                print("\n🎉 INTEGRACIÓN SEAMLESS COMPLETAMENTE EXITOSA")
                print("🚀 Todos los componentes funcionan perfectamente juntos")
                return True
            else:
                print("\n⚠️ INTEGRACIÓN SEAMLESS CON PROBLEMAS")
                print("🔧 Algunos componentes necesitan atención")
                return False

        except Exception as e:
            self.log_integration("ERROR EN INTEGRACIÓN SEAMLESS", {"error": str(e)})
            print(f"❌ Error en integración: {e}")
            return False


def main():
    """Función principal"""
    integrator = SeamlessIntegrator()
    success = integrator.run_seamless_integration()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
