#!/usr/bin/env python3
"""
CORRECCIÓN COMPLETA DEL PROYECTO SHEILY
Script maestro que implementa el plan de corrección completo
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class ProjectCorrector:
    """Clase principal para corrección completa del proyecto"""

    def __init__(self):
        self.base_path = Path(".")
        self.start_time = datetime.now()
        self.log_file = Path("audit_2024/logs/correction_log.jsonl")

    def log_action(self, action, details=None):
        """Registrar acción en log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {},
        }

        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"📋 {action}")

    def create_organized_structure(self):
        """Crear estructura de directorios organizada"""
        self.log_action("CREANDO ESTRUCTURA ORGANIZADA")

        # Crear estructura principal
        dirs_to_create = [
            "models/lora_adapters/functional",
            "models/lora_adapters/corrupted",
            "models/lora_adapters/retraining",
            "models/lora_adapters/production",
            "models/lora_adapters/validation",
            "models/lora_adapters/backup",
            "models/lora_adapters/logs",
            "sheily_train/core/training",
            "sheily_train/core/validation",
            "sheily_train/core/conversion",
            "sheily_train/tools/monitoring",
            "sheily_train/tools/testing",
            "sheily_train/tools/utilities",
            "sheily_train/experimental",
            "sheily_train/deprecated",
            "sheily_train/docs",
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        self.log_action("ESTRUCTURA CREADA", {"directories": len(dirs_to_create)})

    def move_corrupted_adapters(self):
        """Mover adaptadores corruptos identificados"""
        self.log_action("MOVIENDO ADAPTADORES CORRUPTOS")

        # Cargar datos de auditoría
        try:
            with open("audit_2024/reports/lora_audit.json", "r") as f:
                audit_data = json.load(f)
        except FileNotFoundError:
            self.log_action("ERROR: Archivo de auditoría no encontrado")
            return False

        corrupted_branches = list(audit_data["details"]["corrupted"].keys())
        moved_count = 0

        for branch in corrupted_branches:
            source = Path(f"models/lora_adapters/{branch}")
            if source.exists():
                try:
                    # Crear enlace simbólico en lugar de mover
                    link_path = Path(f"models/lora_adapters/corrupted/{branch}")
                    if not link_path.exists():
                        source.rename(link_path)
                        moved_count += 1
                except Exception as e:
                    self.log_action(f"ERROR moviendo {branch}", {"error": str(e)})

        self.log_action("ADAPTADORES CORRUPTOS MOVIDOS", {"count": moved_count})
        return True

    def reorganize_training_scripts(self):
        """Reorganizar scripts de entrenamiento"""
        self.log_action("REORGANIZANDO SCRIPTS DE ENTRENAMIENTO")

        # Crear mapa de organización
        script_organization = {
            "sheily_train/core/training/": [
                "train_lora.py",
                "lora_training.py",
                "lora_sft_train.py",
                "train_multi_rama_lora.py",
                "training.py",
                "training_orchestrator.py",
            ],
            "sheily_train/core/conversion/": [
                "convert_lora_to_gguf.py",
                "convert-model.sh",
                "convert_to_gguf.sh",
                "merge_lora.py",
                "quantize.sh",
            ],
            "sheily_train/core/validation/": [
                "validate_lora.py",
                "test_lora.py",
                "test_training_system.py",
            ],
            "sheily_train/tools/monitoring/": [
                "run_all_tests.sh",
                "test_functional_training_system.py",
                "test_lora_training_system.py",
            ],
            "sheily_train/deprecated/": [
                "convert-model(1).sh",
                "run-converted-model(1).sh",
                "tests(1).sh",
                "tests(2).sh",
                "tests(3).sh",
            ],
        }

        moved_files = 0
        for target_dir, scripts in script_organization.items():
            for script in scripts:
                source = Path(f"sheily_train/{script}")
                if source.exists():
                    try:
                        shutil.move(str(source), target_dir)
                        moved_files += 1
                    except Exception as e:
                        self.log_action(f"ERROR moviendo {script}", {"error": str(e)})

        self.log_action("SCRIPTS REORGANIZADOS", {"files_moved": moved_files})

    def create_validation_system(self):
        """Crear sistema de validación automática"""
        self.log_action("CREANDO SISTEMA DE VALIDACIÓN")

        validation_script = '''
#!/usr/bin/env python3
"""
VALIDADOR AUTOMÁTICO DE ADAPTADORES LoRA
"""

import json
import sys
from pathlib import Path

def validate_adapter(adapter_path):
    """Validar un adaptador LoRA"""
    issues = []

    # Verificar archivos requeridos
    config_file = adapter_path / 'adapter_config.json'
    model_file = adapter_path / 'adapter_model.safetensors'

    if not config_file.exists():
        issues.append("Falta adapter_config.json")
    if not model_file.exists():
        issues.append("Falta adapter_model.safetensors")

    # Verificar tamaño
    if model_file.exists():
        size = model_file.stat().st_size
        if size < 100000:  # Menos de 100KB
            issues.append(f"Tamaño insuficiente: {size/1024:.1f}KB")

    # Verificar config JSON
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            issues.append("Config JSON inválido")

    return len(issues) == 0, issues

def main():
    """Función principal"""
    if len(sys.argv) != 2:
        print("Uso: python validate_adapter.py <ruta_adaptador>")
        return 1

    adapter_path = Path(sys.argv[1])
    if not adapter_path.exists():
        print(f"❌ Adaptador no encontrado: {adapter_path}")
        return 1

    valid, issues = validate_adapter(adapter_path)

    if valid:
        print("✅ Adaptador válido")
        return 0
    else:
        print("❌ Adaptador inválido:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

if __name__ == "__main__":
    exit(main())
'''

        with open("sheily_train/core/validation/validate_adapter.py", "w") as f:
            f.write(validation_script)

        self.log_action("SISTEMA DE VALIDACIÓN CREADO")

    def create_training_pipeline(self):
        """Crear pipeline de entrenamiento estandarizado"""
        self.log_action("CREANDO PIPELINE DE ENTRENAMIENTO")

        pipeline_script = '''
#!/usr/bin/env python3
"""
PIPELINE DE ENTRENAMIENTO ESTANDARIZADO
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def train_branch_with_validation(branch_name, corpus_path, output_path):
    """Entrenar rama con validación completa"""
    print(f"🚀 Entrenando rama: {branch_name}")

    # Verificar datos de entrada
    corpus_file = Path(f"{corpus_path}/{branch_name}/st").glob("*.jsonl")
    corpus_files = list(corpus_file)

    if not corpus_files:
        print(f"❌ No hay datos para {branch_name}")
        return False

    # Usar archivo más grande
    corpus_file = max(corpus_files, key=lambda x: x.stat().st_size)

    # Crear directorio de salida
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ejecutar entrenamiento
    cmd = [
        sys.executable, '../train_lora.py',
        '--data', str(corpus_file),
        '--out', str(output_path),
        '--model', '../../models/gguf/llama-3.2.gguf',
        '--epochs', '3'
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        training_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Entrenamiento exitoso en {training_time:.1f}s")

            # Validar resultado
            success, issues = validate_trained_adapter(output_path)
            if success:
                print("🎉 Adaptador válido generado")
                return True
            else:
                print("❌ Adaptador inválido generado:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Timeout en entrenamiento")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_trained_adapter(adapter_path):
    """Validar adaptador entrenado"""
    issues = []

    config_file = adapter_path / 'adapter_config.json'
    model_file = adapter_path / 'adapter_model.safetensors'

    if not config_file.exists():
        issues.append("Falta config")
    if not model_file.exists():
        issues.append("Falta modelo")

    if model_file.exists():
        size = model_file.stat().st_size
        if size < 100000:
            issues.append(f"Tamaño pequeño: {size/1024:.1f}KB")

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                json.load(f)
        except:
            issues.append("Config JSON inválido")

    return len(issues) == 0, issues

def main():
    """Función principal"""
    if len(sys.argv) != 4:
        print("Uso: python training_pipeline.py <rama> <corpus_path> <output_path>")
        return 1

    branch = sys.argv[1]
    corpus_path = sys.argv[2]
    output_path = sys.argv[3]

    success = train_branch_with_validation(branch, corpus_path, output_path)

    if success:
        print("🎉 Pipeline completado exitosamente")
        return 0
    else:
        print("❌ Pipeline fallido")
        return 1

if __name__ == "__main__":
    exit(main())
'''

        with open("sheily_train/core/training/training_pipeline.py", "w") as f:
            f.write(pipeline_script)

        self.log_action("PIPELINE DE ENTRENAMIENTO CREADO")

    def create_comprehensive_retraining_script(self):
        """Crear script de reentrenamiento comprehensivo"""
        self.log_action("CREANDO SCRIPT DE RETRENAMIENTO COMPREHENSIVO")

        script_content = '''
#!/usr/bin/env python3
"""
RETRENAMIENTO COMPLETO DE TODAS LAS RAMAS
Script maestro para corrección completa del proyecto
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Todas las ramas a procesar
ALL_BRANCHES = [
    'antropologia', 'economia', 'psicologia', 'historia', 'quimica',
    'biologia', 'filosofia', 'sociologia', 'politica', 'ecologia',
    'educacion', 'arte', 'informatica', 'ciberseguridad', 'linguistica',
    'inteligencia_artificial', 'neurociencia', 'robotica', 'etica',
    'tecnologia', 'derecho', 'musica', 'cine', 'literatura',
    'ingenieria', 'antropologia_digital', 'economia_global',
    'filosofia_moderna', 'marketing', 'derecho_internacional',
    'psicologia_social', 'fisica_cuantica', 'astronomia',
    'IA_multimodal', 'voz_emocional', 'metacognicion'
]

def log_session(session_data):
    """Registrar sesión de entrenamiento"""
    log_file = Path('models/lora_adapters/logs/master_training_log.jsonl')
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(session_data) + '\\n')

def validate_environment():
    """Validar entorno completo"""
    print("🔍 VALIDANDO ENTORNO...")

    checks = [
    ('Modelo base', Path('models/gguf/llama-3.2.gguf')),
        ('Corpus', Path('corpus_ES')),
        ('Script entrenamiento', Path('sheily_train/core/training/training_pipeline.py')),
        ('Script validación', Path('sheily_train/core/validation/validate_adapter.py'))
    ]

    for name, path in checks:
        if path.exists():
            print(f"✅ {name}: OK")
        else:
            print(f"❌ {name}: FALTA")
            return False

    print("✅ Entorno validado completamente")
    return True

def process_all_branches():
    """Procesar todas las ramas"""
    if not validate_environment():
        print("❌ Entorno inválido")
        return False

    results = {
        'total': len(ALL_BRANCHES),
        'successful': 0,
        'failed': 0,
        'sessions': []
    }

    print(f"🚀 INICIANDO RETRENAMIENTO COMPLETO")
    print(f"📋 Ramas a procesar: {results['total']}")

    for i, branch in enumerate(ALL_BRANCHES, 1):
        print(f"\\n📍 Progreso: {i}/{results['total']}")
        print(f"🎯 Procesando: {branch}")

        start_time = time.time()

        # Ejecutar pipeline
            cmd = [
            sys.executable, 'sheily_train/core/training/training_pipeline.py',
            branch, 'corpus_ES', f'models/lora_adapters/retraining/{branch}'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            processing_time = time.time() - start_time

            if result.returncode == 0:
                results['successful'] += 1
                status = 'SUCCESS'
                print("✅ Completado exitosamente"            else:
                results['failed'] += 1
                status = 'FAILED'
                print(f"❌ Fallido: {result.stderr.strip()}")

            # Registrar sesión
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'branch': branch,
                'status': status,
                'processing_time': processing_time,
                'return_code': result.returncode
            }
            results['sessions'].append(session_data)
            log_session(session_data)

        except subprocess.TimeoutExpired:
            results['failed'] += 1
            print("⏰ Timeout"
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'branch': branch,
                'status': 'TIMEOUT',
                'processing_time': time.time() - start_time
            }
            log_session(session_data)

        except Exception as e:
            results['failed'] += 1
            print(f"❌ Error: {e}")
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'branch': branch,
                'status': 'ERROR',
                'error': str(e)
            }
            log_session(session_data)

        # Pausa entre ramas
        if i < results['total']:
            print("⏳ Pausa breve...")
            time.sleep(5)

    return results

def generate_final_report(results):
    """Generar reporte final"""
    print("\\n" + "=" * 60)
    print("📊 REPORTE FINAL DE CORRECCIÓN")
    print("=" * 60)

    print(f"✅ Exitosos: {results['successful']}")
    print(f"❌ Fallidos: {results['failed']}")
    print(f"📈 Tasa de éxito: {results['successful']/results['total']*100:.1f}%")

    # Guardar reporte detallado
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_branches': results['total'],
            'successful': results['successful'],
            'failed': results['failed'],
            'success_rate': results['successful']/results['total']*100
        },
        'sessions': results['sessions']
    }

    with open('audit_2024/reports/final_correction_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\\n✅ Reporte guardado: audit_2024/reports/final_correction_report.json")

    if results['successful']/results['total'] >= 0.95:
        print("🎉 CORRECCIÓN COMPLETA EXITOSA")
        return True
    else:
        print("⚠️ CORRECCIÓN PARCIAL - REQUIERE REVISIÓN")
        return False

def main():
    """Función principal"""
    print("🔥 CORRECCIÓN COMPLETA DEL PROYECTO SHEILY")
    print("=" * 60)

    # Procesar todas las ramas
    results = process_all_branches()

    # Generar reporte final
    success = generate_final_report(results)

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
'''

        with open("audit_2024/complete_retraining.py", "w") as f:
            f.write(script_content)

        self.log_action("SCRIPT DE RETRENAMIENTO COMPREHENSIVO CREADO")

    def run_complete_correction(self):
        """Ejecutar corrección completa"""
        print("🚀 EJECUTANDO CORRECCIÓN COMPLETA")
        print("=" * 60)

        # Ejecutar pasos de corrección
        self.create_organized_structure()
        self.move_corrupted_adapters()
        self.reorganize_training_scripts()
        self.create_validation_system()
        self.create_training_pipeline()
        self.create_comprehensive_retraining_script()

        print("\\n✅ CORRECCIÓN COMPLETA FINALIZADA")
        print("📋 Siguientes pasos:")
        print("  1. Ejecutar: python3 audit_2024/complete_retraining.py")
        print("  2. Verificar resultados en audit_2024/reports/")
        print("  3. Validar adaptadores generados")


def main():
    """Función principal"""
    corrector = ProjectCorrector()

    # Ejecutar corrección completa
    corrector.run_complete_correction()

    return 0


if __name__ == "__main__":
    exit(main())
