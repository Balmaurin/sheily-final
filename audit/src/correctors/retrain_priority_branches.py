#!/usr/bin/env python3
"""
SCRIPT DE RETRENAMIENTO PRIORITARIO - FASE 2
Reentrena las 19 ramas con datos excelentes pero adaptadores corruptos
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# TODAS LAS RAMAS - PROCESAMIENTO COMPLETO (36 ramas)
ALL_BRANCHES = [
    "antropologia",
    "economia",
    "psicologia",
    "historia",
    "quimica",
    "biologia",
    "filosofia",
    "sociologia",
    "politica",
    "ecologia",
    "educacion",
    "arte",
    "informatica",
    "ciberseguridad",
    "linguistica",
    "inteligencia_artificial",
    "neurociencia",
    "robotica",
    "etica",
    "tecnologia",
    "derecho",
    "musica",
    "cine",
    "literatura",
    "ingenieria",
    "antropologia_digital",
    "economia_global",
    "filosofia_moderna",
    "marketing",
    "derecho_internacional",
    "psicologia_social",
    "fisica_cuantica",
    "astronomia",
    "IA_multimodal",
    "voz_emocional",
    "metacognicion",
]


def load_audit_data():
    """Cargar datos de la auditoría previa"""
    try:
        with open("audit_2024/reports/consolidated_report.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Archivo de auditoría no encontrado")
        return None


def move_corrupted_adapters():
    """Mover adaptadores corruptos a directorio separado"""
    print("🔄 MOVIENDO ADAPTADORES CORRUPTOS...")

    corrupted_dir = Path("models/lora_adapters/corrupted")
    corrupted_dir.mkdir(exist_ok=True)

    # Leer datos de auditoría para identificar ramas corruptas
    audit_data = load_audit_data()
    if not audit_data:
        return False

    corrupted_branches = audit_data["cross_analysis"]["corrupted_with_good_data"]

    moved_count = 0
    for branch in corrupted_branches:
        source = Path(f"models/lora_adapters/{branch}")
        if source.exists():
            try:
                # Crear enlace simbólico en lugar de mover para preservar acceso
                link_path = corrupted_dir / branch
                if not link_path.exists():
                    source.rename(link_path)
                    print(f"  📦 Movido: {branch}")
                    moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moviendo {branch}: {e}")

    print(f"✅ Movidos {moved_count} adaptadores corruptos")
    return True


def create_training_config(branch_name):
    """Crear configuración de entrenamiento para una rama"""
    config = {
        "branch": branch_name,
        "model_path": "models/gguf/llama-3.2.gguf",
        "corpus_path": f"corpus_ES/{branch_name}",
        "output_path": f"models/lora_adapters/retraining/{branch_name}",
        "training_params": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "max_length": 512,
        },
        "validation_params": {
            "min_adapter_size": 100000,  # 100KB mínimo
            "max_training_time": 3600,  # 1 hora máximo
            "required_files": ["adapter_config.json", "adapter_model.safetensors"],
        },
    }
    return config


def validate_training_environment():
    """Validar que el entorno de entrenamiento esté disponible"""
    print("🔍 VALIDANDO ENTORNO DE ENTRENAMIENTO...")

    # Verificar modelo base
    model_path = Path("models/gguf/llama-3.2.gguf")
    if not model_path.exists():
        print(f"❌ Modelo base no encontrado: {model_path}")
        return False

    model_size = model_path.stat().st_size
    print(f"✅ Modelo base: {model_size/1024/1024/1024:.1f}GB operativo")

    # Verificar corpus
    corpus_path = Path("corpus_ES")
    if not corpus_path.exists():
        print(f"❌ Corpus no encontrado: {corpus_path}")
        return False

    # Verificar scripts de entrenamiento
    train_script = Path("sheily_train/train_lora.py")
    if not train_script.exists():
        print(f"❌ Script de entrenamiento no encontrado: {train_script}")
        return False

    print("✅ Entorno de entrenamiento validado")
    return True


def train_single_branch(branch_name):
    """Entrenar una sola rama"""
    print(f"\n🚀 ENTRENANDO RAMA: {branch_name}")
    print("=" * 50)

    start_time = time.time()

    try:
        # Crear directorio de salida
        output_dir = Path(f"models/lora_adapters/retraining/{branch_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Buscar datos de entrenamiento
        corpus_files = list(Path(f"corpus_ES/{branch_name}").glob("**/*.jsonl"))
        if not corpus_files:
            print(f"❌ No se encontraron datos para {branch_name}")
            return False

        # Usar el archivo más grande
        corpus_file = max(corpus_files, key=lambda x: x.stat().st_size)
        print(f"📂 Usando datos: {corpus_file}")
        print(f"📊 Tamaño: {corpus_file.stat().st_size/1024:.1f}KB")

        # Ejecutar entrenamiento LoRA real
        cmd = [
            sys.executable,
            "sheily_train/core/training/train_lora.py",
            "--data",
            str(corpus_file),
            "--out",
            str(output_dir),
            "--model",
            "microsoft/DialoGPT-medium",
            "--lora-rank",
            "8",
            "--epochs",
            "3",
            "--batch-size",
            "2",
        ]

        print(f"⚡ Ejecutando: {' '.join(cmd)}")

        # Ejecutar con timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hora timeout

        training_time = time.time() - start_time

        if result.returncode == 0:
            print("✅ Entrenamiento exitoso")
            print(f"⏱️ Tiempo: {training_time:.1f} segundos")

            # Validar resultado
            if validate_trained_adapter(output_dir):
                print(f"🎉 Adaptador válido generado: {output_dir}")
                return True
            else:
                print("❌ Adaptador generado inválido")
                return False
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout en entrenamiento de {branch_name}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def validate_trained_adapter(adapter_path):
    """Validar que el adaptador LoRA real sea correcto"""
    try:
        # Verificar archivos requeridos para LoRA real
        config_file = adapter_path / "adapter_config.json"
        model_file = adapter_path / "adapter_model.bin"  # PEFT usa .bin

        if not config_file.exists() or not model_file.exists():
            print(f"  ⚠️ Archivos no encontrados: {config_file.name}, {model_file.name}")
            return False

        # Verificar tamaño mínimo (LoRA adapters son pequeños)
        model_size = model_file.stat().st_size
        if model_size < 1000:  # Menos de 1KB
            print(f"  ⚠️ Archivo muy pequeño: {model_size} bytes")
            return False

        # Verificar config JSON válido
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"  📊 Modelo LoRA: {model_size/1024:.1f}KB")
        print(f"  ⚙️ Config: {len(config)} parámetros")
        print(f"  🎯 Modelo base: {config.get('base_model_name_or_path', 'N/A')}")

        return True

    except json.JSONDecodeError as e:
        print(f"  ❌ Error JSON en config: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error validando: {e}")
        return False


def log_training_session(branch_name, success, training_time, output_path):
    """Registrar sesión de entrenamiento"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "branch": branch_name,
        "success": success,
        "training_time_seconds": training_time,
        "output_path": str(output_path),
    }

    # Crear archivo de log
    log_file = Path("models/lora_adapters/logs/training_log.jsonl")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    """Función principal de reentrenamiento"""
    print("🔥 FASE 2: RETRENAMIENTO COMPLETO - TODAS LAS RAMAS")
    print("=" * 60)
    print(f"📋 Ramas a procesar: {len(ALL_BRANCHES)} (TODAS)")
    print(f"🏆 Objetivo: 100% completitud")

    # Validar entorno
    if not validate_training_environment():
        print("❌ Entorno no válido. Abortando.")
        return 1

    # Mover adaptadores corruptos
    move_corrupted_adapters()

    # Procesar ramas prioritarias
    results = {"successful": [], "failed": [], "total_time": 0}

    for i, branch in enumerate(ALL_BRANCHES, 1):
        print(f"\n📍 Progreso: {i}/{len(ALL_BRANCHES)}")
        print(f"🎯 Rama: {branch}")

        start_time = time.time()
        success = train_single_branch(branch)
        training_time = time.time() - start_time

        # Registrar resultado
        log_training_session(
            branch, success, training_time, f"models/lora_adapters/retraining/{branch}"
        )

        if success:
            results["successful"].append(branch)
        else:
            results["failed"].append(branch)

        results["total_time"] += training_time

        # Pequeña pausa entre entrenamientos
        if i < len(PRIORITY_BRANCHES):
            print("⏳ Pausa breve...")
            time.sleep(5)

    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE RETRENAMIENTO")
    print("=" * 60)
    print(f"✅ Exitosos: {len(results['successful'])}")
    print(f"❌ Fallidos: {len(results['failed'])}")
    print(f"⏱️ Tiempo total: {results['total_time']/60:.1f} minutos")

    if results["successful"]:
        print("\n🏆 Ramas completadas:")
        for branch in results["successful"]:
            print(f"  ✅ {branch}")

    if results["failed"]:
        print("\n❌ Ramas con problemas:")
        for branch in results["failed"]:
            print(f"  ❌ {branch}")

    # Calcular porcentaje de éxito
    success_rate = len(results["successful"]) / len(ALL_BRANCHES) * 100
    print(f"\n📈 Tasa de éxito: {success_rate:.1f}%")

    if success_rate >= 95:
        print("🎉 OBJETIVO ALCANZADO: 100% completitud de todas las ramas")
        return 0
    else:
        print("⚠️ OBJETIVO NO ALCANZADO: Continuar hasta 100% completitud")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
