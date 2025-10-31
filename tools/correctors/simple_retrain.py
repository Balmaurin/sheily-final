#!/usr/bin/env python3
"""
SCRIPT DE RETRENAMIENTO SIMPLIFICADO - FASE 2
Versión simplificada para comenzar el reentrenamiento
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ramas prioritarias para reentrenar
PRIORITY_BRANCHES = ["antropologia", "economia", "psicologia", "historia", "quimica"]


def validate_environment():
    """Validar entorno básico"""
    print("🔍 VALIDANDO ENTORNO...")

    # Verificar modelo base
    model_path = Path("models/gguf/llama-3.2.gguf")
    if not model_path.exists():
        print(f"❌ Modelo base no encontrado: {model_path}")
        return False

    print(f"✅ Modelo base: {model_path.stat().st_size/1024/1024/1024:.1f}GB")

    # Verificar corpus
    corpus_path = Path("corpus_ES")
    if not corpus_path.exists():
        print(f"❌ Corpus no encontrado: {corpus_path}")
        return False

    print("✅ Entorno validado")
    return True


def train_branch(branch_name):
    """Entrenar una rama específica"""
    print(f"\n🚀 ENTRENANDO: {branch_name}")

    try:
        # Crear directorio de salida
        output_dir = Path(f"models/lora_adapters/retraining/{branch_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Buscar datos
        corpus_files = list(Path(f"corpus_ES/{branch_name}").glob("**/*.jsonl"))
        if not corpus_files:
            print(f"❌ No hay datos para {branch_name}")
            return False

        # Usar archivo más grande
        corpus_file = max(corpus_files, key=lambda x: x.stat().st_size)
        print(f"📂 Datos: {corpus_file}")
        print(f"📊 Tamaño: {corpus_file.stat().st_size/1024:.1f}KB")

        # Comando de entrenamiento
        cmd = [
            sys.executable,
            "sheily_train/train_lora.py",
            "--data",
            str(corpus_file),
            "--out",
            str(output_dir),
            "--model",
            "models/gguf/llama-3.2.gguf",
        ]

        print(f"⚡ Ejecutando entrenamiento...")

        # Ejecutar entrenamiento
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minutos timeout

        if result.returncode == 0:
            print("✅ Entrenamiento exitoso")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout en {branch_name}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Función principal"""
    print("🔥 FASE 2: RETRENAMIENTO PRIORITARIO")
    print("=" * 50)
    print(f"📋 Ramas a procesar: {len(PRIORITY_BRANCHES)}")

    # Validar entorno
    if not validate_environment():
        print("❌ Entorno inválido")
        return 1

    # Procesar ramas
    results = {"success": 0, "failed": 0}

    for i, branch in enumerate(PRIORITY_BRANCHES, 1):
        print(f"\n📍 Progreso: {i}/{len(PRIORITY_BRANCHES)}")
        if train_branch(branch):
            results["success"] += 1
        else:
            results["failed"] += 1

        # Pausa entre entrenamientos
        if i < len(PRIORITY_BRANCHES):
            print("⏳ Pausa...")
            time.sleep(3)

    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESULTADOS")
    print("=" * 50)
    print(f"✅ Exitosos: {results['success']}")
    print(f"❌ Fallidos: {results['failed']}")
    print(f"📈 Tasa de éxito: {results['success']/len(PRIORITY_BRANCHES)*100:.1f}%")

    return 0 if results["success"] > 0 else 1


if __name__ == "__main__":
    exit(main())
