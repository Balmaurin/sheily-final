#!/usr/bin/env python3
"""
SISTEMA DE MONITOREO CONTINUO DEL PROYECTO
"""

import json
import time
from datetime import datetime
from pathlib import Path

import psutil


def get_system_metrics():
    """Obtener métricas del sistema"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "timestamp": datetime.now().isoformat(),
    }


def monitor_project_health():
    """Monitorear salud del proyecto"""
    health_metrics = {
        "timestamp": datetime.now().isoformat(),
        "system": get_system_metrics(),
        "project": {},
    }

    # Verificar componentes críticos
    critical_paths = [
        "models/gguf/llama-3.2.gguf",
        "corpus_ES",
        "sheily_train/core/training/training_pipeline.py",
        "audit_2024/reports",
    ]

    for path in critical_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                size = path_obj.stat().st_size
                health_metrics["project"][path] = {"exists": True, "size_mb": size / 1024 / 1024}
            else:
                items = len(list(path_obj.iterdir()))
                health_metrics["project"][path] = {"exists": True, "items": items}
        else:
            health_metrics["project"][path] = {"exists": False}

    return health_metrics


def log_health_metrics(metrics):
    """Registrar métricas de salud"""
    log_file = Path("audit_2024/logs/health_monitoring.jsonl")
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def main():
    """Función principal de monitoreo"""
    print("🔍 MONITOREO CONTINUO DEL PROYECTO")
    print("=" * 50)

    # Monitorear salud
    health = monitor_project_health()

    # Registrar métricas
    log_health_metrics(health)

    # Mostrar estado
    print(f"⏰ Timestamp: {health['timestamp']}")
    print(f"💻 CPU: {health['system']['cpu_percent']}%")
    print(f"🧠 Memoria: {health['system']['memory_percent']}%")
    print(f"💾 Disco: {health['system']['disk_usage']}%")

    print("\n📋 Estado de componentes críticos:")
    for component, status in health["project"].items():
        if status["exists"]:
            if "size_mb" in status:
                print(f"  ✅ {component}: {status['size_mb']:.1f}MB")
            elif "items" in status:
                print(f"  ✅ {component}: {status['items']} elementos")
        else:
            print(f"  ❌ {component}: NO ENCONTRADO")

    print(f"\n📊 Métricas registradas en: audit_2024/logs/health_monitoring.jsonl")
    return 0


if __name__ == "__main__":
    exit(main())
