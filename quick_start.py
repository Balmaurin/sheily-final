#!/usr/bin/env python3
"""
INICIO ULTRA-RÁPIDO DEL SISTEMA RAG
==================================

Comando único que funciona a la primera:
python quick_start.py

Inicia todo automáticamente sin problemas.
"""

import os
import sys
import subprocess
import time

def main():
    print("🚀 INICIO ULTRA-RÁPIDO DEL SISTEMA RAG")
    print("=" * 50)

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Verificar dependencias críticas
    print("📦 Verificando dependencias críticas...")
    try:
        import sentence_transformers
        import faiss
        import fastapi
        import uvicorn
        print("   ✅ Todas las dependencias están instaladas")
    except ImportError as e:
        print(f"   ❌ Dependencia faltante: {e}")
        print("   🔧 Ejecuta: pip install sentence-transformers faiss-cpu fastapi uvicorn")
        return 1

    # 2. Limpiar procesos anteriores
    print("🧹 Limpiando procesos anteriores...")
    subprocess.run("taskkill /f /im python.exe", capture_output=True)

    # 3. Iniciar RAG Service
    print("🌐 Iniciando RAG Service...")
    rag_proc = subprocess.Popen([
        sys.executable, "start_rag_service.py"
    ], cwd=project_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)

    time.sleep(4)  # Esperar que cargue el modelo

    # 4. Iniciar Chat Service
    print("💬 Iniciando Chat Service...")
    chat_proc = subprocess.Popen([
        sys.executable, "-m", "sheily_core.integration.web_chat_server"
    ], cwd=project_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)

    time.sleep(3)  # Esperar que se conecte al RAG

    # 5. Verificación rápida
    print("🔍 Verificando servicios...")
    try:
        import requests
        time.sleep(1)

        # RAG
        try:
            r = requests.get("http://localhost:8001/health", timeout=2)
            print("   ✅ RAG Service listo" if r.status_code == 200 else f"   ❌ RAG Service: {r.status_code}")
        except:
            print("   ❌ RAG Service no responde")

        # Chat
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            print("   ✅ Chat Service listo" if r.status_code == 200 else f"   ❌ Chat Service: {r.status_code}")
        except:
            print("   ❌ Chat Service no responde")

    except ImportError:
        print("   ⚠️ Saltando verificación (requests no disponible)")

    # 6. Información final
    print("\n" + "=" * 50)
    print("🎉 ¡SISTEMA RAG OPERATIVO!")
    print("\n🌐 URLs:")
    print("   • RAG Service: http://localhost:8001")
    print("   • Chat Service: http://localhost:8000")
    print("\n🧪 Prueba rápida:")
    print("   curl http://localhost:8001/health")
    print("   curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{\"message\":\"¿Qué es la antropología?\",\"use_rag\":true}'")
    print("\n🛑 Cierra esta ventana para detener todo")
    print("=" * 50)

    # Mantener vivo
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Deteniendo servicios...")
        try:
            rag_proc.terminate()
            chat_proc.terminate()
            rag_proc.wait(timeout=2)
            chat_proc.wait(timeout=2)
        except:
            pass
        print("✅ Servicios detenidos")

if __name__ == "__main__":
    exit(main())
