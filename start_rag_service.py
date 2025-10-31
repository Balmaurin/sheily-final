#!/usr/bin/env python3
"""
Launcher para RAG Service - Ejecutar desde directorio raíz
"""

import os
import sys

# Asegurar que estamos en el directorio correcto
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configurar PYTHONPATH
os.environ["PYTHONPATH"] = script_dir

import asyncio

# Importar y ejecutar
from sheily_core.integration.rag_service import main

if __name__ == "__main__":
    print("🚀 Iniciando RAG Service desde launcher...")
    asyncio.run(main())
