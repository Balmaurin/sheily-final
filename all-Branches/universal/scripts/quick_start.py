"""
Quick Start - Inicialización rápida del Sistema Universal
==========================================================

Script para inicializar el sistema universal desde cero.
"""

import sys
import logging
from pathlib import Path

# Añadir path del sistema universal
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_manager import UniversalManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*70)
    print("SISTEMA UNIVERSAL SHEILY - INICIALIZACIÓN")
    print("="*70)
    
    logger.info("🚀 Inicializando Sistema Universal Sheily...")
    
    # Inicializar manager
    manager = UniversalManager()
    
    # Inicializar todos los componentes
    status = manager.initialize_all()
    
    # Mostrar estado
    print("\n📊 ESTADO DEL SISTEMA:")
    print(f"  Nombre: {status['system']}")
    print(f"  Versión: {status['version']}")
    print(f"  Timestamp: {status['timestamp']}")
    
    print("\n📚 CORPUS:")
    corpus = status['components']['corpus']
    print(f"  Archivos: {corpus['files']}")
    print(f"  Documentos totales: {corpus['total_documents']}")
    print(f"  Ruta: {corpus['path']}")
    
    print("\n🔍 RAG:")
    rag = status['components']['rag']
    print(f"  Estado: {rag['status']}")
    print(f"  Habilitado: {rag['enabled']}")
    print(f"  Top-K: {rag['top_k']}")
    
    print("\n🤖 MODELO:")
    model = status['components']['model']
    print(f"  Estado: {model['status']}")
    print(f"  Modelo base: {model['base_model']}")
    print(f"  Dispositivo: {model['device']}")
    print(f"  Parámetros entrenables: {model['trainable_params']:,}")
    print(f"  Porcentaje entrenable: {model['trainable_percentage']}")
    
    if 'training' in model and model['training'].get('total_examples', 0) > 0:
        print(f"  Ejemplos entrenados: {model['training']['total_examples']:,}")
        print(f"  Última actualización: {model['training']['last_update']}")
    else:
        print(f"  Estado adaptador: Nuevo (sin entrenar)")
    
    if 'incoming' in status['components']:
        incoming = status['components']['incoming']
        if incoming['files_processed'] > 0:
            print("\n📥 DATOS PROCESADOS:")
            print(f"  Archivos procesados: {incoming['files_processed']}")
            for file in incoming['files']:
                print(f"    - {file}")
    
    print("\n" + "="*70)
    print("✅ SISTEMA UNIVERSAL LISTO PARA USAR")
    print("="*70)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("  1. Migrar datos de antropologia:")
    print("     python scripts\\migrate_from_branch.py antropologia")
    print()
    print("  2. Entrenar el adaptador universal:")
    print("     python scripts\\train_universal.py")
    print()
    print("  3. Añadir nuevo conocimiento:")
    print("     python scripts\\add_knowledge.py <dataset.jsonl>")
    print("="*70)


if __name__ == "__main__":
    main()
