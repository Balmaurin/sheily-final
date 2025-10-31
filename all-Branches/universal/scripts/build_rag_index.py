"""
Build RAG Index - Construye el índice RAG universal
====================================================

Crea índices TF-IDF y Sentence Transformers sobre el corpus unificado
para habilitar búsqueda semántica y RAG.

Uso:
    python build_rag_index.py
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
    print("CONSTRUYENDO ÍNDICE RAG UNIVERSAL")
    print("="*70)
    
    # Inicializar sistema universal
    logger.info("🚀 Inicializando Sistema Universal...")
    manager = UniversalManager()
    
    # Verificar que hay datos
    corpus_files = list(manager.corpus_path.glob("*.jsonl"))
    if not corpus_files:
        print("\n❌ ERROR: No hay archivos en el corpus unificado")
        print("   Ejecuta primero: python migrate_from_branch.py <rama>")
        return 1
    
    print(f"\n📚 Corpus actual:")
    total_examples = 0
    for file in corpus_files:
        with open(file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"  - {file.name}: {count} ejemplos")
        total_examples += count
    
    print(f"\n  Total: {total_examples} ejemplos en {len(corpus_files)} archivos")
    
    # Construir índice
    print("\n🔨 Construyendo índice RAG...")
    print("   Esto puede tardar varios minutos...")
    
    try:
        result = manager.build_rag_index()
        
        print("\n" + "="*70)
        print("✅ ÍNDICE RAG UNIVERSAL CREADO")
        print("="*70)
        print(f"Total documentos indexados: {result['total_documents']}")
        print(f"Términos TF-IDF: {result['tfidf_terms']}")
        print(f"Dimensiones embeddings: {result['embedding_dimensions']}")
        print(f"Ubicación: {result['index_path']}")
        print("="*70)
        
        # Probar búsqueda
        print("\n🔍 Probando búsqueda RAG...")
        test_query = "¿Qué es la antropología?"
        results = manager.search_rag(test_query, top_k=3, method="hybrid")
        
        print(f"\nResultados para: '{test_query}'")
        for i, result in enumerate(results, 1):
            doc = result["document"]
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Instrucción: {doc['metadata']['instruction'][:100]}...")
            print(f"   Respuesta: {doc['metadata']['output'][:150]}...")
        
        print("\n" + "="*70)
        print("✅ SISTEMA RAG UNIVERSAL OPERACIONAL")
        print("="*70)
        
        logger.info("Índice RAG creado exitosamente")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR construyendo índice RAG: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
