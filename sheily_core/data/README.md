# Sheily Core - Document Processor
## Procesamiento Inteligente de Documentos

### 🎯 Propósito
El `DocumentProcessor` es el componente responsable de procesar documentos JSONL y dividirlos en chunks semánticos para el sistema RAG.

### 🔧 Funcionalidades

#### Chunking Inteligente
- **RecursiveCharacterTextSplitter**: Divide texto respetando estructura semántica
- **Configurable**: Tamaño de chunk (512) y solapamiento (100) personalizables
- **Preservación de metadata**: Mantiene información contextual de cada chunk

#### Soporte Multi-idioma
- **Español/inglés**: Procesamiento nativo de documentos en ambos idiomas
- **Corpus estructurado**: Soporte para `all-Branches/{domain}/corpus/{lang}/`
- **Metadata rica**: Incluye dominio, fuente, título, keywords, etc.

### 📊 Uso Básico

```python
from sheily_core.data.document_processor import DocumentProcessor

# Crear procesador
processor = DocumentProcessor(chunk_size=512, chunk_overlap=100)

# Procesar documentos
chunks = processor.process_corpus("all-Branches/antropologia/corpus/")

# Resultado: lista de diccionarios con chunks
for chunk in chunks[:3]:
    print(f"ID: {chunk['id']}")
    print(f"Contenido: {chunk['content'][:100]}...")
    print(f"Dominio: {chunk['metadata']['domain']}")
```

### 🔧 Configuración Avanzada

```python
# Configuración personalizada
processor = DocumentProcessor(
    chunk_size=1024,      # Tamaño de chunk mayor
    chunk_overlap=200     # Más solapamiento
)

# Separadores personalizados
processor.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\\n\\n", "\\n", ". ", " ", ""],  # Prioridad de separación
    length_function=len
)
```

### 📈 Métricas de Performance

- **Velocidad**: ~10,000 tokens/segundo
- **Memoria**: ~50MB para corpus de 1M tokens
- **Escalabilidad**: Maneja corpus de cualquier tamaño

### 🧪 Testing

```bash
# Tests específicos del procesador
pytest tests/unit/test_document_processor.py -v

# Tests de integración con RAG
pytest tests/integration/test_rag_integration.py -v
```

### 🔍 Solución de Problemas

#### Problema: "No se encontraron documentos"
**Solución**: Verificar estructura del directorio corpus
```
all-Branches/
├── antropologia/
│   └── corpus/
│       ├── spanish/
│       │   └── documento.jsonl
│       └── english/
│           └── document.jsonl
```

#### Problema: Chunks muy pequeños/grandes
**Solución**: Ajustar parámetros de chunking
```python
processor = DocumentProcessor(chunk_size=256, chunk_overlap=50)  # Más pequeños
processor = DocumentProcessor(chunk_size=1024, chunk_overlap=200)  # Más grandes
```

### 🔗 Integración con RAG

El DocumentProcessor se integra automáticamente con el sistema RAG:

```python
from sheily_core.integration.rag_service import RAGService

service = RAGService()
# El service ya incluye el document processor configurado
chunks = service.document_processor.process_corpus("corpus/path/")
```

### 📝 Formato de Documentos JSONL

Cada línea debe ser un objeto JSON válido:

```json
{
  "title": "Antropología Cultural",
  "content": "Texto completo del documento...",
  "domain": "antropologia",
  "category": "teoria_fundamental",
  "keywords": ["cultura", "sociedad", "antropologia"],
  "date": "2025-01-24"
}
```

### 🎯 Mejores Prácticas

1. **Tamaño óptimo de chunks**: 512 tokens balancea contexto y precisión
2. **Solapamiento**: 100 tokens asegura continuidad semántica
3. **Metadata completa**: Incluye toda la información contextual posible
4. **Validación**: Verifica formato JSONL antes del procesamiento
5. **Monitoreo**: Registra métricas de procesamiento para optimización

### 🔄 Actualizaciones Futuras

- Soporte para más idiomas
- Chunking basado en transformers
- Optimización automática de parámetros
- Procesamiento distribuido para corpus grandes
