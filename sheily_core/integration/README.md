# Sheily Core - Integration Layer
## Servicios y APIs de Integración

### 🎯 Propósito
La capa de integración proporciona servicios API y conectores para unir todos los componentes de Sheily AI en un sistema cohesivo.

### 🔧 Componentes Principales

#### 🌐 Web Chat Server (`web_chat_server.py`)
**API REST completa para chat con RAG**

- **Endpoints**:
  - `GET /health` - Estado del sistema
  - `POST /chat` - Consulta con RAG integrado

- **Características**:
  - Soporte múltiple LLM (Ollama, llama.cpp)
  - RAG opcional por consulta
  - Rate limiting y seguridad
  - Logging estructurado

```python
# Uso básico
from sheily_core.integration.web_chat_server import app

# El servidor se inicia automáticamente con:
# python -m sheily_core.integration.web_chat_server
```

#### 🤖 RAG Service (`rag_service.py`)
**Servicio dedicado de Retrieval Augmented Generation**

- **Arquitectura FAISS**: Búsqueda vectorial ultrarrápida
- **Embeddings multilingües**: Soporte español/inglés nativo
- **API REST completa**:
  - `POST /process_corpus` - Indexar documentos
  - `POST /add_document` - Agregar documento individual
  - `POST /search` - Buscar similitud semántica
  - `GET /health` - Estado del servicio

```python
from sheily_core.integration.rag_service import RAGService

service = RAGService()
await service.initialize()  # Carga modelo de embeddings
```

#### 🔗 RAG Client (`rag_client.py`)
**Cliente HTTP para conectar con RAG Service**

- **Integración transparente**: Funciona como puente HTTP
- **Fallback inteligente**: Maneja desconexiones graceful
- **Cache automático**: Optimiza performance

```python
from sheily_core.integration.rag_client import get_rag_context

# Obtener contexto RAG para una consulta
context, chunks = get_rag_context("¿Qué es la antropología?", max_length=2000)
```

### 🚀 Inicio de Servicios

#### Opción 1: Todo automático (Recomendado)
```bash
python quick_start.py
```
Inicia RAG Service + Chat Server simultáneamente.

#### Opción 2: Servicios individuales
```bash
# Terminal 1: RAG Service
python start_rag_service.py

# Terminal 2: Chat Server
python -m sheily_core.integration.web_chat_server
```

#### Opción 3: Desarrollo manual
```python
# RAG Service standalone
from sheily_core.integration.rag_service import main
import asyncio
asyncio.run(main())

# Chat Server standalone
uvicorn sheily_core.integration.web_chat_server:app --host 0.0.0.0 --port 8000
```

### 📡 APIs Externas Integradas

#### Ollama Client (`ollama_client.py`)
```python
from sheily_core.integration.ollama_client import generate_completion

response = await generate_completion(
    prompt="¿Qué es la IA?",
    model="llama3.2",
    options={"temperature": 0.7}
)
```

#### Llama.cpp Client (`llama_cpp_client.py`)
```python
from sheily_core.integration.llama_cpp_client import generate_completion

response = generate_completion(
    prompt="Explica machine learning",
    model_path="/path/to/model.gguf",
    max_tokens=512
)
```

### 🔧 Configuración

#### Variables de Entorno
```bash
# URLs de servicios
SHEILY_RAG_SERVICE_URL=http://localhost:8001
SHEILY_CHAT_PORT=8000

# LLM Provider
SHEILY_LLM_PROVIDER=ollama  # o llama_cpp

# Modelo llama.cpp
SHEILY_LLAMA_MODEL_PATH=/path/to/model.gguf

# Seguridad
SHEILY_MCP_ENABLED=true
```

#### Configuración Programática
```python
from sheily_core.integration.rag_client import initialize_rag_service

# Inicializar con URL personalizada
success = initialize_rag_service("http://localhost:8001")
if not success:
    print("RAG Service no disponible")
```

### 🧪 Testing de Integración

```bash
# Tests unitarios
pytest tests/unit/ -v

# Tests de integración
pytest tests/integration/ -v

# Tests de API
pytest tests/integration/test_rag_api.py -v

# Tests de seguridad
pytest tests/security/ -v

# Tests de performance
pytest tests/performance/ -v
```

### 📊 Monitoreo y Health Checks

#### Health Check Completo
```bash
# Verificar estado de todos los servicios
curl http://localhost:8000/health  # Chat Service
curl http://localhost:8001/health  # RAG Service
```

#### Métricas de Performance
- **Latencia**: <100ms para búsquedas RAG
- **Throughput**: >100 consultas/minuto
- **Disponibilidad**: 99.9% uptime esperado
- **Memoria**: <500MB para servicios completos

### 🔒 Seguridad Integrada

#### Autenticación
- JWT opcional para APIs
- Rate limiting automático
- Validación de entrada

#### Auditoría
- Logs estructurados completos
- Trazabilidad de consultas
- Monitoreo de seguridad en tiempo real

### 🔄 Escalabilidad

#### Arquitectura de Microservicios
- **Independiente**: Cada servicio puede escalar por separado
- **Stateless**: Los servicios no mantienen estado entre llamadas
- **Load Balancing**: Fácil distribución de carga

#### Optimizaciones
- **Cache inteligente**: Resultados de búsquedas frecuentes
- **Async/Await**: Procesamiento no bloqueante
- **Pooling de conexiones**: Optimización de recursos

### 🚨 Troubleshooting

#### Problema: "RAG Service no disponible"
```bash
# Verificar que está ejecutándose
curl http://localhost:8001/health

# Reiniciar servicio
python start_rag_service.py
```

#### Problema: "Connection refused"
```bash
# Verificar puertos
netstat -ano | findstr :8001
netstat -ano | findstr :8000

# Cambiar puertos si hay conflictos
export SHEILY_RAG_SERVICE_URL=http://localhost:8003
export SHEILY_CHAT_PORT=8004
```

#### Problema: "Timeout en consultas"
```bash
# Verificar recursos del sistema
top  # o Task Manager

# Reducir carga o aumentar timeouts
# en configuración del servicio
```

### 🎯 Mejores Prácticas

1. **Inicio ordenado**: Siempre RAG Service antes que Chat Server
2. **Health checks**: Verificar estado antes de usar APIs
3. **Timeouts apropiados**: Configurar timeouts según carga esperada
4. **Logging**: Monitorear logs para debugging
5. **Backup**: Mantener backups del índice FAISS

### 🔮 Evolución Futura

- **GraphQL**: API más flexible y eficiente
- **WebSockets**: Comunicación en tiempo real
- **Kubernetes**: Orquestación nativa
- **Multi-tenancy**: Aislamiento por usuario/organización
- **API Gateway**: Punto único de entrada centralizado
