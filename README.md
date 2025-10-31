# 🤖 Sheily AI - Sistema Enterprise de IA Multidominio

[![Licencia](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Score de Calidad](https://img.shields.io/badge/quality-80.1%2F100-brightgreen.svg)](AUDITORIA_DEFINITIVA_COMPLETA.md)

Sistema enterprise de inteligencia artificial con arquitectura modular, soporte para 50+ dominios especializados, RAG (Retrieval-Augmented Generation), y entrenamiento LoRA. Este repositorio refleja el proyecto real de producción, con artefactos pesados excluidos por política.

---

## Índice

- [Estado del Proyecto](#-estado-del-proyecto-auditoría-31102025)
- [Inicio Rápido](#-inicio-rápido)
- [Arquitectura](#-arquitectura)
- [Componentes Principales](#-componentes-principales)
- [Testing](#-testing)
- [Seguridad](#-seguridad)
- [Métricas y Recomendaciones](#-métricas-de-calidad)
- [Flujo de Trabajo](#-flujo-de-trabajo)
- [Herramientas](#-herramientas-disponibles)
- [Documentación adicional](#-documentación-adicional)
- [Contribuir](#-contribuir)
- [Changelog](#-changelog)

---

## 📊 Estado del Proyecto (Auditoría 31/10/2025)

```
Score General: 80.1/100 - AVANZADO (****)

Arquitectura........... 85/100 ✓ [****]
Calidad de Código...... 78/100 ✓ [***]
Seguridad.............. 80/100 ✓ [****]
Testing................ 80/100 ✓ [****]
Documentación.......... 80/100 ✓ [****]
Dependencias........... 88/100 ✓ [****]
Rendimiento............ 70/100 ✓ [***]
DevOps................. 80/100 ✓ [****]
```

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.11+
- 16GB RAM (mínimo)
- GPU CUDA opcional (recomendada)

### Instalación (Windows PowerShell)

```bash
# 1. Clonar repositorio
git clone https://github.com/Balmaurin/sheily-final.git
cd sheily-final

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 4. Iniciar sistema RAG
python quick_start.py
```

### Instalación (Linux/Mac)

```bash
# 1. Clonar repositorio
git clone https://github.com/Balmaurin/sheily-final.git
cd sheily-final

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env

# 4. Iniciar sistema RAG
python3 quick_start.py
```

### Variables .env más comunes

Estas variables están documentadas en `.env.example`:

- MODEL_NAME, MODEL_PATH, USE_GPU
- BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, USE_LORA
- API_HOST, API_PORT, API_WORKERS
- LOG_LEVEL, LOG_DIR
- Opcionales: HF_TOKEN, DB_*, SENTRY_DSN, CORS_ORIGINS

---

## 🏗️ Arquitectura

### Estructura del Proyecto

```
sheily-final/
├── sheily_core/          # Núcleo del sistema (206 archivos)
│   ├── integration/      # Servicios RAG y APIs
│   ├── security/         # Módulos de seguridad
│   ├── data/             # Procesamiento de datos e índices
│   ├── llm_engine/       # Motor de LLM
│   ├── blockchain/       # Integración blockchain
│   └── ...
├── sheily_train/         # Sistema de entrenamiento (19 archivos)
├── tests/                # Suite de tests (23 archivos)
│   ├── unit/             # Tests unitarios
│   ├── integration/      # Tests de integración
│   ├── security/         # Tests de seguridad
│   └── ...
├── all-Branches/         # Dominios especializados
├── tools/                # Herramientas y utilidades
├── docs/                 # Documentación
└── var/, data/, logs/    # Datos y artefactos en tiempo de ejecución (excluidos del repo)

Código Python principal: 376 archivos (ver detalles en AUDITORIA_DEFINITIVA_COMPLETA.md)
```

### Patrones Arquitectónicos

- ✅ **Microservicios/Integración** - Servicios independientes y escalables
- ✅ **Seguridad por Diseño** - Módulo de seguridad dedicado
- ✅ **Observabilidad** - Sistema de monitoring integrado

---

## 🔧 Componentes Principales

### 1. Sistema RAG (Retrieval-Augmented Generation)

```python
from sheily_core.integration.rag_service import rag_service

# Inicializar servicio
await rag_service.initialize()

# Procesar corpus
await rag_service.process_corpus("all-Branches", ["antropologia"])

# Buscar documentos relevantes
results = await rag_service.search("¿Qué es la antropología cultural?")
```

### 2. Sistema de Entrenamiento LoRA

```bash
# Entrenar rama específica
python sheily_train/train_branch.py --branch antropologia --lora

# Con Make
make train BRANCH=antropologia
```

### 3. APIs y Servicios

```bash
# Iniciar servicio RAG (Puerto 8002)
python start_rag_service.py

# Iniciar sistema completo
python quick_start.py
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Tests unitarios (rápidos)
pytest tests/unit/ -v

# Tests de integración
pytest tests/integration/ -v -m integration

# Tests de seguridad
pytest tests/security/ -v -m security

# Con cobertura
pytest tests/ --cov=sheily_core --cov-report=html
```

### Cobertura de la suite

- ✅ Estructura completa: unit, integration, security, e2e, performance
- ✅ 20+ archivos de test; 200+ casos totales (ver auditoría)
- ✅ `pytest.ini` configurado

---

## 🔒 Seguridad

### Características de Seguridad

- ✅ `.secrets.baseline` - Detección de secretos
- ✅ `.pre-commit-config.yaml` - Hooks pre-commit
- ✅ `.env.example` - Ejemplo de configuración segura
- ✅ `sheily_core/security/` - Módulo de seguridad dedicado
- ✅ `.env` en `.gitignore`

### Análisis de Seguridad

```bash
# Ejecutar análisis de seguridad
python -m bandit -r sheily_core -f json -o security_report.json

# Verificar secretos
detect-secrets scan
```

Nota: artefactos sensibles y pesados están excluidos del repositorio por política (.gitignore). Revisa NOTAS_DEL_PROYECTO_REAL.md para instrucciones de reconstrucción local.

---

## 📦 Dependencias

### Dependencias Principales (56 paquetes)

```bash
# Core
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
accelerate>=0.24.0

# RAG System
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2

# Web & API
fastapi>=0.100.0
uvicorn>=0.23.0

# Testing & Quality
pytest>=7.4.0
black>=23.12.0
flake8>=7.0.0
mypy>=1.8.0
bandit>=1.7.6
```

Ver archivo completo: [`requirements.txt`](requirements.txt)

---

## 🐳 Docker

### Uso con Docker

```bash
# Construir imagen
docker build -t sheily-ai:latest .

# Ejecutar con docker-compose
docker-compose up -d

# Verificar servicios
docker-compose ps
```

### Servicios Disponibles

- `sheily-ai` - Servicio principal (Puerto 8000)
- `redis` - Caché (Puerto 6379)
- `postgres` - Base de datos (Puerto 5432)
- `prometheus` - Métricas (Puerto 9090)
- `grafana` - Visualización (Puerto 3000)

### Volúmenes y datos persistentes

El `docker-compose.yml` monta volúmenes locales para no perder datos ni modelos:

- `./var/central_models` → `/app/var/central_models`
- `./var/central_logs` → `/app/var/central_logs`
- `./var/central_cache` → `/app/var/central_cache`
- `./logs` → `/app/logs`

Consulta `docs/NOTAS_DEL_PROYECTO_REAL.md` para conocer qué artefactos están excluidos del repo y cómo reconstruirlos.

---

## 📈 Métricas de Calidad

### Recomendaciones Prioritarias

Consulta el informe: [AUDITORIA_DEFINITIVA_COMPLETA.md](AUDITORIA_DEFINITIVA_COMPLETA.md)

---

## 🔄 Flujo de Trabajo

### 1. Desarrollo

```bash
# Crear rama
git checkout -b feature/nueva-funcionalidad

# Instalar dependencias de desarrollo
pip install -r requirements.txt

# Ejecutar pre-commit hooks
pre-commit install
```

### 2. Testing

```bash
# Ejecutar tests antes de commit
pytest tests/ -v

# Verificar calidad
black sheily_core/
flake8 sheily_core/
mypy sheily_core/
```

### 3. Deployment

```bash
# Construir y desplegar
docker-compose up -d --build

# Verificar salud del sistema
curl http://localhost:8000/health
```

### Makefile útil

```bash
make help          # Ver comandos disponibles
make install       # Instalar dependencias
make test          # Ejecutar tests
make lint          # Lint (flake8/mypy)
make format        # Formato (black)
make audit         # Auditoría completa
```

---

## 🛠️ Herramientas Disponibles

### Auditoría

```bash
# Auditoría empresarial completa
python enterprise_audit.py

# Ver reportes generados
cat enterprise_audit_report.md
```

### Análisis de Branches

```bash
# Listar todas las branches
make list-branches

# Validar branches
make check-branches
```

### Utilidades

```bash
# Limpieza
make clean

# Formateo de código
make format

# Linting
make lint
```

---

## 📚 Documentación Adicional

- 📖 [Tests README](tests/README.md) - Guía completa de testing
- 📖 [Auditoría Definitiva](AUDITORIA_DEFINITIVA_COMPLETA.md) - Informe consolidado (78.2/100)
- 📖 [Notas del Proyecto Real](docs/NOTAS_DEL_PROYECTO_REAL.md) - Exclusiones y reconstrucción local
- 📖 [Configuración de Seguridad](docs/SECURITY_POLICIES.md) - Políticas de seguridad
- 📖 [API Documentation](sheily_core/integration/README.md) - Documentación de APIs

---

## ❗ Troubleshooting

- CUDA no detectada: establece `USE_GPU=false` en `.env` o instala drivers/CUDA adecuados.
- Puerto en uso (8000): cambia `API_PORT` en `.env` o en `docker-compose.yml`.
- Falta FAISS o modelos: revisa `docs/NOTAS_DEL_PROYECTO_REAL.md` para reconstruir índices y ubicar modelos en `var/central_models/`.
- Token HuggingFace: configura `HF_TOKEN` en `.env` si usas modelos privados.
- Rutas en Windows: usa PowerShell y el activador `.\.venv\Scripts\Activate.ps1`.

---

## 🤝 Contribuir

Lee la guía completa en [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📝 Changelog

Consulta [CHANGELOG.md](CHANGELOG.md) para el historial completo de cambios.

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## 👥 Equipo

**Sheily AI Research Team**

- 📧 Email: contact@sheily-ai.dev
- 🌐 Website: https://sheily-ai.dev
- 💬 Discord: [Únete a la comunidad](https://discord.gg/sheily-ai)

---

## 🙏 Agradecimientos

- Comunidad de Python y PyTorch
- Equipo de Transformers (Hugging Face)
- Contribuidores del proyecto
- Usuarios y testers

---

## 🔗 Links Útiles

- [Documentación Oficial](https://docs.sheily-ai.dev)
- [Ejemplos y Tutoriales](https://github.com/sheily-ai/examples)
- [Blog](https://blog.sheily-ai.dev)
- [Roadmap](https://github.com/sheily-ai/roadmap)

---

<div align="center">

**⭐ Si te gusta el proyecto, por favor dale una estrella ⭐**

Hecho con ❤️ por el equipo de Sheily AI

</div>
