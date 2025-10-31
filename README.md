# 🤖 Sheily AI - Sistema Enterprise de IA Multidominio

[![Licencia](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Score de Calidad](https://img.shields.io/badge/quality-61.9%2F100-yellow.svg)](enterprise_audit_report.md)

Sistema enterprise de inteligencia artificial con arquitectura modular, soporte para 50+ dominios especializados, RAG (Retrieval-Augmented Generation), y entrenamiento LoRA.

---

## 📊 Estado del Proyecto (Auditoría 31/10/2025)

```
Score General: 61.9/100 - ACEPTABLE (***)

Arquitectura........... 85/100 ✓ [****]
Documentación.......... 80/100 ✓ [****]
Seguridad.............. 80/100 ✓ [****]
Testing................ 70/100 ~ [***]
Calidad de Código...... 50/100 ~ [**]
Dependencias........... 50/100 ~ [**]
Performance............ 40/100 ! [**]
DevOps................. 40/100 ! [**]
```

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.11+
- 16GB RAM (mínimo)
- GPU CUDA opcional (recomendada)

### Instalación

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-org/sheily-ai.git
cd sheily-ai

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 4. Iniciar sistema RAG
python quick_start.py
```

---

## 🏗️ Arquitectura

### Estructura del Proyecto

```
sheily-ai/
├── sheily_core/          # 206 archivos - Core del sistema (3.9 MB)
│   ├── integration/      # Servicios RAG y APIs
│   ├── security/         # Módulos de seguridad
│   ├── data/            # Procesamiento de datos
│   ├── llm_engine/      # Motor de LLM
│   ├── blockchain/      # Integración blockchain
│   └── ...
├── sheily_train/         # 19 archivos - Sistema de entrenamiento
├── tests/               # 23 archivos - Suite de tests
│   ├── unit/           # Tests unitarios
│   ├── integration/    # Tests de integración
│   ├── security/       # Tests de seguridad
│   └── ...
├── all-Branches/        # 50+ dominios especializados
├── tools/              # Herramientas y utilidades
└── docs/               # Documentación

Total: 34,992 archivos Python
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

### Estadísticas de Testing

- ✅ **16 archivos de test** encontrados
- ✅ **8 tests unitarios**
- ✅ **2 tests de integración**
- ✅ **3 tests de seguridad**
- ✅ **1 test e2e**
- ✅ **pytest.ini** configurado

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

⚠️ **Hallazgos:** 3 archivos con patrones potencialmente inseguros detectados (severity: MEDIUM)

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

---

## 📈 Métricas de Calidad

### Análisis de Código

- ✅ **94%** de archivos con docstrings
- ✅ **92%** de archivos con type hints
- ⚠️ **Alta complejidad** - 34,992 archivos Python

### Recomendaciones Prioritarias

#### [ALTA] Calidad de Código
- Mejorar docstrings en archivos sin documentación
- Agregar type hints faltantes
- Considerar modularización adicional

#### [BAJA] DevOps
- Implementar pipeline CI/CD
- Configurar monitoring avanzado (Prometheus/Grafana)
- Automatizar deployment

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
- 📖 [Reporte de Auditoría](enterprise_audit_report.md) - Auditoría empresarial
- 📖 [Configuración de Seguridad](docs/SECURITY_POLICIES.md) - Políticas de seguridad
- 📖 [API Documentation](sheily_core/integration/README.md) - Documentación de APIs

---

## 🤝 Contribuir

### Guías de Contribución

1. Fork el proyecto
2. Crear rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Estándares de Código

- Seguir PEP 8
- Incluir docstrings en todas las funciones
- Agregar type hints
- Escribir tests para nueva funcionalidad
- Cobertura mínima de tests: 70%

---

## 📝 Changelog

### [2.0.0] - 2025-10-31

#### Agregado
- ✅ Sistema de auditoría empresarial completo
- ✅ Unificación de requirements
- ✅ Limpieza de archivos obsoletos (30 archivos removidos)
- ✅ Script de inicio rápido mejorado

#### Mejorado
- 🔧 Estructura de directorios optimizada
- 🔧 Documentación actualizada
- 🔧 Configuración de seguridad reforzada

#### Removido
- ❌ Archivos duplicados de requirements
- ❌ Scripts de un solo uso
- ❌ Reportes obsoletos

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
