# Sheily AI - Dockerfile Multi-Stage
# ===================================

# Stage 1: Builder
FROM python:3.11-slim as builder

LABEL maintainer="Sheily AI Team"
LABEL description="Sheily AI - Sistema Multidominio de IA"

# Variables de entorno para build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /build

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias Python en un virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Variables de entorno para runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SHEILY_ENV=production

# Instalar solo dependencias runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r sheily && \
    useradd -r -g sheily -u 1000 sheily && \
    mkdir -p /app /app/logs /app/data /app/var && \
    chown -R sheily:sheily /app

# Copiar virtual env desde builder
COPY --from=builder /opt/venv /opt/venv

# Cambiar a directorio de trabajo
WORKDIR /app

# Copiar código de la aplicación
COPY --chown=sheily:sheily . .

# Cambiar a usuario no-root
USER sheily

# Crear directorios necesarios
RUN mkdir -p \
    var/central_models \
    var/central_logs \
    var/central_cache \
    logs

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto (apunta a la app real de FastAPI)
CMD ["python", "-m", "uvicorn", "sheily_core.core.app:app", "--host", "0.0.0.0", "--port", "8000"]
