# Notas del Proyecto Real

Este repositorio refleja el estado real del sistema Sheily AI en producción, pero por política no versiona artefactos pesados ni sensibles. Aquí encontrarás qué se excluye y cómo reconstruirlo localmente.

## Qué NO se versiona (por política)

- Modelos y pesos:
  - `models/`, `**/*.safetensors`, `**/*.gguf`, `**/*.pt`, `**/*.bin`, `**/*.onnx`
- Datasets y artefactos generados:
  - `**/*.jsonl` (datasets de entrenamiento), `**/*.npy`, `**/*.npz`, `**/*.pkl`, `**/*.arrow`
- Índices y BBDD locales:
  - `**/*faiss.index`, `*.db`, `*.sqlite*`, `data/**/*.idx`, `data/**/*.index`
- Tokenizers y checkpoints de entrenamiento:
  - `**/tokenizer.*`, `trainer_state.json`, `training_args.bin`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`
- Logs y datos de runtime:
  - `logs/`, `var/central_*/*`

Revisa `.gitignore` para la lista completa de exclusiones.

## Cómo reconstruir artefactos localmente

1. Preparar entorno
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   cp .env.example .env
   # Edita .env con tus valores (DB_*, REDIS_PASSWORD, etc.)
   ```

2. Arrancar servicios (opcional)
   ```powershell
   docker-compose up -d
   ```

3. Reconstruir índices RAG (ejemplo)
   - Procesar corpus de una rama especializada (p.ej., antropología)
   ```powershell
   python start_rag_service.py
   # o
   python quick_start.py
   ```
   - También puedes usar herramientas en `tools/` para verificar corpus e índices.

4. Modelos locales
   - Utiliza modelos GGUF/transformers compatibles colocándolos en `var/central_models/` (montado en Docker)
   - Alternativamente, configura rutas en `.env` para que el sistema los descubra

## Buenas prácticas en entorno real

- Mantén los modelos y datasets fuera del control de versiones
- Documenta siempre comandos para regenerar artefactos
- Asegura `.env` y usa variables de entorno en producción
- Monitoriza con Prometheus/Grafana (ver `docker-compose.yml` y `monitoring/prometheus.yml`)

## Referencias

- Auditoría consolidada: `AUDITORIA_DEFINITIVA_COMPLETA.md`
- Políticas de seguridad: `docs/SECURITY_POLICIES.md`
- Guía de contribución: `CONTRIBUTING.md`
