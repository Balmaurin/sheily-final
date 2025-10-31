# Guía de Contribución

¡Gracias por tu interés en contribuir a Sheily AI! Este documento resume el flujo de trabajo, estándares y checklist para aportar cambios de forma segura y efectiva.

## Requisitos

- Python 3.11+ (recomendado 3.12)
- pip actualizado y virtualenv/venv
- pre-commit (opcional, recomendado)
- Windows PowerShell o Bash

```powershell
# Entorno local recomendado (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Flujo de trabajo

1. Crea una rama desde `main`:
   - feature/<nombre>
   - fix/<nombre>
   - chore/<nombre>

2. Desarrolla con calidad:
   - Añade tests (unit/integration) cuando aplique
   - Ejecuta linters y formateadores

3. Abre un Pull Request (PR):
   - Describe claramente el cambio y el porqué
   - Añade evidencias (logs, capturas, resultados de tests)

## Estilo de commits

Se recomienda seguir Conventional Commits (opcional):

- feat: nueva funcionalidad
- fix: corrección de bug
- docs: documentación
- refactor: refactor sin cambiar comportamiento
- test: añadir o arreglar tests
- chore: mantenimiento (build, deps, etc.)

Ejemplos:

```
feat(rag): añadir endpoint /search con filtro por dominio
fix(training): corregir ruta de carga de dataset
```

## Estándares de código

- PEP8 y tipado estático cuando sea posible
- Docstrings en funciones/módulos relevantes
- Tests para nueva funcionalidad o bugs críticos

```powershell
# Validaciones locales
pytest tests/ -v
flake8 sheily_core/ sheily_train/ tools/
mypy sheily_train/ --ignore-missing-imports
black --check sheily_core/ sheily_train/ tools/
```

## Checklist de PR

- [ ] Descripción clara del cambio
- [ ] Tests pasan en local (`pytest`)
- [ ] Linting y formato sin errores
- [ ] No introduce artefactos pesados (respeta `.gitignore`)
- [ ] Documentación/README actualizados cuando aplique

## Seguridad y datos

- Nunca subas secretos ni credenciales (usa `.env` local)
- Artefactos pesados (modelos, índices, datasets) están excluidos por política
- Si tu cambio requiere artefactos, documenta cómo reconstruirlos localmente

## Preguntas

Abre un issue o discute en el PR. ¡Gracias por contribuir! 🙌
