# Sistema Universal - Scripts

Scripts para gestionar el Sistema Universal Sheily.

## 📜 Scripts Disponibles

### `quick_start.py`
Inicializa y muestra el estado del sistema.

```powershell
python quick_start.py
```

### `add_knowledge.py`
Añade cualquier dataset al corpus unificado.

```powershell
python add_knowledge.py <dataset.jsonl> [--auto-train]
```

**Ejemplos:**
```powershell
# Añadir dataset simple
python add_knowledge.py nuevos_datos.jsonl

# Añadir y entrenar automáticamente
python add_knowledge.py datos.jsonl --auto-train

# Desde otra rama
python add_knowledge.py ..\..\antropologia\training\premium_dataset.jsonl
```

### `train_universal.py`
Entrena el adaptador universal con todo el corpus.

```powershell
python train_universal.py [opciones]
```

**Opciones:**
- `--batch-size`: Tamaño del batch (default: 2)
- `--epochs`: Número de épocas (default: 3)
- `--learning-rate`: Learning rate (default: 2e-4)

**Ejemplos:**
```powershell
# Entrenamiento estándar
python train_universal.py

# Entrenamiento rápido
python train_universal.py --epochs 1

# Entrenamiento intensivo
python train_universal.py --epochs 5 --learning-rate 1e-4
```

### `build_rag_index.py`
Construye el índice RAG universal con TF-IDF y Sentence Transformers.

```powershell
python build_rag_index.py
```

**Crea:**
- Índice TF-IDF para búsqueda por palabras clave
- Embeddings con Sentence Transformers para búsqueda semántica
- Índice híbrido combinando ambos métodos

**Cuándo usar:**
- Después de migrar datos al corpus
- Después de añadir nuevo conocimiento
- Para habilitar búsqueda RAG

### `migrate_from_branch.py`
Migra datos desde una rama existente al sistema universal.

```powershell
python migrate_from_branch.py <branch_name> [opciones]
```

**Opciones:**
- `--skip-training`: No migrar datasets de training/
- `--skip-corpus`: No migrar corpus/spanish/

**Ejemplos:**
```powershell
# Migrar todo de antropologia
python migrate_from_branch.py antropologia

# Solo migrar corpus (sin training)
python migrate_from_branch.py astronomia --skip-training

# Solo migrar training (sin corpus)
python migrate_from_branch.py biologia --skip-corpus
```

## 🔄 Workflow Típico

### Primer Uso

```powershell
# 1. Inicializar sistema
python quick_start.py

# 2. Migrar datos de antropologia
python migrate_from_branch.py antropologia

# 3. Construir índice RAG
python build_rag_index.py

# 4. Entrenar adaptador
python train_universal.py
```

### Añadir Nuevo Conocimiento

```powershell
# 1. Añadir dataset
python add_knowledge.py nuevos_datos.jsonl

# 2. Reconstruir índice RAG
python build_rag_index.py

# 3. Re-entrenar
python train_universal.py
```

### Migrar Múltiples Ramas

```powershell
# Migrar varias ramas al sistema universal
python migrate_from_branch.py antropologia
python migrate_from_branch.py astronomia
python migrate_from_branch.py biologia

# Entrenar una sola vez con TODO
python train_universal.py
```

## 📊 Verificar Estado

```powershell
# Ver estado completo del sistema
python quick_start.py

# Ver solo información del manager
cd ..
python universal_manager.py
```

## 🐛 Troubleshooting

### Error: "No se encuentra system_config.json"
**Solución:** Ejecuta los scripts desde `all-Branches/universal/scripts/`

### Error: "Rama no encontrada"
**Solución:** Verifica que la rama existe en `all-Branches/<nombre>/`

### Error: "No hay archivos JSONL"
**Solución:** El corpus está vacío. Migra datos primero con `migrate_from_branch.py`

### Error de GPU/DirectML
**Solución:** El sistema detecta automáticamente GPU. Si falla, usará CPU.

---

**Sistema Universal Sheily v1.0**
