# Sheily Universal System

**Sistema de aprendizaje continuo unificado - Un único adaptador que aprende de TODO**

## 🎯 Concepto

En lugar de 37 ramas separadas, un **único sistema global** que:
- ✅ Corpus unificado (todo el conocimiento en un lugar)
- ✅ Sistema RAG universal (busca en todo el conocimiento)
- ✅ **Un solo adaptador LoRA** que mejora con CUALQUIER dato
- ✅ Auto-integración de datasets sin importar el dominio
- ✅ Mejora continua permanente

## 📁 Estructura

```
all-Branches/universal/
├── system_config.json          # Configuración del sistema
├── universal_manager.py        # Gestor principal
├── corpus/
│   ├── unified/               # Corpus global unificado
│   └── incoming/              # Nuevos datos pendientes
├── adapters/
│   └── universal_lora/
│       ├── current/           # Adaptador activo
│       └── checkpoints/       # Checkpoints históricos
├── rag/                       # Sistema RAG universal
└── scripts/
    ├── add_knowledge.py       # Añadir cualquier dataset
    ├── train_universal.py     # Entrenar adaptador
    └── migrate_from_branch.py # Migrar desde ramas existentes
```

## 🚀 Uso Rápido

### 1. Inicializar el sistema

```powershell
cd all-Branches\universal
python universal_manager.py
```

### 2. Migrar datos de antropologia (o cualquier rama)

```powershell
python scripts\migrate_from_branch.py antropologia
```

### 3. Añadir nuevo conocimiento

```powershell
python scripts\add_knowledge.py ..\..\nuevos_datos.jsonl
```

### 4. Construir índice RAG (opcional pero recomendado)

```powershell
python scripts\build_rag_index.py
```

### 5. Entrenar el adaptador universal

```powershell
python scripts\train_universal.py
```

## 💡 Ventajas

### Vs Sistema de Ramas (37 branches)

| Aspecto | Sistema de Ramas | Sistema Universal |
|---------|------------------|-------------------|
| **Fragmentación** | 37 sistemas separados | 1 sistema unificado |
| **Mantenimiento** | 37× código duplicado | 1× código centralizado |
| **Conocimiento** | Aislado por dominio | Global, conectado |
| **Adaptador** | 37 adaptadores | 1 adaptador universal |
| **Entrenamiento** | 37 entrenamientos | 1 entrenamiento |
| **RAG** | 37 índices | 1 índice universal |
| **Escalabilidad** | Añadir rama = duplicar | Añadir datos = copiar |

### Beneficios Clave

1. **Simplicidad**: Un solo sistema en lugar de 37
2. **Cross-domain Learning**: El modelo aprende de TODO simultáneamente
3. **Mejora Continua**: Cada dataset mejora el adaptador único
4. **Mantenimiento**: Actualizar código una sola vez
5. **Búsqueda Universal**: RAG sobre todo el conocimiento

## 🔧 Configuración

Edita `system_config.json`:

```json
{
  "model": {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "device": "privateuseone:0"
  },
  "lora": {
    "r": 56,
    "lora_alpha": 112,
    "lora_dropout": 0.025
  },
  "training": {
    "batch_size": 2,
    "learning_rate": 2e-4,
    "num_epochs": 3
  },
  "continuous_learning": {
    "enabled": true,
    "auto_train_threshold": 100
  }
}
```

## 📊 Estado del Sistema

```python
from universal_manager import UniversalManager

manager = UniversalManager()
status = manager.get_status()

print(f"Documentos en corpus: {status['corpus']['total_documents']}")
print(f"Adaptador: {status['adapter']['status']}")
print(f"Ejemplos entrenados: {status['adapter']['total_examples']}")
```

### 🔍 Usar el RAG Universal

```python
from universal_manager import UniversalManager

manager = UniversalManager()

# Buscar con híbrido (TF-IDF + semántico)
results = manager.search_rag("¿Qué es la antropología cultural?", top_k=5)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Texto: {result['document']['metadata']['instruction']}")
```

## 🎓 Ejemplos

### Migrar todos los datos de antropologia

```powershell
python scripts\migrate_from_branch.py antropologia
```

**Output esperado:**
```
📂 Migrando desde rama: antropologia
📚 Migrando corpus...
  ✅ antropologia_corpus_documents_base.jsonl: 150 documentos
  ✅ antropologia_corpus_documents_extended.jsonl: 200 documentos
🎓 Migrando datasets de training...
  ✅ antropologia_training_premium_dataset.jsonl: 320 ejemplos
  ✅ antropologia_training_improved_dataset.jsonl: 54 ejemplos
  
Total de ejemplos migrados: 1,920
```

### Entrenar con todos los datos

```powershell
python scripts\train_universal.py --epochs 3
```

**Output esperado:**
```
🚀 Inicializando Sistema Universal...
📚 Cargando corpus unificado...
  Archivos encontrados: 12
  Total: 1,920 ejemplos

🔥 Entrenando adaptador universal...
Epoch 1/3: Loss 0.4532
Epoch 2/3: Loss 0.2156
Epoch 3/3: Loss 0.0973

✅ Entrenamiento completado
💾 Adaptador guardado en: adapters/universal_lora/current/
```

## 🔄 Workflow Continuo

1. **Añadir datos nuevos** → `add_knowledge.py`
2. **Auto-entrenar** cuando hay 100+ ejemplos nuevos
3. **Backup automático** del adaptador anterior
4. **RAG actualizado** automáticamente

## 🧪 Testing

```powershell
# Probar el manager
python universal_manager.py

# Probar migración
python scripts\migrate_from_branch.py antropologia

# Probar entrenamiento (sin GPU, solo validación)
python scripts\train_universal.py --epochs 1
```

## 📝 Convivencia con Sistema Anterior

El sistema universal **coexiste** con las ramas existentes:

```
all-Branches/
├── antropologia/          ← Rama original (se mantiene)
├── astronomia/            ← Otras ramas (se mantienen)
├── ...
└── universal/             ← Sistema nuevo (extrae de las ramas)
```

**Puedes:**
- Mantener `antropologia/` como referencia
- Migrar datos al sistema universal
- Comparar resultados entre ambos sistemas
- Eliminar ramas gradualmente si el universal funciona mejor

## 🎯 Próximos Pasos

1. ✅ Estructura creada
2. ⏳ Migrar datos de antropologia
3. ⏳ Entrenar adaptador universal
4. ⏳ Implementar RAG universal con FAISS
5. ⏳ Auto-entrenamiento continuo
6. ⏳ Dashboard de monitoreo

---

**Sistema Universal Sheily v1.0** - Un modelo, todo el conocimiento
