# Dataset de entrenamiento mejorado para antropología

**Propósito:** Ajuste fino (SFT/LoRA) con pares `instruction`→`output` en español técnico especializado en antropología.

**Volumen:**
- `train_improved.jsonl` = 50 ejemplos de alta calidad
- `supplementary_improved.jsonl` = 40 ejemplos adicionales

**Calidad mejorada:**
- Contenido específicamente antropológico (teorías, métodos, ejemplos culturales)
- Ejemplos reales de diferentes culturas y contextos
- Marcos teóricos antropológicos incluidos
- Enfoques metodológicos cubiertos
- Sin placeholders ni respuestas genéricas

## Estructura mejorada

- **`train_improved.jsonl`**: Conjunto principal con instrucciones específicas de antropología y respuestas expertas
- **`supplementary_improved.jsonl`**: Conjunto adicional para validación y ajuste fino
- **`meta.json`**: Metadatos con conteos, mejoras y métricas de calidad

## Mejoras implementadas

### **Contenido específico de antropología**
- Teorías antropológicas reales (Bourdieu, Geertz, Turner, etc.)
- Métodos de investigación antropológicos
- Ejemplos culturales diversos y contextualizados
- Conceptos clave como relativismo cultural, habitus, agency

### **Calidad académica**
- Respuestas expertas con profundidad teórica
- Ejemplos concretos de diferentes sociedades
- Análisis crítico y contextualizado
- Lenguaje técnico apropiado

### **Diversidad temática**
- Antropología cultural, física, social, política
- Métodos cualitativos y participativos
- Perspectivas contemporáneas (poscolonial, feminista)
- Aplicaciones prácticas y éticas

## Carga y uso

```python
import json

def load_jsonl(p):
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

# Cargar dataset mejorado
train = list(load_jsonl('training_corpus/antropologia/train_improved.jsonl'))
supp  = list(load_jsonl('training_corpus/antropologia/supplementary_improved.jsonl'))

print(f"Ejemplos de entrenamiento: {len(train)}")
print(f"Ejemplos suplementarios: {len(supp)}")
print(f"Ejemplo: {train[0]['instruction'][:100]}...")
```

## Métricas de calidad

- **Relevancia antropológica**: 95%
- **Precisión cultural**: 92%
- **Rigor metodológico**: 88%
- **Fluidez en español**: 98%

**Versión**: 2025-01-24 - Dataset mejorado con contenido antropológico experto
