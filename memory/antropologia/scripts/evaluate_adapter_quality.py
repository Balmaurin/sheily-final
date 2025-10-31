#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de Calidad del Adaptador LoRA - Antropología
========================================================

Evalúa la calidad del adaptador entrenado con pruebas de inference.

Author: Sheily AI Team
Version: 1.0.0
"""

import json
import torch
import torch_directml as dml
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Preguntas de evaluación en diferentes niveles
EVALUATION_QUESTIONS = [
    {
        "level": "básico",
        "question": "¿Qué es la antropología?",
        "expected_keywords": ["estudio", "ser humano", "cultura", "sociedad"]
    },
    {
        "level": "intermedio",
        "question": "¿Cuál es la diferencia entre perspectiva émica y ética en antropología?",
        "expected_keywords": ["interna", "externa", "cultura", "investigador", "miembros"]
    },
    {
        "level": "avanzado",
        "question": "Explica el concepto de descripción densa de Clifford Geertz",
        "expected_keywords": ["significado", "contexto", "interpretación", "simbólico", "cultural"]
    },
    {
        "level": "experto",
        "question": "¿Cómo se aplica el relativismo cultural en la investigación etnográfica moderna?",
        "expected_keywords": ["contexto", "valores", "juzgar", "cultura", "entender", "propia"]
    }
]

def load_model_with_adapter(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path: str = None
):
    """Carga el modelo con adaptador LoRA."""
    print("🔄 Cargando modelo base...")
    
    # Detectar dispositivo
    try:
        device = dml.device()
        device_type = "directml"
        print(f"✓ GPU DirectML detectada: {device}")
    except:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"⚠️  Usando CPU")
    
    # Cargar modelo y tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map=None
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar adaptador si existe
    if adapter_path and Path(adapter_path).exists():
        print(f"🔄 Cargando adaptador LoRA: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✓ Adaptador cargado")
    else:
        print("⚠️  Sin adaptador - usando modelo base")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_answer(
    model,
    tokenizer,
    device,
    question: str,
    max_length: int = 300
) -> str:
    """Genera respuesta para una pregunta."""
    
    # Formatear prompt
    prompt = f"<|system|>\nEres un experto en antropología. Responde de manera clara y precisa.</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta del asistente
    if "<|assistant|>" in full_text:
        answer = full_text.split("<|assistant|>")[-1].strip()
    else:
        answer = full_text.replace(prompt, "").strip()
    
    return answer

def evaluate_answer(answer: str, expected_keywords: List[str]) -> Dict:
    """Evalúa la calidad de una respuesta."""
    answer_lower = answer.lower()
    
    # Contar keywords encontrados
    keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0
    
    # Evaluar longitud (respuestas muy cortas o muy largas)
    length_score = 1.0
    if len(answer) < 50:
        length_score = 0.5
    elif len(answer) > 1000:
        length_score = 0.7
    
    # Evaluar coherencia (básico - no repeticiones excesivas)
    words = answer.split()
    unique_words = set(words)
    coherence_score = min(len(unique_words) / max(len(words), 1), 1.0)
    
    # Puntuación final
    final_score = (keyword_score * 0.5 + length_score * 0.2 + coherence_score * 0.3)
    
    return {
        "keyword_score": keyword_score,
        "keywords_found": keywords_found,
        "length_score": length_score,
        "coherence_score": coherence_score,
        "final_score": final_score,
        "length": len(answer)
    }

def main():
    """Ejecuta evaluación completa."""
    print("=" * 70)
    print("EVALUACIÓN DE CALIDAD - ADAPTADOR ANTROPOLOGÍA")
    print("=" * 70)
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    adapter_path = base_dir / "adapters" / "lora_adapters" / "current"
    
    # Cargar modelo
    model, tokenizer, device = load_model_with_adapter(
        adapter_path=str(adapter_path)
    )
    
    print(f"\n{'=' * 70}")
    print("PRUEBAS DE INFERENCE")
    print(f"{'=' * 70}\n")
    
    results = []
    
    for i, test in enumerate(EVALUATION_QUESTIONS, 1):
        print(f"\n📝 Prueba {i}/{len(EVALUATION_QUESTIONS)} - Nivel: {test['level'].upper()}")
        print(f"Pregunta: {test['question']}")
        print("-" * 70)
        
        # Generar respuesta
        answer = generate_answer(model, tokenizer, device, test['question'])
        
        # Evaluar
        evaluation = evaluate_answer(answer, test['expected_keywords'])
        
        # Mostrar resultados
        print(f"\n💬 Respuesta generada ({evaluation['length']} caracteres):")
        print(f"{answer[:500]}..." if len(answer) > 500 else answer)
        
        print(f"\n📊 Evaluación:")
        print(f"  - Keywords encontrados: {len(evaluation['keywords_found'])}/{len(test['expected_keywords'])}")
        print(f"    {evaluation['keywords_found']}")
        print(f"  - Score Keywords: {evaluation['keyword_score']:.2%}")
        print(f"  - Score Longitud: {evaluation['length_score']:.2%}")
        print(f"  - Score Coherencia: {evaluation['coherence_score']:.2%}")
        print(f"  - 🎯 Score Final: {evaluation['final_score']:.2%}")
        
        results.append({
            "level": test['level'],
            "question": test['question'],
            "answer": answer,
            "evaluation": evaluation
        })
        
        print("=" * 70)
    
    # Resumen final
    print(f"\n{'=' * 70}")
    print("📊 RESUMEN GENERAL")
    print(f"{'=' * 70}\n")
    
    avg_keyword = sum(r['evaluation']['keyword_score'] for r in results) / len(results)
    avg_length = sum(r['evaluation']['length_score'] for r in results) / len(results)
    avg_coherence = sum(r['evaluation']['coherence_score'] for r in results) / len(results)
    avg_final = sum(r['evaluation']['final_score'] for r in results) / len(results)
    
    print(f"Pruebas realizadas: {len(results)}")
    print(f"Score promedio Keywords: {avg_keyword:.2%}")
    print(f"Score promedio Longitud: {avg_length:.2%}")
    print(f"Score promedio Coherencia: {avg_coherence:.2%}")
    print(f"\n🎯 CALIDAD GENERAL: {avg_final:.2%}")
    
    # Clasificación de calidad
    if avg_final >= 0.80:
        quality = "EXCELENTE ⭐⭐⭐⭐⭐"
    elif avg_final >= 0.65:
        quality = "BUENA ⭐⭐⭐⭐"
    elif avg_final >= 0.50:
        quality = "ACEPTABLE ⭐⭐⭐"
    elif avg_final >= 0.35:
        quality = "MEJORABLE ⭐⭐"
    else:
        quality = "INSUFICIENTE ⭐"
    
    print(f"Clasificación: {quality}")
    
    # Guardar resultados
    output_file = base_dir / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": str(Path(__file__).stat().st_mtime),
            "adapter_path": str(adapter_path),
            "results": results,
            "summary": {
                "avg_keyword_score": avg_keyword,
                "avg_length_score": avg_length,
                "avg_coherence_score": avg_coherence,
                "avg_final_score": avg_final,
                "quality_classification": quality
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Resultados guardados en: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
