#!/usr/bin/env python3
"""
TRAINER - Sistema de Entrenamiento Real
========================================
Entrenamiento de modelos usando HuggingFace Transformers y LoRA.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
    
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    if not DEPENDENCIES_AVAILABLE:
        print("\n" + "❌" * 35)
        print("  ERROR: DEPENDENCIAS FALTANTES")
        print("❌" * 35 + "\n")
        print("Para entrenar modelos, necesitas instalar:")
        print("\n  pip install transformers datasets peft accelerate bitsandbytes torch\n")
        print("O ejecuta:")
        print("\n  make install\n")
        return False
    return True


def load_training_data(data_path: str, tokenizer):
    """Cargar datos de entrenamiento desde archivos JSONL"""
    print(f"📂 Cargando datos desde: {data_path}")
    
    # Buscar todos los archivos JSONL
    data_files = list(Path(data_path).glob("*.jsonl"))
    
    if not data_files:
        raise ValueError(f"No se encontraron archivos .jsonl en {data_path}")
    
    print(f"   Archivos encontrados: {len(data_files)}")
    
    # Cargar dataset
    dataset = load_dataset(
        'json',
        data_files=[str(f) for f in data_files],
        split='train'
    )
    
    print(f"   Total ejemplos: {len(dataset)}")
    
    # Función para formatear ejemplos
    def format_instruction(example):
        """Formatear ejemplo en formato instruction-following"""
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        # Formato común para instruction tuning
        text = f"### Instrucción:\n{instruction}\n\n### Respuesta:\n{output}"
        
        return {'text': text}
    
    # Formatear dataset
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Tokenizar
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
        )
    
    print("🔤 Tokenizando datos...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
    )
    
    return tokenized_dataset


def create_lora_model(model, lora_config: Dict):
    """Crear modelo con LoRA"""
    print("🔧 Configurando LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    
    # Mostrar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Parámetros totales: {total_params:,}")
    
    return model


def train_model(config: Dict) -> bool:
    """
    Entrenar modelo con configuración dada.
    
    Args:
        config: Diccionario con configuración de entrenamiento
        
    Returns:
        bool: True si el entrenamiento fue exitoso
    """
    
    # Verificar dependencias
    if not check_dependencies():
        return False
    
    print("\n" + "🚀" * 35)
    print("  ENTRENAMIENTO DE MODELO - INICIANDO")
    print("🚀" * 35 + "\n")
    
    # Extraer configuración
    branch = config['branch']
    model_name = config['model']
    data_path = config['data_path']
    output_dir = config['output_dir']
    use_lora = config.get('use_lora', False)
    training_params = config['training_params']
    
    print(f"📚 Rama: {branch}")
    print(f"🤖 Modelo base: {model_name}")
    print(f"💾 Salida: {output_dir}")
    print(f"🔧 LoRA: {'Sí' if use_lora else 'No'}")
    
    try:
        # 1. Cargar tokenizer
        print("\n1️⃣  Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configurar pad token si no existe
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("   ✅ Tokenizer cargado")
        
        # 2. Cargar datos
        print("\n2️⃣  Cargando datos de entrenamiento...")
        dataset = load_training_data(data_path, tokenizer)
        print("   ✅ Datos cargados y tokenizados")
        
        # 3. Cargar modelo
        print("\n3️⃣  Cargando modelo...")
        
        # Verificar si hay GPU disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Dispositivo: {device.upper()}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        print("   ✅ Modelo cargado")
        
        # 4. Aplicar LoRA si está configurado
        if use_lora:
            print("\n4️⃣  Aplicando LoRA...")
            model = create_lora_model(model, config.get('lora_config', {}))
            print("   ✅ LoRA aplicado")
        else:
            print("\n4️⃣  Entrenamiento completo (sin LoRA)")
        
        # 5. Configurar entrenamiento
        print("\n5️⃣  Configurando entrenamiento...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_params['epochs'],
            per_device_train_batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=device == "cuda",
            report_to="none",  # Desactivar wandb, tensorboard, etc.
        )
        
        print(f"   Epochs: {training_params['epochs']}")
        print(f"   Batch size: {training_params['batch_size']}")
        print(f"   Learning rate: {training_params['learning_rate']}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Crear trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("   ✅ Trainer configurado")
        
        # 6. Entrenar
        print("\n6️⃣  ENTRENANDO MODELO...")
        print("   (Esto puede tomar varios minutos/horas dependiendo del tamaño)\n")
        
        train_result = trainer.train()
        
        print("\n   ✅ Entrenamiento completado!")
        print(f"   Loss final: {train_result.training_loss:.4f}")
        
        # 7. Guardar modelo
        print("\n7️⃣  Guardando modelo...")
        
        # Guardar modelo entrenado
        trainer.save_model(output_dir)
        
        # Guardar tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Guardar métricas
        metrics = {
            "branch": branch,
            "model": model_name,
            "training_loss": float(train_result.training_loss),
            "epochs": training_params['epochs'],
            "examples_trained": len(dataset),
            "use_lora": use_lora,
            "trained_at": datetime.now().isoformat(),
        }
        
        metrics_file = Path(output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✅ Modelo guardado en: {output_dir}")
        print(f"   ✅ Métricas guardadas en: {metrics_file}")
        
        # 8. Resumen final
        print("\n" + "✅" * 35)
        print("  ENTRENAMIENTO COMPLETADO CON ÉXITO")
        print("✅" * 35 + "\n")
        
        print(f"📊 Resumen:")
        print(f"   • Ejemplos entrenados: {len(dataset):,}")
        print(f"   • Loss final: {train_result.training_loss:.4f}")
        print(f"   • Epochs completados: {training_params['epochs']}")
        print(f"   • Modelo guardado: {output_dir}")
        
        print(f"\n💡 Para usar el modelo entrenado:")
        print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"   model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
        
        return True
        
    except Exception as e:
        print("\n" + "❌" * 35)
        print("  ERROR EN EL ENTRENAMIENTO")
        print("❌" * 35 + "\n")
        print(f"Error: {str(e)}")
        print(f"\nTipo: {type(e).__name__}")
        
        import traceback
        print("\nTraceback completo:")
        traceback.print_exc()
        
        return False


def quick_test(model_path: str):
    """
    Test rápido del modelo entrenado.
    
    Args:
        model_path: Ruta al modelo guardado
    """
    if not check_dependencies():
        return
    
    print(f"\n🧪 Testeando modelo: {model_path}")
    
    try:
        # Cargar modelo y tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test simple
        test_prompt = "### Instrucción:\n¿Qué es la física cuántica?\n\n### Respuesta:\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n📝 Test de generación:")
        print(response)
        print("\n✅ Modelo funciona correctamente")
        
    except Exception as e:
        print(f"\n❌ Error en test: {e}")


if __name__ == "__main__":
    # Ejemplo de uso
    print("Este módulo debe ser llamado desde train_branch.py")
    print("\nUso:")
    print("  python3 sheily_train/train_branch.py --branch physics --lora")
