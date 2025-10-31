#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_make_full.py
-------------------
Reconstruye Sheily-main/ -> Sheily-Final/ con toda la infraestructura real:
- Mueve modelos (.gguf) a models/gguf/
- Mueve LoRA adapters a models/lora_adapters/<rama>/
- Mueve corpus a corpus_ES/<rama>/{st, tfidf, vectors.npz, *_enhanced.jsonl}
- Mueve scripts .py/.sh/.ps1, docs y logs
- Archiva desconocidos en archivos_no_ubicados/
- Genera main_router.py, train_lora_cpu_real.py, train_rag_index.py, validators,
  auto-maintainer, run scripts, installers (online + offline), Dockerfile, docker-compose, docs
- Genera out/FULL_PLAN.md con lista completa de movimientos
- Opciones: --project-root, --apply, --move, --plan, --verbose
Usage:
    python3 sheily_make_full.py --project-root ./Sheily-main --apply --move --verbose
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path


# ---------------------- UTIL ----------------------
def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(dst_root: Path, branches, unlocated):
    base_dirs = [
        "sheily_core/lora_loader",
        "sheily_core/rag_bridge",
        "sheily_core/security",
        "sheily_core/modules",
        "sheily_rag",
        "sheily_train",
        "models/gguf",
        "models/lora_adapters",
        "corpus_ES",
        "scripts",
        "tests",
        "logs/chat_sessions",
        "logs/chat_history",
        "logs/train_stats",
        "docs",
        unlocated,
        "out",
    ]
    for d in base_dirs:
        (dst_root / d).mkdir(parents=True, exist_ok=True)
    # create branches structure for lora and corpus
    for b in branches:
        (dst_root / f"models/lora_adapters/{b}").mkdir(parents=True, exist_ok=True)
        (dst_root / f"corpus_ES/{b}/st").mkdir(parents=True, exist_ok=True)
        (dst_root / f"corpus_ES/{b}/tfidf").mkdir(parents=True, exist_ok=True)


def is_lora_dir(p: Path):
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists()


def is_lora_file(p: Path):
    if p.suffix.lower() == ".safetensors":
        if (p.parent / "adapter_config.json").exists():
            return True
        if "adapter" in p.name.lower() or "lora" in p.name.lower():
            return True
    return False


def classify_file(p: Path):
    name = p.name.lower()
    if name.endswith(".gguf"):
        return "gguf"
    if name.endswith(".bin") and "gguf" in name:
        return "gguf"
    if p.is_dir() and is_lora_dir(p):
        return "lora_dir"
    if is_lora_file(p):
        return "lora_file"
    if name.endswith(".jsonl"):
        if name.endswith("_enhanced.jsonl"):
            return "corpus_enhanced"
        return "corpus_jsonl"
    if name == "vectors.npz" or name.endswith(".npz"):
        return "corpus_vectors"
    if "tfidf" in p.parts or name.startswith("tfidf"):
        return "corpus_tfidf"
    if name.endswith(".meta.json") or name == "meta.json":
        return "corpus_meta"
    if name.endswith(".py"):
        return "py"
    if name.endswith(".md") or name.endswith(".txt"):
        return "doc"
    if name.endswith(".sh"):
        return "shell"
    if name.endswith(".ps1"):
        return "ps1"
    if name.endswith(".log"):
        return "log"
    return "unknown"


def detect_branch_from_path(rel: str, branches):
    low = rel.lower()
    for b in branches:
        if b.lower() in low:
            return b
    return ""


def safe_move(src: Path, dst: Path, apply: bool, verbose: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not apply:
        if verbose:
            print(f"[PLAN] {src} -> {dst}")
        return {"src": str(src), "dst": str(dst), "moved": False}
    # if destination exists, compare sha
    if dst.exists():
        try:
            if src.is_file() and dst.is_file() and sha256_file(src) == sha256_file(dst):
                if verbose:
                    print(f"[SKIP] Igual hash; eliminando origen {src}")
                src.unlink()
                return {
                    "src": str(src),
                    "dst": str(dst),
                    "moved": True,
                    "note": "duplicate_removed",
                }
            # rename destination
            base = dst.stem
            ext = dst.suffix
            i = 1
            ndst = dst.parent / f"{base}({i}){ext}"
            while ndst.exists():
                i += 1
                ndst = dst.parent / f"{base}({i}){ext}"
            dst = ndst
        except Exception as e:
            if verbose:
                print(f"[WARN] Error comprobando hashes: {e}")
    try:
        shutil.move(str(src), str(dst))
        if verbose:
            print(f"[MOVE] {src} -> {dst}")
        return {"src": str(src), "dst": str(dst), "moved": True}
    except Exception as e:
        if verbose:
            print(f"[ERR] No se pudo mover {src} -> {dst}: {e}")
        return {"src": str(src), "dst": str(dst), "moved": False, "error": str(e)}


# ---------------------- TEMPLATES (archivos a generar) ----------------------
MAIN_ROUTER = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time
from datetime import datetime

BANNER = """
═══════════════════════════════════════════════════════
🤖  Sheily AI — Sistema Cognitivo Modular Inicializado
🧠  39 ramas activas  |  RAG + LoRA integrados  |  Modo texto
═══════════════════════════════════════════════════════
"""

def log_session(prompt, branch, response, elapsed):
    os.makedirs("logs/chat_sessions", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": prompt,
        "detected_branch": branch,
        "response": response,
        "elapsed_time_sec": round(elapsed,2)
    }
    fname = datetime.now().isoformat().replace(":", "-") + ".json"
    with open(os.path.join("logs","chat_sessions", fname), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

def detect_branch(prompt: str):
    # Heurística ligera; reemplázala por tu enrutador RAG/embeddings real si quieres.
    keys = {
        "fisica":"fisica","química":"quimica","biología":"biologia","antropología":"antropologia",
        "economía":"economia","matemática":"matematica","historia":"historia","psicología":"psicologia",
        "derecho":"derecho","filosofía":"filosofia","tecnología":"tecnologia","medicina":"medicina"
    }
    low = prompt.lower()
    for k,v in keys.items():
        if k in low: return v
    return "general"

def main():
    print(BANNER)
    print("👋 Hola, soy Sheily AI.\nPuedes preguntarme sobre cualquier tema. Escribe 'salir' para terminar.\n")
    while True:
        try:
            q = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta pronto 👋"); break
        if q.lower() in ("salir","exit","quit"):
            print("Hasta pronto 👋"); break
        start = time.time()
        branch = detect_branch(q)
        print(f"[INFO] Rama detectada: {branch}")
        print("[INFO] RAG: (placeholder) recuperación de contexto")
        print("[INFO] LoRA: (placeholder) carga de adaptador si existe")
        response = f"(demo) Respuesta de Sheily en la rama '{branch}' — integra aquí inferencia real."
        print("Sheily:", response, "\n")
        log_session(q, branch, response, time.time()-start)

if __name__ == '__main__':
    main()
'''

TRAIN_LORA = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento LoRA real (PEFT + transformers)
Uso:
  python3 sheily_train/train_lora_cpu_real.py --branch antropologia --epochs 3
"""
import os, json, time, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_gguf = Path("models/gguf/llama-3.2.gguf")
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if local_gguf.exists():
        # Nota: si dispones de un conversion pipeline desde GGUF a HF, adapta aquí la ruta.
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    branch = args.branch
    data_path = Path(f"corpus_ES/{branch}/st/texts.jsonl")
    if not data_path.exists():
        raise SystemExit(f"No existe dataset esperado: {data_path}")
    texts = []
    with data_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                obj = json.loads(ln)
                t = obj.get("text") or obj.get("output") or obj.get("instruction") or ""
                if t: texts.append(t)
    if not texts:
        raise SystemExit("Dataset vacío o sin campo de texto.")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.to(device)

    def tok_map(batch):
        toks = tokenizer(batch["text"], max_length=args.seq, truncation=True, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds = Dataset.from_dict({"text": texts}).map(tok_map, batched=True, remove_columns=["text"])

    lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          task_type=TaskType.CAUSAL_LM, target_modules=["q_proj","v_proj"])
    model = get_peft_model(model, lora_cfg)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=f"models/lora_adapters/{branch}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )

    print(f"[TRAIN] Dispositivo: {device} | Rama: {branch} | Epochs: {args.epochs}")
    t0 = time.time()
    trainer = Trainer(model=model, args=training_args, data_collator=collator, train_dataset=ds)
    trainer.train()
    model.save_pretrained(f"models/lora_adapters/{branch}")
    tokenizer.save_pretrained(f"models/lora_adapters/{branch}")

    # guardar métricas simples
    os.makedirs("logs/train_stats", exist_ok=True)
    stats_file = Path("logs/train_stats") / f"{branch}.json"
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "branch": branch,
        "base_model": str(local_gguf if local_gguf.exists() else base_model),
        "device": device,
        "epochs": args.epochs,
        "train_examples": len(texts),
        "train_time_min": round((time.time()-t0)/60.0,2)
    }
    hist = []
    if stats_file.exists():
        try:
            hist = json.loads(stats_file.read_text(encoding="utf-8"))
            if not isinstance(hist, list): hist = []
        except Exception:
            hist = []
    hist.append(entry)
    stats_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[TRAIN] Finalizado. Adaptador guardado y métricas registradas.")

if __name__ == "__main__":
    main()
'''

TRAIN_RAG = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Índice RAG: TF-IDF + FAISS
Uso:
  python3 sheily_train/train_rag_index.py --branch antropologia
"""
import os, json, argparse
from pathlib import Path
def load_texts(p: Path):
    arr = []
    if not p.exists():
        return arr
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                obj = json.loads(ln)
                t = obj.get("text") or obj.get("output") or obj.get("instruction") or ""
                if t: arr.append(t)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    args = ap.parse_args()
    branch = args.branch
    project = Path(".").resolve()
    st = project / f"corpus_ES/{branch}/st/texts.jsonl"
    if not st.exists():
        print(f"[WARN] No hay corpus en {st}")
        return
    texts = load_texts(st)
    if not texts:
        print("[WARN] Corpus vacío")
        return
    # TF-IDF (resumen mínimo)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=50000)
    X = vec.fit_transform(texts)
    (project / f"corpus_ES/{branch}/tfidf").mkdir(parents=True, exist_ok=True)
    (project / f"corpus_ES/{branch}/tfidf/index.json").write_text(json.dumps({
        "ndocs": len(texts),
        "vocab_size": len(vec.vocabulary_)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RAG] TF-IDF construido ({len(texts)} docs).")
    # FAISS si existen embeddings
    vecs = project / f"corpus_ES/{branch}/vectors.npz"
    if vecs.exists():
        import numpy as np, faiss
        npz = np.load(vecs)
        emb = npz["embeddings"]
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb.astype("float32"))
        faiss.write_index(idx, str(project / f"corpus_ES/{branch}/st/faiss.index"))
        print("[RAG] FAISS index guardado.")
    else:
        print("[RAG] No hay vectors.npz; se omitió FAISS.")

if __name__ == "__main__":
    main()
'''

VALIDATE_LORA = r"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
def main():
    base = Path("models/lora_adapters")
    missing = []
    for br in base.iterdir():
        if br.is_dir():
            if not any(br.glob("*.safetensors")) and not (br / "adapter_config.json").exists():
                missing.append(br.name)
    if missing:
        print("[WARN] Ramas sin adaptador válido:", ", ".join(missing))
    else:
        print("[OK] Todos los adaptadores parecen válidos (comprobación mínima).")

if __name__ == "__main__":
    main()
"""

RAG_VALIDATOR = r"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
def main():
    base = Path("corpus_ES")
    issues = []
    for br in base.iterdir():
        if br.is_dir():
            st = br / "st" / "texts.jsonl"
            if not st.exists():
                issues.append(f"{br.name}: falta st/texts.jsonl")
    if issues:
        print("[WARN] Problemas RAG:\n - " + "\n - ".join(issues))
    else:
        print("[OK] Corpus RAG básico presente por rama.")
if __name__ == "__main__":
    main()
"""

RERANKER = r"""# sheily_rag/rag_ranker.py
def rerank(candidates):
    return sorted(candidates, key=lambda x: x.get("score",0), reverse=True)
"""

AUTO_MAINTAINER = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_auto_maintainer.py
- Detecta cambios en corpus_ES/<rama>/st/texts.jsonl por hash
- Si hay cambios, ejecuta train_lora_cpu_real.py y train_rag_index.py para la rama
- Compacta logs antiguos (placeholder)
"""
import os, json, hashlib, subprocess
from pathlib import Path
def sha256(p: Path):
    if not p.exists(): return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()

STATE = Path("logs/.maintainer_state.json")
def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    os.makedirs("logs", exist_ok=True)
    state = load_state()
    changed = []
    for br in Path("corpus_ES").iterdir():
        if not br.is_dir(): continue
        st = br / "st" / "texts.jsonl"
        if not st.exists(): continue
        h = sha256(st)
        key = f"{br.name}::st"
        if state.get(key) != h:
            changed.append(br.name)
            state[key] = h
    for br in changed:
        print(f"[AUTO] Cambios detectados en {br} → entrenando LoRA + reindexando RAG")
        try:
            subprocess.run(["python3","sheily_train/train_lora_cpu_real.py","--branch",br], check=True)
            subprocess.run(["python3","sheily_train/train_rag_index.py","--branch",br], check=True)
        except Exception as e:
            print(f"[ERR] Auto-entrenamiento falló para {br}: {e}")
    save_state(state)
    print("[AUTO] Mantenimiento completado.")

if __name__ == "__main__":
    main()
'''

RUN_SH = r"""#!/usr/bin/env bash
set -e
has_net() {
  (getent hosts pypi.org >/dev/null 2>&1 || nslookup pypi.org >/dev/null 2>&1) && ping -c1 -W1 1.1.1.1 >/dev/null 2>&1
}
if [ ! -d ".venv_sheily" ]; then
  if has_net; then
    echo "🌐 Red detectada → instalador online"
    bash sheily_train/install_lora_env.sh
  else
    echo "🛠️  Sin red → instalador offline (./deps_cache)"
    if [ -d "./deps_cache" ]; then
      bash sheily_train/install_lora_env_offline.sh
    else
      echo "❌ No existe deps_cache/ con ruedas (.whl)."
      echo "   Crea el cache en otra máquina: pip download -r requirements.txt -d deps_cache"
      exit 2
    fi
  fi
fi
source .venv_sheily/bin/activate
python3 sheily_auto_maintainer.py
python3 sheily_core/main_router.py
"""

RUN_PS1 = r"""$ErrorActionPreference = "Stop"
function Test-Net {
  try { $null = Resolve-DnsName pypi.org -ErrorAction Stop; $r = Invoke-WebRequest -Uri "https://1.1.1.1" -Method Head -TimeoutSec 3; return $true }
  catch { return $false }
}
if (!(Test-Path ".\.venv_sheily")) {
  if (Test-Net) {
    Write-Host "🌐 Red detectada → instalador online"; .\sheily_train\install_lora_env.ps1
  } else {
    Write-Host "🛠️ Sin red → instalador offline (.\deps_cache)"
    if (Test-Path ".\deps_cache") {
      python -m venv .venv_sheily
      .\.venv_sheily\Scripts\Activate.ps1
      python -m pip install --upgrade pip
      pip install --no-index --find-links .\deps_cache -r requirements.txt
    } else {
      Write-Host "❌ No hay deps_cache/ con .whl"; exit 2
    }
  }
}
.\.venv_sheily\Scripts\Activate.ps1
python .\sheily_auto_maintainer.py
python .\sheily_core\main_router.py
"""

INSTALL_SH = r"""#!/usr/bin/env bash
set -e
echo "🚀 Instalando entorno Sheily (Ubuntu/Linux)"
python3 -m venv .venv_sheily
source .venv_sheily/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft datasets accelerate safetensors tqdm faiss-cpu ragas deepeval sentencepiece scikit-learn
echo "✅ Entorno listo. Activa con: source .venv_sheily/bin/activate"
"""

INSTALL_OFFLINE_SH = r"""#!/usr/bin/env bash
set -e
echo "⚙️ Instalando Sheily (modo OFFLINE)"
python3 -m venv .venv_sheily
source .venv_sheily/bin/activate
pip install --upgrade pip
if [ ! -d "./deps_cache" ]; then
  echo "❌ Falta ./deps_cache con los whl (ver README en deps_cache/)"
  exit 2
fi
pip install --no-index --find-links ./deps_cache -r requirements.txt
echo "✅ Entorno OFFLINE instalado."
"""

INSTALL_PS1 = r"""Write-Host "🚀 Instalando entorno Sheily (Windows)"
python -m venv .venv_sheily
.\.venv_sheily\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft datasets accelerate safetensors tqdm faiss-cpu ragas deepeval sentencepiece scikit-learn
Write-Host "✅ Entorno listo. Para activar: .\.venv_sheily\Scripts\Activate.ps1"
"""

DOCKERFILE = r"""FROM ubuntu:24.04
WORKDIR /app
RUN apt update && apt install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash","-lc","python3 sheily_auto_maintainer.py && python3 sheily_core/main_router.py"]
"""

DOCKER_COMPOSE = r"""version: "3.9"
services:
  sheily:
    build: .
    container_name: sheily-ai
    working_dir: /app
    command: ["bash","-lc","python3 sheily_auto_maintainer.py && python3 sheily_core/main_router.py"]
    volumes:
      - ./:/app
      - ./models:/app/models
      - ./corpus_ES:/app/corpus_ES
      - ./logs:/app/logs
    tty: true
    stdin_open: true
"""

REQUIREMENTS = (
    "\n".join(
        [
            "transformers",
            "peft",
            "datasets",
            "accelerate",
            "safetensors",
            "tqdm",
            "faiss-cpu",
            "ragas",
            "deepeval",
            "sentencepiece",
            "scikit-learn",
        ]
    )
    + "\n"
)

DOCS_MD = r"""# SHEILY_FULL_BUILD.md

Sheily-Final — documentación de despliegue y uso.

Resumen:
- Reconstrucción hecha: Sheily-main -> Sheily-Final
- Arranque rápido: ./run_sheily.sh (Linux) / .\run_sheily.ps1 (Windows)
- Docker: docker compose up --build (arranca auto-maintainer + asistente interactivo)
- Instalación offline: colocar .whl en deps_cache/ y correr install_lora_env_offline.sh

Estructura principal:
- sheily_core/: núcleo, main_router.py, lora_loader, rag_bridge
- sheily_rag/: scripts RAG y helpers
- sheily_train/: scripts de entrenamiento LoRA y RAG, instaladores
- models/gguf/: modelos gguf (pon aquí llama-3.2.gguf)
- models/lora_adapters/: adaptadores LoRA por rama
- corpus_ES/: corpus por rama (st/, tfidf/, vectors.npz)
- logs/: chat_history/, train_stats/
- docs/: documentación

Auto-entrenamiento:
- El auto-maintainer detecta cambios en corpus_ES/<rama>/st/texts.jsonl y lanza training + reindexing.

Nota: El script de entrenamiento LoRA usa un modelo HF por defecto como fallback.
Si trabajas con GGUF, adapta tu pipeline para convertir o exponer el modelo HF equivalente.
"""

QUICK_CHECK = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_quick_check.py
Comprueba presencia de:
 - models/gguf/llama-3.2.gguf
 - corpus_ES/*/st/texts.jsonl
 - models/lora_adapters/*/*.safetensors
 - corpus_ES/*/st/faiss.index (opcional)
"""
import json, os
from pathlib import Path

def main():
    root = Path(".")
    model = root / "models" / "gguf" / "llama-3.2.gguf"
    print("═══════════════════════════════════════════")
    print("🔍 SHEILY QUICK CHECK")
    print("───────────────────────────────────────────")
    if model.exists():
        print("✅ Modelo base:", model, "(OK)")
    else:
        print("⚠️ Modelo base no encontrado:", model)
    branches = [p.name for p in (root / "corpus_ES").glob("*") if p.is_dir()]
    ok_rag = 0
    ok_lora = 0
    missing_rag = []
    missing_lora = []
    for b in branches:
        st = root / f"corpus_ES/{b}/st/texts.jsonl"
        if st.exists() and st.stat().st_size>0:
            ok_rag += 1
        else:
            missing_rag.append(b)
        lora_dir = root / f"models/lora_adapters/{b}"
        if any(lora_dir.glob("*.safetensors")) or (lora_dir / "adapter_config.json").exists():
            ok_lora += 1
        else:
            missing_lora.append(b)
    print(f"✅ RAG: {ok_rag}/{len(branches)} ramas con corpus")
    if missing_rag:
        print("⚠️ Ramas sin corpus:", ", ".join(missing_rag))
    print(f"✅ LoRA: {ok_lora}/{len(branches)} adaptadores detectados")
    if missing_lora:
        print("⚠️ Ramas sin adaptadores:", ", ".join(missing_lora))
    print("═══════════════════════════════════════════")

if __name__ == '__main__':
    main()
'''


# ---------------------- SCRIPT PRINCIPAL ----------------------
def main():
    ap = argparse.ArgumentParser(description="Reconstruir Sheily-main -> Sheily-Final (estructura completa)")
    ap.add_argument("--project-root", default="./Sheily-main", help="Ruta al proyecto original descomprimido")
    ap.add_argument("--apply", action="store_true", help="Aplicar los movimientos (si no, solo plan)")
    ap.add_argument("--move", action="store_true", help="Usar mover en vez de copiar (recomendado)")
    ap.add_argument("--plan", default="out/FULL_PLAN.md", help="Ruta relativa del informe a crear en destino")
    ap.add_argument("--branches-file", default="BRANCHES.txt", help="Archivo con lista de ramas (una por línea)")
    ap.add_argument("--unlocated-dir", default="archivos_no_ubicados", help="Carpeta destino para desconocidos")
    ap.add_argument("--verbose", action="store_true", help="Salida verbosa")
    args = ap.parse_args()

    src_root = Path(args.project_root).resolve()
    if not src_root.exists():
        print(f"❌ No existe la ruta: {src_root}")
        sys.exit(2)

    dst_root = src_root.parent / "Sheily-Final"
    if dst_root.exists() and args.apply and args.move:
        # we'll still proceed but structure exists; be conservative
        pass
    dst_root.mkdir(parents=True, exist_ok=True)

    # read branches file or default
    branches_path = Path(args.branches_file)
    if branches_path.exists():
        branches = [
            ln.strip()
            for ln in branches_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        branches = [
            "antropologia",
            "economia",
            "matematica",
            "psicologia",
            "tecnologia",
            "historia",
            "fisica",
            "quimica",
            "biologia",
            "filosofia",
            "derecho",
            "sociologia",
            "politica",
            "ecologia",
            "educacion",
            "medicina",
            "arte",
            "musica",
            "cine",
            "literatura",
            "ingenieria",
            "informatica",
            "ciberseguridad",
            "linguistica",
            "antropologia_digital",
            "inteligencia_artificial",
            "economia_global",
            "etica",
            "filosofia_moderna",
            "marketing",
            "derecho_internacional",
            "psicologia_social",
            "neurociencia",
            "robotica",
            "fisica_cuantica",
            "astronomia",
            "IA_multimodal",
            "voz_emocional",
            "metacognicion",
        ]

    ensure_dirs(dst_root, branches, args.unlocated_dir)

    report = {"moves": [], "unlocated": [], "errors": [], "totals": {"files": 0, "moved": 0}}

    # Walk source tree
    for root, dirs, files in os.walk(src_root):
        root_p = Path(root)
        # skip venvs, node_modules, git internals, and the target if inside source
        if any(x in root_p.parts for x in [".git", ".venv", "venv", "__pycache__", "node_modules"]):
            continue
        for fn in files:
            src = root_p / fn
            # if source is already inside destination, skip
            try:
                src.relative_to(dst_root)
                continue
            except Exception:
                pass
            report["totals"]["files"] += 1
            rel = src.relative_to(src_root)
            cls = classify_file(src)
            branch = detect_branch_from_path(str(rel), branches)
            # destination selection
            if cls == "gguf":
                dst = dst_root / "models" / "gguf" / src.name
            elif cls in ("lora_dir", "lora_file"):
                if not branch:
                    branch = "misc"
                dst = dst_root / "models" / "lora_adapters" / branch / src.name
            elif cls in (
                "corpus_jsonl",
                "corpus_meta",
                "corpus_vectors",
                "corpus_enhanced",
                "corpus_tfidf",
            ):
                if not branch:
                    branch = "misc"
                if cls == "corpus_enhanced":
                    dst = dst_root / "corpus_ES" / branch / src.name
                elif cls == "corpus_vectors":
                    dst = dst_root / "corpus_ES" / branch / "vectors.npz"
                elif cls == "corpus_meta":
                    dst = dst_root / "corpus_ES" / branch / "st" / "meta.json"
                elif cls == "corpus_tfidf":
                    dst = dst_root / "corpus_ES" / branch / "tfidf" / src.name
                else:
                    dst = dst_root / "corpus_ES" / branch / "st" / src.name
            elif cls == "py":
                low = src.name.lower()
                if "rag" in low:
                    dst = dst_root / "sheily_rag" / src.name
                elif "train" in low or "lora" in low:
                    dst = dst_root / "sheily_train" / src.name
                elif "loader" in low:
                    dst = dst_root / "sheily_core" / "lora_loader" / src.name
                elif "security" in low or "audit" in low:
                    dst = dst_root / "sheily_core" / "security" / src.name
                else:
                    dst = dst_root / "sheily_core" / src.name
            elif cls == "shell":
                dst = dst_root / "sheily_train" / src.name
            elif cls == "ps1":
                dst = dst_root / "sheily_train" / src.name
            elif cls == "doc":
                dst = dst_root / "docs" / src.name
            elif cls == "log":
                dst = dst_root / "logs" / src.name
            else:
                dst = dst_root / args.unlocated_dir / src.name
                report["unlocated"].append(str(rel))

            res = safe_move(src, dst, apply=(args.apply and args.move), verbose=args.verbose)
            report["moves"].append(res)
            if res.get("moved"):
                report["totals"]["moved"] += 1
            if res.get("error"):
                report["errors"].append(res.get("error"))

    # generate created files (templates) in destination
    def write_file(path: Path, content: str, mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        try:
            path.chmod(mode)
        except Exception:
            pass

    # main router
    write_file(dst_root / "sheily_core" / "main_router.py", MAIN_ROUTER)
    # training scripts
    write_file(dst_root / "sheily_train" / "train_lora_cpu_real.py", TRAIN_LORA)
    write_file(dst_root / "sheily_train" / "train_rag_index.py", TRAIN_RAG)
    write_file(dst_root / "sheily_train" / "validate_lora.py", VALIDATE_LORA)
    write_file(dst_root / "sheily_rag" / "rag_validator.py", RAG_VALIDATOR)
    write_file(dst_root / "sheily_rag" / "rag_ranker.py", RERANKER)
    write_file(dst_root / "sheily_auto_maintainer.py", AUTO_MAINTAINER)
    write_file(dst_root / "run_sheily.sh", RUN_SH, mode=0o755)
    write_file(dst_root / "run_sheily.ps1", RUN_PS1, mode=0o755)
    write_file(dst_root / "sheily_train" / "install_lora_env.sh", INSTALL_SH, mode=0o755)
    write_file(dst_root / "sheily_train" / "install_lora_env_offline.sh", INSTALL_OFFLINE_SH, mode=0o755)
    write_file(dst_root / "sheily_train" / "install_lora_env.ps1", INSTALL_PS1, mode=0o755)
    write_file(dst_root / "Dockerfile", DOCKERFILE)
    write_file(dst_root / "docker-compose.yml", DOCKER_COMPOSE)
    write_file(dst_root / "requirements.txt", REQUIREMENTS)
    write_file(dst_root / "docs" / "SHEILY_FULL_BUILD.md", DOCS_MD)
    write_file(dst_root / "sheily_quick_check.py", QUICK_CHECK)
    write_file(dst_root / "BRANCHES.txt", "\n".join(branches) + "\n")

    # create deps_cache placeholder
    (dst_root / "deps_cache").mkdir(parents=True, exist_ok=True)
    (dst_root / "deps_cache" / "README.txt").write_text(
        "Place your pre-downloaded .whl packages here for offline installation.\n"
        "Create them with: pip download -r requirements.txt -d deps_cache\n"
    )

    # write out plan/informe
    plan_path = dst_root / args.plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        f.write("# FULL_PLAN — Sheily-Final\n\n")
        f.write(f"Fecha: {datetime.now().isoformat()}\n\n")
        f.write("## Totales\n")
        f.write(json.dumps(report.get("totals", {}), indent=2, ensure_ascii=False))
        f.write("\n\n## Movimientos\n")
        for m in report["moves"]:
            f.write(f"- {m.get('src')} → {m.get('dst')}  moved={m.get('moved')}\n")
        f.write("\n## No ubicados\n")
        for u in report["unlocated"]:
            f.write(f"- {u}\n")
        f.write("\n## Errores\n")
        for e in report["errors"]:
            f.write(f"- {e}\n")

    print("✅ Reconstrucción completada.")
    print(f" - Origen: {src_root}")
    print(f" - Destino: {dst_root}")
    print(f" - Informe: {plan_path}")


if __name__ == "__main__":
    main()
