#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de dataset local (JSONL) desde el contenido en corpus_ES.

- Recorre corpus_ES/<rama>/** y toma archivos .txt/.md como fuente.
- Crea ejemplos con un prompt neutro y el contenido como "output".
- Permite filtrar por rama y limitar el número de ejemplos.

Salida JSONL por línea con campos: {instruction, input, output, branch, source}
"""
import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Tuple

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus_ES"
DEFAULT_OUT = ROOT / "data" / "dataset.jsonl"


def iter_corpus_files(branch: str | None) -> Iterator[Tuple[str, Path]]:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"WARNING: No existe el directorio de corpus: {CORPUS_DIR}")

    for rama in sorted(d.name for d in CORPUS_DIR.iterdir() if d.is_dir()):
        if branch and rama != branch:
            continue
        rama_dir = CORPUS_DIR / rama
        for root, _, files in os.walk(rama_dir):
            for fn in files:
                if fn.lower().endswith((".txt", ".md")):
                    yield rama, Path(root) / fn


def clean_text(s: str) -> str:
    # Limpieza mínima
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def build_example(branch: str, text: str, source: Path) -> dict:
    instruction = "Lee el siguiente contenido y elabora una respuesta clara y concisa."
    return {
        "instruction": instruction,
        "input": "",
        "output": text,
        "branch": branch,
        "source": str(source.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser(description="Generar dataset JSONL desde corpus_ES")
    ap.add_argument(
        "--model_id",
        default="sheily-ai/Sheily-3B-Instruct",
        help="Se acepta por compatibilidad; no se usa",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--branch", help="Rama a procesar (opcional)")
    ap.add_argument("--min_chars", type=int, default=200, help="Mínimo de caracteres por ejemplo")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.out.open("w", encoding="utf-8") as f:
        for rama, path in iter_corpus_files(args.branch):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"WARNING: No se pudo leer {path}: {e}")
                continue
            text = clean_text(text)
            if len(text) < args.min_chars:
                continue
            ex = build_example(rama, text, path)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.num:
                break

    print(f"✅ Dataset generado: {args.out} ({count} ejemplos)")


if __name__ == "__main__":
    main()
