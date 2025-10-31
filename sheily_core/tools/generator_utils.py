#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades ligeras para generación/ingestión que no requieren dependencias pesadas.
Se usan en tests y como helpers desde scripts más grandes.
"""
import hashlib
import re
from typing import List, Set

__all__ = [
    "normalize_space",
    "sha1_text",
    "tokenize_simple",
    "jaccard_sim",
    "chunk_text",
]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def tokenize_simple(text: str) -> Set[str]:
    toks = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", (text or "").lower())
    return set(toks)


def jaccard_sim(a_tokens: Set[str], b_tokens: Set[str]) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / max(1, union)


def chunk_text(text: str, target_chars: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    # Separar por frases simples.
    sents = re.split(r"(?<=[\.!?…])\s+(?=[A-ZÁÉÍÓÚÑÜ0-9])", text.strip(), flags=re.MULTILINE)
    chunks, buff, size = [], [], 0
    for sent in sents:
        st = sent.strip()
        if not st:
            continue
        if size + len(st) + 1 > target_chars and buff:
            chunk = " ".join(buff).strip()
            if chunk:
                chunks.append(chunk)
            carry = chunk[-overlap:]
            buff = [carry, st] if carry else [st]
            size = len(" ".join(buff))
        else:
            buff.append(st)
            size += len(st) + 1
    if buff:
        chunks.append(" ".join(buff).strip())
    return [c for c in chunks if len(c) >= 120]
