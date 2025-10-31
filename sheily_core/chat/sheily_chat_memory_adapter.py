#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sheily_chat_memory_adapter.py
=============================
- Detección avanzada de órdenes: memoriza/guarda/aprende..., olvida/borra...
- Borra por fragmento exacto o por "relacionado con ..."
- Maneja archivos PDF/TXT en chunks con metadatos
- Responde con naturalidad usando recuerdos (TF-IDF + embeddings)
"""

from __future__ import annotations

import re
from pathlib import Path

from sheily_core.llm import sheily_llm_bridge as bridge
from sheily_core.memory import sheily_memory_vault as vault
from sheily_core.memory import sheily_pdf_extractor as pdf
from sheily_core.memory import sheily_text_cleaner as cleaner

USER_ID = "user_local"
MODE = "strict"  # "strict" | "soft"
SIM_THRESHOLD = 0.35

# ——— Comandos de aprendizaje/memoria ———
LEARN_PATTERNS = [
    r"^\s*(?:sheily)?\s*(?:memoriza|recuerda|aprende|guarda|almacena|quiero que (?:aprendas|guardes|memorices|recuerdes))[:,-]?\s*(.+)$",
]
# ——— Comandos de borrado ———
FORGET_EXACT_PATTERNS = [
    r"^\s*(?:olvida|borra|elimina)\s+(?:este\s+(?:texto|cacho|fragmento)|esto)[:,-]?\s*(.+)$",  # "borra este cacho: …"
    r"^\s*(?:olvida|borra|elimina)[:,-]?\s*(.+)$",  # "borra: …"
]
FORGET_RELATED_PATTERNS = [
    r"^\s*(?:olvida|borra|elimina)\s+lo\s+relacionado\s+con[:,-]?\s*(.+)$",  # "borra lo relacionado con: …"
]


def _match_any(patterns, text):
    for p in patterns:
        m = re.match(p, text, flags=re.I | re.S)
        if m:
            return m.group(m.lastindex).strip()
    return None


def detect_learn_command(t):
    return _match_any(LEARN_PATTERNS, t)


def detect_forget_exact(t):
    return _match_any(FORGET_EXACT_PATTERNS, t)


def detect_forget_related(t):
    return _match_any(FORGET_RELATED_PATTERNS, t)


# ——— Prompt natural ———
def summarize_hits(hits, max_chars=600):
    facts = []
    for h in hits:
        s = h["text"].strip().replace("\n", " ")
        s = s.split(".")[0][:160]
        facts.append(s)
        if sum(len(f) for f in facts) > max_chars:
            break
    return facts


def build_prompt(question: str, facts: list):
    lines = "\n".join(f"- {f}" for f in facts)
    return (
        "Eres Sheily, una IA cercana y natural. Usa tus recuerdos (del chat y documentos) para responder "
        "con tus propias palabras. Si hay conflicto, prioriza los recuerdos del usuario. No cites textualmente.\n\n"
        f"Recuerdos relevantes:\n{lines}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta natural:"
    )


# ——— Aprendizaje: texto o archivos ———
def handle_learning_input(content: str) -> str:
    path = Path(content)
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext == ".pdf":
            chunks = pdf.extract_chunks_with_meta(path)
            total_ids = 0
            for ch in chunks:
                txt = cleaner.clean_text(ch["text"])
                ids = vault.remember_chunked(txt, USER_ID, origin="docs", meta=ch["meta"])
                total_ids += len(ids)
            return f"He leído y recordado {total_ids} fragmentos de {path.name}."
        elif ext == ".txt":
            txt = path.read_text(encoding="utf-8")
            txt = cleaner.clean_text(txt)
            ids = vault.remember_chunked(txt, USER_ID, origin="docs", meta={"source_file": path.name})
            return f"He guardado {len(ids)} fragmento(s) de {path.name}."
    # Texto directo
    txt = cleaner.clean_text(content)
    ids = vault.remember_chunked(txt, USER_ID, origin="chat", meta={"source": "chat"})
    return f"He memorizado {len(ids)} fragmento(s) del texto que me diste 😊"


# ——— Respuesta ———
def respond(message: str) -> str:
    # 1) Borrado relacionado
    related_q = detect_forget_related(message)
    if related_q:
        removed = vault.forget_related(USER_ID, related_q, threshold=0.45, top_k=12)
        return f'He borrado {removed} fragmento(s) relacionados con "{related_q}".'

    # 2) Borrado exacto / por fragmento
    exact_q = detect_forget_exact(message)
    if exact_q:
        removed = vault.forget_exact_fragment(USER_ID, exact_q, top_k=6)
        return f"He borrado {removed} fragmento(s) que contienen o se parecen a ese texto."

    # 3) Aprendizaje
    learn_txt = detect_learn_command(message)
    if learn_txt:
        return handle_learning_input(learn_txt)

    # 4) Responder usando memoria semántica combinada
    hits = vault.search_semantic(message, USER_ID, top_k=5)
    if not hits:
        return bridge.call_llm(message)
    top = hits[0]
    if MODE == "strict" or top["score"] >= SIM_THRESHOLD:
        facts = summarize_hits(hits)
        prompt = build_prompt(message, facts)
        return bridge.call_llm(prompt)
    return bridge.call_llm(message)
