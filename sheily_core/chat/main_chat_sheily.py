#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_chat_sheily.py
===================
Interfaz principal de conversación con Sheily (memoria híbrida + modelo local).
"""

from __future__ import annotations

import sys

from sheily_core.chat import sheily_chat_memory_adapter as chat


def main():
    print("💬 Chat Sheily con memoria híbrida (chat + documentos)")
    print("Comandos:")
    print(' - "Sheily memoriza / guarda / aprende: <texto|ruta>"')
    print(' - "borra este cacho: <fragmento>" o "borra: <fragmento>"')
    print(' - "borra lo relacionado con: <tema>"')
    print(' - "salir" para terminar.\n')
    while True:
        try:
            msg = input("Tú: ").strip()
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in {"salir", "exit", "quit"}:
            print("Sheily: Hasta pronto 💫")
            break
        resp = chat.respond(msg)
        print(f"Sheily: {resp}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSheily: sesión terminada.")
        sys.exit(0)
