#!/usr/bin/env python3
"""
Script de verificación del sistema de corpus bilingüe Sheily-AI
Verifica que ambos corpus (ES y EN) estén correctamente configurados

ZERO-DEPENDENCY VERSION - Solo usa Python stdlib
"""

import json
import os
from pathlib import Path


def verify_corpus(corpus_path, language):
    """Verifica la estructura y configuración de un corpus"""
    print(f"\n🔍 VERIFICANDO CORPUS {language.upper()} ({corpus_path})")

    if not os.path.exists(corpus_path):
        print(f"❌ Error: {corpus_path} no existe")
        return False

    domains = [d for d in os.listdir(corpus_path) if os.path.isdir(os.path.join(corpus_path, d))]
    print(f"📂 Dominios encontrados: {len(domains)}")

    # Verificar estructura de cada dominio
    configs_found = 0
    structures_complete = 0

    for domain in domains:
        domain_path = os.path.join(corpus_path, domain)

        # Verificar config file (YAML, JSON o texto plano)
        config_files = [
            ("domain_config.yaml", "yaml"),
            ("domain_config.json", "json"),
            ("domain_config.txt", "txt"),
        ]

        config_found = False
        for config_name, config_type in config_files:
            config_file = os.path.join(domain_path, config_name)
            if os.path.exists(config_file):
                configs_found += 1
                config_found = True
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        if config_type == "json":
                            config = json.load(f)
                            if config.get("language") == language:
                                print(f"  ✅ {domain}: Configuración JSON OK")
                            else:
                                print(f"  ⚠️  {domain}: Idioma incorrecto en config JSON")
                        elif config_type == "yaml":
                            # Parsing YAML básico con regex para zero-dependency
                            content = f.read()
                            lang_match = None
                            for line in content.split("\n"):
                                if line.strip().startswith("language:"):
                                    lang_match = line.split(":", 1)[1].strip().strip("\"'")
                                    break
                            if lang_match == language:
                                print(f"  ✅ {domain}: Configuración YAML OK")
                            else:
                                print(f"  ⚠️  {domain}: Idioma incorrecto en YAML ({lang_match} != {language})")
                        else:  # txt
                            content = f.read().lower()
                            if f"language={language}" in content or f"lang={language}" in content:
                                print(f"  ✅ {domain}: Configuración TXT OK")
                            else:
                                print(f"  ⚠️  {domain}: Idioma no encontrado en TXT")
                except Exception as e:
                    print(f"  ❌ {domain}: Error en config {config_type} - {e}")
                break

        if not config_found:
            print(f"  ❌ {domain}: Sin archivo de configuración")

        # Verificar estructura de directorios
        timestamp_dirs = [
            d for d in os.listdir(domain_path) if os.path.isdir(os.path.join(domain_path, d)) and d.startswith("2025")
        ]

        if timestamp_dirs:
            latest_dir = os.path.join(domain_path, timestamp_dirs[0])
            required_subdirs = ["chunks", "clean", "embeddings", "index", "logs", "raw"]
            existing_subdirs = [d for d in os.listdir(latest_dir) if os.path.isdir(os.path.join(latest_dir, d))]

            if all(subdir in existing_subdirs for subdir in required_subdirs):
                structures_complete += 1

    print(f"📊 RESUMEN {language.upper()}:")
    print(f"   - Dominios totales: {len(domains)}")
    print(f"   - Configuraciones: {configs_found}/{len(domains)}")
    print(f"   - Estructuras completas: {structures_complete}/{len(domains)}")

    if configs_found == len(domains) and structures_complete == len(domains):
        print(f"   🎉 ESTADO: ✅ COMPLETO")
        return True
    else:
        print(f"   ⚠️  ESTADO: INCOMPLETO")
        return False


def main():
    print("=" * 60)
    print("🌐 VERIFICACIÓN SISTEMA CORPUS BILINGÜE SHEILY-AI")
    print("=" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Verificar corpus español
    corpus_es = verify_corpus(os.path.join(base_path, "corpus_ES"), "es")

    # Verificar corpus inglés
    corpus_en = verify_corpus(os.path.join(base_path, "corpus_EN"), "en")

    print("\n" + "=" * 60)
    print("🏁 RESULTADO FINAL")
    print("=" * 60)

    if corpus_es and corpus_en:
        print("🎉 ¡SISTEMA BILINGÜE COMPLETAMENTE OPERATIVO!")
        print("✅ Corpus Español: Listo")
        print("✅ Corpus Inglés: Listo")
        print("🚀 El sistema RAG puede procesar material en ambos idiomas")
    else:
        print("⚠️  Sistema bilingüe requiere ajustes:")
        if not corpus_es:
            print("❌ Corpus Español: Incompleto")
        if not corpus_en:
            print("❌ Corpus Inglés: Incompleto")


if __name__ == "__main__":
    main()
