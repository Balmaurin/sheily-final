#!/usr/bin/env python3
"""
BÚSQUEDA PROFUNDA DE ISSUES Y OPORTUNIDADES
============================================
Encuentra problemas ocultos, archivos importantes, y mejoras.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path("/home/yo/sheily-pruebas-1.0-final")


def search_todo_fixme():
    """Buscar TODOs y FIXMEs en el código"""
    print("\n" + "=" * 70)
    print("  1. TODO/FIXME EN EL CÓDIGO")
    print("=" * 70 + "\n")

    patterns = [r"TODO", r"FIXME", r"XXX", r"HACK", r"BUG"]
    found = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    matches = re.findall(f".*{pattern}.*", content, re.IGNORECASE)
                    for match in matches[:3]:  # Max 3 per file
                        found.append((py_file.relative_to(PROJECT_ROOT), match.strip()))
        except:
            pass

    if found:
        print(f"Encontrados {len(found)} comentarios de desarrollo:\n")
        for file, comment in found[:20]:  # Show first 20
            print(f"📝 {file}")
            print(f"   {comment[:100]}")
            print()
    else:
        print("✅ No se encontraron TODOs/FIXMEs pendientes")


def search_empty_files():
    """Buscar archivos vacíos o casi vacíos"""
    print("=" * 70)
    print("  2. ARCHIVOS VACÍOS O MUY PEQUEÑOS")
    print("=" * 70 + "\n")

    empty = []
    tiny = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        size = py_file.stat().st_size

        if size == 0:
            empty.append(py_file.relative_to(PROJECT_ROOT))
        elif size < 100 and py_file.name != "__init__.py":
            tiny.append((py_file.relative_to(PROJECT_ROOT), size))

    if empty:
        print(f"⚠️  Archivos vacíos ({len(empty)}):")
        for f in empty[:10]:
            print(f"   • {f}")
        print()
    else:
        print("✅ No hay archivos vacíos\n")

    if tiny:
        print(f"⚠️  Archivos muy pequeños (<100 bytes, {len(tiny)}):")
        for f, size in tiny[:10]:
            print(f"   • {f} ({size} bytes)")
        print()
    else:
        print("✅ No hay archivos sospechosamente pequeños\n")


def search_large_files():
    """Buscar archivos muy grandes"""
    print("=" * 70)
    print("  3. ARCHIVOS MUY GRANDES (>5 MB)")
    print("=" * 70 + "\n")

    large = []

    for file in PROJECT_ROOT.rglob("*"):
        if file.is_file() and "venv" not in str(file):
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb > 5:
                large.append((file.relative_to(PROJECT_ROOT), size_mb))

    if large:
        print(f"⚠️  Archivos grandes encontrados ({len(large)}):\n")
        for f, size in sorted(large, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {f}")
            print(f"     Tamaño: {size:.1f} MB")
            print()
    else:
        print("✅ No hay archivos excesivamente grandes\n")


def search_missing_requirements():
    """Buscar imports sin requirements.txt"""
    print("=" * 70)
    print("  4. IMPORTS EXTERNOS (verificar requirements.txt)")
    print("=" * 70 + "\n")

    imports = set()

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            # Find import statements
            for line in content.split("\n"):
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    match = re.match(r"(?:from|import)\s+([a-zA-Z0-9_]+)", line.strip())
                    if match:
                        pkg = match.group(1)
                        # Skip stdlib
                        if pkg not in [
                            "os",
                            "sys",
                            "json",
                            "pathlib",
                            "datetime",
                            "re",
                            "time",
                            "typing",
                            "dataclasses",
                            "functools",
                            "asyncio",
                            "logging",
                            "argparse",
                            "collections",
                            "itertools",
                        ]:
                            imports.add(pkg)
        except:
            pass

    # Check if requirements.txt exists
    req_file = PROJECT_ROOT / "requirements.txt"

    if req_file.exists():
        print(f"✅ requirements.txt existe\n")
        print("📦 Imports externos encontrados:")
        for imp in sorted(imports)[:20]:
            print(f"   • {imp}")
    else:
        print(f"⚠️  requirements.txt NO EXISTE\n")
        print("📦 Imports externos que necesitan requirements.txt:")
        for imp in sorted(imports)[:20]:
            print(f"   • {imp}")

    print()


def search_duplicate_names():
    """Buscar archivos con nombres duplicados"""
    print("=" * 70)
    print("  5. ARCHIVOS CON NOMBRES DUPLICADOS")
    print("=" * 70 + "\n")

    names = {}

    for file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(file) or "__pycache__" in str(file):
            continue

        name = file.name
        if name not in names:
            names[name] = []
        names[name].append(file.relative_to(PROJECT_ROOT))

    duplicates = {k: v for k, v in names.items() if len(v) > 1 and k != "__init__.py"}

    if duplicates:
        print(f"⚠️  Archivos con nombres duplicados ({len(duplicates)}):\n")
        for name, paths in sorted(duplicates.items())[:10]:
            print(f"📝 {name}:")
            for p in paths:
                print(f"   • {p}")
            print()
    else:
        print("✅ No hay nombres duplicados (excepto __init__.py)\n")


def search_broken_imports():
    """Buscar posibles imports rotos"""
    print("=" * 70)
    print("  6. POSIBLES IMPORTS ROTOS")
    print("=" * 70 + "\n")

    issues = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            # Look for relative imports that might be broken
            if "from ." in content or "from .." in content:
                # Check if __init__.py exists in parent
                parent = py_file.parent
                if not (parent / "__init__.py").exists():
                    issues.append(py_file.relative_to(PROJECT_ROOT))
        except:
            pass

    if issues:
        print(f"⚠️  Archivos con imports relativos sin __init__.py ({len(issues)}):\n")
        for f in issues[:10]:
            print(f"   • {f}")
        print()
    else:
        print("✅ No se detectaron imports rotos obvios\n")


def search_important_files():
    """Buscar archivos importantes que deberían existir"""
    print("=" * 70)
    print("  7. ARCHIVOS IMPORTANTES DEL PROYECTO")
    print("=" * 70 + "\n")

    important = {
        "README.md": "Documentación principal",
        "requirements.txt": "Dependencias Python",
        ".gitignore": "Control de versiones",
        "LICENSE": "Licencia del proyecto",
        "setup.py": "Instalación del proyecto",
        ".env.example": "Variables de entorno ejemplo",
        "docker-compose.yml": "Configuración Docker",
        "Makefile": "Automatización de tareas",
    }

    for file, desc in important.items():
        path = PROJECT_ROOT / file
        if path.exists():
            print(f"✅ {file:25} | {desc}")
        else:
            print(f"⚠️  {file:25} | {desc} (FALTA)")

    print()


def search_test_coverage():
    """Verificar cobertura de tests"""
    print("=" * 70)
    print("  8. COBERTURA DE TESTS")
    print("=" * 70 + "\n")

    test_files = list(PROJECT_ROOT.rglob("test_*.py"))
    test_files += list(PROJECT_ROOT.rglob("*_test.py"))

    tests_dir = PROJECT_ROOT / "tests"
    if tests_dir.exists():
        test_in_tests = len(list(tests_dir.rglob("*.py")))
    else:
        test_in_tests = 0

    print(f"Archivos de test encontrados: {len(test_files)}")
    print(f"Tests en carpeta tests/: {test_in_tests}")

    if len(test_files) > 10:
        print("✅ Buena cobertura de tests")
    elif len(test_files) > 0:
        print("⚠️  Cobertura de tests básica")
    else:
        print("❌ Sin tests detectados")

    print()


def main():
    print("\n" + "🔍 " * 35)
    print("BÚSQUEDA PROFUNDA DE ISSUES Y OPORTUNIDADES")
    print("🔍 " * 35)

    search_todo_fixme()
    search_empty_files()
    search_large_files()
    search_missing_requirements()
    search_duplicate_names()
    search_broken_imports()
    search_important_files()
    search_test_coverage()

    print("=" * 70)
    print("  RESUMEN DE LA BÚSQUEDA")
    print("=" * 70)
    print("✅ Búsqueda completa finalizada")
    print("📊 Revisa los resultados anteriores para mejorar el proyecto")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
