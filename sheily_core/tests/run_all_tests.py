#!/usr/bin/env python3
"""
🧪 RUNNER MAESTRO - SUITE PRESIDENCIAL DE TESTS
Ejecuta TODOS los tests del proyecto de forma organizada

Suite Presidencial:
- test_generator.py       → Generador de datasets
- test_trainer.py         → Sistema de entrenamiento
- test_chat.py            → Aplicación de chat
- test_interactuar.py     → Script de interacción
- test_merge.py           → Fusión de modelos
- test_listar_ramas.py    → Listado de ramas
- test_branches.py        → Sistema de ramas
- test_sheily_model.py    → Modelo Sheily
- test_datasets.py        → Datasets y validación
- test_config.py          → Configuración
- test_documentation.py   → Documentación
- test_project_structure.py → Estructura del proyecto
- test_rag_core.py        → Sistema RAG vMAX
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_and_run_tests():
    """Descubre y ejecuta todos los tests (excluyendo RAG que tiene su propia suite)"""

    print("=" * 80)
    print("🏨 SUITE PRESIDENCIAL DE TESTS - SHEILY-AI")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("ℹ️  Nota: Tests RAG se ejecutan con run_complete_rag_evaluation.py")
    print()

    # Descubrir tests excluyendo RAG
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent

    # Tests no-RAG útiles
    non_rag_tests = [
        "test_branches.py",
        "test_chat.py",
        "test_config.py",
        "test_datasets.py",
        "test_documentation.py",
        "test_generator.py",
        "test_interactuar.py",
        "test_listar_ramas.py",
        "test_merge.py",
        "test_project_structure.py",
        "test_rag_core.py",
        "test_trainer.py",
    ]

    suite = unittest.TestSuite()
    for test_file in non_rag_tests:
        test_path = start_dir / test_file
        if test_path.exists():
            try:
                test_suite = loader.discover(str(start_dir), pattern=test_file)
                suite.addTest(test_suite)
            except Exception as e:
                print(f"⚠️  Error cargando {test_file}: {e}")

    # Contar tests
    def count_tests(suite_or_test):
        try:
            # unittest TestSuite es iterable de tests o sub-suites
            if isinstance(suite_or_test, unittest.TestSuite):
                return sum(count_tests(t) for t in suite_or_test)
            else:
                return 1
        except Exception:
            return 0

    total_tests = count_tests(suite)
    print(f"📊 Total de tests descubiertos: {total_tests}")
    print()

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumen detallado
    print()
    print("=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    print(f"✅ Tests pasados:  {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests fallados: {len(result.failures)}")
    print(f"⚠️  Errores:        {len(result.errors)}")
    print(f"⏭️  Saltados:       {len(result.skipped)}")
    print(f"📈 Total:          {result.testsRun}")
    print()

    # Calcular porcentaje
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f"🎯 Tasa de éxito: {success_rate:.1f}%")

    print("=" * 80)

    # Detalles de fallos
    if result.failures:
        print()
        print("❌ TESTS FALLADOS:")
        for test, tb in result.failures:
            print(f"- {test}")
            print(tb)

    if result.errors:
        print()
        print("⚠️  ERRORES:")
        for test, tb in result.errors:
            print(f"- {test}")
            print(tb)

    print()

    return result.wasSuccessful()


def list_test_modules():
    """Lista todos los módulos de test"""
    print()
    print("📦 MÓDULOS DE TEST DISPONIBLES:")
    print()

    test_files = sorted(Path(__file__).parent.glob("test_*.py"))

    for i, test_file in enumerate(test_files, 1):
        if test_file.name == "test_dataset.py":
            continue  # Saltar el antiguo

        # Leer docstring
        content = test_file.read_text()
        lines = content.split("\n")
        docstring = ""

        for line in lines:
            if '"""' in line and not line.strip().startswith("#"):
                docstring = line.split('"""')[1].strip()
                break

        print(f"   {i:2d}. {test_file.name:30s} - {docstring}")

    print()


if __name__ == "__main__":
    list_test_modules()
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)
