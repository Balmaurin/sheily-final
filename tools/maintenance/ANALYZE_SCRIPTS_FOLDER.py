#!/usr/bin/env python3
"""
ANÁLISIS DE CARPETA SCRIPTS/
=============================
Categoriza scripts en: importantes, duplicados, obsoletos, movibles
"""

import re
from pathlib import Path

SCRIPTS_DIR = Path("/home/yo/sheily-pruebas-1.0-final/scripts")


def categorize_scripts():
    """Categorize all scripts"""
    categories = {
        "setup_initialization": [],
        "testing": [],
        "ci_cd": [],
        "deployment": [],
        "quality_security": [],
        "legacy_cline": [],
        "ultra_fast": [],
        "llm_specific": [],
        "utilities": [],
    }

    for script in sorted(SCRIPTS_DIR.glob("*")):
        if script.name == "__init__.py":
            continue

        name = script.name.lower()

        # Categorize
        if "cline" in name or "migrate" in name:
            categories["legacy_cline"].append(script)
        elif "ultra_fast" in name:
            categories["ultra_fast"].append(script)
        elif "llama" in name or "llm" in name or "local" in name:
            categories["llm_specific"].append(script)
        elif "test" in name:
            categories["testing"].append(script)
        elif "setup" in name or "initialize" in name or "activate" in name:
            categories["setup_initialization"].append(script)
        elif "ci" in name or "pipeline" in name or "pre-commit" in name:
            categories["ci_cd"].append(script)
        elif "deploy" in name or "blue" in name:
            categories["deployment"].append(script)
        elif (
            "quality" in name
            or "security" in name
            or "audit" in name
            or "validate" in name
            or "analyze" in name
            or "standards" in name
        ):
            categories["quality_security"].append(script)
        else:
            categories["utilities"].append(script)

    return categories


def analyze_importance():
    """Determine which scripts are important"""
    print("\n" + "=" * 70)
    print("  ANÁLISIS DE IMPORTANCIA - scripts/")
    print("=" * 70 + "\n")

    categories = categorize_scripts()

    # Critical
    print("🔴 CRÍTICOS (mantener):")
    critical = ["activate_sheily.sh", "ci_pipeline.sh", "quality_gate_local.sh", "validate_structure.py"]
    for name in critical:
        path = SCRIPTS_DIR / name
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {name:40} ({size:>6} bytes)")
    print()

    # Important
    print("🟡 IMPORTANTES (revisar uso):")
    important = ["audit_quick.py", "quality_dashboard.py", "security_remediation.sh", "pre-commit"]
    for name in important:
        path = SCRIPTS_DIR / name
        if path.exists():
            size = path.stat().st_size
            print(f"   • {name:40} ({size:>6} bytes)")
    print()

    # Legacy/Obsolete
    print("⚠️  LEGACY/OBSOLETOS (considerar eliminar):")
    legacy = categories["legacy_cline"]
    for script in legacy:
        size = script.stat().st_size
        print(f"   🗑️  {script.name:40} ({size:>6} bytes)")
    print()

    # Ultra Fast (duplicados?)
    print("🔍 ULTRA_FAST (posibles duplicados):")
    ultra = categories["ultra_fast"]
    for script in ultra:
        size = script.stat().st_size
        print(f"   • {script.name:40} ({size:>6} bytes)")
    print()

    # LLM Specific
    print("🤖 LLM ESPECÍFICOS:")
    llm = categories["llm_specific"]
    for script in llm:
        size = script.stat().st_size
        print(f"   • {script.name:40} ({size:>6} bytes)")
    print()


def generate_recommendations():
    """Generate reorganization recommendations"""
    print("=" * 70)
    print("  RECOMENDACIONES DE REORGANIZACIÓN")
    print("=" * 70 + "\n")

    recommendations = [
        {
            "action": "ELIMINAR",
            "files": ["cline_tasks.sh", "migrate_from_cline_workflows.sh"],
            "reason": "Scripts legacy de Cline - ya no se usan",
            "priority": "ALTA",
        },
        {
            "action": "CONSOLIDAR",
            "files": ["initialize_ultra_fast.py", "setup_ultra_fast.py", "test_ultra_fast.py"],
            "reason": "3 scripts 'ultra_fast' - verificar si son duplicados",
            "priority": "MEDIA",
        },
        {
            "action": "MOVER A tools/maintenance/",
            "files": ["validate_structure.py", "analyze_dead_code.py", "workspace_cleanup.sh"],
            "reason": "Son herramientas de mantenimiento, no scripts operacionales",
            "priority": "MEDIA",
        },
        {
            "action": "MOVER A tools/deployment/",
            "files": ["enterprise_blue_green_deployment.py"],
            "reason": "Script de deployment debería estar en tools/",
            "priority": "BAJA",
        },
        {
            "action": "MANTENER EN scripts/",
            "files": ["activate_sheily.sh", "ci_pipeline.sh", "pre-commit", "quality_gate_local.sh"],
            "reason": "Scripts operacionales críticos del día a día",
            "priority": "N/A",
        },
        {
            "action": "REVISAR USO",
            "files": ["test_api.py", "test_local_llm_api.py", "test_quick_web_system.sh", "test_sheily_web_system.sh"],
            "reason": "Múltiples scripts de testing - verificar cuáles se usan",
            "priority": "MEDIA",
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['action']}")
        print(f"   Archivos: {', '.join(rec['files'])}")
        print(f"   Razón: {rec['reason']}\n")


def count_stats():
    """Count statistics"""
    print("=" * 70)
    print("  ESTADÍSTICAS")
    print("=" * 70 + "\n")

    categories = categorize_scripts()

    total = sum(len(files) for files in categories.values())

    print(f"Total scripts: {total}")
    print(f"Legacy/Cline: {len(categories['legacy_cline'])} ⚠️")
    print(f"Ultra Fast: {len(categories['ultra_fast'])} 🔍")
    print(f"Testing: {len(categories['testing'])} 🧪")
    print(f"Quality/Security: {len(categories['quality_security'])} 🛡️")
    print(f"CI/CD: {len(categories['ci_cd'])} 🔄")
    print(f"Setup/Init: {len(categories['setup_initialization'])} 🚀")
    print(f"LLM Specific: {len(categories['llm_specific'])} 🤖")
    print(f"Deployment: {len(categories['deployment'])} 📦")
    print()


def main():
    print("\n" + "🔍 " * 35)
    print("ANÁLISIS DE CARPETA SCRIPTS/")
    print("🔍 " * 35)

    analyze_importance()
    generate_recommendations()
    count_stats()

    print("=" * 70)
    print("✅ Análisis completado")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
