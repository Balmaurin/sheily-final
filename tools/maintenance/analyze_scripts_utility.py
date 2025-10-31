#!/usr/bin/env python3
"""
ANALYZE SCRIPTS UTILITY - Para proyecto EN MARCHA
==================================================
Determina qué scripts son útiles ahora vs obsoletos/legacy
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def categorize_by_current_utility():
    """Categorizar por utilidad ACTUAL"""
    
    categories = {
        "ÚTILES_DIARIO": {
            "desc": "Scripts que SÍ usarías regularmente",
            "scripts": [
                ("activate_sheily.sh", "Activar entorno - útil si trabajas con venv"),
                ("ci_pipeline.sh", "Pipeline CI/CD - útil para validar antes de commits"),
                ("pre-commit", "Hook de git - validación automática"),
                ("quality_gate_local.sh", "Validar calidad antes de push"),
            ]
        },
        "ÚTILES_OCASIONAL": {
            "desc": "Scripts útiles de vez en cuando",
            "scripts": [
                ("audit_quick.py", "Auditoría rápida del código"),
                ("quality_dashboard.py", "Dashboard de calidad del proyecto"),
                ("security_remediation.sh", "Remediación de seguridad"),
                ("validate_structure.py", "Validar estructura del proyecto"),
                ("workspace_cleanup.sh", "Limpiar workspace"),
            ]
        },
        "SETUP_INICIAL": {
            "desc": "Solo para setup inicial - YA NO NECESARIOS",
            "scripts": [
                ("initialize_ultra_fast.py", "Inicialización - YA HECHO"),
                ("setup_ultra_fast.py", "Setup - YA HECHO"),
                ("setup_llama_3_2_1b_q4.sh", "Setup LLM - YA HECHO si usas LLM"),
            ]
        },
        "TESTING": {
            "desc": "Scripts de testing - útiles para QA",
            "scripts": [
                ("test_api.py", "Test API - útil si tienes API"),
                ("test_local_llm_api.py", "Test LLM - útil si usas LLM local"),
                ("test_quick_web_system.sh", "Test web rápido"),
                ("test_sheily_web_system.sh", "Test web completo"),
                ("test_ultra_fast.py", "Test sistema ultra fast"),
            ]
        },
        "LEGACY_ELIMINAR": {
            "desc": "Scripts OBSOLETOS - ELIMINAR",
            "scripts": [
                ("cline_tasks.sh", "Legacy Cline - YA NO SE USA"),
                ("migrate_from_cline_workflows.sh", "Migración Cline - YA NO SE USA"),
            ]
        },
        "DESARROLLO_AVANZADO": {
            "desc": "Features avanzadas - opcional",
            "scripts": [
                ("advanced_ml_features.py", "Features ML avanzadas - solo si desarrollas ML"),
                ("advanced_optimization.py", "Optimizaciones avanzadas - solo si optimizas"),
                ("analyze_dead_code.py", "Análisis código muerto - mantenimiento"),
                ("apply_code_standards.sh", "Aplicar estándares - mantenimiento"),
                ("enterprise_blue_green_deployment.py", "Deployment enterprise - solo en producción"),
            ]
        },
        "SERVIDORES": {
            "desc": "Scripts de servidor - útiles si corres servidores",
            "scripts": [
                ("start_local_llm_server.sh", "Servidor LLM local - útil si usas LLM"),
            ]
        }
    }
    
    return categories

def print_analysis():
    """Imprimir análisis detallado"""
    print("\n" + "="*70)
    print("  ANÁLISIS DE UTILIDAD - PROYECTO EN MARCHA")
    print("="*70 + "\n")
    
    categories = categorize_by_current_utility()
    
    total_scripts = 0
    useful_count = 0
    obsolete_count = 0
    
    for cat_name, cat_data in categories.items():
        print(f"\n📁 {cat_name}")
        print(f"   {cat_data['desc']}")
        print("   " + "-"*66)
        
        for script_name, description in cat_data['scripts']:
            script_path = SCRIPTS_DIR / script_name
            exists = "✅" if script_path.exists() else "❌"
            
            print(f"   {exists} {script_name:40} - {description}")
            total_scripts += 1
            
            if cat_name in ["ÚTILES_DIARIO", "ÚTILES_OCASIONAL", "TESTING", "SERVIDORES"]:
                useful_count += 1
            elif cat_name == "LEGACY_ELIMINAR":
                obsolete_count += 1
    
    return total_scripts, useful_count, obsolete_count

def generate_recommendations():
    """Generar recomendaciones específicas"""
    print("\n" + "="*70)
    print("  RECOMENDACIONES PARA PROYECTO EN MARCHA")
    print("="*70 + "\n")
    
    print("🗑️  ELIMINAR INMEDIATAMENTE:")
    print("   • cline_tasks.sh")
    print("   • migrate_from_cline_workflows.sh")
    print("   Razón: Legacy de Cline, ya no se usan\n")
    
    print("📦 MOVER A tools/maintenance/:")
    print("   • validate_structure.py")
    print("   • analyze_dead_code.py")
    print("   • workspace_cleanup.sh")
    print("   Razón: Son herramientas de mantenimiento\n")
    
    print("📦 MOVER A tools/deployment/:")
    print("   • enterprise_blue_green_deployment.py")
    print("   Razón: Es deployment, no script diario\n")
    
    print("📦 ARCHIVAR (mover a scripts/archive/):")
    print("   • initialize_ultra_fast.py")
    print("   • setup_ultra_fast.py")
    print("   • setup_llama_3_2_1b_q4.sh")
    print("   Razón: Solo necesarios en setup inicial\n")
    
    print("✅ MANTENER EN scripts/:")
    print("   • activate_sheily.sh")
    print("   • ci_pipeline.sh")
    print("   • pre-commit")
    print("   • quality_gate_local.sh")
    print("   • Scripts de testing (test_*.py, test_*.sh)")
    print("   • start_local_llm_server.sh (si usas LLM)")
    print("   Razón: Uso diario/frecuente\n")

def print_quick_decision():
    """Decisión rápida para usuario apurado"""
    print("="*70)
    print("  RESPUESTA RÁPIDA")
    print("="*70 + "\n")
    
    print("❓ ¿Valen para algo AHORA que el proyecto está en marcha?\n")
    
    print("SÍ, SON ÚTILES (10-12 scripts):")
    print("   ✅ activate_sheily.sh - Activar entorno")
    print("   ✅ ci_pipeline.sh - Validar antes de commits")
    print("   ✅ pre-commit - Git hooks")
    print("   ✅ quality_gate_local.sh - Control de calidad")
    print("   ✅ Scripts test_* - Testing")
    print("   ✅ quality_dashboard.py - Monitoreo")
    print("   ✅ audit_quick.py - Auditorías")
    print("\nNO, SON OBSOLETOS/INNECESARIOS (15+ scripts):")
    print("   🗑️  cline_tasks.sh - ELIMINAR")
    print("   🗑️  migrate_from_cline_workflows.sh - ELIMINAR")
    print("   📦 initialize_ultra_fast.py - Archivar (ya hecho)")
    print("   📦 setup_ultra_fast.py - Archivar (ya hecho)")
    print("   📦 advanced_*.py - Solo si desarrollas features avanzadas")
    print("   📦 enterprise_blue_green_deployment.py - Solo en producción")
    print("\n💡 RESUMEN: De 26 scripts, solo ~10-12 son útiles regularmente")
    print("="*70 + "\n")

def main():
    print("\n" + "🔍 " * 35)
    print("UTILIDAD DE SCRIPTS - PROYECTO EN MARCHA")
    print("🔍 " * 35)
    
    total, useful, obsolete = print_analysis()
    generate_recommendations()
    print_quick_decision()
    
    print("="*70)
    print(f"Total scripts: {total}")
    print(f"Útiles: ~{useful} ({useful*100//total}%)")
    print(f"Obsoletos: {obsolete}")
    print(f"Archivar/Reorganizar: {total - useful - obsolete}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
