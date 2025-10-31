#!/usr/bin/env python3
"""
ANÁLISIS DE DUPLICADOS Y OBSOLETOS EN SHEILY_CORE
=================================================
Detecta archivos duplicados, obsoletos o innecesarios.
"""

from pathlib import Path
import hashlib

SHEILY_CORE = Path("/home/yo/sheily-pruebas-1.0-final/sheily_core")

def file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of file"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return "error"

def read_first_lines(filepath: Path, n=50) -> str:
    """Read first n lines of file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return ''.join(f.readlines()[:n])
    except:
        return ""

def analyze_duplicates():
    """Analyze potential duplicates"""
    print("\n" + "="*70)
    print("  ANÁLISIS DE DUPLICADOS EN SHEILY_CORE")
    print("="*70 + "\n")
    
    # Files to compare
    comparisons = [
        ("sheily_core/config.py", "sheily_core/utils/config.py"),
        ("sheily_core/logger.py", "sheily_core/utils/logger.py"),
        ("sheily_core/utils/utility.py", "sheily_core/utils/utils.py"),
    ]
    
    duplicates = []
    different = []
    
    for file1, file2 in comparisons:
        path1 = SHEILY_CORE.parent / file1
        path2 = SHEILY_CORE.parent / file2
        
        if not path1.exists() or not path2.exists():
            print(f"⚠️  Missing: {file1} or {file2}")
            continue
        
        hash1 = file_hash(path1)
        hash2 = file_hash(path2)
        
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        
        if hash1 == hash2:
            duplicates.append((file1, file2, size1))
            print(f"🔴 DUPLICADO EXACTO:")
            print(f"   • {file1} ({size1} bytes)")
            print(f"   • {file2} ({size2} bytes)")
            print(f"   Hash: {hash1}\n")
        else:
            # Check if one is a wrapper
            content2 = read_first_lines(path2, 10)
            if "from .." in content2 and "import *" in content2:
                print(f"🟡 WRAPPER (redirige a otro archivo):")
                print(f"   • {file2} -> redirige a otro módulo")
                print(f"   • Tamaño: {size2} bytes (muy pequeño)\n")
            else:
                different.append((file1, file2, size1, size2))
                print(f"✅ DIFERENTES (propósitos distintos):")
                print(f"   • {file1} ({size1} bytes)")
                print(f"   • {file2} ({size2} bytes)\n")
    
    return duplicates, different

def analyze_suspicious_files():
    """Analyze suspicious or potentially obsolete files"""
    print("="*70)
    print("  ARCHIVOS SOSPECHOSOS / POTENCIALMENTE OBSOLETOS")
    print("="*70 + "\n")
    
    suspicious = []
    
    # Very small files (< 100 bytes) might be wrappers or stubs
    utils_dir = SHEILY_CORE / "utils"
    if utils_dir.exists():
        for file in utils_dir.glob("*.py"):
            size = file.stat().st_size
            if size < 100 and file.name != "__init__.py":
                content = read_first_lines(file, 10)
                print(f"⚠️  ARCHIVO MUY PEQUEÑO: {file.name} ({size} bytes)")
                print(f"   Contenido: {content[:100]}...\n")
                suspicious.append(file)
    
    # Multiple error handling files
    error_files = [
        SHEILY_CORE / "utils" / "error_decorators.py",
        SHEILY_CORE / "utils" / "functional_errors.py",
        SHEILY_CORE / "utils" / "result.py"
    ]
    
    print("🔍 MÚLTIPLES ARCHIVOS DE MANEJO DE ERRORES:")
    for ef in error_files:
        if ef.exists():
            size = ef.stat().st_size
            content = read_first_lines(ef, 3)
            print(f"   • {ef.name} ({size} bytes)")
            print(f"     {content.split('\\n')[0] if content else 'N/A'}")
    print()
    
    return suspicious

def analyze_structure():
    """Analyze overall structure"""
    print("="*70)
    print("  ESTRUCTURA GENERAL")
    print("="*70 + "\n")
    
    # Count files
    py_files = list(SHEILY_CORE.rglob("*.py"))
    utils_files = list((SHEILY_CORE / "utils").glob("*.py")) if (SHEILY_CORE / "utils").exists() else []
    
    print(f"Total archivos Python: {len(py_files)}")
    print(f"Archivos en utils/: {len(utils_files)}")
    print(f"Archivos en root: {len(list(SHEILY_CORE.glob('*.py')))}\n")
    
    # List root files
    print("Archivos en sheily_core/ (root):")
    for f in sorted(SHEILY_CORE.glob("*.py")):
        size = f.stat().st_size
        print(f"  • {f.name:30} ({size:>6} bytes)")
    print()

def generate_recommendations():
    """Generate cleanup recommendations"""
    print("="*70)
    print("  RECOMENDACIONES")
    print("="*70 + "\n")
    
    recommendations = []
    
    # Check for wrappers
    wrapper = SHEILY_CORE / "utils" / "config.py"
    if wrapper.exists():
        content = read_first_lines(wrapper, 10)
        if "from .." in content and "import *" in content:
            recommendations.append({
                "action": "MANTENER",
                "file": "sheily_core/utils/config.py",
                "reason": "Es un wrapper de compatibilidad (compat layer)",
                "priority": "Bajo"
            })
    
    # Different logger files
    recommendations.append({
        "action": "REVISAR",
        "file": "sheily_core/logger.py vs utils/logger.py",
        "reason": "Dos implementaciones de logger diferentes - verificar cuál se usa",
        "priority": "Medio"
    })
    
    # utility.py vs utils.py
    recommendations.append({
        "action": "MANTENER AMBOS",
        "file": "utility.py y utils.py",
        "reason": "Tienen propósitos diferentes (utility=modelos, utils=HTTP/server)",
        "priority": "Bajo"
    })
    
    # Multiple error files
    recommendations.append({
        "action": "CONSOLIDAR",
        "file": "error_decorators.py, functional_errors.py, result.py",
        "reason": "3 archivos de manejo de errores - posible consolidación",
        "priority": "Medio"
    })
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['action']}: {rec['file']}")
        print(f"   Razón: {rec['reason']}")
        print(f"   Prioridad: {rec['priority']}\n")

def main():
    print("\n" + "🔍 " * 35)
    print("ANÁLISIS DE SHEILY_CORE")
    print("🔍 " * 35 + "\n")
    
    duplicates, different = analyze_duplicates()
    suspicious = analyze_suspicious_files()
    analyze_structure()
    generate_recommendations()
    
    print("="*70)
    print("  RESUMEN")
    print("="*70)
    print(f"Duplicados exactos encontrados: {len(duplicates)}")
    print(f"Archivos sospechosos: {len(suspicious)}")
    print(f"Pares de archivos diferentes: {len(different)}")
    print("\n✅ Análisis completado")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
