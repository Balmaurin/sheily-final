#!/usr/bin/env python3
"""
ANALYZE CORE STRUCTURE - Maintenance Tool
==========================================
Analyze sheily_core/ for duplicates, obsolete files, and structure issues.
Part of Sheily AI Enterprise Maintenance System.
"""

from pathlib import Path
import hashlib
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SHEILY_CORE = PROJECT_ROOT / "sheily_core"

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
    print("  DUPLICATE ANALYSIS")
    print("="*70 + "\n")
    
    comparisons = [
        ("sheily_core/config.py", "sheily_core/utils/config.py"),
        ("sheily_core/logger.py", "sheily_core/utils/logger.py"),
        ("sheily_core/utils/utility.py", "sheily_core/utils/utils.py"),
    ]
    
    duplicates = []
    
    for file1, file2 in comparisons:
        path1 = PROJECT_ROOT / file1
        path2 = PROJECT_ROOT / file2
        
        if not path1.exists() or not path2.exists():
            continue
        
        hash1 = file_hash(path1)
        hash2 = file_hash(path2)
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        
        if hash1 == hash2:
            duplicates.append((file1, file2, size1))
            print(f"🔴 EXACT DUPLICATE:")
            print(f"   • {file1} ({size1} bytes)")
            print(f"   • {file2} ({size2} bytes)\n")
        else:
            content2 = read_first_lines(path2, 10)
            if "from .." in content2 and "import *" in content2:
                print(f"🟡 WRAPPER (redirects):")
                print(f"   • {file2} -> redirects to another module\n")
            else:
                print(f"✅ DIFFERENT (distinct purposes):")
                print(f"   • {file1} ({size1} bytes)")
                print(f"   • {file2} ({size2} bytes)\n")
    
    return duplicates

def analyze_structure():
    """Analyze overall structure"""
    print("="*70)
    print("  STRUCTURE OVERVIEW")
    print("="*70 + "\n")
    
    py_files = list(SHEILY_CORE.rglob("*.py"))
    utils_files = list((SHEILY_CORE / "utils").glob("*.py")) if (SHEILY_CORE / "utils").exists() else []
    
    print(f"Total Python files: {len(py_files)}")
    print(f"Files in utils/: {len(utils_files)}")
    print(f"Files in root: {len(list(SHEILY_CORE.glob('*.py')))}\n")

def main():
    print("\n" + "🔍 " * 35)
    print("SHEILY CORE STRUCTURE ANALYSIS")
    print("🔍 " * 35 + "\n")
    
    duplicates = analyze_duplicates()
    analyze_structure()
    
    print("="*70)
    print(f"Exact duplicates found: {len(duplicates)}")
    print("\n✅ Analysis completed")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
