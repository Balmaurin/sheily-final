#!/usr/bin/env python3
"""
FIND DUPLICATE SCRIPTS - Maintenance Tool
==========================================
Find scripts with duplicate functionality in scripts/ folder.
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def file_hash(filepath: Path) -> str:
    """Calculate MD5 hash"""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return "error"


def read_file_content(filepath: Path) -> str:
    """Read file content"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def find_exact_duplicates():
    """Find exact duplicate files by hash"""
    print("\n" + "=" * 70)
    print("  EXACT DUPLICATES (same hash)")
    print("=" * 70 + "\n")

    files = [f for f in SCRIPTS_DIR.glob("*") if f.is_file() and f.name != "__init__.py"]

    hash_dict = {}
    for file in files:
        h = file_hash(file)
        if h not in hash_dict:
            hash_dict[h] = []
        hash_dict[h].append(file)

    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}

    if duplicates:
        for hash_val, files in duplicates.items():
            print(f"🔴 EXACT MATCH (hash: {hash_val[:16]}...):")
            for f in files:
                size = f.stat().st_size
                print(f"   • {f.name:40} ({size:>6} bytes)")
            print()
    else:
        print("✅ No exact duplicates found\n")

    return len(duplicates)


def find_similar_names():
    """Find files with similar names (potential duplicates)"""
    print("=" * 70)
    print("  SIMILAR NAMES (potential functional duplicates)")
    print("=" * 70 + "\n")

    groups = {"testing": [], "setup": [], "ultra_fast": [], "llm": [], "quality": [], "cline": []}

    for file in SCRIPTS_DIR.glob("*"):
        if file.name == "__init__.py":
            continue
        name = file.name.lower()

        if "test" in name:
            groups["testing"].append(file)
        if "setup" in name or "initialize" in name:
            groups["setup"].append(file)
        if "ultra" in name or "fast" in name:
            groups["ultra_fast"].append(file)
        if "llm" in name or "llama" in name or "local" in name:
            groups["llm"].append(file)
        if "quality" in name or "audit" in name:
            groups["quality"].append(file)
        if "cline" in name or "migrate" in name:
            groups["cline"].append(file)

    for group_name, files in groups.items():
        if len(files) > 1:
            print(f"⚠️  {group_name.upper()} group ({len(files)} files):")
            for f in sorted(files):
                size = f.stat().st_size
                print(f"   • {f.name:40} ({size:>6} bytes)")
            print()

    return sum(1 for files in groups.values() if len(files) > 1)


def compare_ultra_fast_scripts():
    """Compare the 3 ultra_fast scripts in detail"""
    print("=" * 70)
    print("  ULTRA_FAST SCRIPTS DETAILED COMPARISON")
    print("=" * 70 + "\n")

    ultra_files = [
        SCRIPTS_DIR / "initialize_ultra_fast.py",
        SCRIPTS_DIR / "setup_ultra_fast.py",
        SCRIPTS_DIR / "test_ultra_fast.py",
    ]

    for f in ultra_files:
        if f.exists():
            content = read_file_content(f)
            lines = content.split("\n")

            # Extract main functions
            functions = [line.strip() for line in lines if line.strip().startswith("def ")]

            print(f"📄 {f.name}:")
            print(f"   Size: {f.stat().st_size} bytes")
            print(f"   Lines: {len(lines)}")
            print(f"   Functions: {len(functions)}")
            if functions:
                print(f"   Main functions:")
                for func in functions[:5]:  # Show first 5
                    print(f"      • {func[:60]}")
            print()


def compare_test_scripts():
    """Compare test scripts"""
    print("=" * 70)
    print("  TEST SCRIPTS COMPARISON")
    print("=" * 70 + "\n")

    test_files = [
        SCRIPTS_DIR / "test_api.py",
        SCRIPTS_DIR / "test_local_llm_api.py",
        SCRIPTS_DIR / "test_quick_web_system.sh",
        SCRIPTS_DIR / "test_sheily_web_system.sh",
        SCRIPTS_DIR / "test_ultra_fast.py",
    ]

    for f in test_files:
        if f.exists():
            content = read_file_content(f)

            # Check what they test
            purpose = "Unknown"
            if "api" in f.name.lower():
                purpose = "API Testing"
            elif "web" in f.name.lower():
                purpose = "Web System Testing"
            elif "ultra" in f.name.lower():
                purpose = "Ultra Fast System Testing"

            print(f"🧪 {f.name:40} -> {purpose}")
            print(f"   Size: {f.stat().st_size:>6} bytes")

            # Check for key indicators
            if "curl" in content or "requests.get" in content:
                print(f"   Type: HTTP/API calls")
            if "pytest" in content or "unittest" in content:
                print(f"   Framework: Testing framework detected")
            if "#!/bin/bash" in content:
                print(f"   Type: Shell script")
            print()


def main():
    print("\n" + "🔍 " * 35)
    print("DUPLICATE SCRIPTS ANALYSIS")
    print("🔍 " * 35)

    exact_dups = find_exact_duplicates()
    similar_groups = find_similar_names()

    compare_ultra_fast_scripts()
    compare_test_scripts()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Exact duplicate groups: {exact_dups}")
    print(f"Similar name groups: {similar_groups}")

    print("\n📋 RECOMMENDATIONS:")
    print("   1. ultra_fast scripts: Revisar si initialize/setup son duplicados")
    print("   2. test scripts: Consolidar o documentar diferencias")
    print("   3. cline scripts: Eliminar (legacy)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
