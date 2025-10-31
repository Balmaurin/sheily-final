#!/usr/bin/env python3
"""
VERIFY PROJECT STRUCTURE - Maintenance Tool
============================================
Verify that the complete project structure is correct.
Part of Sheily AI Enterprise Maintenance System.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_STRUCTURE = {
    "tools": {
        "subdirs": ["branch_management", "automation", "development", "maintenance"],
        "files": ["README.md", "__init__.py"],
    },
    "var": {"subdirs": ["central_data", "central_logs", "central_cache", "central_models"], "files": ["README.md"]},
    "docs": {"files": ["README.md", "TOOLS_GUIDE.md", "BRANCH_STATUS.md", "ARCHITECTURE.md"]},
    "all-Branches": {"type": "branches"},
}


def check_structure(base_path: Path, structure: dict, path_name: str = "") -> list:
    """Verify directory structure"""
    issues = []

    for key, requirements in structure.items():
        full_path = base_path / key if path_name == "" else base_path
        display_name = f"{path_name}/{key}" if path_name else key

        if not full_path.exists():
            issues.append(f"❌ Missing: {display_name}")
            continue

        if "subdirs" in requirements:
            for subdir in requirements["subdirs"]:
                subdir_path = full_path / subdir
                if not subdir_path.exists():
                    issues.append(f"❌ Missing subdir: {display_name}/{subdir}")

        if "files" in requirements:
            for file in requirements["files"]:
                file_path = full_path / file
                if not file_path.exists():
                    issues.append(f"⚠️  Missing file: {display_name}/{file}")

        if requirements.get("type") == "branches":
            branches = [d for d in full_path.iterdir() if d.is_dir()]
            if len(branches) < 50:
                issues.append(f"⚠️  Only {len(branches)} branches (expected 50+)")

    return issues


def count_files():
    """Count important files"""
    stats = {
        "tools_py": len(list((PROJECT_ROOT / "tools").rglob("*.py"))),
        "docs_md": len(list((PROJECT_ROOT / "docs").rglob("*.md"))),
        "branches": len([d for d in (PROJECT_ROOT / "all-Branches").iterdir() if d.is_dir()]),
    }
    return stats


def main():
    print("\n" + "=" * 70)
    print("  PROJECT STRUCTURE VERIFICATION")
    print("=" * 70 + "\n")

    all_issues = []

    for path_key, requirements in EXPECTED_STRUCTURE.items():
        if "/" not in path_key:
            issues = check_structure(PROJECT_ROOT, {path_key: requirements})
            all_issues.extend(issues)

    if all_issues:
        print("⚠️  ISSUES FOUND:\n")
        for issue in all_issues:
            print(f"  {issue}")
        print()
    else:
        print("✅ ALL STRUCTURE CHECKS PASSED!\n")

    stats = count_files()
    print("=" * 70)
    print("  PROJECT STATISTICS")
    print("=" * 70)
    print(f"  • Tools Python files: {stats['tools_py']}")
    print(f"  • Documentation files: {stats['docs_md']}")
    print(f"  • Domain branches: {stats['branches']}")
    print()

    print("=" * 70)
    if all_issues:
        print("  STATUS: ⚠️  NEEDS ATTENTION")
    else:
        print("  STATUS: ✅ PERFECTLY ORGANIZED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
