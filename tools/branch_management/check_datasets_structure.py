#!/usr/bin/env python3
"""
CHECK DATASETS STRUCTURE - Branch Management Tool
=================================================
Validate dataset structure across all branches.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_PATH = PROJECT_ROOT / "all-Branches"


def check_branch(branch_path):
    """Check branch dataset structure"""
    datasets = branch_path / "datasets"
    if not datasets.exists():
        return "❌ No datasets/"

    # Count JSON files
    json_files = list(datasets.rglob("*.json"))
    json_files = [f for f in json_files if f.name != "master_index.json"]

    # Count subdirectories
    subdirs = [d for d in datasets.iterdir() if d.is_dir()]

    if len(json_files) == 0:
        return f"⚠️  {len(subdirs)} folders, 0 files"

    return f"✅ {len(subdirs)} folders, {len(json_files)} files"


def main():
    print("\n" + "=" * 70)
    print("  DATASET STRUCTURE VALIDATION")
    print("=" * 70 + "\n")

    for branch_dir in sorted(BASE_PATH.iterdir()):
        if branch_dir.is_dir():
            status = check_branch(branch_dir)
            print(f"{branch_dir.name:30} | {status}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
