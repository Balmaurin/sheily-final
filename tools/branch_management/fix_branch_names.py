#!/usr/bin/env python3
"""
FIX BRANCH NAMES - Branch Management Tool
==========================================
Standardize branch names from Spanish to English.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_PATH = PROJECT_ROOT / "all-Branches"

# Spanish to English mapping
BRANCH_NAME_MAPPING = {
    "antropologia": "anthropology",
    "arquitectura": "architecture", 
    "arte": "art",
    "inteligencia_artificial": "artificial_intelligence",
    "astronomia": "astronomy",
    "biologia": "biology",
    "quimica": "chemistry",
    "informatica": "computer_science",
    "ciberseguridad": "cybersecurity",
    "ecologia": "ecology",
    "educacion": "education",
    "ingenieria": "engineering",
    "etica": "ethics",
    "fisica": "physics",
    # Add more mappings as needed
}

def fix_jsonl_file(file_path: Path, old_branch: str, new_branch: str):
    """Fix branch names in a JSONL file"""
    lines = []
    updated = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("branch") == old_branch:
                    data["branch"] = new_branch
                    updated += 1
                lines.append(json.dumps(data, ensure_ascii=False))
            except:
                lines.append(line.strip())
    
    if updated > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    return updated

def fix_branch_datasets(branch_path: Path, branch_name: str):
    """Fix all datasets of a branch"""
    training_data = branch_path / "training" / "data"
    
    if not training_data.exists():
        return 0, 0
    
    jsonl_files = list(training_data.glob("*.jsonl"))
    if not jsonl_files:
        return 0, 0
    
    # Detect current branch name
    try:
        with open(jsonl_files[0], 'r', encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            current_branch = data.get("branch", "")
            
            if current_branch == branch_name:
                return 0, 0
            
            if current_branch in BRANCH_NAME_MAPPING:
                expected = BRANCH_NAME_MAPPING[current_branch]
                if expected == branch_name:
                    total_updated = 0
                    for jsonl_file in jsonl_files:
                        updated = fix_jsonl_file(jsonl_file, current_branch, branch_name)
                        total_updated += updated
                    
                    return len(jsonl_files), total_updated
    except:
        pass
    
    return 0, 0

def main():
    print("\n" + "="*70)
    print("  FIX BRANCH NAMES")
    print("="*70 + "\n")
    
    total_files = 0
    total_entries = 0
    branches_fixed = 0
    
    for branch_dir in sorted(BASE_PATH.iterdir()):
        if not branch_dir.is_dir():
            continue
        
        branch_name = branch_dir.name
        files, entries = fix_branch_datasets(branch_dir, branch_name)
        
        if files > 0:
            print(f"✅ {branch_name:30} | {files} archivos | {entries} entradas corregidas")
            total_files += files
            total_entries += entries
            branches_fixed += 1
    
    print("\n" + "="*70)
    print(f"Ramas corregidas: {branches_fixed}")
    print(f"Archivos actualizados: {total_files}")
    print(f"Entradas modificadas: {total_entries}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
