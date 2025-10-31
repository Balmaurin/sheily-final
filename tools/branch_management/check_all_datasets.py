#!/usr/bin/env python3
"""
CHECK ALL DATASETS - Branch Management Tool
============================================
Comprehensive validation of all 50 branches datasets.
Part of Sheily AI Enterprise Branch Management System.
"""

import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_PATH = PROJECT_ROOT / "all-Branches"

def check_branch_dataset(branch_path: Path, branch_name: str):
    """Verify dataset of a branch"""
    training_data = branch_path / "training" / "data"
    
    if not training_data.exists():
        return {"status": "❌", "reason": "No training/data folder", "files": 0, "branch": "unknown"}
    
    jsonl_files = list(training_data.glob("*.jsonl"))
    
    if not jsonl_files:
        return {"status": "❌", "reason": "No JSONL files", "files": 0, "branch": "unknown"}
    
    # Read first file to verify content
    try:
        with open(jsonl_files[0], 'r', encoding='utf-8') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            
            # Verify branch matches
            if data.get("branch") == branch_name:
                return {"status": "✅", "reason": "Valid & specific", "files": len(jsonl_files), "branch": data.get("branch")}
            elif "fisica" in first_line:
                return {"status": "⚠️", "reason": "Contains 'fisica'", "files": len(jsonl_files), "branch": data.get("branch", "unknown")}
            else:
                return {"status": "❓", "reason": f"Branch mismatch: {data.get('branch')}", "files": len(jsonl_files), "branch": data.get("branch", "unknown")}
    except Exception as e:
        return {"status": "❌", "reason": f"Error: {str(e)[:30]}", "files": len(jsonl_files), "branch": "error"}

def main():
    print("\n" + "="*80)
    print("  CHECK ALL DATASETS - Branch Validation Tool")
    print("="*80 + "\n")
    
    branches_by_status = {
        "✅": [],
        "❌": [],
        "⚠️": [],
        "❓": []
    }
    
    # Verify all branches
    for branch_dir in sorted(BASE_PATH.iterdir()):
        if not branch_dir.is_dir():
            continue
        
        branch_name = branch_dir.name
        result = check_branch_dataset(branch_dir, branch_name)
        
        print(f"{result['status']} {branch_name:30} | {result['files']:2} files | {result['reason']:30} | branch={result['branch']}")
        
        branches_by_status[result['status']].append(branch_name)
    
    print("\n" + "="*80)
    print("  SUMMARY BY STATUS")
    print("="*80)
    print(f"✅ Valid and specific: {len(branches_by_status['✅'])}")
    for branch in branches_by_status['✅']:
        print(f"   • {branch}")
    
    print(f"\n❌ Missing datasets or error: {len(branches_by_status['❌'])}")
    for branch in branches_by_status['❌']:
        print(f"   • {branch}")
    
    print(f"\n⚠️  With problems: {len(branches_by_status['⚠️'])}")
    for branch in branches_by_status['⚠️']:
        print(f"   • {branch}")
    
    print(f"\n❓ Require review: {len(branches_by_status['❓'])}")
    for branch in branches_by_status['❓']:
        print(f"   • {branch}")
    
    print("\n" + "="*80)
    total = sum(len(v) for v in branches_by_status.values())
    valid = len(branches_by_status['✅'])
    print(f"TOTAL: {valid}/{total} branches with valid datasets ({valid*100//total}%)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
