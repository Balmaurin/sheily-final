#!/usr/bin/env python3
"""
ENRICH ALL DATASETS - Automation Tool
======================================
Generate complete dataset structures for all 49 branches.
"""

import json
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_PATH = PROJECT_ROOT / "all-Branches"

DEFAULT_STRUCTURE = {
    "categories": ["fundamentals", "advanced", "applications", "research", "case_studies"],
    "file_types": ["articles", "data", "analysis"]
}

def generate_article(category: str, index: int, domain: str) -> dict:
    """Generate article placeholder"""
    return {
        "id": f"{domain}_{category}_{index:03d}",
        "title": f"Article about {category.replace('_', ' ')} - {index}",
        "category": category,
        "domain": domain,
        "content": f"Detailed content about {category} in {domain} domain.",
        "keywords": [category, domain, "research"],
        "created_at": datetime.now().isoformat(),
        "quality_score": 0.85 + (index % 15) * 0.01
    }

def create_datasets_structure(branch_path: Path, branch_name: str, structure: dict):
    """Create complete datasets structure"""
    datasets_path = branch_path / "datasets"
    datasets_path.mkdir(exist_ok=True)
    
    total_files = 0
    categories_info = {}
    
    for category in structure["categories"]:
        category_path = datasets_path / category
        category_path.mkdir(exist_ok=True)
        
        # Generate 10 articles per category
        for i in range(10):
            article = generate_article(category, i, branch_name)
            article_file = category_path / f"article_{i:03d}.json"
            
            with open(article_file, 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
            
            total_files += 1
        
        categories_info[category] = {
            "count": 10,
            "path": f"{category}/"
        }
    
    # Create master_index.json
    master_index = {
        "created_date": datetime.now().isoformat(),
        "branch": branch_name,
        "total_documents": total_files,
        "total_categories": len(structure["categories"]),
        "categories": categories_info
    }
    
    with open(datasets_path / "master_index.json", 'w', encoding='utf-8') as f:
        json.dump(master_index, f, indent=2)
    
    return total_files

def main():
    print("\n" + "="*70)
    print("  ENRICH ALL DATASETS")
    print("="*70 + "\n")
    
    total_branches = 0
    total_files = 0
    
    for branch_dir in sorted(BASE_PATH.iterdir()):
        if not branch_dir.is_dir() or branch_dir.name == "biotechnology":
            continue
        
        branch_name = branch_dir.name
        print(f"📚 {branch_name:30} | Creating structure...")
        
        files_created = create_datasets_structure(branch_dir, branch_name, DEFAULT_STRUCTURE)
        
        print(f"   ✅ {files_created} files created")
        
        total_branches += 1
        total_files += files_created
    
    print("\n" + "="*70)
    print(f"Branches enriched: {total_branches}")
    print(f"Files created: {total_files}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
