import os
import re
import sys
from pathlib import Path

# Look for print( not inside comments (best effort)
PATTERN = re.compile(r"(^|[^#])\bprint\s*\(")
SEARCH_DIRS = ("sheily_core",)

def main() -> int:
    bad: list[str] = []
    for base in SEARCH_DIRS:
        if not Path(base).exists():
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = Path(root) / fn
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if "print(" in text:
                    # Simple filter to ignore commented lines entirely
                    lines = text.splitlines()
                    for i, line in enumerate(lines, 1):
                        if "print(" in line and not line.strip().startswith("#"):
                            bad.append(f"{path}:{i}: use logger instead of print() in core modules")
                            break
    if bad:
        print("\n".join(bad))
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
