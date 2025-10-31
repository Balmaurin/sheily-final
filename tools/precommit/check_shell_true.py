import os
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\bshell\s*=\s*True\b")
SEARCH_DIRS = ("sheily_core", "sheily_train")

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
                if PATTERN.search(text):
                    bad.append(str(path))
    if bad:
        print("shell=True found in:\n" + "\n".join(bad))
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
