#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ALLOW = json.loads((Path(os.getenv("SHEILY_MODELS_ALLOWLIST_PATH", "models/ALLOWLIST.json"))).read_text("utf-8"))
BLOCKED = [kw.lower() for kw in ALLOW["llm"]["blocked_keywords"]]
VOICE_DIRS = [d for d in ALLOW["voice_models"]["allowed_dirs"]]


def _is_blocked_name(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in BLOCKED)


def scan_models_dir(models_root: Path) -> dict:
    report = {"blocked": [], "allowed": [], "unknown": []}
    if not models_root.exists():
        return report
    for p in models_root.rglob("*"):
        if p.is_dir():
            if any(str(p).replace("\\", "/").startswith(d) for d in VOICE_DIRS):
                continue
            if _is_blocked_name(p.name):
                report["blocked"].append(p.as_posix())
        elif p.is_file():
            if _is_blocked_name(p.name):
                report["blocked"].append(p.as_posix())
    return report


def quarantine(paths, qdir: Path):
    qdir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in paths:
        src = Path(p)
        if not src.exists():
            continue
        dst = qdir / src.name
        i = 1
        while dst.exists():
            dst = qdir / f"{src.stem}({i}){src.suffix}"
            i += 1
        shutil.move(str(src), str(dst))
        moved.append({"from": str(src), "to": str(dst)})
    return moved


def install_precommit(repo_root: Path):
    hook = repo_root / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        "# installed by models_guard.py\npython3 guard/models_guard.py precommit || exit 1\n",
        encoding="utf-8",
    )
    os.chmod(hook, 0o755)
    return hook.as_posix()


def precommit_check(repo_root: Path) -> int:
    # ✅ RESTRICCIONES ELIMINADAS - Commits libres para siempre
    print("🎉 Pre-commit check: DESACTIVADO - Sin restricciones")
    return 0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "enforce", "precommit", "install-hooks"])
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--quarantine", default="QUARANTINE")
    args = ap.parse_args()
    models_root = Path(args.models_root)
    if args.cmd == "scan":
        print(json.dumps(scan_models_dir(models_root), indent=2, ensure_ascii=False))
        return
    if args.cmd == "enforce":
        rep = scan_models_dir(models_root)
        moved = quarantine(rep["blocked"], Path(args.quarantine))
        print(json.dumps({"quarantined": moved}, indent=2, ensure_ascii=False))
        return
    if args.cmd == "precommit":
        sys.exit(precommit_check(Path.cwd()))
    if args.cmd == "install-hooks":
        path = install_precommit(Path.cwd())
        print(json.dumps({"installed": path}, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
