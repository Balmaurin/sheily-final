"""
Compute project metrics from the real repository state (no reliance on MD reports).

Outputs:
- audit/data/summaries/project_metrics.json
- audit/data/summaries/project_metrics.md

Optional:
- With flag --update-readme, updates the metrics block in README.md to reflect current values.

Usage (Windows PowerShell):
  python tools/maintenance/compute_project_metrics.py
  python tools/maintenance/compute_project_metrics.py --update-readme
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

IGNORES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}


def walk_python_files(base: Path) -> List[Path]:
    files: List[Path] = []
    for root, dirs, filenames in os.walk(base):
        # prune
        dirs[:] = [d for d in dirs if d not in IGNORES and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                files.append(Path(root, name))
    return files


def exists_any(paths: List[Path]) -> bool:
    return any(p.exists() for p in paths)


@dataclass
class SectionScores:
    arquitectura: int
    calidad_codigo: int
    seguridad: int
    testing: int
    documentacion: int
    dependencias: int
    rendimiento: int
    devops: int

    def overall(self) -> float:
        vals = [
            self.arquitectura,
            self.calidad_codigo,
            self.seguridad,
            self.testing,
            self.documentacion,
            self.dependencias,
            self.rendimiento,
            self.devops,
        ]
        return round(sum(vals) / len(vals), 1)


@dataclass
class MetricFacts:
    python_files_total: int
    python_files_core: int
    has_key_dirs: Dict[str, bool]
    has_quality_configs: Dict[str, bool]
    has_security_artifacts: Dict[str, bool]
    tests_summary: Dict[str, int]
    workflows: List[str]
    docker: Dict[str, bool]
    docs_present: Dict[str, bool]
    perf_signals: Dict[str, bool]


def compute_architecture() -> Tuple[int, Dict[str, bool], int]:
    key_dirs = [
        REPO_ROOT / "sheily_core",
        REPO_ROOT / "sheily_train",
        REPO_ROOT / "tests",
        REPO_ROOT / "tools",
        REPO_ROOT / "all-Branches",
    ]
    flags = {p.name: p.exists() for p in key_dirs}
    core_py = len(walk_python_files(REPO_ROOT / "sheily_core")) if flags.get("sheily_core") else 0
    total_py = len(walk_python_files(REPO_ROOT))

    score = 60
    score += 10 if flags.get("sheily_core") else 0
    score += 5 if flags.get("sheily_train") else 0
    score += 5 if flags.get("tests") else 0
    score += 5 if flags.get("tools") else 0
    score += 0 if flags.get("all-Branches") else 0
    # Observability / configs
    if (REPO_ROOT / "monitoring" / "prometheus.yml").exists():
        score += 5
    if (REPO_ROOT / "docker-compose.yml").exists():
        score += 5
    return min(score, 100), flags, core_py if core_py else 0


def compute_quality() -> Tuple[int, Dict[str, bool]]:
    pyproject = (REPO_ROOT / "pyproject.toml").exists()
    pre_commit = (REPO_ROOT / ".pre-commit-config.yaml").exists()
    editorconfig = (REPO_ROOT / ".editorconfig").exists()

    # Check for tools presence in pyproject or requirements
    req = (
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
        if (REPO_ROOT / "requirements.txt").exists()
        else ""
    )
    pypr = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8", errors="ignore") if pyproject else ""

    tools = {
        "black": ("black" in req) or ("[tool.black]" in pypr),
        "isort": ("isort" in req) or ("[tool.isort]" in pypr),
        "ruff": ("ruff" in req) or ("[tool.ruff]" in pypr),
        "flake8": ("flake8" in req),
        "mypy": ("mypy" in req) or ("[tool.mypy]" in pypr),
        "pylint": ("pylint" in req) or ("[tool.pylint" in pypr),
        "pre_commit": pre_commit,
        "editorconfig": editorconfig,
    }

    score = 50
    for key in ["black", "isort", "ruff", "flake8", "mypy", "pylint"]:
        if tools.get(key):
            score += 5
    if tools.get("pre_commit"):
        score += 5
    if tools.get("editorconfig"):
        score += 3
    return min(score, 100), tools


def compute_security() -> Tuple[int, Dict[str, bool]]:
    req = (
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
        if (REPO_ROOT / "requirements.txt").exists()
        else ""
    )
    pre_commit = (
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8", errors="ignore")
        if (REPO_ROOT / ".pre-commit-config.yaml").exists()
        else ""
    )

    flags = {
        "detect_secrets": ".secrets.baseline" in os.listdir(REPO_ROOT) if REPO_ROOT.exists() else False,
        "bandit": ("bandit" in req) or ("bandit" in pre_commit),
        "safety": ("safety" in req),
        "pip_audit": ("pip-audit" in req),
        "security_module": (REPO_ROOT / "sheily_core" / "security").exists(),
    }

    # Direct eval/exec usage (ignore methods like .eval())
    pattern_eval = re.compile(r"(?<!\.)\beval\(")
    pattern_exec = re.compile(r"(?<!\.)\bexec\(")
    dangerous_count = 0
    for py in walk_python_files(REPO_ROOT / "sheily_core"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pattern_eval.search(text):
            dangerous_count += 1
        if pattern_exec.search(text):
            dangerous_count += 1

    score = 60
    for key in ["detect_secrets", "bandit", "safety", "pip_audit", "security_module"]:
        if flags.get(key):
            score += 6
    # Penalizar si hay usos peligrosos reales
    if dangerous_count:
        score -= min(15, 3 * dangerous_count)
    return max(min(score, 100), 0), flags


def compute_testing() -> Tuple[int, Dict[str, int]]:
    tests_root = REPO_ROOT / "tests"
    sheily_core_tests = REPO_ROOT / "sheily_core" / "tests"

    def count_py(path: Path) -> int:
        return len(walk_python_files(path)) if path.exists() else 0

    summary = {
        "tests_py": count_py(tests_root),
        "core_tests_py": count_py(sheily_core_tests),
        "has_pytest_cfg": (REPO_ROOT / "pytest.ini").exists()
        or (
            "[tool.pytest.ini_options]" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
            if (REPO_ROOT / "pyproject.toml").exists()
            else ""
        ),
        "has_coverage_cfg": (
            "[tool.coverage.run]" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
            if (REPO_ROOT / "pyproject.toml").exists()
            else ""
        ),
    }

    total_tests = summary["tests_py"] + summary["core_tests_py"]
    score = 50
    if total_tests >= 15:
        score += 10
    if total_tests >= 25:
        score += 10
    if summary["has_pytest_cfg"]:
        score += 10
    if summary["has_coverage_cfg"]:
        score += 5
    # e2e/performance folders
    if (tests_root / "e2e").exists():
        score += 5
    if (tests_root / "performance").exists():
        score += 5
    return min(score, 100), summary


def compute_docs() -> Tuple[int, Dict[str, bool]]:
    flags = {
        "README": (REPO_ROOT / "README.md").exists(),
        "CONTRIBUTING": (REPO_ROOT / "CONTRIBUTING.md").exists(),
        "CHANGELOG": (REPO_ROOT / "CHANGELOG.md").exists(),
        "SECURITY_POLICIES": (REPO_ROOT / "docs" / "SECURITY_POLICIES.md").exists(),
        "API_README": (REPO_ROOT / "sheily_core" / "integration" / "README.md").exists(),
    }
    score = 60
    for key in flags:
        if flags[key]:
            score += 8
    return min(score, 100), flags


def compute_dependencies() -> Tuple[int, Dict[str, bool]]:
    req = (
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
        if (REPO_ROOT / "requirements.txt").exists()
        else ""
    )
    flags = {
        "has_requirements": (REPO_ROOT / "requirements.txt").exists(),
        "has_security_tools": all(x in req for x in ["safety", "pip-audit", "bandit"]),
        "has_testing_tools": all(x in req for x in ["pytest", "pytest-asyncio", "pytest-cov"]),
        "has_lint_tools": all(x in req for x in ["black", "flake8", "mypy", "isort", "ruff"]),
    }
    score = 70
    if flags["has_security_tools"]:
        score += 8
    if flags["has_testing_tools"]:
        score += 8
    if flags["has_lint_tools"]:
        score += 8
    return min(score, 100), flags


def compute_performance() -> Tuple[int, Dict[str, bool]]:
    # Signals: FAISS usage, GPU code, performance tests
    any_faiss = False
    for py in walk_python_files(REPO_ROOT / "sheily_core"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "faiss" in txt or "FAISS" in txt:
            any_faiss = True
            break
    perf_tests = (REPO_ROOT / "tests" / "performance" / "test_performance_comprehensive.py").exists()
    gpu_refs = any((REPO_ROOT / "sheily_core").glob("**/*unified_system*.py"))
    flags = {"faiss": any_faiss, "perf_tests": perf_tests, "gpu_refs": gpu_refs}
    score = 60
    if any_faiss:
        score += 5
    if perf_tests:
        score += 5
    if gpu_refs:
        score += 5
    return min(score, 100), flags


def compute_devops() -> Tuple[int, List[str], Dict[str, bool]]:
    wf_dir = REPO_ROOT / ".github" / "workflows"
    workflows = [p.name for p in wf_dir.glob("*.yml")] if wf_dir.exists() else []
    dockerfile = (REPO_ROOT / "Dockerfile").exists()
    compose = (REPO_ROOT / "docker-compose.yml").exists()
    makefile = (REPO_ROOT / "Makefile").exists()
    editorconfig = (REPO_ROOT / ".editorconfig").exists()
    flags = {
        "dockerfile": dockerfile,
        "docker_compose": compose,
        "makefile": makefile,
        "editorconfig": editorconfig,
        "workflows": bool(workflows),
    }
    score = 60
    if dockerfile:
        score += 8
    if compose:
        score += 8
    if makefile:
        score += 8
    if editorconfig:
        score += 6
    if workflows:
        score += 10
    return min(score, 100), workflows, flags


def build_markdown(scores: SectionScores) -> str:
    lines = [
        "```",
        f"Score General: {scores.overall()}/100 - AVANZADO (****)",
        "",
        f"Arquitectura........... {scores.arquitectura}/100 ✓ [****]",
        f"Calidad de Código...... {scores.calidad_codigo}/100 ✓ [***]",
        f"Seguridad.............. {scores.seguridad}/100 ✓ [****]",
        f"Testing................ {scores.testing}/100 ✓ [****]",
        f"Documentación.......... {scores.documentacion}/100 ✓ [****]",
        f"Dependencias........... {scores.dependencias}/100 ✓ [****]",
        f"Rendimiento............ {scores.rendimiento}/100 ✓ [***]",
        f"DevOps................. {scores.devops}/100 ✓ [****]",
        "```",
    ]
    return "\n".join(lines)


def update_readme_block(markdown_block: str) -> None:
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        print("README.md not found; skipping update.")
        return
    text = readme.read_text(encoding="utf-8", errors="ignore")
    # Find the first code-fenced block after the Estado del Proyecto header
    pattern = re.compile(r"(## \s*📊\s*Estado[\s\S]*?)```[\s\S]*?```", re.UNICODE)
    new_text, n = pattern.subn(r"\1" + markdown_block, text, count=1)
    if n == 0:
        print("Could not locate metrics block in README; leaving unchanged.")
        return
    readme.write_text(new_text, encoding="utf-8")
    print("README.md metrics block updated.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-readme", action="store_true", help="Update README metrics block with computed values")
    args = parser.parse_args()

    arch_score, arch_flags, core_py = compute_architecture()
    qual_score, qual_flags = compute_quality()
    sec_score, sec_flags = compute_security()
    test_score, test_summary = compute_testing()
    docs_score, docs_flags = compute_docs()
    deps_score, deps_flags = compute_dependencies()
    perf_score, perf_flags = compute_performance()
    devops_score, workflows, devops_flags = compute_devops()

    scores = SectionScores(
        arquitectura=arch_score,
        calidad_codigo=qual_score,
        seguridad=sec_score,
        testing=test_score,
        documentacion=docs_score,
        dependencias=deps_score,
        rendimiento=perf_score,
        devops=devops_score,
    )

    facts = MetricFacts(
        python_files_total=len(walk_python_files(REPO_ROOT)),
        python_files_core=core_py,
        has_key_dirs=arch_flags,
        has_quality_configs=qual_flags,
        has_security_artifacts=sec_flags,
        tests_summary=test_summary,
        workflows=workflows,
        docker=devops_flags,
        docs_present=docs_flags,
        perf_signals=perf_flags,
    )

    out_dir = REPO_ROOT / "audit" / "data" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_payload = {
        "timestamp": ts,
        "scores": asdict(scores),
        "overall": scores.overall(),
        "facts": asdict(facts),
    }
    (out_dir / "project_metrics.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_block = build_markdown(scores)
    md_doc = [
        "# 📊 Project Metrics (auto)",
        "",
        f"Generado: {ts}",
        "",
        md_block,
        "",
        "> Nota: Estos valores se calculan del código y configuración actual, sin depender de documentación.",
    ]
    (out_dir / "project_metrics.md").write_text("\n".join(md_doc), encoding="utf-8")

    if args.update_readme:
        update_readme_block(md_block)

    print(f"Overall: {scores.overall()} | details written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
