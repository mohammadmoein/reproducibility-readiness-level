"""Mechanical environment score E from a retrieved repository tree. No model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MANIFEST_NAMES = (
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "renv.lock",
)
PACKRAT_DIR = "packrat"
LOCKFILE_OR_CONTAINER = (
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "Dockerfile",
)
RUNTIME_FILES = (
    ".python-version",
    "runtime.txt",
    ".nvmrc",
    ".ruby-version",
    ".tool-versions",
)
SOURCE_SUFFIXES = {".py", ".r", ".R", ".jl", ".js", ".ts", ".ipynb", ".rmd", ".Rmd"}
SEED_PATTERNS = (
    re.compile(r"\brandom\.seed\s*\("),
    re.compile(r"\bnp\.random\.seed\s*\("),
    re.compile(r"\bnumpy\.random\.seed\s*\("),
    re.compile(r"\btorch\.manual_seed\s*\("),
    re.compile(r"\btf\.random\.set_seed\s*\("),
    re.compile(r"\bset\.seed\s*\("),
    re.compile(r"\bRandomState\s*\("),
    re.compile(r"\bseed\s*=\s*\d+"),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)(?:[A-Z]:\\|\\\\)[^\s\"']+"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])/(?:Users|home|usr/local|opt|data|mnt)/[^\s\"']+"),
)
OUTPUT_DIR_PATTERNS = (
    re.compile(r"\bos\.makedirs\s*\("),
    re.compile(r"\bos\.mkdir\s*\("),
    re.compile(r"\bPath\([^)]*\)\.mkdir\s*\("),
    re.compile(r"\bmkdir\s*\("),
    re.compile(r"\bdir\.create\s*\("),
    re.compile(r"\bexist_ok\s*="),
)
PINNED_EQ = re.compile(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*==\s*[^\s#]+")
FLOATING = re.compile(
    r"(?m)^\s*([A-Za-z0-9_.-]+)\s*(?:>=|<=|>|<|~=|\^)\s*"
)
TOML_PINNED = re.compile(
    r'(?i)(?:==|===)\s*["\']?[0-9]|"[A-Za-z0-9_.-]+==[0-9]'
)
PYTHON_REQUIRES = re.compile(r"(?i)python_requires|requires-python|python-version")


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = {part.lower() for part in path.relative_to(root).parts}
        if parts & {".git", "node_modules", "__pycache__", ".venv"}:
            continue
        files.append(path)
    return files


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _has_named(files: list[Path], names: tuple[str, ...]) -> bool:
    wanted = {name.lower() for name in names}
    return any(path.name.lower() in wanted for path in files)


def _manifest_paths(root: Path, files: list[Path]) -> list[Path]:
    found: list[Path] = []
    wanted = {name.lower() for name in MANIFEST_NAMES}
    for path in files:
        if path.name.lower() in wanted:
            found.append(path)
        elif path.name.lower() == PACKRAT_DIR or (
            path.parent.name.lower() == PACKRAT_DIR
        ):
            found.append(path)
    if (root / PACKRAT_DIR).is_dir():
        found.append(root / PACKRAT_DIR)
    return found


def _versions_pinned(manifests: list[Path]) -> bool:
    saw_requirement = False
    saw_floating = False
    saw_pinned = False
    for path in manifests:
        if path.is_dir():
            continue
        text = _read_text(path)
        name = path.name.lower()
        if name in {"requirements.txt"}:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                    continue
                saw_requirement = True
                if FLOATING.search(stripped) and "==" not in stripped:
                    saw_floating = True
                elif "==" in stripped or stripped.endswith(".whl"):
                    saw_pinned = True
        elif name in {"environment.yml", "environment.yaml"}:
            if re.search(r"(?m)^\s+-\s+[A-Za-z0-9_.-]+=[^=]", text):
                saw_pinned = True
                saw_requirement = True
            if re.search(r"(?m)^\s+-\s+[A-Za-z0-9_.-]+\s*$", text):
                saw_floating = True
                saw_requirement = True
        elif name == "pyproject.toml":
            if re.search(r"\[(?:project\.)?dependencies\]|dependencies\s*=", text):
                saw_requirement = True
            if TOML_PINNED.search(text) or "==" in text:
                saw_pinned = True
            if re.search(r">=|~=", text):
                saw_floating = True
        elif name == "renv.lock":
            saw_requirement = True
            saw_pinned = True
    if not saw_requirement:
        return False
    return saw_pinned and not saw_floating


def _runtime_declared(root: Path, files: list[Path], manifests: list[Path]) -> bool:
    if _has_named(files, RUNTIME_FILES):
        return True
    if (root / ".devcontainer").is_dir():
        return True
    for path in manifests:
        if path.is_file() and PYTHON_REQUIRES.search(_read_text(path)):
            return True
    return False


def _scan_source(files: list[Path]) -> tuple[bool, bool, bool]:
    seeds = False
    absolute = False
    creates = False
    for path in files:
        if path.suffix not in SOURCE_SUFFIXES and path.name.lower() not in {
            "makefile",
            "makefile.r",
        }:
            continue
        text = _read_text(path)
        if any(pattern.search(text) for pattern in SEED_PATTERNS):
            seeds = True
        if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
            absolute = True
        if any(pattern.search(text) for pattern in OUTPUT_DIR_PATTERNS):
            creates = True
    return seeds, not absolute, creates


def score_repository(root: Path) -> dict[str, Any]:
    """Return individual E components plus a simple mean composite."""
    files = _iter_files(root)
    manifests = _manifest_paths(root, files)
    lockfile = _has_named(files, LOCKFILE_OR_CONTAINER) or (root / ".devcontainer").is_dir()
    seeds, relative_paths, creates_output_dirs = _scan_source(files)
    components = {
        "manifest_present": bool(manifests),
        "versions_pinned": _versions_pinned(manifests),
        "lockfile_or_container": lockfile,
        "seeds_set": seeds,
        "relative_paths": relative_paths if files else False,
        "creates_output_dirs": creates_output_dirs,
        "runtime_declared": _runtime_declared(root, files, manifests),
    }
    values = [1.0 if value else 0.0 for value in components.values()]
    components["E"] = sum(values) / len(values)
    components["file_count"] = len(files)
    return components
