"""Phase 3: documentation-quality payloads from retrieved artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifact import utc_now

DEFAULT_CHAR_BUDGET = 8000
DEFAULT_TREE_CAP = 200
DOCS_NAMES = ("readme", "readme.md", "readme.txt", "readme.rst", "contributing.md", "usage.md")
MANIFEST_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "renv.lock",
    "packrat.lock",
    "uv.lock",
    "poetry.lock",
    "dockerfile",
    "package.json",
    "description",
}
HEADER_EXTENSIONS = {".csv", ".tsv"}
SAMPLING_SEED = 20260917


def _truncate(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    return text[:budget], True


def _tree_names(root: Path, cap: int) -> list[str]:
    names: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        names.append(path.relative_to(root).as_posix())
        if len(names) >= cap:
            break
    return names


def _top_level_docs(root: Path, budget: int) -> tuple[str, int, bool]:
    chunks: list[str] = []
    used = 0
    truncated = False
    for path in sorted(root.iterdir() if root.is_dir() else []):
        if not path.is_file():
            continue
        if path.name.lower() not in DOCS_NAMES and not path.name.lower().startswith("readme"):
            continue
        remaining = budget - used
        if remaining <= 0:
            truncated = True
            break
        text, was_cut = _truncate(path.read_text(encoding="utf-8", errors="replace"), remaining)
        chunks.append(f"## {path.name}\n{text}")
        used += len(text)
        truncated = truncated or was_cut
    return "\n\n".join(chunks), used, truncated


def _manifests(root: Path) -> str:
    chunks: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name.lower() not in MANIFEST_NAMES:
            continue
        chunks.append(
            f"## {path.relative_to(root).as_posix()}\n"
            + path.read_text(encoding="utf-8", errors="replace")[:20_000]
        )
    return "\n\n".join(chunks)


def _data_headers(root: Path, limit_files: int = 5) -> str:
    chunks: list[str] = []
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in HEADER_EXTENSIONS:
            continue
        try:
            line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            continue
        chunks.append(f"## {path.relative_to(root).as_posix()} header\n{line}")
        count += 1
        if count >= limit_files:
            break
    return "\n\n".join(chunks)


def build_docs_payload(
    store_path: Path,
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    tree_cap: int = DEFAULT_TREE_CAP,
) -> dict[str, Any]:
    content = store_path / "content"
    root = content if content.is_dir() else store_path
    docs, used, truncated = _top_level_docs(root, char_budget)
    tree = _tree_names(root, tree_cap)
    payload = (
        f"character_budget={char_budget}\n"
        f"tree_entry_cap={tree_cap}\n"
        f"docs_chars_used={used}\n"
        f"docs_truncated={str(truncated).lower()}\n\n"
        "# Documentation files\n"
        f"{docs or '(none)'}\n\n"
        "# File tree (names only)\n"
        + "\n".join(tree)
        + "\n\n# Dependency manifests\n"
        + (_manifests(root) or "(none)")
        + "\n\n# Data column headers\n"
        + (_data_headers(root) or "(none)")
    )
    return {
        "payload": payload,
        "char_budget": char_budget,
        "tree_entry_cap": tree_cap,
        "docs_chars_used": used,
        "docs_truncated": truncated,
        "tree_entries": len(tree),
        "built_at": utc_now(),
    }


async def docs_payload_for_paper(catalog: Any, experiment: Any, paper_id: int) -> str:
    cursor = await catalog.db.execute(
        "SELECT store_path FROM artifact_snapshots WHERE paper_id=? ORDER BY id",
        (paper_id,),
    )
    rows = await cursor.fetchall()
    if not rows:
        raise ValueError(f"No retrieved artifacts for paper {paper_id}")
    budget = experiment["docs_char_budget"] or DEFAULT_CHAR_BUDGET
    cap = experiment["docs_tree_entry_cap"] or DEFAULT_TREE_CAP
    parts: list[str] = []
    for row in rows:
        built = build_docs_payload(Path(row["store_path"]), char_budget=budget, tree_cap=cap)
        parts.append(built["payload"])
    return "\n\n----\n\n".join(parts)


def stratified_sample(paper_ids: list[int], *, seed: int = SAMPLING_SEED, size: int | None = None) -> list[int]:
    """Deterministic frame. If size is None, return every id (this run processes all)."""
    import random

    ordered = sorted(paper_ids)
    if size is None or size >= len(ordered):
        return ordered
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:size])
