"""Build Arm B samples with a redacted paper and sanitized data, never author code."""
from __future__ import annotations

import io
import json
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

from arm_b.prompt import build_user_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_TAG = "repro-sandbox:1"  # exact Arm A image: environment is held constant
SANDBOX_PAPER = "/workspace/paper"
SANDBOX_DATA = "/workspace/data"
SANDBOX_WORK = "/workspace/work"
DEFAULT_TREE_CAP = 300
# Never mount analysis recipes. ZIPs are recursively rewritten; other archives are excluded
# because they cannot be proven code-free with the same guarantee.
CODE_EXTENSIONS = {".do", ".r", ".rmd", ".py", ".ipynb", ".sps", ".sas", ".smcl", ".m", ".jl", ".sh", ".java", ".scala", ".c", ".cc", ".cpp", ".js", ".ts", ".rb", ".pl", ".sql"}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zst"}
CODE_FILENAMES = {"dockerfile", "makefile", "implementation", "analysis", "run", "main", "script"}
_CODE_TEXT = re.compile(r"(?m)^\s*(?:#!|import\s+|from\s+\S+\s+import\s+|def\s+|class\s+|library\s*\(|require\s*\(|source\s*\(|function\s*\()")


class WithheldValueError(RuntimeError):
    pass


def _db_path(database: str | Path) -> Path:
    path = Path(database)
    return path if path.is_absolute() else REPO_ROOT / path


def _value_forms(value: float | None, verbatim: str | None) -> set[str]:
    """Local redaction forms; mirrors the runner's leak guard without importing it."""
    values: list[float] = []
    if value is not None:
        values += [float(value), float(value) * 100, float(value) / 100]
    values += [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", verbatim or "")]
    forms: set[str] = set()
    for number in values:
        if not number:
            continue
        for form in (f"{number:g}", f"{number:.1f}", f"{number:.2f}", f"{number:.3f}"):
            if len(form) >= 3 and float(form) == number:
                forms.add(form)
    return forms


def redact(text: str, needles: set[str]) -> tuple[str, int]:
    count = 0
    for needle in sorted(needles, key=len, reverse=True):
        # Numeric boundaries prevent .8 from redacting the middle of 10.85.
        pattern = rf"(?<![\d.]){re.escape(needle)}(?![\d.])"
        text, found = re.subn(pattern, "[REDACTED TARGET VALUE]", text)
        count += found
    return text, count


def _claim(db: sqlite3.Connection, paper_id: int, claim_experiment: str | None) -> dict[str, Any]:
    sql = (
        "SELECT d.claim_statement,d.claim_type,d.reported_value_verbatim,"
        "d.reported_value_parsed_json,d.comparison_target_json "
        "FROM claim_decisions d JOIN claim_experiments e ON e.id=d.experiment_id "
        "WHERE d.paper_id=?"
    )
    params: tuple[Any, ...] = (paper_id,)
    if claim_experiment:
        sql += " AND e.name=?"
        params += (claim_experiment,)
    else:
        sql += " ORDER BY d.experiment_id DESC LIMIT 1"
    row = db.execute(sql, params).fetchone()
    if row is None:
        raise ValueError(f"Paper {paper_id} has no Stage 3 focal claim")
    try:
        parsed = json.loads(row["reported_value_parsed_json"] or "{}")
        target = json.loads(row["comparison_target_json"] or "{}")
    except ValueError as exc:
        raise ValueError(f"Paper {paper_id} has malformed focal-claim JSON") from exc
    return {
        "claim_statement": row["claim_statement"], "claim_type": row["claim_type"],
        "reported_value": parsed.get("point_estimate"), "verbatim": row["reported_value_verbatim"],
        "metric_name": parsed.get("metric_name") or target.get("metric_name"),
        "analysis_population": target.get("analysis_population"),
    }


def _paper_markdown(db: sqlite3.Connection, paper_id: int) -> str:
    row = db.execute(
        "SELECT cv.markdown FROM content_versions cv JOIN file_assets fa ON fa.id=cv.file_asset_id "
        "WHERE fa.paper_id=? AND fa.is_primary_pdf=1 AND cv.is_current=1", (paper_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Paper {paper_id} has no current parsed Markdown")
    return str(row["markdown"])


def data_snapshots(db: sqlite3.Connection, paper_id: int) -> list[sqlite3.Row]:
    """Retrieved snapshots that may contain data.

    Most of this seven-paper set stores its data inside a code repository. Arm B may use that
    data, but `stage_materials` removes every analysis script before mounting it; selecting only
    URLs labelled ``data`` would silently discard five eligible papers.
    """
    rows = db.execute(
        "SELECT s.id,s.store_path,s.sha256,s.size_bytes,u.original_url,u.final_url "
        "FROM artifact_snapshots s JOIN artifact_urls u ON u.id=s.url_id "
        "WHERE s.paper_id=? AND s.size_bytes>0 AND u.artifact_kind IN ('code','data','both') "
        "ORDER BY s.id", (paper_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"Paper {paper_id} has no retrieved artifact snapshot for Arm B")
    return rows


def _bytes_look_like_code(name: str, data: bytes) -> bool:
    path = Path(name)
    return path.suffix.lower() in CODE_EXTENSIONS or path.name.lower() in CODE_FILENAMES or bool(
        _CODE_TEXT.search(data[:16_384].decode("utf-8", errors="ignore"))
    )


def _looks_like_code(path: Path) -> bool:
    """Conservatively reject extensionless/textual analysis recipes as well as known suffixes."""
    try:
        with path.open("rb") as source:
            return _bytes_look_like_code(path.name, source.read(16_384))
    except OSError:
        return True


def _sanitize_zip_bytes(data: bytes, removed: list[str], prefix: str = "", depth: int = 0) -> bytes:
    """Recursively remove executable analysis files from a ZIP without touching the source."""
    if depth > 8:
        raise ValueError("archive nesting exceeds the Arm B sanitisation limit")
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if _bytes_look_like_code(name, src.read(info)) or "__MACOSX" in Path(name).parts or Path(name).name.startswith("._"): 
                removed.append(prefix + name)
                continue
            body = src.read(info)
            if suffix == ".zip" and zipfile.is_zipfile(io.BytesIO(body)):
                body = _sanitize_zip_bytes(body, removed, prefix=f"{prefix}{name}:", depth=depth + 1)
            dst.writestr(info, body)
    return result.getvalue()


def _sanitize_zip(source: Path, target: Path, removed: list[str], prefix: str = "") -> None:
    target.write_bytes(_sanitize_zip_bytes(source.read_bytes(), removed, prefix=prefix))


def stage_materials(paper_id: int, snapshots: list[sqlite3.Row], markdown: str, claim: dict[str, Any], sandbox_dir: str | Path) -> tuple[Path, Path, list[str], int, int]:
    """Create a deterministic code-free host mount and a paper with target values redacted."""
    root = Path(sandbox_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root / str(paper_id)
    paper_dir, data_dir = root / "paper", root / "data"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    needles = _value_forms(claim["reported_value"], claim["verbatim"])
    redacted_paper, paper_redactions = redact(markdown, needles)
    (paper_dir / "article.md").write_text(redacted_paper, encoding="utf-8")
    removed: list[str] = []
    for snapshot in snapshots:
        content = Path(snapshot["store_path"]) / "content"
        if not content.is_dir():
            continue
        destination_root = data_dir / f"snapshot-{snapshot['id']}"
        for source in content.rglob("*"):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(content)
            suffix = source.suffix.lower()
            if ".git" in relative.parts or _looks_like_code(source) or suffix in (ARCHIVE_EXTENSIONS - {".zip"}):
                removed.append(str(relative))
                continue
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if suffix == ".zip":
                try:
                    _sanitize_zip(source, destination, removed, prefix=f"{relative}:")
                except (OSError, zipfile.BadZipFile):
                    removed.append(str(relative))
                    destination.unlink(missing_ok=True)
            else:
                shutil.copy2(source, destination)
    files = sorted(str(p.relative_to(data_dir)).replace("\\", "/") for p in data_dir.rglob("*") if p.is_file())
    return paper_dir, data_dir, files, len(removed), paper_redactions


def _compose_text(paper_dir: Path, data_dir: Path, image_tag: str, network: bool) -> str:
    build_context = (REPO_ROOT / "arm_a").resolve().as_posix()
    lines = [
        "# Generated by arm_b.dataset; sanitized data and redacted paper only.", "services:", "  default:",
        f'    image: "{image_tag}"', "    build:", f'      context: "{build_context}"', "    init: true",
        "    command: tail -f /dev/null", "    cpus: 2.0", "    mem_limit: 8gb", "    volumes:",
        f'      - "{paper_dir.resolve().as_posix()}:{SANDBOX_PAPER}:ro"',
        f'      - "{data_dir.resolve().as_posix()}:{SANDBOX_DATA}:ro"',
    ]
    if not network:
        lines.append("    network_mode: none")
    return "\n".join(lines) + "\n"


def build_sample(database: str | Path, paper_id: int, claim_experiment: str | None, sandbox_dir: str | Path = "artifacts/arm_b/sandbox", image_tag: str = DEFAULT_IMAGE_TAG, network: bool = True) -> Sample:
    db = sqlite3.connect(f"file:{_db_path(database).as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        claim = _claim(db, paper_id, claim_experiment)
        markdown = _paper_markdown(db, paper_id)
        snapshots = data_snapshots(db, paper_id)
    finally:
        db.close()
    paper_dir, data_dir, files, stripped, paper_redactions = stage_materials(paper_id, snapshots, markdown, claim, sandbox_dir)
    compose = Path(sandbox_dir)
    if not compose.is_absolute():
        compose = REPO_ROOT / compose
    compose = compose / str(paper_id) / "compose.yaml"
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text(_compose_text(paper_dir, data_dir, image_tag, network), encoding="utf-8")
    claim_text, claim_redactions = redact(str(claim["claim_statement"] or ""), _value_forms(claim["reported_value"], claim["verbatim"]))
    return Sample(
        id=f"paper-{paper_id}",
        input=build_user_prompt(claim_text, claim["metric_name"], claim["analysis_population"], files, stripped),
        sandbox=SandboxEnvironmentSpec("docker", str(compose.resolve())),
        metadata={"paper_id": paper_id, "snapshot_ids": [int(s["id"]) for s in snapshots], "data_files": files, "stripped_file_count": stripped, "paper_redaction_count": paper_redactions + claim_redactions, "compose_path": str(compose.resolve()), "network_available": network},
    )


def load_arm_b_dataset(**kwargs: Any) -> MemoryDataset:
    return MemoryDataset([build_sample(**kwargs)])
