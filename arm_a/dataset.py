"""Build the single Arm A `Sample` for one paper, without ever reading the reported value.

The rule that the agent never sees the reported value is enforced here rather than trusted: every
query this module issues goes through `_select`, which refuses SQL naming a column that
carries the paper's own number or its wording. The task module imports only this module and
`prompt`, so there is no code path from an Arm A run to `reported_value_parsed_json`.

What the agent gets: the retrieved repository (bind-mounted read-only, copied to a writable
tree by `task.prepare_repo`), the shipped file list, and the *name* of the focal metric.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec


from arm_a.prompt import build_user_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the read-only retrieval is mounted, and where the agent's writable copy lives.
SANDBOX_MOUNT = "/mnt/repo"
SANDBOX_REPO = "/workspace/repo"
SANDBOX_STATE = "/workspace/.arma"
BASELINE_MANIFEST = f"{SANDBOX_STATE}/baseline.sha256"

DEFAULT_IMAGE_TAG = "repro-sandbox:1"
DEFAULT_TREE_CAP = 300

#: Columns that carry the withheld value or its wording. Reading any of them from an Arm A
#: code path is the one mistake that voids the comparison, so it is a hard error rather than
#: a review comment. `comparison_target_json` is *not* here — but see `_metric_name`, which
#: takes only its `metric_name` field, because the free-text fields of some papers' targets
#: quote the printed number back (paper 482 names "the Table 1 value of 0.79").
WITHHELD_COLUMNS = (
    "reported_value_parsed_json",
    "reported_value_verbatim",
    "claim_statement",
    "source_quote",
)

NETWORK_NOTE_OPEN = (
    "The sandbox has outbound network access for the whole run, so `pip install` works. It "
    "is NOT switched off for the analysis itself — compose cannot toggle it mid-run — so "
    "treat network access as available throughout and say so if the analysis depends on it."
)
NETWORK_NOTE_CLOSED = (
    "The sandbox has NO outbound network access. Anything the analysis needs must already be "
    "in the image or in the repository. If a dependency cannot be installed, that is an "
    "`environment` fault and a legitimate result — record it and submit."
)


class WithheldValueError(RuntimeError):
    """Raised when an Arm A query would read the paper's own reported value."""


def _select(db: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    lowered = sql.lower()
    for column in WITHHELD_COLUMNS:
        if column in lowered:
            raise WithheldValueError(
                f"Arm A must not read {column!r}. The reported value is withheld until "
                f"`lakdb arm-a compare`, which runs outside the sandbox."
            )
    if "*" in sql.split("FROM")[0]:
        raise WithheldValueError(
            "Arm A queries must name their columns; SELECT * can pick up a withheld column."
        )
    return list(db.execute(sql, params).fetchall())


def _resolve(store_path: str) -> Path:
    path = Path(store_path.replace("\\", "/"))
    return path if path.is_absolute() else (REPO_ROOT / path)


# Human choices where a paper has more than one code snapshot, made by the authors
# before the runs. Kept in code rather than inferred so the choice is version-controlled and
# reviewable; `snapshot_for_paper` still refuses to guess for any paper not listed here.
# Keyed by the repository URL rather than the snapshot's row id, because row ids depend on
# the order in which a given database fetched its artifacts.
SNAPSHOT_OVERRIDES: dict[int, str] = {
    # Paper 69: LAK21_metadata holds only youtube_dataset.csv; ICALT2020_metadata holds
    # Data-set.csv AND Implementation.py, the only code. The paper's own open_materials
    # statement names ICALT2020_metadata as where the implementation lives.
    69: "https://github.com/rezatavakoli/ICALT2020_metadata",
}


def snapshot_for_paper(
    db: sqlite3.Connection, paper_id: int, snapshot_id: int | None = None
) -> sqlite3.Row:
    """The retrieved code repository for this paper.

    Stage 4 runs only on the papers whose data ships inside the repository, so one code snapshot is usually the whole artifact. A paper
    with two is an error rather than a guess: paper 69 has `LAK21_metadata` (a dataset, no
    code) and `ICALT2020_metadata` (the only `Implementation.py`), and picking blind would
    have been a coin flip between them.

    `snapshot_id` resolves that by hand. The choice is recorded on the run so it is auditable
    rather than a shell argument nobody remembers.
    """
    rows = _select(
        db,
        "SELECT s.id, s.paper_id, s.store_path, s.sha256, s.size_bytes, s.manifest_json, "
        "u.artifact_kind, u.original_url, u.final_url "
        "FROM artifact_snapshots s JOIN artifact_urls u ON u.id = s.url_id "
        "WHERE s.paper_id = ? AND s.size_bytes > 0 AND u.artifact_kind IN ('code', 'both') "
        "ORDER BY u.artifact_kind = 'code' DESC, s.size_bytes DESC",
        (paper_id,),
    )
    if not rows:
        raise ValueError(
            f"Paper {paper_id} has no retrieved code snapshot; Arm A has nothing to run."
        )
    if snapshot_id is None and paper_id in SNAPSHOT_OVERRIDES:
        chosen = [r for r in rows if r["original_url"] == SNAPSHOT_OVERRIDES[paper_id]]
        if len(chosen) != 1:
            available = [f"{r['id']}:{r['original_url']}" for r in rows]
            raise ValueError(
                f"Override {SNAPSHOT_OVERRIDES[paper_id]} matches {len(chosen)} code snapshots "
                f"of paper {paper_id}. Available: {available}"
            )
        return chosen[0]
    if snapshot_id is not None:
        chosen = [r for r in rows if int(r["id"]) == snapshot_id]
        if not chosen:
            available = [f"{r['id']}:{r['original_url']}" for r in rows]
            raise ValueError(
                f"Snapshot {snapshot_id} is not a code snapshot of paper {paper_id}. "
                f"Available: {available}"
            )
        return chosen[0]
    if len(rows) > 1:
        kinds = [f"{r['id']}:{r['artifact_kind']}:{r['original_url']}" for r in rows]
        raise ValueError(
            f"Paper {paper_id} has {len(rows)} code snapshots ({kinds}); pass "
            f"--snapshot-id to choose one before running Arm A."
        )
    return rows[0]


def _focal_target(
    db: sqlite3.Connection, paper_id: int, claim_experiment: str | None
) -> tuple[str | None, str | None]:
    """The focal metric's name and the analysis population, from Stage 3.

    `metric_name` alone is not enough. Paper 34's claim is an average "between the three
    platforms"; told only to find `recall@5`, the pilot agent ran the first line of the
    shipped example script, produced one platform pair, and `compare` scored a single cell
    against a three-platform average. The population is what tells the agent how much of the
    analysis to run.

    These two fields only. The rest of `comparison_target_json` is free text the extractor
    wrote about the paper and never leaves this function. `analysis_population` is itself free
    text and could in principle repeat the printed value, so it is not trusted here: the
    prompt is screened by `arm_a_runner.value_needles` before any run starts, and a leak
    stops the run rather than being silently tolerated.
    """
    if claim_experiment:
        rows = _select(
            db,
            "SELECT d.comparison_target_json, d.claim_type FROM claim_decisions d "
            "JOIN claim_experiments e ON e.id = d.experiment_id "
            "WHERE d.paper_id = ? AND e.name = ?",
            (paper_id, claim_experiment),
        )
    else:
        rows = _select(
            db,
            "SELECT d.comparison_target_json, d.claim_type FROM claim_decisions d "
            "WHERE d.paper_id = ? ORDER BY d.experiment_id DESC LIMIT 1",
            (paper_id,),
        )
    if not rows:
        return None, None
    try:
        target = json.loads(rows[0]["comparison_target_json"] or "null") or {}
    except ValueError:
        return None, None
    name = (target.get("metric_name") or "").strip()
    population = (target.get("analysis_population") or "").strip()
    return (name or None), (population or None)


def _compose_text(
    content_dir: Path,
    image_tag: str,
    cpus: float,
    mem_limit: str,
    network: bool,
) -> str:
    """A per-paper compose file: the shared image plus this paper's read-only bind mount.

    `image:` alongside `build:` means the image is built once and reused for every
    subsequent paper; without it compose derives a name from the (per-paper) project
    directory and rebuilds each time.
    """
    host_path = content_dir.resolve().as_posix()
    build_context = (REPO_ROOT / "arm_a").resolve().as_posix()
    lines = [
        "# Generated by arm_a.dataset. Do not edit; regenerate with `lakdb arm-a run`.",
        "services:",
        "  default:",
        f'    image: "{image_tag}"',
        "    build:",
        f'      context: "{build_context}"',
        "    init: true",
        "    command: tail -f /dev/null",
        f"    cpus: {cpus}",
        f"    mem_limit: {mem_limit}",
        "    volumes:",
        f'      - "{host_path}:{SANDBOX_MOUNT}:ro"',
    ]
    if not network:
        lines.append("    network_mode: none")
    return "\n".join(lines) + "\n"


def write_compose(
    paper_id: int,
    content_dir: Path,
    sandbox_dir: Path,
    image_tag: str = DEFAULT_IMAGE_TAG,
    cpus: float = 2.0,
    mem_limit: str = "8gb",
    network: bool = True,
) -> Path:
    text = _compose_text(content_dir, image_tag, cpus, mem_limit, network)
    root = Path(sandbox_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    target = root / str(paper_id) / "compose.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


def build_sample(
    database: str | Path = "data/lak.sqlite",
    paper_id: int = 0,
    claim_experiment: str | None = None,
    sandbox_dir: str | Path = "artifacts/arm_a/sandbox",
    image_tag: str = DEFAULT_IMAGE_TAG,
    cpus: float = 2.0,
    mem_limit: str = "8gb",
    network: bool = True,
    tree_cap: int = DEFAULT_TREE_CAP,
    snapshot_id: int | None = None,
) -> Sample:
    db_path = Path(database)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        snapshot = snapshot_for_paper(db, paper_id, snapshot_id)
        metric, population = _focal_target(db, paper_id, claim_experiment)
    finally:
        db.close()

    content_dir = _resolve(snapshot["store_path"]) / "content"
    if not content_dir.is_dir():
        raise FileNotFoundError(
            f"Snapshot content for paper {paper_id} is not on disk at {content_dir}. "
            f"Re-run `lakdb fetch code` before Arm A."
        )
    manifest = json.loads(snapshot["manifest_json"] or "{}")
    file_tree = list((manifest.get("inventory") or {}).get("file_tree") or [])

    compose = write_compose(
        paper_id, content_dir, Path(sandbox_dir), image_tag, cpus, mem_limit, network
    )
    prompt = build_user_prompt(
        metric_name=metric,
        analysis_population=population,
        repo_path=SANDBOX_REPO,
        file_tree=file_tree,
        tree_cap=tree_cap,
        network_note=NETWORK_NOTE_OPEN if network else NETWORK_NOTE_CLOSED,
    )
    return Sample(
        id=f"paper-{paper_id}",
        input=prompt,
        sandbox=SandboxEnvironmentSpec("docker", str(compose.resolve())),
        metadata={
            "paper_id": paper_id,
            "metric_name": metric,
            "analysis_population": population,
            "snapshot_id": int(snapshot["id"]),
            "snapshot_sha256": snapshot["sha256"],
            "snapshot_size_bytes": int(snapshot["size_bytes"]),
            "artifact_url": snapshot["final_url"] or snapshot["original_url"],
            "shipped_file_count": len(file_tree),
            "compose_path": str(compose.resolve()),
            "host_repo_path": content_dir.resolve().as_posix(),
            "network_available": network,
            "network_isolated_during_analysis": False,
        },
    )


def load_arm_a_dataset(**kwargs: Any) -> MemoryDataset:
    """One paper, one sample. Repetitions are separate evals, not epochs, so that each one
    gets its own row in `arm_a_runs` and can be resumed independently."""
    return MemoryDataset([build_sample(**kwargs)])
