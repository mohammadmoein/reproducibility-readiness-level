"""Pinned upstream ARA execution, persistence, scoring, and corpus summaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

ARA_REPOSITORY = (
    "https://github.com/AndresLaverdeMarin/agentic_reproducibility_assessment.git"
)
ARA_COMMIT = "e015be37d82c57e6da826ad50fe8023e9f1255f5"
ARA_INTEGRATION_VERSION = "1.1.0"
LAYER_WEIGHTS = {"source": 0.30, "process": 0.40, "sink": 0.30}
STRUCTURAL_WEIGHTS = {
    "sources_consumed_ratio": 0.25,
    "sinks_produced_ratio": 0.25,
    "resolved_input_ratio": 0.20,
    "source_to_sink_reachability": 0.15,
    "lwcc_fraction": 0.15,
}

ARA_SCHEMA = """
CREATE TABLE IF NOT EXISTS ara_experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applicability_experiment_id INTEGER NOT NULL REFERENCES study_experiments(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK(status IN ('frozen','closed')),
    provider TEXT NOT NULL CHECK(provider='openai'),
    model TEXT NOT NULL,
    temperature REAL,
    repeat_count INTEGER NOT NULL CHECK(repeat_count > 0),
    upstream_repository TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    upstream_source_sha256 TEXT NOT NULL,
    integration_version TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    frozen_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS ara_runs (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES ara_experiments(id) ON DELETE RESTRICT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,
    file_asset_id INTEGER NOT NULL REFERENCES file_assets(id) ON DELETE RESTRICT,
    repetition_index INTEGER NOT NULL CHECK(repetition_index > 0),
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
    execution_mode TEXT NOT NULL CHECK(execution_mode='realtime'),
    source_pdf_sha256 TEXT NOT NULL,
    upstream_commit TEXT NOT NULL,
    profile_json TEXT,
    raw_response_json TEXT,
    usage_json TEXT,
    source_score REAL,
    methods_score REAL,
    experiments_score REAL,
    sink_score REAL,
    content_score REAL,
    structural_score REAL,
    reproducibility_index REAL,
    wiring_metrics_json TEXT,
    source_node_count INTEGER,
    method_node_count INTEGER,
    experiment_node_count INTEGER,
    sink_node_count INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    estimated_cost_usd REAL,
    duration_ms INTEGER,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    error_type TEXT,
    error_text TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(experiment_id,paper_id,file_asset_id,repetition_index)
) STRICT;
CREATE INDEX IF NOT EXISTS ara_runs_experiment_status ON ara_runs(experiment_id,status);
CREATE INDEX IF NOT EXISTS ara_runs_paper ON ara_runs(paper_id,experiment_id);
"""


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_score(nodes: list[dict[str, Any]]) -> float | None:
    if not nodes:
        return None
    return sum(float(node.get("reproducibility_score", 0)) for node in nodes) / (
        100 * len(nodes)
    )


def _normalized(value: Any) -> str:
    return "".join(str(value).lower().split())


def _process_id(node: dict[str, Any], index: int) -> str:
    for key in ("node_id", "step_id", "id"):
        if node.get(key):
            return str(node[key])
    return f"meth_{index + 1:02d}"


def _reachable(adjacency: dict[str, set[str]], source: str, sink: str) -> bool:
    pending = [source]
    seen = {source}
    while pending:
        node = pending.pop()
        if node == sink:
            return True
        for neighbor in adjacency.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def _largest_weak_component(nodes: set[str], edges: set[tuple[str, str]]) -> float:
    if not nodes:
        return 0.0
    neighbors = {node: set() for node in nodes}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    remaining = set(nodes)
    largest = 0
    while remaining:
        start = remaining.pop()
        component = {start}
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor in neighbors[node] & remaining:
                remaining.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)
        largest = max(largest, len(component))
    return largest / len(nodes)


def score_ara_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Compute the published ARA content, structural, and geometric-mean scores."""
    sources = list(profile.get("nodes_source") or [])
    processes = list(profile.get("nodes_process") or [])
    sinks = list(profile.get("nodes_sink") or [])
    methods = [
        node for node in processes if node.get("process_type", "method") == "method"
    ]
    experiments = [
        node for node in processes if node.get("process_type") == "experiment"
    ]

    stage_scores = {
        "source_score": _mean_score(sources),
        "methods_score": _mean_score(methods),
        "experiments_score": _mean_score(experiments),
        "sink_score": _mean_score(sinks),
    }
    layer_scores = {
        "source": _mean_score(sources),
        "process": _mean_score(processes),
        "sink": _mean_score(sinks),
    }
    present = {name: value for name, value in layer_scores.items() if value is not None}
    present_weight = sum(LAYER_WEIGHTS[name] for name in present)
    content = (
        sum(LAYER_WEIGHTS[name] * value for name, value in present.items())
        / present_weight
        if present_weight
        else 0.0
    )

    source_ids = {
        _normalized(node.get("node_id")) for node in sources if node.get("node_id")
    }
    sink_ids = {
        _normalized(node.get("node_id")) for node in sinks if node.get("node_id")
    }
    process_rows: list[tuple[str, list[str], list[str]]] = []
    for index, process in enumerate(processes):
        process_rows.append(
            (
                _normalized(_process_id(process, index)),
                [
                    _normalized(value)
                    for value in (
                        process.get("input_ids") or process.get("data_inputs") or []
                    )
                ],
                [
                    _normalized(value)
                    for value in (
                        process.get("outcomes") or process.get("expected_outputs") or []
                    )
                ],
            )
        )
    process_ids = {row[0] for row in process_rows}
    nodes = source_ids | process_ids | sink_ids
    edges: set[tuple[str, str]] = set()
    all_outputs = {value for _, _, outputs in process_rows for value in outputs}
    for process_id, inputs, outputs in process_rows:
        edges.update((value, process_id) for value in inputs if value in source_ids)
        edges.update((process_id, value) for value in outputs if value in sink_ids)
    for left_id, _, outputs in process_rows:
        intermediate = set(outputs) - source_ids - sink_ids
        if not intermediate:
            continue
        for right_id, inputs, _ in process_rows:
            if left_id != right_id and intermediate.intersection(inputs):
                edges.add((left_id, right_id))

    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
    total_inputs = sum(len(inputs) for _, inputs, _ in process_rows)
    dangling_inputs = sum(
        value not in (source_ids | sink_ids | all_outputs)
        for _, inputs, _ in process_rows
        for value in inputs
    )
    source_sink_pairs = len(source_ids) * len(sink_ids)
    wiring = {
        "sources_consumed_ratio": (
            sum(any(left == source for left, _ in edges) for source in source_ids)
            / len(source_ids)
            if source_ids
            else 0.0
        ),
        "sinks_produced_ratio": (
            sum(any(right == sink for _, right in edges) for sink in sink_ids)
            / len(sink_ids)
            if sink_ids
            else 0.0
        ),
        "resolved_input_ratio": 1 - dangling_inputs / total_inputs
        if total_inputs
        else 0.0,
        "source_to_sink_reachability": (
            sum(
                _reachable(adjacency, source, sink)
                for source in source_ids
                for sink in sink_ids
            )
            / source_sink_pairs
            if source_sink_pairs
            else 0.0
        ),
        "lwcc_fraction": _largest_weak_component(nodes, edges),
    }
    structural = sum(
        STRUCTURAL_WEIGHTS[name] * wiring[name] for name in STRUCTURAL_WEIGHTS
    )
    return {
        **stage_scores,
        "content_score": content,
        "structural_score": structural,
        "reproducibility_index": math.sqrt(max(0.0, content) * max(0.0, structural)),
        "wiring_metrics": wiring,
        "source_node_count": len(sources),
        "method_node_count": len(methods),
        "experiment_node_count": len(experiments),
        "sink_node_count": len(sinks),
    }


async def ensure_ara_schema(catalog: Any) -> None:
    from screening import ensure_schema

    await ensure_schema(catalog)
    await catalog.db.executescript(ARA_SCHEMA)
    await catalog.db.commit()


def _checkout_revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot read ARA checkout revision: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_checkout(checkout: Path, expected_commit: str) -> tuple[str, str]:
    checkout = checkout.resolve()
    source = checkout / "src" / "ara_pipeline" / "online_llm_pipelines.py"
    if not source.is_file():
        raise FileNotFoundError(f"ARA source is missing: {source}")
    revision = _checkout_revision(checkout)
    if revision != expected_commit:
        raise ValueError(
            f"ARA checkout is {revision}; experiment requires {expected_commit}"
        )
    return revision, _sha256_file(source)


async def _get_ara_experiment(catalog: Any, name: str) -> Any:
    cursor = await catalog.db.execute(
        "SELECT * FROM ara_experiments WHERE name=?", (name,)
    )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown ARA experiment: {name}")
    return row


async def command_ara_experiment_list(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_ara_schema(catalog)
        cursor = await catalog.db.execute(
            "SELECT e.id,e.name,e.status,s.name applicability_experiment,e.model,"
            "e.repeat_count,e.configuration_json,e.created_at,"
            "(SELECT COUNT(DISTINCT r.paper_id) FROM ara_runs r WHERE r.experiment_id=e.id) papers,"
            "(SELECT COUNT(*) FROM ara_runs r WHERE r.experiment_id=e.id) runs,"
            "(SELECT COUNT(*) FROM ara_runs r WHERE r.experiment_id=e.id AND r.status='succeeded') succeeded,"
            "(SELECT COUNT(*) FROM ara_runs r WHERE r.experiment_id=e.id AND r.status='failed') failed "
            "FROM ara_experiments e JOIN study_experiments s ON s.id=e.applicability_experiment_id "
            "ORDER BY e.id"
        )
        rows = list(await cursor.fetchall())

    table = Table(title="ARA experiments", header_style="bold cyan")
    for name, justify in (
        ("ID", "right"),
        ("Name", "left"),
        ("Status", "left"),
        ("Applicability", "left"),
        ("Model", "left"),
        ("Effort", "left"),
        ("Repeats", "right"),
        ("Papers", "right"),
        ("Runs", "right"),
        ("Succeeded", "right"),
        ("Failed", "right"),
        ("Created", "left"),
    ):
        table.add_column(name, justify=justify)
    for row in rows:
        configuration = json.loads(row["configuration_json"])
        table.add_row(
            str(row["id"]),
            row["name"],
            row["status"],
            row["applicability_experiment"],
            row["model"],
            configuration.get("reasoning_effort") or "none",
            str(row["repeat_count"]),
            str(row["papers"]),
            str(row["runs"]),
            str(row["succeeded"]),
            str(row["failed"]),
            row["created_at"],
        )
    Console().print(table)
    logger.info("ara_experiments_listed", count=len(rows))
    return 0


async def command_ara_create(args: Any, logger: Any, Catalog: Any) -> int:
    reasoning_model = args.model.startswith(("gpt-5", "o1", "o3", "o4"))
    if reasoning_model and args.reasoning_effort is None:
        raise ValueError("Reasoning-model ARA experiments require --reasoning-effort")
    if not reasoning_model and args.reasoning_effort is not None:
        raise ValueError(
            "--reasoning-effort is only valid for GPT-5, o1, o3, or o4 models"
        )
    temperature = None if reasoning_model else args.temperature
    checkout = Path(args.checkout)
    revision, source_hash = await asyncio.to_thread(
        validate_checkout, checkout, args.upstream_commit
    )
    async with Catalog(Path(args.database)) as catalog:
        await ensure_ara_schema(catalog)
        cursor = await catalog.db.execute(
            "SELECT id,status FROM study_experiments WHERE name=?",
            (args.applicability_experiment,),
        )
        applicability = await cursor.fetchone()
        if applicability is None:
            raise ValueError(
                f"Unknown applicability experiment: {args.applicability_experiment}"
            )
        if applicability["status"] != "frozen":
            raise ValueError("ARA requires a frozen applicability experiment")
        now = _utc_now()
        configuration = {
            "selection": (
                "final_label=applicable AND majority claim_type NOT IN "
                "(non_quantitative,none,unclear)"
            ),
            "selection_version": "2.0.0",
            "input": "original PDF",
            "execution": "upstream six-query GPTPaperAnalyst, realtime Responses API",
            "reasoning_effort": args.reasoning_effort,
            "upstream_batch_api": False,
            "batch_reason": "nodes_sink depends on nodes_source; nodes_process depends on both",
            "checkout_used_at_creation": str(checkout.resolve()),
        }
        try:
            cursor = await catalog.db.execute(
                "INSERT INTO ara_experiments(name,applicability_experiment_id,status,provider,model,temperature,repeat_count,upstream_repository,upstream_commit,upstream_source_sha256,integration_version,configuration_json,created_at,frozen_at) "
                "VALUES (?,?,'frozen','openai',?,?,?,?,?,?,?,?,?,?)",
                (
                    args.name,
                    applicability["id"],
                    args.model,
                    temperature,
                    args.repeats,
                    ARA_REPOSITORY,
                    revision,
                    source_hash,
                    ARA_INTEGRATION_VERSION,
                    _json(configuration),
                    now,
                    now,
                ),
            )
        except Exception:
            await catalog.db.rollback()
            raise
        await catalog.db.commit()
        experiment_id = int(cursor.lastrowid)
    logger.info(
        "ara_experiment_created",
        experiment_id=experiment_id,
        name=args.name,
        applicability_experiment=args.applicability_experiment,
        model=args.model,
        repeats=args.repeats,
        upstream_commit=revision,
    )
    return 0


async def _ara_candidates(catalog: Any, experiment: Any, args: Any) -> list[Any]:
    clauses = [
        "d.experiment_id=?",
        "d.final_label='applicable'",
        "cv.id=d.content_version_id",
        "fa.id=cv.file_asset_id",
        "fa.is_primary_pdf=1",
    ]
    params: list[Any] = [experiment["applicability_experiment_id"]]
    if not getattr(args, "include_non_quantitative", False):
        # Protocol: L0 papers carry no recomputable quantitative claim, so ARA has no
        # workflow to reconstruct for them. final_label alone does not encode this;
        # claim type lives per repetition in applicability_runs, so take the majority.
        # Stated positively against the canonical set, so any claim type added later is
        # excluded until someone decides otherwise rather than silently assessed.
        from screening import QUANTITATIVE_CLAIM_TYPES

        keep = sorted(c.value for c in QUANTITATIVE_CLAIM_TYPES)
        clauses.append(
            "d.paper_id IN (SELECT paper_id FROM ("
            "SELECT paper_id,claim_type,ROW_NUMBER() OVER ("
            "PARTITION BY paper_id ORDER BY COUNT(*) DESC,claim_type) rn "
            "FROM applicability_runs WHERE experiment_id=? AND status='succeeded' "
            "AND claim_type IS NOT NULL GROUP BY paper_id,claim_type"
            ") WHERE rn=1 AND claim_type IN (" + ",".join("?" * len(keep)) + "))"
        )
        params.append(experiment["applicability_experiment_id"])
        params.extend(keep)
    if args.paper_id is not None:
        clauses.append("p.id=?")
        params.append(args.paper_id)
    if args.year is not None:
        clauses.append("p.collection_year=?")
        params.append(args.year)
    if args.kind is not None:
        clauses.append("fa.paper_kind=?")
        params.append(args.kind)
    sql = (
        "SELECT p.id paper_id,p.title,p.collection_year,fa.id file_asset_id,fa.sha256 source_pdf_sha256,"
        "fa.relative_path,sr.dataset_path FROM applicability_decisions d "
        "JOIN papers p ON p.id=d.paper_id JOIN content_versions cv ON cv.id=d.content_version_id "
        "JOIN file_assets fa ON fa.id=cv.file_asset_id JOIN source_roots sr ON sr.id=fa.source_root_id WHERE "
        + " AND ".join(clauses)
        + " ORDER BY p.collection_year,fa.paper_kind,fa.relative_path"
    )
    if args.limit is not None:
        sql += " LIMIT ?"
        params.append(args.limit)
    cursor = await catalog.db.execute(sql, params)
    return list(await cursor.fetchall())


async def _ensure_runs(
    catalog: Any, experiment: Any, rows: list[Any], retry_failed: bool
) -> list[dict[str, Any]]:
    now = _utc_now()
    pending: list[dict[str, Any]] = []
    for row in rows:
        for repetition in range(1, experiment["repeat_count"] + 1):
            await catalog.db.execute(
                "INSERT INTO ara_runs(experiment_id,paper_id,file_asset_id,repetition_index,status,execution_mode,source_pdf_sha256,upstream_commit,created_at) "
                "VALUES (?,?,?,?,'pending','realtime',?,?,?) ON CONFLICT(experiment_id,paper_id,file_asset_id,repetition_index) DO NOTHING",
                (
                    experiment["id"],
                    row["paper_id"],
                    row["file_asset_id"],
                    repetition,
                    row["source_pdf_sha256"],
                    experiment["upstream_commit"],
                    now,
                ),
            )
            cursor = await catalog.db.execute(
                "SELECT id,status FROM ara_runs WHERE experiment_id=? AND paper_id=? AND file_asset_id=? AND repetition_index=?",
                (experiment["id"], row["paper_id"], row["file_asset_id"], repetition),
            )
            run = await cursor.fetchone()
            if run["status"] == "succeeded":
                continue
            if run["status"] == "failed" and not retry_failed:
                continue
            if run["status"] in {"failed", "running"}:
                await catalog.db.execute(
                    "UPDATE ara_runs SET status='pending',error_type=NULL,error_text=NULL WHERE id=?",
                    (run["id"],),
                )
            pending.append({**dict(row), "run_id": run["id"], "repetition": repetition})
    await catalog.db.commit()
    return pending


async def _execute_run(
    database: Path,
    experiment_name: str,
    run: dict[str, Any],
    checkout: Path,
    output_root: Path,
    logger: Any,
    Catalog: Any,
) -> bool:
    async with Catalog(database) as catalog:
        await ensure_ara_schema(catalog)
        experiment = await _get_ara_experiment(catalog, experiment_name)
        started = _utc_now()
        await catalog.db.execute(
            "UPDATE ara_runs SET status='running',started_at=?,finished_at=NULL WHERE id=?",
            (started, run["run_id"]),
        )
        await catalog.db.commit()

    pdf = Path(run["dataset_path"]) / run["relative_path"]
    artifact = (
        output_root
        / experiment_name
        / str(run["collection_year"])
        / f"paper-{run['paper_id']}-run-{run['repetition']}.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    bridge = Path(__file__).resolve().with_name("ara_bridge.py")
    command = [
        "uv",
        "run",
        "--project",
        str(checkout.resolve() / "src"),
        "--locked",
        "python",
        str(bridge),
        "--checkout",
        str(checkout.resolve()),
        "--pdf",
        str(pdf),
        "--model",
        experiment["model"],
        "--temperature",
        str(
            experiment["temperature"] if experiment["temperature"] is not None else 0.0
        ),
    ]
    reasoning_effort = json.loads(experiment["configuration_json"]).get(
        "reasoning_effort"
    )
    if reasoning_effort:
        command.extend(["--reasoning-effort", reasoning_effort])
    command.extend(["--output", str(artifact.resolve())])
    tick = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout, stderr = await process.communicate()
    duration_ms = round((time.perf_counter() - tick) * 1000)
    finished = _utc_now()

    async with Catalog(database) as catalog:
        await ensure_ara_schema(catalog)
        if process.returncode != 0:
            error = (
                stderr.decode("utf-8", errors="replace").strip()
                or stdout.decode("utf-8", errors="replace").strip()
            )
            await catalog.db.execute(
                "UPDATE ara_runs SET status='failed',duration_ms=?,error_type='UpstreamProcessError',error_text=?,finished_at=? WHERE id=?",
                (duration_ms, error[-20000:], finished, run["run_id"]),
            )
            await catalog.db.commit()
            logger.error(
                "ara_run_failed",
                run_id=run["run_id"],
                paper_id=run["paper_id"],
                error=error[-1000:],
            )
            return False
        try:
            raw_text = artifact.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            profile = payload["profile"]
            if not all(
                key in profile
                for key in ("metadata", "nodes_source", "nodes_process", "nodes_sink")
            ):
                raise ValueError("Upstream output is not a PaperProfile")
            scores = score_ara_profile(profile)
            usage = payload.get("usage") or {}
            # Upstream does not report spend, so total_usd is absent and the column was
            # silently NULL for every run before 2026-09-18. Derive it from the token
            # counts the API did return.
            cost_usd = usage.get("total_usd")
            if cost_usd is None:
                from pricing import assessment_cost_usd

                cost_usd = assessment_cost_usd(
                    experiment["model"],
                    usage.get("per_query"),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    responses=payload.get("responses"),
                    tier="standard",
                )
            await catalog.db.execute(
                "UPDATE ara_runs SET status='succeeded',profile_json=?,raw_response_json=?,usage_json=?,"
                "source_score=?,methods_score=?,experiments_score=?,sink_score=?,content_score=?,structural_score=?,reproducibility_index=?,wiring_metrics_json=?,"
                "source_node_count=?,method_node_count=?,experiment_node_count=?,sink_node_count=?,input_tokens=?,output_tokens=?,total_tokens=?,estimated_cost_usd=?,"
                "duration_ms=?,artifact_path=?,artifact_sha256=?,error_type=NULL,error_text=NULL,finished_at=? WHERE id=?",
                (
                    _json(profile),
                    raw_text,
                    _json(usage),
                    scores["source_score"],
                    scores["methods_score"],
                    scores["experiments_score"],
                    scores["sink_score"],
                    scores["content_score"],
                    scores["structural_score"],
                    scores["reproducibility_index"],
                    _json(scores["wiring_metrics"]),
                    scores["source_node_count"],
                    scores["method_node_count"],
                    scores["experiment_node_count"],
                    scores["sink_node_count"],
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("total_tokens"),
                    cost_usd,
                    duration_ms,
                    str(artifact),
                    _sha256_file(artifact),
                    finished,
                    run["run_id"],
                ),
            )
            await catalog.db.commit()
        except (KeyError, OSError, TypeError, ValueError) as exc:
            await catalog.db.execute(
                "UPDATE ara_runs SET status='failed',duration_ms=?,error_type=?,error_text=?,finished_at=? WHERE id=?",
                (duration_ms, type(exc).__name__, str(exc), finished, run["run_id"]),
            )
            await catalog.db.commit()
            logger.error(
                "ara_run_invalid",
                run_id=run["run_id"],
                paper_id=run["paper_id"],
                error=str(exc),
            )
            return False
    logger.info(
        "ara_run_complete",
        run_id=run["run_id"],
        paper_id=run["paper_id"],
        repetition=run["repetition"],
        duration_ms=duration_ms,
        reproducibility_index=scores["reproducibility_index"],
    )
    return True


async def command_ara_run(args: Any, logger: Any, Catalog: Any) -> int:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GPT_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY or GPT_API_KEY in the environment")
    database = Path(args.database)
    checkout = Path(args.checkout)
    async with Catalog(database) as catalog:
        await ensure_ara_schema(catalog)
        experiment = await _get_ara_experiment(catalog, args.experiment)
        await asyncio.to_thread(
            validate_checkout, checkout, experiment["upstream_commit"]
        )
        rows = await _ara_candidates(catalog, experiment, args)
        pending = await _ensure_runs(catalog, experiment, rows, args.retry_failed)
    if not pending:
        logger.info(
            "ara_nothing_to_do", experiment=args.experiment, selected_papers=len(rows)
        )
        return 0

    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def bounded(run: dict[str, Any]) -> bool:
        async with semaphore:
            return await _execute_run(
                database,
                args.experiment,
                run,
                checkout,
                Path(args.output_dir),
                logger,
                Catalog,
            )

    succeeded = failed = 0
    tasks = [asyncio.create_task(bounded(run)) for run in pending]
    with tqdm(
        total=len(tasks), desc="Running ARA", unit="run", disable=args.no_progress
    ) as progress:
        for task in asyncio.as_completed(tasks):
            if await task:
                succeeded += 1
            else:
                failed += 1
            progress.update(1)
    logger.info(
        "ara_execution_complete",
        experiment=args.experiment,
        selected_papers=len(rows),
        runs=len(pending),
        succeeded=succeeded,
        failed=failed,
    )
    return 0 if failed == 0 else 2


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


async def build_ara_report(
    catalog: Any, experiment_name: str, survey_path: Path
) -> dict[str, Any]:
    from ara_analysis import SCORE_FIELDS, aggregate_paper_runs, compare_with_haim

    experiment = await _get_ara_experiment(catalog, experiment_name)
    cursor = await catalog.db.execute(
        "SELECT r.id run_id,r.paper_id,r.repetition_index,r.status,"
        "r.source_score,r.methods_score,r.experiments_score,r.sink_score,"
        "r.content_score,r.structural_score,r.reproducibility_index,"
        "r.source_node_count,r.method_node_count,r.experiment_node_count,r.sink_node_count,"
        "r.input_tokens,r.output_tokens,r.total_tokens,r.estimated_cost_usd,r.duration_ms,"
        "r.artifact_path,r.artifact_sha256,r.error_type,r.error_text,"
        "p.canonical_doi,p.title,p.collection_year,fa.paper_kind FROM ara_runs r "
        "JOIN papers p ON p.id=r.paper_id JOIN file_assets fa ON fa.id=r.file_asset_id "
        "WHERE r.experiment_id=? ORDER BY p.collection_year,fa.paper_kind,p.id,r.repetition_index",
        (experiment["id"],),
    )
    rows = [dict(row) for row in await cursor.fetchall()]
    successful = [row for row in rows if row["status"] == "succeeded"]
    distributions = {
        field: _distribution(
            [row[field] for row in successful if row[field] is not None]
        )
        for field in SCORE_FIELDS
    }
    by_year = {}
    for year in sorted({row["collection_year"] for row in successful}):
        year_rows = [row for row in successful if row["collection_year"] == year]
        by_year[str(year)] = {
            field: _distribution(
                [row[field] for row in year_rows if row[field] is not None]
            )
            for field in SCORE_FIELDS
        }
    paper_summaries = aggregate_paper_runs(rows)
    comparison = compare_with_haim(paper_summaries, survey_path, rows)
    return {
        "analysis_version": ARA_INTEGRATION_VERSION,
        "experiment": dict(experiment),
        "run_counts": dict(
            sorted(
                __import__("collections").Counter(row["status"] for row in rows).items()
            )
        ),
        "distributions": distributions,
        "by_year": by_year,
        "paper_summaries": paper_summaries,
        "haim_comparison": comparison,
        "decision_rule": {
            "binary_threshold": None,
            "reason": "Haim's applicable LAK benchmark has no reproducible positive cases, so it cannot identify or validate a binary R threshold.",
            "interpretation": "R is retained as a continuous document-level reconstructability index, not a reproduction outcome.",
            "required_threshold_validation": "Use an external benchmark containing both human-verified reproducible and non-reproducible outcomes; report ordinal MAE, weighted kappa, rank correlation, and threshold sensitivity analysis.",
        },
        "runs": rows,
    }


def render_ara_report(report: dict[str, Any]) -> None:
    console = Console()
    console.print(
        f"[bold]ARA report: {report['experiment']['name']}[/bold]  runs={report['run_counts']}"
    )
    table = Table(title="ARA score distributions")
    for column in ("Score", "n", "Mean", "Median", "SD", "Min", "Max"):
        table.add_column(column, justify="right" if column != "Score" else "left")
    for name, values in report["distributions"].items():
        formatted = [
            "n/a" if values[field] is None else f"{values[field]:.3f}"
            for field in ("mean", "median", "stdev", "min", "max")
        ]
        table.add_row(name, str(values["n"]), *formatted)
    console.print(table)
    candidates = report["haim_comparison"]["haim_likely_reproducible_candidates"]
    candidate_table = Table(
        title="Haim likely-reproducible candidates (both open data and full materials)"
    )
    for column in ("DOI", "ARA status", "Rc", "Rs", "R", "R rank", "Haim 15 min"):
        candidate_table.add_column(
            column,
            justify="right" if column in {"Rc", "Rs", "R", "R rank"} else "left",
        )
    for candidate in candidates:
        scores = [
            "pending" if candidate[field] is None else f"{candidate[field]:.3f}"
            for field in (
                "content_score",
                "structural_score",
                "reproducibility_index",
            )
        ]
        candidate_table.add_row(
            candidate["doi"],
            ", ".join(candidate["ara_statuses"]) or "not scheduled",
            *scores,
            (
                f"{candidate['ara_rank_by_r']}/{candidate['ara_completed_papers']}"
                if candidate["ara_rank_by_r"] is not None
                else "pending"
            ),
            candidate["haim_reproducible_15_minutes"],
        )
    console.print(candidate_table)
    console.print(
        "[bold yellow]No binary reproducible/not-reproducible cutoff is estimated from Haim:[/bold yellow] its applicable LAK cases are all labeled non-reproducible."
    )


async def command_ara_report(args: Any, logger: Any, Catalog: Any) -> int:
    from ara_analysis import write_analysis_artifacts

    async with Catalog(Path(args.database)) as catalog:
        await ensure_ara_schema(catalog)
        report = await build_ara_report(catalog, args.experiment, Path(args.survey))
    render_ara_report(report)
    artifacts = write_analysis_artifacts(report, Path(args.charts_dir))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    logger.info(
        "ara_report_complete",
        experiment=args.experiment,
        run_counts=report["run_counts"],
        output=str(args.output) if args.output else None,
        artifacts=artifacts,
    )
    return 0
