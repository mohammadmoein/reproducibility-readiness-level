"""Stage 4 Arm A: drive, record, verify and compare artifact-execution runs.

See docs/stage4-execution.md. The Inspect AI task in
`arm_a/` runs the authors' code; everything in this module happens outside the sandbox.

The division of labour matters and is not incidental:

- `arm_a/` never reads the paper's reported value. This module does, but only in
  `command_arm_a_compare` and in the pre-run leak check, both of which run in a process the
  agent has no contact with. There is no Inspect scorer for exactly this reason.
- The two axes of the measurement land in one `arm_a_decisions` row: the intervention rung
  (how much help the artifact needed to run at all, L3-vs-L4) and the tolerance verdict
  (whether what it produced agrees, L5/L6).
- Nine papers cannot estimate a reproduction rate. `report` refuses to print one.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from screening import sha256_text, utc_now

EXPERIMENT_KIND = "arm_a_execution"
HARNESS = "inspect_ai"

#: Rung ordering, lowest intervention first. A tie in the majority vote resolves *upward*:
#: claiming an artifact needed less help than it did is the error that matters.
RUNG_ORDER = ("A0", "A1", "A2", "A3", "A4")

#: Tie-break order for the fault class, most artifact-attributable first. A tie also sets
#: `needs_adjudication`, so this only decides what is displayed while a human looks.
FAULT_PRECEDENCE = (
    "code_error",
    "data_mismatch",
    "environment",
    "underspecified",
    "value_mismatch",
    "agent_fault",
    "none",
)

#: `docs/protocol.md` §6, frozen. Note T2 broad is **0.05 absolute or 10% relative**.
TOLERANCES: dict[str, dict[str, float]] = {
    "inferential_effect": {"strict_abs": 0.05, "broad_abs": 0.20},
    "predictive_performance": {
        "strict_abs": 0.01,
        "strict_rel": 0.02,
        "broad_abs": 0.05,
        "broad_rel": 0.10,
    },
    "comparative_ranking": {"broad_tau": 0.6},
    "descriptive_structural": {"strict_ari": 0.7},
}

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


ARM_A_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS arm_a_experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK(status IN ('draft','frozen','closed')),
    claim_experiment_id INTEGER NOT NULL REFERENCES claim_experiments(id) ON DELETE RESTRICT,
    prompt_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    submission_schema TEXT NOT NULL,
    schema_sha256 TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    reasoning_effort TEXT,
    repeat_count INTEGER NOT NULL CHECK(repeat_count > 0),
    message_limit INTEGER,
    cost_limit_usd REAL,
    time_limit_seconds INTEGER NOT NULL,
    command_timeout_seconds INTEGER NOT NULL,
    network_available INTEGER NOT NULL,
    harness TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    image_tag TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    frozen_at TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS arm_a_runs (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES arm_a_experiments(id) ON DELETE RESTRICT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,
    repetition_index INTEGER NOT NULL CHECK(repetition_index > 0),
    snapshot_id INTEGER REFERENCES artifact_snapshots(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
    custom_id TEXT NOT NULL UNIQUE,
    prompt_sha256 TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    leak_check TEXT NOT NULL,
    log_path TEXT,
    eval_status TEXT,
    limit_hit TEXT,
    ran INTEGER,
    intervention_rung TEXT,
    entrypoint TEXT,
    exit_code INTEGER,
    install_commands_json TEXT,
    code_changes_json TEXT,
    pip_freeze TEXT,
    produced_value REAL,
    produced_value_units TEXT,
    produced_metric_name TEXT,
    produced_value_location TEXT,
    produced_ranking_json TEXT,
    produced_group_count INTEGER,
    stdout_excerpt TEXT,
    no_output_reason TEXT,
    unrunnable_reason TEXT,
    agent_fault_class TEXT,
    agent_reasoning TEXT,
    submission_json TEXT,
    submission_error TEXT,
    contradictions_json TEXT,
    evidence_source TEXT,
    evidence_text TEXT,
    evidence_file_state TEXT,
    files_written_json TEXT,
    location_verified INTEGER,
    location_verify_note TEXT,
    setup_json TEXT,
    network_observed TEXT,
    wall_clock_seconds REAL,
    working_seconds REAL,
    message_count INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    error_type TEXT,
    error_text TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE(experiment_id, paper_id, repetition_index)
) STRICT;

CREATE TABLE IF NOT EXISTS arm_a_decisions (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES arm_a_experiments(id) ON DELETE RESTRICT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,
    claim_type TEXT,
    ran INTEGER,
    intervention_rung TEXT,
    rung_votes_json TEXT NOT NULL,
    fault_class TEXT,
    fault_votes_json TEXT NOT NULL,
    produced_value REAL,
    produced_value_spread REAL,
    produced_values_json TEXT NOT NULL,
    reported_value REAL,
    reported_units TEXT,
    metric_name TEXT,
    scale_note TEXT,
    abs_delta REAL,
    rel_delta REAL,
    comparator TEXT NOT NULL,
    tolerance_verdict TEXT NOT NULL,
    qualitative_verdict TEXT NOT NULL,
    rrl_level TEXT NOT NULL,
    comparison_note TEXT,
    location_verified INTEGER,
    value_disagreement INTEGER NOT NULL DEFAULT 0,
    needs_adjudication INTEGER NOT NULL DEFAULT 0,
    completed_run_count INTEGER NOT NULL,
    wall_clock_seconds REAL,
    aggregated_at TEXT NOT NULL,
    UNIQUE(experiment_id, paper_id)
) STRICT;
"""


async def ensure_arm_a_schema(catalog: Any) -> None:
    for statement in ARM_A_SCHEMA.split(";"):
        if statement.strip():
            await catalog.db.execute(statement)
    await catalog.db.commit()


# --- helpers -----------------------------------------------------------------------


async def get_experiment(catalog: Any, name: str) -> Any:
    cursor = await catalog.db.execute("SELECT * FROM arm_a_experiments WHERE name=?", (name,))
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Unknown Arm A experiment: {name}")
    return row


async def _claim_experiment(catalog: Any, name: str | None) -> Any:
    if name:
        cursor = await catalog.db.execute(
            "SELECT id,name,model FROM claim_experiments WHERE name=?", (name,)
        )
    else:
        cursor = await catalog.db.execute(
            "SELECT id,name,model FROM claim_experiments WHERE status='frozen' "
            "ORDER BY id DESC LIMIT 1"
        )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(
            "No Stage 3 claim experiment to attach to. Arm A targets the focal claim's "
            "metric; run `lakdb claim aggregate` first."
        )
    return row


async def _candidate_papers(catalog: Any, experiment: Any, paper_id: int | None) -> list[int]:
    """Papers Arm A can run: a Stage 3 focal claim plus a retrieved code repository.

    That intersection is recomputed rather
    than hardcoded so a later retrieval round widens it without a code change.
    """
    if paper_id is not None:
        return [paper_id]
    cursor = await catalog.db.execute(
        "SELECT DISTINCT d.paper_id FROM claim_decisions d "
        "JOIN artifact_snapshots s ON s.paper_id = d.paper_id AND s.size_bytes > 0 "
        "JOIN artifact_urls u ON u.id = s.url_id AND u.artifact_kind IN ('code','both') "
        "WHERE d.experiment_id = ? ORDER BY d.paper_id",
        (experiment["claim_experiment_id"],),
    )
    return [int(row["paper_id"]) for row in await cursor.fetchall()]


async def _withheld(catalog: Any, claim_experiment_id: int, paper_id: int) -> dict[str, Any]:
    """The reported value and its comparison target. Only `compare` and the leak check
    may call this; nothing here may be passed to `arm_a.dataset` or into a prompt."""
    cursor = await catalog.db.execute(
        "SELECT claim_type, reported_value_verbatim, reported_value_parsed_json, "
        "comparison_target_json FROM claim_decisions WHERE experiment_id=? AND paper_id=?",
        (claim_experiment_id, paper_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return {}
    try:
        parsed = json.loads(row["reported_value_parsed_json"] or "null") or {}
    except ValueError:
        parsed = {}
    try:
        target = json.loads(row["comparison_target_json"] or "null") or {}
    except ValueError:
        target = {}
    return {
        "claim_type": row["claim_type"],
        "verbatim": row["reported_value_verbatim"],
        "point_estimate": parsed.get("point_estimate"),
        "units": parsed.get("units"),
        "metric_name": parsed.get("metric_name") or target.get("metric_name"),
        "ranked_items": target.get("ranked_items"),
        "expected_group_count": target.get("expected_group_count"),
        # Stage 3 records what the paper leaves undefined. The comparator must honour it:
        # paper 34's claim is an average "between the three platforms", the pilot ran one
        # platform pair, and the mismatch verdict compared a single cell against an average.
        "unspecified_by_paper": target.get("unspecified_by_paper") or [],
        "analysis_population": target.get("analysis_population"),
    }


def value_needles(point_estimate: float | None, verbatim: str | None) -> set[str]:
    """String forms of the withheld value that must not appear in an Arm A prompt.

    Short forms are dropped: a bare `8` or `.8` would fire on any file tree, and a check
    that always fires teaches the operator to ignore it.
    """
    needles: set[str] = set()
    candidates: list[float] = []
    if point_estimate is not None:
        candidates.extend([float(point_estimate), float(point_estimate) * 100.0,
                           float(point_estimate) / 100.0])
    for text in (verbatim or "",):
        for match in _NUMBER.findall(text):
            try:
                candidates.append(float(match))
            except ValueError:
                continue
    for value in candidates:
        if not value:
            continue
        for form in (f"{value:g}", f"{value:.1f}", f"{value:.2f}", f"{value:.3f}"):
            # Lossless forms only. A rounded form is a different number, and a needle that
            # is not the value fires on unrelated file names and trains the operator to
            # pass --skip-leak-check without looking.
            if len(form) >= 3 and float(form) == value:
                needles.add(form)
    return needles


def leak_check(prompt: str, needles: set[str]) -> list[str]:
    haystack = prompt.lower()
    return sorted(needle for needle in needles if needle.lower() in haystack)


def numbers_in(text: str | None) -> list[float]:
    values: list[float] = []
    for match in _NUMBER.findall(text or ""):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def value_present(value: float, text: str | None) -> bool:
    """Is `value` actually at this location?

    Compares parsed numbers rather than substrings, so `0.8` matches a stored
    `0.7999999998` and a printed `80.0%`, and does not match the `0.8` inside `10.85`.
    """
    if not text:
        return False
    decimals = len(f"{value!r}".partition(".")[2])
    for candidate in numbers_in(text):
        for scaled in (candidate, candidate / 100.0, candidate * 100.0):
            if math.isclose(scaled, value, rel_tol=1e-4, abs_tol=1e-9):
                return True
            if decimals and round(scaled, decimals) == round(value, decimals):
                return True
    return False


def _modal(values: list[str], precedence: tuple[str, ...]) -> tuple[str | None, dict[str, int], bool]:
    if not values:
        return None, {}, False
    counts = Counter(values)
    top = max(counts.values())
    tied = sorted((v for v, n in counts.items() if n == top),
                  key=lambda v: precedence.index(v) if v in precedence else len(precedence))
    return tied[0], dict(counts), len(tied) > 1


# --- create ------------------------------------------------------------------------


async def command_arm_a_create(args: Any, logger: Any, Catalog: Any) -> int:
    from arm_a.dataset import DEFAULT_IMAGE_TAG
    from arm_a.prompt import (
        ARM_A_SYSTEM_PROMPT,
        PROMPT_NAME,
        PROMPT_VERSION,
        SCHEMA_VERSION,
        SUBMISSION_SCHEMA,
    )

    import importlib.metadata

    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        claim = await _claim_experiment(catalog, getattr(args, "claim_experiment", None))
        now = utc_now()
        configuration = {
            "arm": "A",
            "question": "does the shipped artifact run, and does it recover the claim",
            "harness": "Inspect AI react() in the Holtdirk container, unmodified image",
            "paper_mounted": False,
            "author_code_mounted": True,
            "reported_value_withheld": True,
            "scorer": None,
            "comparison": "lakdb arm-a compare, outside the sandbox, docs/protocol.md §6",
            "intervention_ladder": list(RUNG_ORDER),
            "claim_experiment": claim["name"],
            "model_deviation": (
                "Holtdirk used Claude Opus 4.7; this project uses a single frozen model "
                "across every stage."
            ),
            "network_isolated_during_analysis": False,
        }
        schema_text = SUBMISSION_SCHEMA
        cursor = await catalog.db.execute(
            "INSERT INTO arm_a_experiments(name,status,claim_experiment_id,prompt_name,"
            "prompt_version,system_prompt,prompt_sha256,schema_version,submission_schema,"
            "schema_sha256,provider,model,reasoning_effort,repeat_count,message_limit,"
            "cost_limit_usd,time_limit_seconds,command_timeout_seconds,network_available,"
            "harness,harness_version,image_tag,configuration_json,created_at,frozen_at) "
            "VALUES (?,'frozen',?,?,?,?,?,?,?,?,'openai',?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                args.name, claim["id"], PROMPT_NAME, PROMPT_VERSION, ARM_A_SYSTEM_PROMPT,
                sha256_text(ARM_A_SYSTEM_PROMPT), SCHEMA_VERSION, schema_text,
                sha256_text(schema_text), args.model, args.reasoning_effort, args.repeats,
                args.message_limit, args.cost_limit, args.time_limit,
                args.command_timeout, int(not args.no_network), HARNESS,
                importlib.metadata.version("inspect_ai"), DEFAULT_IMAGE_TAG,
                json.dumps(configuration, sort_keys=True), now, now,
            ),
        )
        await catalog.db.commit()
        logger.info(
            "arm_a_experiment_created",
            name=args.name,
            experiment_id=cursor.lastrowid,
            model=args.model,
            repeats=args.repeats,
            claim_experiment=claim["name"],
            prompt_sha256=sha256_text(ARM_A_SYSTEM_PROMPT)[:12],
            time_limit_seconds=args.time_limit,
        )
    return 0


# --- run ---------------------------------------------------------------------------


async def command_arm_a_run(args: Any, logger: Any, Catalog: Any) -> int:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        # Inspect reads `.env` itself before dispatching, so refusing here on an unset
        # environment variable would reject runs that would otherwise have worked.
        from dotenv import load_dotenv

        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required (environment or .env)")
    if getattr(args, "no_progress", False):
        # `eval_async` takes no display argument; Inspect reads this instead.
        os.environ.setdefault("INSPECT_DISPLAY", "plain")

    from inspect_ai import eval_async

    from arm_a.dataset import build_sample
    from arm_a.task import arm_a as arm_a_task

    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        repeats = int(getattr(args, "repeats", None) or experiment["repeat_count"])
        papers = await _candidate_papers(catalog, experiment, getattr(args, "paper_id", None))
        if not papers:
            logger.warning("arm_a_no_candidates", experiment=args.experiment)
            return 0
        log_root = Path(getattr(args, "log_dir", None) or "artifacts/arm_a/logs")
        network = bool(experiment["network_available"])
        cursor = await catalog.db.execute(
            "SELECT name FROM claim_experiments WHERE id=?",
            (experiment["claim_experiment_id"],),
        )
        claim_name = (await cursor.fetchone())["name"]
        logger.info(
            "arm_a_run_start", experiment=args.experiment, papers=papers, repeats=repeats,
            model=experiment["model"], time_limit_seconds=experiment["time_limit_seconds"],
        )

        ok = failed = skipped = 0
        for paper_id in papers:
            sample = build_sample(
                database=str(args.database),
                paper_id=paper_id,
                claim_experiment=claim_name,
                network=network,
                image_tag=experiment["image_tag"],
            )
            prompt = str(sample.input)
            withheld = await _withheld(catalog, experiment["claim_experiment_id"], paper_id)
            hits = leak_check(
                prompt + experiment["system_prompt"],
                value_needles(withheld.get("point_estimate"), withheld.get("verbatim")),
            )
            if hits and not getattr(args, "skip_leak_check", False):
                raise ValueError(
                    f"Paper {paper_id}: the reported value would reach the agent — the "
                    f"prompt contains {hits}. Inspect the prompt before proceeding; rerun "
                    f"with --skip-leak-check only once you have established it is a "
                    f"coincidence, which is then recorded on the run."
                )
            leak_state = "clean" if not hits else f"acknowledged:{','.join(hits)}"

            for repetition in range(1, repeats + 1):
                custom_id = f"arma-{experiment['id']}-{paper_id}-{repetition}"
                cursor = await catalog.db.execute(
                    "SELECT status FROM arm_a_runs WHERE custom_id=?", (custom_id,)
                )
                existing = await cursor.fetchone()
                if (
                    existing
                    and existing["status"] == "succeeded"
                    and not getattr(args, "force", False)
                ):
                    skipped += 1
                    continue
                await catalog.db.execute(
                    "INSERT INTO arm_a_runs(experiment_id,paper_id,repetition_index,"
                    "snapshot_id,status,custom_id,prompt_sha256,user_prompt,leak_check,"
                    "created_at) VALUES (?,?,?,?,'running',?,?,?,?,?) "
                    "ON CONFLICT(custom_id) DO UPDATE SET status='running',"
                    "user_prompt=excluded.user_prompt,prompt_sha256=excluded.prompt_sha256,"
                    "leak_check=excluded.leak_check",
                    (
                        experiment["id"], paper_id, repetition,
                        (sample.metadata or {}).get("snapshot_id"), custom_id,
                        sha256_text(prompt), prompt, leak_state, utc_now(),
                    ),
                )
                await catalog.db.commit()

                log_dir = log_root / args.experiment / str(paper_id) / f"rep{repetition}"
                try:
                    logs = await eval_async(
                        arm_a_task(
                            paper_id=paper_id,
                            database=str(args.database),
                            claim_experiment=claim_name,
                            reasoning_effort=experiment["reasoning_effort"] or "medium",
                            message_limit=experiment["message_limit"],
                            cost_limit=experiment["cost_limit_usd"],
                            time_limit=experiment["time_limit_seconds"],
                            command_timeout=experiment["command_timeout_seconds"],
                            image_tag=experiment["image_tag"],
                            network=network,
                        ),
                        model=f"openai/{experiment['model']}",
                        log_dir=str(log_dir),
                        fail_on_error=False,
                        score=False,
                    )
                except Exception as exc:  # noqa: BLE001 — a harness crash is a recorded state
                    await catalog.db.execute(
                        "UPDATE arm_a_runs SET status='failed',error_type=?,error_text=?,"
                        "finished_at=? WHERE custom_id=?",
                        (type(exc).__name__, str(exc)[:4000], utc_now(), custom_id),
                    )
                    await catalog.db.commit()
                    failed += 1
                    logger.error(
                        "arm_a_eval_failed", paper_id=paper_id, repetition=repetition,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                log = logs[0]
                status = "succeeded" if log.status == "success" else "failed"
                await catalog.db.execute(
                    "UPDATE arm_a_runs SET status=?,log_path=?,eval_status=?,error_type=?,"
                    "error_text=?,finished_at=? WHERE custom_id=?",
                    (
                        status, log.location, log.status,
                        type(log.error).__name__ if log.error else None,
                        (log.error.message if log.error else None),
                        utc_now(), custom_id,
                    ),
                )
                await catalog.db.commit()
                ok += int(status == "succeeded")
                failed += int(status != "succeeded")
                logger.info(
                    "arm_a_eval_complete", paper_id=paper_id, repetition=repetition,
                    eval_status=log.status, log=log.location,
                )
        logger.info("arm_a_run_complete", succeeded=ok, failed=failed, skipped=skipped)
    return 0


# --- collect -----------------------------------------------------------------------


def _usage_blocks(sample: Any) -> tuple[list[Any], bool]:
    """Per-request usage from the transcript, falling back to the sample aggregate.

    The aggregate is not safe to price directly. `pricing.run_cost_usd` selects the
    long-context tier on a *single request's* input, and an Arm A sample makes around
    fifty calls; summing them first crossed 128k and doubled the bill on the pilot. This
    is the same trap `pricing.assessment_cost_usd` documents for ARA's six queries.
    """
    blocks = [
        event.output.usage
        for event in (sample.events or [])
        if getattr(event, "event", None) == "model"
        and getattr(getattr(event, "output", None), "usage", None) is not None
    ]
    if blocks:
        return blocks, True
    return list((sample.model_usage or {}).values()), False


def _usage_cost(model: str, sample: Any) -> tuple[int, int, float | None, bool]:
    from pricing import UnknownModelError, request_cost_usd

    blocks, per_request = _usage_blocks(sample)
    total_in = total_out = 0
    cost = 0.0
    priced = False
    for usage in blocks:
        # Inspect reports `input_tokens` as the *uncached* remainder, with cache reads and
        # writes alongside it; `pricing.request_cost_usd` expects OpenAI's convention,
        # where `input_tokens` is the total and the details are carved out of it. Summing
        # reconciles the two, and stops the token count reading an order of magnitude low
        # once prompt caching engages (1.08M cache reads on the pilot against 144 fresh).
        cached = int(getattr(usage, "input_tokens_cache_read", 0) or 0)
        written = int(getattr(usage, "input_tokens_cache_write", 0) or 0)
        fresh = int(getattr(usage, "input_tokens", 0) or 0)
        output = int(getattr(usage, "output_tokens", 0) or 0)
        total_in += fresh + cached + written
        total_out += output
        try:
            cost += request_cost_usd(
                model,
                {
                    "input_tokens": fresh + cached + written,
                    "output_tokens": output,
                    "input_tokens_details": {
                        "cached_tokens": cached,
                        "cache_write_tokens": written,
                    },
                },
            )
            priced = True
        except UnknownModelError:
            continue
    return total_in, total_out, (cost if priced else None), per_request


def _transcript_matches(messages: list[Any], value: float | None, cap: int = 12) -> str:
    """Lines in the agent's tool output that carry `value`.

    The fallback evidence path for a `stdout:` location: the container is gone by the time
    `verify` runs, but everything the tools printed is in the log.
    """
    if value is None:
        return ""
    found: list[str] = []
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        text = getattr(message, "text", None) or ""
        for line in text.splitlines():
            if value_present(value, line):
                found.append(line.strip()[:400])
                if len(found) >= cap:
                    return "\n".join(found)
    return "\n".join(found)


async def command_arm_a_collect(args: Any, logger: Any, Catalog: Any) -> int:
    from inspect_ai.log import read_eval_log

    from arm_a.submission import parse_submission
    from arm_a.task import EVIDENCE_KEY, SETUP_KEY

    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        cursor = await catalog.db.execute(
            "SELECT custom_id, paper_id, repetition_index, log_path FROM arm_a_runs "
            "WHERE experiment_id=? ORDER BY paper_id, repetition_index",
            (experiment["id"],),
        )
        rows = list(await cursor.fetchall())
        parsed = missing = 0
        for row in rows:
            if not row["log_path"]:
                missing += 1
                continue
            try:
                log = read_eval_log(row["log_path"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "arm_a_log_unreadable", custom_id=row["custom_id"], error=str(exc)
                )
                missing += 1
                continue
            if not log.samples:
                missing += 1
                continue
            sample = log.samples[0]
            metadata = sample.metadata or {}
            store = sample.store or {}
            evidence = metadata.get(EVIDENCE_KEY) or store.get(EVIDENCE_KEY) or {}
            setup = metadata.get(SETUP_KEY) or store.get(SETUP_KEY) or {}
            completion = sample.output.completion if sample.output else None
            submission, raw, submission_error = parse_submission(completion)

            if submission is not None:
                produced = submission.produced_value
                fields = {
                    "ran": None if submission.ran is None else int(submission.ran),
                    "intervention_rung": (submission.intervention_rung or "").upper() or None,
                    "entrypoint": submission.entrypoint,
                    "exit_code": submission.exit_code,
                    "install_commands_json": json.dumps(submission.install_commands),
                    "code_changes_json": json.dumps(
                        [c.model_dump() for c in submission.code_changes]
                    ),
                    "pip_freeze": submission.pip_freeze or evidence.get("pip_freeze"),
                    "produced_value": produced,
                    "produced_value_units": submission.produced_value_units,
                    "produced_metric_name": submission.produced_metric_name,
                    "produced_value_location": submission.produced_value_location,
                    "produced_ranking_json": json.dumps(submission.produced_ranking)
                    if submission.produced_ranking is not None else None,
                    "produced_group_count": submission.produced_group_count,
                    "stdout_excerpt": (submission.stdout_excerpt or "")[:8000] or None,
                    "no_output_reason": submission.no_output_reason,
                    "unrunnable_reason": submission.unrunnable_reason,
                    "agent_fault_class": submission.fault_class,
                    "agent_reasoning": submission.reasoning,
                    "contradictions_json": json.dumps(submission.contradictions()),
                }
            else:
                produced = None
                fields = {
                    "ran": None, "intervention_rung": None, "entrypoint": None,
                    "exit_code": None, "install_commands_json": None,
                    "code_changes_json": None,
                    "pip_freeze": evidence.get("pip_freeze"),
                    "produced_value": None, "produced_value_units": None,
                    "produced_metric_name": None, "produced_value_location": None,
                    "produced_ranking_json": None, "produced_group_count": None,
                    "stdout_excerpt": None, "no_output_reason": None,
                    "unrunnable_reason": None, "agent_fault_class": None,
                    "agent_reasoning": None, "contradictions_json": None,
                }

            if evidence.get("location_kind") == "file" and evidence.get("evidence_text"):
                evidence_source = "file"
                evidence_text = evidence.get("evidence_text")
            else:
                evidence_source = "transcript"
                evidence_text = _transcript_matches(sample.messages or [], produced)
                if not evidence_text:
                    evidence_source = "none"

            tokens_in, tokens_out, cost, per_request = _usage_cost(
                experiment["model"], sample
            )
            if not per_request and cost is not None:
                logger.warning(
                    "arm_a_cost_from_aggregate", custom_id=row["custom_id"],
                    note="no per-request usage in the log; cost may be over-tiered",
                )
            limit_hit = sample.limit.type if sample.limit else None
            # A sample that exhausted its budget never reaches submit(), so every outcome
            # field parses as null and the row reads as though nothing happened. That is a
            # real A4 — "did not run within budget" — and must be recorded as one rather
            # than left blank. Paper 468 hit this on all three repetitions: its artifact is
            # an R Markdown file needing tidymodels, tidyverse and xgboost, and the sandbox
            # is python:3.12-slim with no R, so the agent spent the whole budget installing
            # a language rather than running an analysis. The rung says it did not run; the
            # reason says whose environment was at fault.
            if limit_hit and fields["intervention_rung"] is None and raw is None:
                fields["ran"] = 0
                fields["intervention_rung"] = "A4"
                fields["agent_fault_class"] = "environment"
                fields["unrunnable_reason"] = (
                    f"budget exhausted ({limit_hit} limit) before the agent submitted a "
                    f"result; no outcome was reported"
                )
            await catalog.db.execute(
                "UPDATE arm_a_runs SET ran=?,intervention_rung=?,entrypoint=?,exit_code=?,"
                "install_commands_json=?,code_changes_json=?,pip_freeze=?,produced_value=?,"
                "produced_value_units=?,produced_metric_name=?,produced_value_location=?,"
                "produced_ranking_json=?,produced_group_count=?,stdout_excerpt=?,"
                "no_output_reason=?,unrunnable_reason=?,agent_fault_class=?,"
                "agent_reasoning=?,submission_json=?,submission_error=?,"
                "contradictions_json=?,evidence_source=?,evidence_text=?,"
                "evidence_file_state=?,files_written_json=?,setup_json=?,network_observed=?,"
                "limit_hit=?,wall_clock_seconds=?,working_seconds=?,message_count=?,"
                "input_tokens=?,output_tokens=?,estimated_cost_usd=? WHERE custom_id=?",
                (
                    fields["ran"], fields["intervention_rung"], fields["entrypoint"],
                    fields["exit_code"], fields["install_commands_json"],
                    fields["code_changes_json"],
                    (fields["pip_freeze"] or "")[:20000] or None,
                    fields["produced_value"], fields["produced_value_units"],
                    fields["produced_metric_name"], fields["produced_value_location"],
                    fields["produced_ranking_json"], fields["produced_group_count"],
                    fields["stdout_excerpt"], fields["no_output_reason"],
                    fields["unrunnable_reason"], fields["agent_fault_class"],
                    fields["agent_reasoning"],
                    json.dumps(raw, sort_keys=True) if raw is not None else None,
                    submission_error, fields["contradictions_json"],
                    evidence_source, (evidence_text or "")[:20000] or None,
                    evidence.get("file_state"),
                    json.dumps(evidence.get("files_written_by_run") or []),
                    json.dumps(setup, sort_keys=True) if setup else None,
                    evidence.get("network_observed") or setup.get("network_observed"),
                    limit_hit, sample.total_time, sample.working_time,
                    len(sample.messages or []), tokens_in, tokens_out, cost,
                    row["custom_id"],
                ),
            )
            parsed += 1
        await catalog.db.commit()
        logger.info(
            "arm_a_collected", experiment=args.experiment, runs=len(rows),
            parsed=parsed, without_log=missing,
        )
    return 0


# --- verify ------------------------------------------------------------------------



async def command_arm_a_verify(args: Any, logger: Any, Catalog: Any) -> int:
    """Open the location the agent named and confirm the number is there.

    Same guard as Stage 3's `source_quote`, which caught two fabricated quotes in 27 runs.
    A run fails verification for three distinct reasons, and the note says which: the number
    is not at the named location, the location could not be read, or the location names a
    file byte-identical to what the authors shipped — in which case the value was read, not
    produced.
    """
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        cursor = await catalog.db.execute(
            "SELECT custom_id, paper_id, repetition_index, produced_value, "
            "produced_value_location, evidence_source, evidence_text, evidence_file_state "
            "FROM arm_a_runs WHERE experiment_id=? AND status='succeeded' "
            "ORDER BY paper_id, repetition_index",
            (experiment["id"],),
        )
        rows = list(await cursor.fetchall())
        verified = rejected = skipped = 0
        for row in rows:
            value = row["produced_value"]
            if value is None:
                await catalog.db.execute(
                    "UPDATE arm_a_runs SET location_verified=NULL,location_verify_note=? "
                    "WHERE custom_id=?",
                    ("no produced_value to verify", row["custom_id"]),
                )
                skipped += 1
                continue
            notes: list[str] = []
            present = value_present(float(value), row["evidence_text"])
            if not row["produced_value_location"]:
                notes.append("no produced_value_location given")
            if row["evidence_source"] == "none":
                notes.append("location could not be read and the value is not in the transcript")
            elif not present:
                notes.append(f"value not found at the {row['evidence_source']} evidence")
            if row["evidence_file_state"] == "unmodified":
                notes.append(
                    "location names a file byte-identical to the shipped repository: the "
                    "value was read from the authors' committed output, not regenerated"
                )
            passed = bool(present and row["produced_value_location"]) and (
                row["evidence_file_state"] != "unmodified"
            )
            await catalog.db.execute(
                "UPDATE arm_a_runs SET location_verified=?,location_verify_note=? "
                "WHERE custom_id=?",
                (int(passed), "; ".join(notes) or "value found at the named location",
                 row["custom_id"]),
            )
            verified += int(passed)
            rejected += int(not passed)
            if not passed:
                logger.warning(
                    "arm_a_location_unverified", custom_id=row["custom_id"],
                    paper_id=row["paper_id"], note="; ".join(notes),
                )
        await catalog.db.commit()
        logger.info(
            "arm_a_verified", experiment=args.experiment, verified=verified,
            rejected=rejected, no_value=skipped,
        )
    return 0


# --- compare -----------------------------------------------------------------------


def _rescale(
    produced: float,
    produced_units: str | None,
    reported: float,
    reported_units: str | None,
) -> tuple[float, str | None]:
    """Put the produced value on the reported value's scale.

    The extractor records units as printed, so `94.6` carries `units="%"` while the code
    that produced it will normally print `0.946`. Every adjustment is returned as a note and
    stored, because a silent factor of 100 is the kind of thing that makes a mismatch look
    like an agreement.
    """
    produced_is_percent = (produced_units or "").strip().lower() in {"%", "percent", "pct"}
    reported_is_percent = (reported_units or "").strip().lower() in {"%", "percent", "pct"}
    if produced_is_percent == reported_is_percent:
        if reported_is_percent and abs(reported) > 1.5 and abs(produced) <= 1.5:
            return produced * 100.0, "produced value read as a proportion, scaled to percent"
        return produced, None
    if reported_is_percent:
        return produced * 100.0, "produced value scaled from proportion to percent"
    return produced / 100.0, "produced value scaled from percent to proportion"


def _numeric_verdict(
    claim_type: str, produced: float, reported: float
) -> tuple[str, float, float | None, str]:
    absolute = abs(produced - reported)
    relative = absolute / abs(reported) if reported else None
    limits = TOLERANCES.get(claim_type, TOLERANCES["predictive_performance"])
    if claim_type == "inferential_effect":
        if absolute <= limits["strict_abs"]:
            return "strict", absolute, relative, "abs delta-d"
        if absolute <= limits["broad_abs"]:
            return "broad", absolute, relative, "abs delta-d"
        return "mismatch", absolute, relative, "abs delta-d"
    strict = absolute <= limits["strict_abs"] or (
        relative is not None and relative <= limits["strict_rel"]
    )
    broad = absolute <= limits["broad_abs"] or (
        relative is not None and relative <= limits["broad_rel"]
    )
    verdict = "strict" if strict else ("broad" if broad else "mismatch")
    return verdict, absolute, relative, "headline metric, abs or rel"


def _ranking_verdict(
    produced: list[str], reported: list[str]
) -> tuple[str, float | None, str]:
    from scipy.stats import kendalltau

    order = {name.strip().lower(): index for index, name in enumerate(reported)}
    pairs = [
        (order[name.strip().lower()], index)
        for index, name in enumerate(produced)
        if name.strip().lower() in order
    ]
    if len(pairs) < 2:
        return "not_comparable", None, (
            f"only {len(pairs)} of the paper's {len(reported)} ranked items were matched "
            f"by name in the produced ranking"
        )
    tau = float(kendalltau([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
    top_preserved = bool(produced) and bool(reported) and (
        produced[0].strip().lower() == reported[0].strip().lower()
    )
    # docs/protocol.md §6 defines T3 strict as "top-1 claim preserved" and broad as tau >= 0.6,
    # which as written are not nested — a ranking could be strict without being broad. The
    # RRL requires L6 to imply L5, so strict additionally requires the broad condition.
    # Nothing is loosened by this; the top-1 outcome is recorded either way.
    broad = tau >= TOLERANCES["comparative_ranking"]["broad_tau"]
    detail = (
        f"top-1 {'preserved' if top_preserved else 'not preserved'}, "
        f"Kendall tau {tau:.3f} over {len(pairs)} matched items"
    )
    if top_preserved and broad:
        return "strict", tau, detail
    if broad:
        return "broad", tau, detail
    return "mismatch", tau, detail


# `unspecified_by_paper` records what the PAPER'S TEXT omits. That is the right blocker for
# Arm B, which writes fresh code and must guess. In Arm A the artifact supplies most of it:
# hyperparameters, seeds, splits and preprocessing are all in the code that just ran, so the
# text omitting them does not stop us comparing what the code produced.
#
# Only ambiguity the code CANNOT resolve blocks Arm A: which of several outputs corresponds
# to the reported number. Paper 34 is the case — its code emits one value per platform pair
# and the paper reports an unexplained average, so no output is the reported quantity.
# Paper 69 is the counter-case: the text omits hyperparameters, the code contains them, and
# blocking on that hid a reproduction agreeing to four decimal places. The nine-paper run of
# 2026-09-22 made the distinction unavoidable: the first version of this guard fired on every
# paper that ran, so nothing was ever scored. Paper 34's gap ("which conditions were
# averaged") is material; paper 69's ("the train/test split and random seed") is not, and
# blocking on it hid a reproduction that agreed to four decimal places.
#: Blocks Arm A: the reader cannot tell which produced number is the reported one.
_MATERIAL_MARKERS = (
    "which", "aggregat", "averag", "included", "direction", "pair", "subset", "subsample",
    "population", "cutoff", "denominator", "reported average", "across",
)
#: Does not block Arm A: the artifact determines it, whatever the text leaves out.
_IMMATERIAL_MARKERS = (
    "seed", "random state", "split", "fold", "shuffl", "hyperparameter", "preprocessing",
    "composition", "configuration", "implementation detail",
)


def _split_unspecified(
    items: list[Any], for_arm: str = "A"
) -> tuple[list[Any], list[Any]]:
    """Partition Stage 3 omissions by the information available to an execution arm.

    Arm A executes the shipped artifact, so its code settles seeds, folds and similar
    implementation details. Arm B writes fresh code and must guess every omitted detail;
    those omissions therefore all block a numeric comparison there. Keeping this switch in
    the shared comparator prevents the two arms from drifting into separate tolerance rules.
    """
    if for_arm.upper() == "B":
        return list(items), []
    if for_arm.upper() != "A":
        raise ValueError(f"Unknown execution arm: {for_arm!r}")
    material: list[Any] = []
    immaterial: list[Any] = []
    for item in items:
        text = str(item).lower()
        if any(marker in text for marker in _MATERIAL_MARKERS):
            material.append(item)      # "which" beats everything: the target is ambiguous
        elif any(marker in text for marker in _IMMATERIAL_MARKERS):
            immaterial.append(item)    # the code settles it
        else:
            # Unrecognised: treat as material. Declining to score is the safe failure.
            material.append(item)
    return material, immaterial


# Paper 321 reports "15.6% on the AUC metric" — a relative improvement with none of the words
# the first version of this check looked for, so it was compared against an absolute AUC of
# 0.49 and scored as a mismatch. Stage 3 should mark this explicitly; until it does, widen the
# net and accept the false positives, which merely decline to score.
_RELATIVE_PATTERNS = (
    "improvement", "increase of", "increased by", "gain", "relative",
    "higher than", "lower than", "better than", "outperform", "reduction",
    "% on the", "% over", "% above", "percentage point", "pp ",
)


def _looks_relative(verbatim: str) -> bool:
    return any(pattern in verbatim for pattern in _RELATIVE_PATTERNS)


def compare_paper(
    claim_type: str | None,
    produced: float | None,
    produced_units: str | None,
    produced_ranking: list[str] | None,
    produced_group_count: int | None,
    withheld: dict[str, Any],
    ran: bool,
    for_arm: str = "A",
) -> dict[str, Any]:
    """Type-specific comparison, `docs/protocol.md` §6. No model call anywhere in here.

    The qualitative conclusion is the primary outcome for every type (§110). `opposite` is
    only assigned where the type defines a direction — a sign for T1, a negative rank
    correlation for T3. A T2 metric that simply misses is `inconclusive` with a
    `value_mismatch` fault, not `opposite`: inventing a direction for a headline metric
    would be a tolerance we made up after seeing the data.
    """
    result: dict[str, Any] = {
        "comparator": claim_type or "unknown",
        "tolerance_verdict": "not_comparable",
        "qualitative_verdict": "inconclusive",
        "abs_delta": None,
        "rel_delta": None,
        "scale_note": None,
        "note": "",
        "fault": None,
    }
    reported = withheld.get("point_estimate")
    # Scale is handled by `_rescale` at the numeric comparison below, which puts the produced
    # value on the reported value's scale. An earlier revision added a second converter here
    # that moved the reported value the other way; the two cancelled and turned paper 69's
    # four-decimal agreement into a 99% mismatch. One converter, at one place.

    if not ran:
        result["note"] = "the artifact did not run, so there is nothing to compare"
        return result
    if reported is None:
        result["note"] = (
            "Stage 3 extracted no numeric value for this paper's focal claim, so 'does it "
            "run' is answerable and 'does it agree' is not"
        )
        return result

    verbatim = (withheld.get("verbatim") or "").lower()
    if _looks_relative(verbatim):
        result["note"] = (
            f"the reported value is stated as a relative change ({withheld['verbatim']!r}); "
            f"a produced absolute value is not comparable to it without the baseline"
        )
        return result

    unspecified = withheld.get("unspecified_by_paper") or []
    material, immaterial = _split_unspecified(unspecified, for_arm=for_arm)
    if material:
        result["note"] = (
            "the paper does not specify "
            + "; ".join(str(item) for item in material[:2])
            + " — that changes which quantity is computed, so a produced value cannot be "
            "scored against the reported one"
        )
        result["tolerance_verdict"] = "underspecified"
        result["fault"] = "underspecified"
        return result
    if immaterial:
        result["note"] = (
            "scored despite under-specification the artifact settles: "
            + "; ".join(str(item) for item in immaterial[:2])
        )

    if claim_type == "comparative_ranking" and withheld.get("ranked_items"):
        if not produced_ranking:
            result["note"] = "the run produced no ordering to compare against the paper's"
            result["fault"] = "no_comparable_output"
            return result
        verdict, tau, note = _ranking_verdict(produced_ranking, withheld["ranked_items"])
        result.update(
            tolerance_verdict=verdict, rel_delta=tau, note=note,
            comparator="comparative_ranking",
        )
        if verdict in {"strict", "broad"}:
            result["qualitative_verdict"] = "supports"
        elif tau is not None and tau < 0:
            result["qualitative_verdict"] = "opposite"
        if verdict == "mismatch":
            result["fault"] = "value_mismatch"
        return result

    if claim_type == "descriptive_structural" and withheld.get("expected_group_count"):
        expected = int(withheld["expected_group_count"])
        if produced_group_count is None and produced is None:
            result["note"] = "the run produced no group count to compare"
            result["fault"] = "no_comparable_output"
            return result
        observed = produced_group_count if produced_group_count is not None else int(produced)
        matched = observed == expected
        result.update(
            tolerance_verdict="broad" if matched else "mismatch",
            abs_delta=float(abs(observed - expected)),
            comparator="descriptive_structural",
            note=(
                f"count {observed} against {expected}; strict additionally requires "
                f"ARI >= {TOLERANCES['descriptive_structural']['strict_ari']} against the "
                f"paper's partition, which was not recovered, so strict is unreachable here"
            ),
        )
        result["qualitative_verdict"] = "supports" if matched else "inconclusive"
        if not matched:
            result["fault"] = "value_mismatch"
        return result

    if produced is None:
        result["note"] = "the artifact ran but emitted no comparable quantity"
        result["fault"] = "no_comparable_output"
        return result

    scaled, scale_note = _rescale(
        float(produced), produced_units, float(reported), withheld.get("units")
    )
    verdict, absolute, relative, comparator = _numeric_verdict(
        claim_type or "predictive_performance", scaled, float(reported)
    )
    result.update(
        tolerance_verdict=verdict,
        abs_delta=absolute,
        rel_delta=relative,
        scale_note=scale_note,
        comparator=comparator,
        note=(
            f"produced {scaled:g} against reported {float(reported):g}: "
            f"abs {absolute:.4g}"
            + (f", rel {relative:.2%}" if relative is not None else "")
        ),
    )
    if verdict in {"strict", "broad"}:
        result["qualitative_verdict"] = "supports"
    elif claim_type == "inferential_effect" and (scaled * float(reported)) < 0:
        result["qualitative_verdict"] = "opposite"
        result["fault"] = "value_mismatch"
    else:
        result["fault"] = "value_mismatch"
    return result


def _rrl(ran: bool, verdict: str) -> str:
    if not ran:
        return "L3"
    return {"strict": "L6", "broad": "L5"}.get(verdict, "L4")


async def command_arm_a_compare(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        cursor = await catalog.db.execute(
            "SELECT paper_id, intervention_rung, ran, produced_value, produced_value_units, "
            "produced_ranking_json, produced_group_count, agent_fault_class, "
            "location_verified, wall_clock_seconds FROM arm_a_runs "
            "WHERE experiment_id=? AND status='succeeded' ORDER BY paper_id, repetition_index",
            (experiment["id"],),
        )
        by_paper: dict[int, list[Any]] = {}
        for row in await cursor.fetchall():
            by_paper.setdefault(int(row["paper_id"]), []).append(row)

        now = utc_now()
        written = 0
        for paper_id, runs in sorted(by_paper.items()):
            withheld = await _withheld(
                catalog, experiment["claim_experiment_id"], paper_id
            )
            rung, rung_votes, rung_tied = _modal(
                [r["intervention_rung"] for r in runs if r["intervention_rung"]],
                tuple(reversed(RUNG_ORDER)),
            )
            ran_votes = [bool(r["ran"]) for r in runs if r["ran"] is not None]
            ran = bool(ran_votes) and sum(ran_votes) > len(ran_votes) / 2

            values = [float(r["produced_value"]) for r in runs if r["produced_value"] is not None]
            unique = sorted(set(values))
            value_disagreement = len(unique) > 1
            produced = None
            spread = None
            if values:
                # Never average a disputed value (claim.py's rule): the median of an odd
                # number of runs is a number some run actually produced.
                produced = statistics.median(values) if value_disagreement else values[0]
                spread = (max(values) - min(values)) if value_disagreement else 0.0

            units = next(
                (r["produced_value_units"] for r in runs if r["produced_value_units"]), None
            )
            ranking = next(
                (json.loads(r["produced_ranking_json"]) for r in runs
                 if r["produced_ranking_json"]),
                None,
            )
            group_count = next(
                (r["produced_group_count"] for r in runs
                 if r["produced_group_count"] is not None),
                None,
            )
            comparison = compare_paper(
                claim_type=withheld.get("claim_type"),
                produced=produced,
                produced_units=units,
                produced_ranking=ranking,
                produced_group_count=group_count,
                withheld=withheld,
                ran=ran,
            )

            agent_faults = [r["agent_fault_class"] for r in runs if r["agent_fault_class"]]
            if comparison["fault"] == "value_mismatch":
                agent_faults = ["value_mismatch"]
            fault, fault_votes, fault_tied = _modal(agent_faults, FAULT_PRECEDENCE)
            if comparison["fault"] == "no_comparable_output":
                fault = fault if fault and fault != "none" else "no_comparable_output"

            verified = [r["location_verified"] for r in runs if r["location_verified"] is not None]
            location_verified = int(all(verified)) if verified else None
            needs_adjudication = int(
                value_disagreement
                or rung_tied
                or fault_tied
                or fault == "agent_fault"
                or (location_verified == 0)
                or len(runs) < experiment["repeat_count"]
                or comparison["tolerance_verdict"] in {"not_comparable", "underspecified"}
            )
            wall = [r["wall_clock_seconds"] for r in runs if r["wall_clock_seconds"]]

            await catalog.db.execute(
                "INSERT INTO arm_a_decisions(experiment_id,paper_id,claim_type,ran,"
                "intervention_rung,rung_votes_json,fault_class,fault_votes_json,"
                "produced_value,produced_value_spread,produced_values_json,reported_value,"
                "reported_units,metric_name,scale_note,abs_delta,rel_delta,comparator,"
                "tolerance_verdict,qualitative_verdict,rrl_level,comparison_note,"
                "location_verified,value_disagreement,needs_adjudication,"
                "completed_run_count,wall_clock_seconds,aggregated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(experiment_id,paper_id) DO UPDATE SET "
                "claim_type=excluded.claim_type,ran=excluded.ran,"
                "intervention_rung=excluded.intervention_rung,"
                "rung_votes_json=excluded.rung_votes_json,fault_class=excluded.fault_class,"
                "fault_votes_json=excluded.fault_votes_json,"
                "produced_value=excluded.produced_value,"
                "produced_value_spread=excluded.produced_value_spread,"
                "produced_values_json=excluded.produced_values_json,"
                "reported_value=excluded.reported_value,"
                "reported_units=excluded.reported_units,metric_name=excluded.metric_name,"
                "scale_note=excluded.scale_note,abs_delta=excluded.abs_delta,"
                "rel_delta=excluded.rel_delta,comparator=excluded.comparator,"
                "tolerance_verdict=excluded.tolerance_verdict,"
                "qualitative_verdict=excluded.qualitative_verdict,"
                "rrl_level=excluded.rrl_level,comparison_note=excluded.comparison_note,"
                "location_verified=excluded.location_verified,"
                "value_disagreement=excluded.value_disagreement,"
                "needs_adjudication=excluded.needs_adjudication,"
                "completed_run_count=excluded.completed_run_count,"
                "wall_clock_seconds=excluded.wall_clock_seconds,"
                "aggregated_at=excluded.aggregated_at",
                (
                    experiment["id"], paper_id, withheld.get("claim_type"), int(ran),
                    rung, json.dumps(rung_votes), fault, json.dumps(fault_votes),
                    produced, spread, json.dumps(values), withheld.get("point_estimate"),
                    withheld.get("units"), withheld.get("metric_name"),
                    comparison["scale_note"], comparison["abs_delta"],
                    comparison["rel_delta"], comparison["comparator"],
                    comparison["tolerance_verdict"], comparison["qualitative_verdict"],
                    _rrl(ran, comparison["tolerance_verdict"]), comparison["note"],
                    location_verified, int(value_disagreement), needs_adjudication,
                    len(runs), (max(wall) if wall else None), now,
                ),
            )
            written += 1
            logger.info(
                "arm_a_compared", paper_id=paper_id, rung=rung, ran=ran,
                produced=produced, reported=withheld.get("point_estimate"),
                verdict=comparison["tolerance_verdict"],
                qualitative=comparison["qualitative_verdict"],
                note=comparison["note"],
            )
        await catalog.db.commit()
        logger.info("arm_a_compare_complete", experiment=args.experiment, papers=written)
    return 0


# --- report ------------------------------------------------------------------------


async def command_arm_a_report(args: Any, logger: Any, Catalog: Any) -> int:
    from rich.console import Console
    from rich.table import Table

    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_a_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        cursor = await catalog.db.execute(
            "SELECT d.*, p.collection_year y, p.title FROM arm_a_decisions d "
            "JOIN papers p ON p.id=d.paper_id WHERE d.experiment_id=? ORDER BY d.paper_id",
            (experiment["id"],),
        )
        decisions = list(await cursor.fetchall())
        cursor = await catalog.db.execute(
            "SELECT COUNT(*) n, SUM(status='succeeded') ok, SUM(estimated_cost_usd) cost, "
            "SUM(wall_clock_seconds) wall, SUM(location_verified=0) unverified "
            "FROM arm_a_runs WHERE experiment_id=?",
            (experiment["id"],),
        )
        stats = await cursor.fetchone()

        table = Table(title=f"Stage 4 Arm A — {args.experiment}")
        for column in ("paper", "yr", "type", "rung", "ran", "produced", "reported",
                       "verdict", "conclusion", "RRL", "fault", "loc", "flag"):
            table.add_column(column)
        for row in decisions:
            table.add_row(
                str(row["paper_id"]), str(row["y"]), (row["claim_type"] or "-")[:12],
                row["intervention_rung"] or "-", "yes" if row["ran"] else "no",
                "-" if row["produced_value"] is None else f"{row['produced_value']:g}",
                "-" if row["reported_value"] is None else f"{row['reported_value']:g}",
                row["tolerance_verdict"], row["qualitative_verdict"], row["rrl_level"],
                row["fault_class"] or "-",
                {1: "ok", 0: "FAIL"}.get(row["location_verified"], "-"),
                "ADJUDICATE" if row["needs_adjudication"] else "",
            )
        console = Console()
        console.print(table)

        rungs = Counter(r["intervention_rung"] for r in decisions)
        payload = {
            "experiment": args.experiment,
            "model": experiment["model"],
            "reasoning_effort": experiment["reasoning_effort"],
            "repeat_count": experiment["repeat_count"],
            "time_limit_seconds": experiment["time_limit_seconds"],
            "message_limit": experiment["message_limit"],
            "cost_limit_usd": experiment["cost_limit_usd"],
            "network_isolated_during_analysis": False,
            "papers": len(decisions),
            "runs": stats["n"],
            "runs_succeeded": stats["ok"],
            "runs_with_unverified_location": stats["unverified"],
            "spend_usd": stats["cost"],
            "wall_clock_seconds": stats["wall"],
            "rung_counts": {rung or "unreported": count for rung, count in rungs.items()},
            "ran_count": sum(1 for r in decisions if r["ran"]),
            "verdicts": dict(Counter(r["tolerance_verdict"] for r in decisions)),
            "conclusions": dict(Counter(r["qualitative_verdict"] for r in decisions)),
            "rrl_levels": dict(Counter(r["rrl_level"] for r in decisions)),
            "needs_adjudication": [
                r["paper_id"] for r in decisions if r["needs_adjudication"]
            ],
            "no_rate_reported": (
                "At this n a reproduction rate is not estimable and is "
                "not reported. These are counts of papers, not an estimate of anything."
            ),
            "agent_fault_adjudication": (
                "agent_fault must be confirmed by a human, not by the "
                "agent that reported it. Every such paper is in needs_adjudication."
            ),
        }
        if getattr(args, "output", None):
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        console.print(
            f"[dim]{payload['no_rate_reported']}[/dim]"
        )
        logger.info("arm_a_report", **{
            key: value for key, value in payload.items()
            if key not in {"no_rate_reported", "agent_fault_adjudication"}
        })
    return 0
