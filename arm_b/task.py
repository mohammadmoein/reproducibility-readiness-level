"""Inspect task for Arm B independent reanalysis."""
from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.agent._types import AgentPrompt
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from arm_a.task import _analysis_tools, _register_model_costs
from arm_b.dataset import DEFAULT_IMAGE_TAG, SANDBOX_WORK, load_arm_b_dataset
from arm_b.prompt import ARM_B_SYSTEM_PROMPT

EVIDENCE_KEY = "arm_b_evidence"
SETUP_KEY = "arm_b_setup"
EVIDENCE_BYTE_CAP = 200_000


@solver
def prepare_workspace() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await sandbox().exec(["bash", "-c", f"mkdir -p {SANDBOX_WORK}"])
        setup = {"success": result.success, "returncode": result.returncode, "author_code_mounted": False}
        state.metadata[SETUP_KEY] = setup
        state.store.set(SETUP_KEY, setup)
        if not result.success:
            raise RuntimeError("Arm B could not create its analysis workspace")
        return state
    return solve


async def _exec(command: str) -> tuple[bool, str]:
    try:
        result = await sandbox().exec(["bash", "-c", command], timeout=120)
        return result.success, result.stdout or result.stderr or ""
    except Exception as exc:  # cleanup must preserve the run even if evidence capture fails
        return False, f"{type(exc).__name__}: {exc}"


async def capture_evidence(state: TaskState) -> None:
    from arm_b.submission import parse_location, parse_submission

    evidence: dict[str, Any] = {"captured": False}
    try:
        submission, _, parse_error = parse_submission(state.output.completion if state.output else None)
        evidence["parse_error"] = parse_error
        location = submission.produced_value_location if submission else None
        kind, path, locator = parse_location(location)
        evidence.update({"location": location, "location_kind": kind, "locator": locator})
        if kind == "file" and path:
            quoted = json.dumps(f"{SANDBOX_WORK}/{path}")
            exists, _ = await _exec(f"test -f {quoted}")
            evidence["file_exists"] = exists
            if exists:
                _, contents = await _exec(f"head -c {EVIDENCE_BYTE_CAP} {quoted}")
                evidence["evidence_text"] = contents
        evidence["captured"] = True
    except Exception as exc:  # noqa: BLE001
        evidence["capture_error"] = f"{type(exc).__name__}: {exc}"
    state.metadata[EVIDENCE_KEY] = evidence
    try:
        state.store.set(EVIDENCE_KEY, evidence)
    except Exception:  # noqa: BLE001
        pass


@task
def arm_b(
    paper_id: int,
    database: str = "data/lak.sqlite",
    claim_experiment: str | None = None,
    reasoning_effort: str = "medium",
    message_limit: int | None = 100,
    cost_limit: float | None = 2.0,
    time_limit: int = 1800,
    command_timeout: int = 300,
    sandbox_dir: str = "artifacts/arm_b/sandbox",
    image_tag: str = DEFAULT_IMAGE_TAG,
    network: bool = True,
) -> Task:
    if cost_limit is not None:
        _register_model_costs()
    return Task(
        dataset=load_arm_b_dataset(database=database, paper_id=paper_id, claim_experiment=claim_experiment, sandbox_dir=sandbox_dir, image_tag=image_tag, network=network),
        setup=prepare_workspace(),
        solver=[react(prompt=AgentPrompt(instructions=ARM_B_SYSTEM_PROMPT, assistant_prompt=None, handoff_prompt=None), tools=_analysis_tools(command_timeout))],
        cleanup=capture_evidence,
        sandbox="docker",
        message_limit=int(message_limit) if message_limit is not None else None,
        time_limit=int(time_limit) if time_limit else None,
        cost_limit=float(cost_limit) if cost_limit is not None else None,
        config=GenerateConfig(reasoning_effort=reasoning_effort),
    )
