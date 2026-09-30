"""Arm A: run the authors' own code in a sandbox and report what it produced.

A sibling of `pipeline/agentic_analysis/task.py` in https://github.com/tobihol/agentic-reproducibility, which stays pristine —
it is Arm B's harness and method fidelity depends on it being unmodified. The container,
the tool set and the stateless-shell convention are theirs; the dataset, the prompt and the
submission schema are ours.

Three things differ from their task by design:

- **No scorer.** Inspect's scorer runs inside the eval and would need the withheld reported
  value to score against. Comparison happens afterwards in `lakdb arm-a compare`, in a
  process the agent never touches.
- **A setup solver**, `prepare_repo`, copies the read-only retrieval into a writable tree and
  hashes it first, so `capture_evidence` can later tell a regenerated result file from one
  the authors committed.
- **A cleanup hook**, `capture_evidence`, reads the location the agent named while the
  sandbox is still alive. It runs in a `finally` inside the sandbox context, so it still
  fires when the sample hit its message, cost or time limit.

Run it through the CLI (`lakdb arm-a run`), which records the run in SQLite. The bare
Inspect form works too, for debugging:

    uv run inspect eval arm_a/task.py@arm_a -T paper_id=34 --model openai/gpt-5.6-terra
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.agent._types import AgentPrompt
from inspect_ai.model import GenerateConfig, ModelCost, ModelInfo, set_model_info
from inspect_ai.model._model import active_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import (
    ContentImage,
    ContentText,
    Tool,
    ToolError,
    bash,
    python,
    think,
    tool,
)
from inspect_ai.util import sandbox

from arm_a.dataset import (
    BASELINE_MANIFEST,
    DEFAULT_IMAGE_TAG,
    SANDBOX_MOUNT,
    SANDBOX_REPO,
    SANDBOX_STATE,
    load_arm_a_dataset,
)
from arm_a.prompt import ARM_A_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

EVIDENCE_KEY = "arm_a_evidence"
SETUP_KEY = "arm_a_setup"

#: Bytes of the located file kept as evidence. Enough for a results.json or a metrics CSV;
#: a location pointing at something larger is itself worth seeing in the record.
EVIDENCE_BYTE_CAP = 200_000

_costs_registered = False


def _register_model_costs() -> None:
    """Register `pricing.py` rates with Inspect so `cost_limit` can be enforced.

    Inspect hard-requires cost data for the active model whenever a cost limit is set.
    Holtdirk fetch LiteLLM's live table; we do not, because this project already keeps its
    own captured price table (`pricing.py`) and all spend must trace to it.
    """
    global _costs_registered
    if _costs_registered:
        return
    from pricing import CACHE_PRICES, PRICES

    registered = 0
    for model, tiers in PRICES.items():
        short_in, short_out, _, _ = tiers["standard"]
        short_cached, short_write, _, _ = CACHE_PRICES[model]["standard"]
        info = ModelInfo(
            cost=ModelCost(
                input=short_in,
                output=short_out,
                input_cache_read=short_cached,
                input_cache_write=short_write,
            )
        )
        for alias in (model, f"openai/{model}"):
            set_model_info(alias, info)
        registered += 1
    _costs_registered = True
    logger.info("Registered pricing.py rates for %d models", registered)


# ---------------------------------------------------------------------------
# Image viewer. Copied from Holtdirk's task.py — it is not part of Inspect's standard tool
# set. Safe here in a way it is not in Arm B: the paper is never mounted in Arm A, so the
# only figures in the sandbox are ones the authors' code just wrote. Do not mount the paper.
# ---------------------------------------------------------------------------
_IMAGE_MIME = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_TEXT_ONLY_MODEL_HINTS = ("glm",)


def _model_supports_images(model: Any) -> bool:
    name = f"{model}".lower() if model is not None else ""
    return not any(hint in name for hint in _TEXT_ONLY_MODEL_HINTS)


@tool
def view_image() -> Tool:
    async def execute(path: str) -> list[ContentText | ContentImage]:
        """View an image file so you can actually see it.

        The `python`/`bash` tools only return text, so use this tool whenever the analysis
        writes its result as a figure rather than a number.

        Args:
            path: Path to an image file in the sandbox (.jpeg/.jpg/.png/.gif/.webp),
                e.g. "/workspace/repo/output/figure3.png".

        Returns:
            The image rendered in your context.
        """
        if not _model_supports_images(active_model()):
            raise ToolError(
                "This model cannot view images — read the numbers out of the files the "
                "run wrote instead."
            )
        mime = _IMAGE_MIME.get(os.path.splitext(path)[1].lower())
        if mime is None:
            raise ToolError(
                f"{path!r} is not a supported image file "
                f"(expected one of {sorted(_IMAGE_MIME)})."
            )
        try:
            data = await sandbox().read_file(path, text=False)
        except FileNotFoundError:
            raise ToolError(f"No such image file: {path!r}")
        b64 = base64.b64encode(data).decode()
        return [
            ContentText(text=f"Image {os.path.basename(path)}:"),
            ContentImage(image=f"data:{mime};base64,{b64}"),
        ]

    return execute


def _analysis_tools(command_timeout: int) -> list[Tool]:
    """Holtdirk's four tools. `bash` gets the same budget as `python` here because Arm A's
    long operation is the authors' training run, which is a bash command, not a snippet."""
    return [
        bash(timeout=command_timeout),
        python(timeout=command_timeout),
        think(),
        view_image(),
    ]


# ---------------------------------------------------------------------------
# Setup: writable copy + baseline hashes
# ---------------------------------------------------------------------------
_PREPARE_SCRIPT = f"""\
set -eu
mkdir -p {SANDBOX_REPO} {SANDBOX_STATE}
cp -a {SANDBOX_MOUNT}/. {SANDBOX_REPO}/ 2>/dev/null || cp -R {SANDBOX_MOUNT}/. {SANDBOX_REPO}/
rm -rf {SANDBOX_REPO}/.git
chmod -R u+rwX {SANDBOX_REPO}
find {SANDBOX_REPO} -name '*.sh' -exec chmod +x {{}} +
cd {SANDBOX_REPO}
find . -type f -print0 | sort -z | xargs -0 -r sha256sum > {BASELINE_MANIFEST}
echo "files=$(wc -l < {BASELINE_MANIFEST})"
echo "bytes=$(du -sb {SANDBOX_REPO} | cut -f1)"
getent hosts pypi.org >/dev/null 2>&1 && echo "network=on" || echo "network=off"
"""


@solver
def prepare_repo(timeout: int = 1800) -> Solver:
    """Copy the read-only retrieval into `/workspace/repo` and hash it before the agent runs.

    The hash is the guard that makes `produced_value_location` mean something. Several of
    these repositories ship the authors' own committed `output/` directory, so the target
    number is already on disk at t=0; without a baseline, "the value is in
    output/results.json" is unfalsifiable.

    `.git` is removed: the retrieval was a shallow clone, and leaving it in would let the
    agent check out something other than what was shipped. Recorded in the run metadata
    rather than done quietly.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await sandbox().exec(["bash", "-c", _PREPARE_SCRIPT], timeout=timeout)
        fields: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            key, _, value = line.partition("=")
            if value:
                fields[key.strip()] = value.strip()
        setup = {
            "success": result.success,
            "returncode": result.returncode,
            "baseline_file_count": int(fields.get("files") or 0),
            "copied_bytes": int(fields.get("bytes") or 0),
            "network_observed": fields.get("network"),
            "git_metadata_removed": True,
            "stderr": (result.stderr or "")[:2000],
        }
        state.metadata[SETUP_KEY] = setup
        state.store.set(SETUP_KEY, setup)
        if not result.success or not setup["baseline_file_count"]:
            raise RuntimeError(
                f"Arm A setup failed for {state.sample_id}: rc={result.returncode} "
                f"stderr={setup['stderr']!r}"
            )
        return state

    return solve


# ---------------------------------------------------------------------------
# Cleanup: open the location the agent named, while the container is still up
# ---------------------------------------------------------------------------
async def _exec(cmd: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        result = await sandbox().exec(["bash", "-c", cmd], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — evidence capture must never fail a sample
        return False, f"{type(exc).__name__}: {exc}"
    return result.success, (result.stdout or result.stderr or "")


async def capture_evidence(state: TaskState) -> None:
    """Post-check for `produced_value_location`.

    Reads the named file out of the sandbox and compares its hash with the baseline taken
    in `prepare_repo`. Three outcomes, all recorded and none of them fatal: the file was
    regenerated by this run (`modified`), it is byte-identical to what shipped
    (`unmodified` — the agent may have read the authors' answer), or it is new since the
    baseline (`created`). `lakdb arm-a verify` turns that into a pass/fail.
    """
    from arm_a.submission import parse_location, parse_submission

    evidence: dict[str, Any] = {"captured": False}
    try:
        submission, _, parse_error = parse_submission(
            state.output.completion if state.output else None
        )
        evidence["parse_error"] = parse_error
        location = submission.produced_value_location if submission else None
        kind, path, locator = parse_location(location)
        evidence.update({"location": location, "location_kind": kind, "locator": locator})

        if kind == "file" and path:
            quoted = json.dumps(f"{SANDBOX_REPO}/{path}")
            exists, _ = await _exec(f"test -f {quoted}")
            evidence["file_exists"] = exists
            if exists:
                _, head = await _exec(f"head -c {EVIDENCE_BYTE_CAP} {quoted}")
                evidence["evidence_text"] = head
                _, digest = await _exec(f"sha256sum {quoted} | cut -d' ' -f1")
                digest = digest.strip()
                evidence["sha256"] = digest
                _, baseline = await _exec(
                    f"grep -F {json.dumps('  ./' + path)} {BASELINE_MANIFEST} "
                    f"| head -1 | cut -d' ' -f1"
                )
                baseline = baseline.strip()
                evidence["baseline_sha256"] = baseline or None
                if not baseline:
                    evidence["file_state"] = "created"
                elif baseline == digest:
                    evidence["file_state"] = "unmodified"
                else:
                    evidence["file_state"] = "modified"

        _, freeze = await _exec("pip freeze")
        evidence["pip_freeze"] = freeze[:20_000]
        _, changed = await _exec(
            f"cd {SANDBOX_REPO} && find . -type f -newer {BASELINE_MANIFEST} "
            f"| head -200"
        )
        evidence["files_written_by_run"] = [
            line for line in changed.splitlines() if line.strip()
        ]
        network_ok, _ = await _exec("getent hosts pypi.org >/dev/null 2>&1")
        evidence["network_observed"] = "on" if network_ok else "off"
        evidence["captured"] = True
    except Exception as exc:  # noqa: BLE001 — never turn a captured run into an error
        evidence["capture_error"] = f"{type(exc).__name__}: {exc}"

    state.metadata[EVIDENCE_KEY] = evidence
    try:
        state.store.set(EVIDENCE_KEY, evidence)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@task
def arm_a(
    paper_id: int,
    database: str = "data/lak.sqlite",
    claim_experiment: str | None = None,
    reasoning_effort: str = "medium",
    message_limit: int | None = 100,
    cost_limit: float | None = 2.0,
    time_limit: int = 1800,
    command_timeout: int = 300,
    sandbox_dir: str = "artifacts/arm_a/sandbox",
    image_tag: str = DEFAULT_IMAGE_TAG,
    cpus: float = 2.0,
    mem_limit: str = "8gb",
    network: bool = True,
    setup_timeout: int = 1800,
) -> Task:
    """Execute one paper's shipped artifact.

    Args:
        paper_id: catalog id of a paper with a focal claim and retrieved code.
        database: SQLite catalog, relative to the repository root.
        claim_experiment: Stage 3 experiment whose focal metric *name* is given to the
            agent. Latest wins when omitted. The value itself is never read.
        reasoning_effort: passed through to the model; `medium` everywhere by frozen decision.
        message_limit / cost_limit / time_limit: per-sample budgets. `time_limit` is the
            30-minute wall clock per sample, recorded and reported with the
            result so a reader can discount it.
        command_timeout: per `bash`/`python` call, in seconds.
        network: outbound access in the sandbox. On by default because dependency
            installation is rung A1 and must be possible; the run records that isolation
            was not enforced rather than claiming it was.
    """
    if cost_limit is not None:
        _register_model_costs()
    return Task(
        dataset=load_arm_a_dataset(
            database=database,
            paper_id=paper_id,
            claim_experiment=claim_experiment,
            sandbox_dir=sandbox_dir,
            image_tag=image_tag,
            cpus=cpus,
            mem_limit=mem_limit,
            network=network,
        ),
        setup=prepare_repo(timeout=setup_timeout),
        solver=[
            react(
                prompt=AgentPrompt(
                    instructions=ARM_A_SYSTEM_PROMPT,
                    assistant_prompt=None,
                    handoff_prompt=None,
                ),
                tools=_analysis_tools(command_timeout),
            ),
        ],
        cleanup=capture_evidence,
        sandbox="docker",
        message_limit=int(message_limit) if message_limit is not None else None,
        time_limit=int(time_limit) if time_limit else None,
        cost_limit=float(cost_limit) if cost_limit is not None else None,
        config=GenerateConfig(reasoning_effort=reasoning_effort),
    )
