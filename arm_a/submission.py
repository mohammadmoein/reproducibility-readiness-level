"""The structured report an Arm A agent submits, and a tolerant parser for it.

Shared by the task (which reads `produced_value_location` while the sandbox is still up)
and by `arm_a_runner` (which writes the fields into `arm_a_runs`). Deliberately lenient:
a malformed submission is recorded as a malformed submission, never discarded, because
"the agent could not produce a clean report" is itself an Arm A observation.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

RUNGS = ("A0", "A1", "A2", "A3", "A4")

#: `value_mismatch` is absent on purpose: the agent does not know the reported value, so it
#: cannot assign that class. `lakdb arm-a compare` assigns it. `artifact_absent` is absent
#: because Stage 4 runs only on papers whose artifact was retrieved.
AGENT_FAULT_CLASSES = (
    "none",
    "environment",
    "data_mismatch",
    "code_error",
    "underspecified",
    "agent_fault",
)


class CodeChange(BaseModel):
    model_config = ConfigDict(extra="allow")
    file: str | None = None
    change: str | None = None
    diff: str | None = None


class ArmASubmission(BaseModel):
    model_config = ConfigDict(extra="allow")

    ran: bool | None = None
    intervention_rung: str | None = None
    entrypoint: str | None = None
    exit_code: int | None = None
    install_commands: list[str] = []
    code_changes: list[CodeChange] = []
    pip_freeze: str | None = None
    produced_value: float | None = None
    produced_value_units: str | None = None
    produced_metric_name: str | None = None
    produced_value_location: str | None = None
    produced_ranking: list[str] | None = None
    produced_group_count: int | None = None
    stdout_excerpt: str | None = None
    no_output_reason: str | None = None
    unrunnable_reason: str | None = None
    fault_class: str | None = None
    reasoning: str | None = None

    def contradictions(self) -> list[str]:
        """Internal inconsistencies the prompt told the agent would invalidate a submission.

        Recorded rather than acted on: a contradictory report is evidence about the harness,
        and silently repairing it would hide that.
        """
        problems: list[str] = []
        rung = (self.intervention_rung or "").upper()
        if rung and rung not in RUNGS:
            problems.append(f"unknown rung {self.intervention_rung!r}")
        if rung == "A0" and self.install_commands:
            problems.append("rung A0 with install_commands")
        if rung == "A3" and not self.code_changes:
            problems.append("rung A3 with no code_changes")
        if rung == "A4" and self.ran:
            problems.append("rung A4 with ran=true")
        if rung and rung != "A4" and self.ran is False:
            problems.append(f"rung {rung} with ran=false")
        if self.produced_value is not None and not self.produced_value_location:
            problems.append("produced_value with no produced_value_location")
        if self.code_changes and rung in {"A0", "A1", "A2"}:
            problems.append(f"code_changes recorded at rung {rung}")
        return problems


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """First balanced `{...}` in the completion, fences and prose tolerated."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(candidate[start : index + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_submission(
    text: str | None,
) -> tuple[ArmASubmission | None, dict[str, Any] | None, str | None]:
    """Returns (submission, raw_dict, error). Any of the three may be None."""
    raw = extract_json_object(text)
    if raw is None:
        return None, None, "no JSON object in the submission"
    try:
        return ArmASubmission.model_validate(raw), raw, None
    except Exception as exc:  # noqa: BLE001 — record the shape we got, do not repair it
        return None, raw, f"{type(exc).__name__}: {exc}"


def parse_location(location: str | None) -> tuple[str, str | None, str | None]:
    """Split `produced_value_location` into (kind, path, locator).

    kind is 'file', 'stdout' or 'unparsed'. A file location is `<path>:<locator>`; the
    path is taken as relative to the repository root even when the agent wrote an absolute
    one, because that is the only tree the checker can address.
    """
    if not location:
        return "unparsed", None, None
    text = location.strip()
    lowered = text.lower()
    if lowered.startswith("stdout") or lowered.startswith("stderr"):
        _, _, rest = text.partition(":")
        return "stdout", None, rest.strip() or text
    path, separator, locator = text.partition(":")
    path = path.strip()
    if not separator or not path:
        return "unparsed", None, text
    path = path.replace("\\", "/").lstrip("./")
    for prefix in ("workspace/repo/", "/workspace/repo/", "repo/"):
        if path.startswith(prefix.lstrip("/")):
            path = path[len(prefix.lstrip("/")) :]
    if path.startswith("/"):
        return "unparsed", None, text
    return "file", path, locator.strip() or None
