"""Structured, auditable submissions for independent Arm B reanalysis."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class ArmBSubmission(BaseModel):
    model_config = ConfigDict(extra="allow")

    computed: bool | None = None
    produced_value: float | None = None
    produced_value_units: str | None = None
    produced_metric_name: str | None = None
    produced_value_location: str | None = None
    produced_ranking: list[str] | None = None
    produced_group_count: int | None = None
    analysis_script: str | None = None
    operationalisation: str | None = None
    assumptions_made: list[str] = []
    could_not_compute_reason: str | None = None
    fault_class: str | None = None
    reasoning: str | None = None

    def contradictions(self) -> list[str]:
        problems: list[str] = []
        if self.computed and self.produced_value is None and not self.produced_ranking and self.produced_group_count is None:
            problems.append("computed=true with no comparable result")
        if self.computed and not self.produced_value_location:
            problems.append("computed=true with no produced_value_location")
        if self.computed and not self.analysis_script:
            problems.append("computed=true with no analysis_script")
        if self.computed and not self.operationalisation:
            problems.append("computed=true with no operationalisation")
        if self.computed and self.could_not_compute_reason:
            problems.append("computed=true with could_not_compute_reason")
        if self.computed is False and not self.could_not_compute_reason:
            problems.append("computed=false with no could_not_compute_reason")
        return problems


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) > 1:
            candidate = parts[1]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(candidate[start:index + 1])
                except ValueError:
                    return None
                return result if isinstance(result, dict) else None
    return None


def parse_submission(text: str | None) -> tuple[ArmBSubmission | None, dict[str, Any] | None, str | None]:
    raw = extract_json_object(text)
    if raw is None:
        return None, None, "no JSON object in the submission"
    try:
        return ArmBSubmission.model_validate(raw), raw, None
    except Exception as exc:  # record malformed output instead of silently dropping it
        return None, raw, f"{type(exc).__name__}: {exc}"


def parse_location(location: str | None) -> tuple[str, str | None, str | None]:
    """Return (kind, path, locator) for an agent-created output location."""
    if not location:
        return "unparsed", None, None
    text = location.strip()
    if text.lower().startswith(("stdout", "stderr")):
        _, _, rest = text.partition(":")
        return "stdout", None, rest.strip() or text
    path, separator, locator = text.partition(":")
    path = path.strip().replace("\\", "/")
    if not separator or not path:
        return "unparsed", None, text
    normalised = path.lstrip("/")
    for prefix in ("workspace/work/", "work/"):
        if normalised.startswith(prefix):
            path = normalised[len(prefix):]
            break
    if path.startswith("/") or ".." in path.split("/"):
        return "unparsed", None, text
    return "file", path.lstrip("./"), locator.strip() or None
