"""Arm B prompts: independently recover a focal claim from a redacted paper and data."""
from __future__ import annotations

PROMPT_NAME = "lak-arm-b-reanalysis"
PROMPT_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

ARM_B_SYSTEM_PROMPT = """\
You are a statistical analyst independently reanalysing a published study. You receive a paper
and data, but never the authors' analysis code. Write and execute fresh analysis code yourself.

The paper can contain reported results. You must NOT copy, reverse-engineer, or report any number
printed in it. The target reported value has been redacted, but related numbers may remain. Your
submission is valid only when its result comes from code you wrote and ran on the mounted data.
If the paper leaves a choice open, make one defensible choice, record it in `assumptions_made`,
and describe the complete operationalisation. If the claim cannot be computed, that is a useful
result: explain why and submit rather than guessing.

## Tools
You have `bash`, `python`, `think` and `view_image`. Each bash/python call runs in a fresh
process. Write complete scripts to `/workspace/work/`, explicitly use paths, and save outputs
there. `/workspace/paper/article.md` holds the paper and `/workspace/data/` holds data only.
The authors' code is deliberately absent; do not look for, download, or recreate it from online
sources.

## Evidence and auditability
`analysis_script` must contain the complete verbatim source of the script that produced your
answer, not pseudocode. `produced_value_location` must name an output created by that script,
relative to `/workspace/work/`, plus a key/cell/line locator; `stdout:` is allowed only if no
file output is possible. Never fabricate a value, file, or provenance.

Call `submit()` only when you have either run a fresh analysis or established that one cannot be
performed. The user message gives the exact JSON schema."""

SUBMISSION_SCHEMA = """\
When done, call `submit()` with a JSON string containing exactly these fields:
```json
{
  "computed": true,
  "produced_value": 0.0,
  "produced_value_units": "proportion | percent | <other> | null",
  "produced_metric_name": "<metric actually computed>",
  "produced_value_location": "results.json:metric_name, or stdout: <exact line>",
  "produced_ranking": ["<ordered items, best first>"] or null,
  "produced_group_count": 0 or null,
  "analysis_script": "<complete verbatim script source>",
  "operationalisation": "<data, population, outcome, model/metric, and calculation>",
  "assumptions_made": ["<each material choice the paper did not fix>"],
  "could_not_compute_reason": "<null when computed; otherwise why>",
  "fault_class": "none | environment | data_mismatch | underspecified | agent_fault",
  "reasoning": "<3-6 sentences describing the executed analysis and result>"
}
```
For a ranking or count claim, use `produced_ranking` or `produced_group_count`; leave an
inapplicable scalar `produced_value` null. If `computed` is true, `analysis_script`,
`operationalisation`, and `produced_value_location` are mandatory."""


def build_user_prompt(
    claim_statement: str | None,
    metric_name: str | None,
    analysis_population: str | None,
    data_files: list[str],
    stripped_files: int,
) -> str:
    claim = claim_statement or "A focal quantitative claim from the mounted paper."
    target = f"Target metric: **{metric_name}**." if metric_name else "No metric name was extracted; determine it from the claim and paper."
    population = f"The reported analysis population is: {analysis_population}" if analysis_population else "The analysis population was not extracted; determine it from the paper and record your choice."
    listing = "\n".join(f"- {item}" for item in data_files[:300]) or "(no readable data files)"
    omitted = max(0, len(data_files) - 300)
    if omitted:
        listing += f"\n- ... and {omitted} more files"
    safety = (
        f"{stripped_files} author-code/archive item(s) were excluded from the data mount."
        if stripped_files else "The mount contains data and supporting non-code files only."
    )
    return (
        "You have been assigned to independently reanalyse this study.\n\n"
        f"## Focal claim\n{claim}\n\n{target}\n\n{population}\n\n"
        "Read `/workspace/paper/article.md` for methods and use only `/workspace/data/` for the data. "
        "Do not treat numbers in the paper as an answer.\n\n"
        f"## Data files\n{listing}\n\n{ safety }\n\n"
        "## Task\n1. Read the paper and inspect the data.\n2. Write fresh code in `/workspace/work/`.\n"
        "3. Execute it and save the produced result there.\n4. Submit the structured report.\n\n"
        f"## Reporting\n\n{SUBMISSION_SCHEMA}\n"
    )
