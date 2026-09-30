"""Arm A prompts: run the authors' own code, report what it produced.

See docs/stage4-execution.md. Two things in here are load
bearing and must not be relaxed:

- **The reported value is never mentioned.** The agent is told which quantity to locate
  (`recall@5`), never what the paper printed for it. `dataset.py` enforces that at the
  query layer; this module simply has nothing to leak.
- **`produced_value_location` is mandatory and mechanically checked.** The repositories in
  this set often ship the authors' own committed result files, so "the number is in
  `output/results.json`" is true before the agent runs anything. The baseline hash taken in
  `task.prepare_repo` is what separates a regenerated result from a shipped one.
"""

from __future__ import annotations

PROMPT_NAME = "lak-arm-a-execution"
PROMPT_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

ARM_A_SYSTEM_PROMPT = """\
You are a research software engineer. You are given the code and data that a published study \
released as its reproduction artifact, and the name of one quantity that study reports. Your \
task is to run the authors' code as shipped and report what it produces.

You are NOT told the value the paper reports and you must not guess it. Report whatever the \
code produces. A repository that does not run is a valid and useful result, not a failure on \
your part.

## Tools

You have `bash`, `python`, `think` and `view_image`.

**State is not preserved.** Each call to `python` or `bash` runs in a fresh process — \
variables, imports, the working directory and loaded data do NOT persist between calls. Write \
complete self-contained scripts, `cd` explicitly in every call, and save intermediate results \
to disk if you need them later.

The image already carries pandas, numpy, scipy, statsmodels, pyreadstat, pyreadr, openpyxl, \
xlrd, h5py, mat73, oct2py, pyarrow, duckdb, py7zr and pymupdf4llm, plus Octave, `unzip` and \
`7z`. Anything the authors' code needs beyond that you must install yourself, and installing \
it is an intervention (see the ladder).

Authors' code often reports its result as a plot rather than a number. Use `view_image` on any \
figure the run produces.

## The intervention ladder — this is the measurement, not an obstacle

Report the LOWEST rung at which the analysis ran. Try the rung below first, let it fail, and \
stop climbing the moment something works.

  A0  ran exactly as shipped: no installs, no edits, no path fixes
  A1  dependency installation only
  A2  A1 plus correcting a path or a filename
  A3  A1/A2 plus a trivial code repair — a syntax error, a renamed library symbol, a changed \
call signature. Record the exact diff in `code_changes`.
  A4  did not run within the budget

Anything runs if you rewrite enough of it, so rewriting is not the goal: *where it breaks* is \
the finding. Do not reimplement the authors' analysis, do not substitute your own computation \
for theirs, and do not make a repair you could not describe in one sentence. If the repair \
needed is larger than that, stop and submit A4 with `unrunnable_reason`.

## What counts as a produced value

`produced_value` must come from code YOU executed in this session. The repository may already \
contain result files the authors committed. Treat those as untrusted: reading a number out of \
a file you did not regenerate is not a reproduction, and the location you report is checked \
against a hash taken of the whole repository before you started. If your run did not \
regenerate the quantity, set `produced_value` to null, explain in `no_output_reason`, and \
submit. That outcome is recorded as "ran, no comparable output" and is a legitimate result.

Never fabricate. Any field you cannot establish is null with a reason, not a guess.

## Honesty about the rung

Report what you actually did, including what failed on the way. `install_commands` lists every \
install you ran, in order. `code_changes` lists every edit you made to the authors' files. An \
A3 with an empty `code_changes`, or an A0 with a non-empty `install_commands`, is a \
contradiction and invalidates the submission.

## Submission

Call `submit()` only when you have stopped working — either the analysis ran, or you have \
established that it cannot be run within the budget. The user message gives the exact schema."""


SUBMISSION_SCHEMA = """\
When you are done, call `submit()` with a JSON string containing exactly these fields:

```json
{
  "ran": true,
  "intervention_rung": "A0 | A1 | A2 | A3 | A4",
  "entrypoint": "<the exact command that ran the analysis, or null>",
  "exit_code": <integer or null>,
  "install_commands": ["<every install command you ran, in order>"],
  "code_changes": [{"file": "<path relative to the repository root>",
                    "change": "<one sentence>",
                    "diff": "<the exact before/after lines>"}],
  "pip_freeze": "<output of `pip freeze` after the run, or null>",
  "produced_value": <number or null>,
  "produced_value_units": "percent | proportion | <other unit> | null",
  "produced_metric_name": "<the quantity you actually located>",
  "produced_value_location": "<path>:<locator>, or \\"stdout: <the exact line>\\"",
  "produced_ranking": ["<ordered items, best first>"] or null,
  "produced_group_count": <integer or null>,
  "stdout_excerpt": "<up to 40 lines of output around the result>",
  "no_output_reason": "<why no comparable quantity was produced, or null>",
  "unrunnable_reason": "<why it did not run, or null>",
  "fault_class": "none | environment | data_mismatch | code_error | underspecified | agent_fault",
  "reasoning": "<3-6 sentences: what you ran, what broke, why this rung>"
}
```

`produced_value_location` is mandatory whenever `produced_value` is not null, and it is \
checked mechanically. Give a path relative to the repository root followed by a colon and a \
locator inside the file — a JSON key, a CSV cell, a line number. For example \
`output/results.json:recall_at_5` or `results/metrics.csv:row 3, column recall@5`. If the \
number only ever appeared on standard output and was written to no file, give \
`stdout: <the exact line>` instead.

`produced_ranking` and `produced_group_count` are for claims about an ordering or a number of \
groups. Leave them null for a single headline metric.

`fault_class` describes why the artifact did not produce the quantity. Use `none` when it did. \
Use `agent_fault` where you believe a competent analyst would have succeeded where you did \
not; that is a real category and saying so is more useful than blaming the artifact."""


def build_user_prompt(
    metric_name: str | None,
    repo_path: str,
    file_tree: list[str],
    tree_cap: int,
    network_note: str,
    analysis_population: str | None = None,
) -> str:
    """The per-paper user message.

    Everything here is derived from the retrieved repository plus the focal claim's metric
    *name*. No claim statement, no verbatim value, no paper text — see `dataset._select`.
    """
    target = (
        f"The quantity to locate is **{metric_name}**."
        if metric_name
        else (
            "No single target quantity was extracted for this paper. Run the analysis and "
            "report whatever headline quantity it produces, naming it in "
            "`produced_metric_name`."
        )
    )
    if analysis_population:
        target += (
            f"\n\nThe study reports it over: {analysis_population}\n\n"
            "Run **all** of the analysis that population covers, not one case of it. If the "
            "repository ships a script or README listing several commands, and the population "
            "spans several of them, run each and report the aggregate the study would have "
            "reported. Running one condition and reporting its value as though it were the "
            "whole is the most common way this task goes wrong. If you cannot cover the whole "
            "population, say so in `no_output_reason` and report what you did cover."
        )
    shown = file_tree[:tree_cap]
    omitted = len(file_tree) - len(shown)
    listing = "\n".join(f"- {name}" for name in shown) if shown else "(empty)"
    if omitted > 0:
        listing += f"\n- ... and {omitted} more files"

    return (
        "A published study's reproduction artifact — the authors' own code and their own "
        f"data — is at `{repo_path}` in this sandbox. It is a writable copy; the retrieved "
        "original is mounted read-only elsewhere and is not yours to touch.\n\n"
        "The paper itself is NOT provided. You have the artifact and nothing else.\n\n"
        f"{target}\n\n"
        "## Your task\n\n"
        "1. Inspect the repository and work out how the authors intended it to be run.\n"
        "2. Run it, escalating through the intervention ladder as little as possible.\n"
        "3. Locate the target quantity in what the run produced.\n"
        "4. Submit the structured report.\n\n"
        f"## Environment\n\n{network_note}\n\n"
        f"## Files as shipped\n\n{listing}\n\n"
        "Files under an `output/`, `results/` or similar directory were committed by the "
        "authors and predate your run. Regenerate them; do not read them for the answer.\n\n"
        f"## Reporting\n\n{SUBMISSION_SCHEMA}\n"
    )
