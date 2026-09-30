# Stage 4: execution in two arms

**Question.** Can the focal claim from Stage 3 be recomputed? Two arms ask this in different
ways, and both run the same model in the same sandbox:

| | Arm A: authors' code | Arm B: independent reanalysis |
|---|---|---|
| Agent gets | the retrieved repository (code + data), the metric **name** | the paper with the target value redacted, the data with all author code removed, the claim statement |
| Agent does not get | the paper, the reported value | the authors' code, the reported value |
| Question | does the shipped artifact run, and what does it produce? | can the paper plus data recover the claim without the code? |
| Rung / outcome | lowest intervention rung that ran: A0 as shipped, A1 installs, A2 path fixes, A3 trivial code repair, A4 did not run | computed or not, with the full analysis script and every assumption |

**Code.** `arm_a/`, `arm_a_runner.py`, `arm_b/`, `arm_b_runner.py` · **Commands.**
`lakdb arm-a …`, `lakdb arm-b …`

## Harness

- **Sandbox:** the Docker image from the agentic reanalysis study of Holtdirk et al.
  (`arm_a/Dockerfile`, copied verbatim: Python 3.12 with pandas, numpy, scipy, statsmodels,
  pyreadstat, pyreadr, Octave and others). Arm B uses the **same image** so that the contrast
  between arms is not an environment effect. The image has no R runtime, so R artifacts cannot
  run in either arm.
- **Agent:** Inspect AI `react()` with four tools, `bash`, `python`, `think` and `view_image`.
  Each call runs in a fresh process.
- **Model:** `gpt-5.6-terra`, reasoning effort `medium`, 3 repetitions per paper, majority
  vote. Budgets per run are 100 messages, US$2, 30 minutes wall clock and 300 s per command.
  Network access is on so that dependencies can be installed, and this is recorded on the run.
- **No scorer inside the sandbox.** The agent submits structured JSON. Everything after that
  happens outside the sandbox, in a process the agent never touches:
  - `collect` parses the Inspect logs.
  - `verify` opens the file and locator the agent named and checks that the number is there,
    and that the file is not byte-identical to one the authors shipped (a value that was read,
    not produced, fails).
  - `compare` joins the withheld reported value and applies the tolerances of
    [protocol §6](protocol.md#6-claim-taxonomy-and-tolerances) to assign an RRL level.
- **Leak check:** before a run, the prompt is searched for the reported value's digits, and the
  run stops if they are found.

## Reference scope

The executable set: papers with a Stage 3 focal claim and a retrieved repository holding both
code and data. Arm A ran all nine. Arm B ran seven. Paper 431 has no recomputable value, and
paper 468 ships an R artifact that the Python-only sandbox cannot run.

| paper_id | DOI |
|---|---|
| 34 | 10.1145/3448139.3448173 |
| 69 | 10.1145/3448139.3448208 |
| 244 | 10.1145/3636555.3636898 |
| 321 | 10.1145/3706468.3706500 |
| 408 | 10.1145/3785022.3785031 |
| 431 | 10.1145/3785022.3785070 (Arm A only) |
| 468 | 10.1145/3785022.3785034 (Arm A only) |
| 482 | 10.1145/3785022.3785073 |
| 491 | 10.1145/3785022.3785086 |

Paper ids are catalog ids. They match the study only if the corpus is ingested in the order
of `docs/corpus.csv`, which `scripts/organize_acm_download.py` guarantees. Arm B's paper list
(`ARM_B_PAPERS` in `arm_b_runner.py`) is fixed in code by paper id. Paper 69 links two
repositories, and the choice between them was made by hand before the runs. It is fixed in
`SNAPSHOT_OVERRIDES` in `arm_a/dataset.py` by repository URL.

Experiments: `s4-arma-all9-r3` (Arm A, prompt `lak-arm-a-execution` v1.0.0, SHA-256
`6e53ae1fc4513c252b61ed8396d7c4b0d499cd0b4b7319454eff778bb7c52780`) and `s4-armb-7-r3`
(Arm B, prompt `lak-arm-b-reanalysis` v1.0.0, SHA-256
`e091ff41f079a086392c2ea4d87bcb033c0ca5860d6af112208341494b7fd294`).

## Run it

Docker must be running.

```bash
docker compose -f arm_a/compose.yaml build        # builds image repro-sandbox:1 once

uv run lakdb arm-a create s4a-demo --claim-experiment s3-demo
uv run lakdb arm-a run --experiment s4a-demo --paper-id 69 --repeats 1
uv run lakdb arm-a collect --experiment s4a-demo
uv run lakdb arm-a verify --experiment s4a-demo
uv run lakdb arm-a compare --experiment s4a-demo
uv run lakdb arm-a report --experiment s4a-demo

uv run lakdb arm-b create s4b-demo --claim-experiment s3-demo
uv run lakdb arm-b run --experiment s4b-demo --paper-id 69 --repeats 1
uv run lakdb arm-b collect --experiment s4b-demo
uv run lakdb arm-b verify --experiment s4b-demo
uv run lakdb arm-b compare --experiment s4b-demo
uv run lakdb arm-b report --experiment s4b-demo
uv run lakdb arm-b contrast --arm-a s4a-demo --arm-b s4b-demo
```

Inspect logs go to `artifacts/arm_a/logs/` and `artifacts/arm_b/logs/`. Open them with
`uv run inspect view --log-dir artifacts/arm_a/logs`.

## Arm A: system prompt (verbatim)

```text
You are a research software engineer. You are given the code and data that a published study released as its reproduction artifact, and the name of one quantity that study reports. Your task is to run the authors' code as shipped and report what it produces.

You are NOT told the value the paper reports and you must not guess it. Report whatever the code produces. A repository that does not run is a valid and useful result, not a failure on your part.

## Tools

You have `bash`, `python`, `think` and `view_image`.

**State is not preserved.** Each call to `python` or `bash` runs in a fresh process — variables, imports, the working directory and loaded data do NOT persist between calls. Write complete self-contained scripts, `cd` explicitly in every call, and save intermediate results to disk if you need them later.

The image already carries pandas, numpy, scipy, statsmodels, pyreadstat, pyreadr, openpyxl, xlrd, h5py, mat73, oct2py, pyarrow, duckdb, py7zr and pymupdf4llm, plus Octave, `unzip` and `7z`. Anything the authors' code needs beyond that you must install yourself, and installing it is an intervention (see the ladder).

Authors' code often reports its result as a plot rather than a number. Use `view_image` on any figure the run produces.

## The intervention ladder — this is the measurement, not an obstacle

Report the LOWEST rung at which the analysis ran. Try the rung below first, let it fail, and stop climbing the moment something works.

  A0  ran exactly as shipped: no installs, no edits, no path fixes
  A1  dependency installation only
  A2  A1 plus correcting a path or a filename
  A3  A1/A2 plus a trivial code repair — a syntax error, a renamed library symbol, a changed call signature. Record the exact diff in `code_changes`.
  A4  did not run within the budget

Anything runs if you rewrite enough of it, so rewriting is not the goal: *where it breaks* is the finding. Do not reimplement the authors' analysis, do not substitute your own computation for theirs, and do not make a repair you could not describe in one sentence. If the repair needed is larger than that, stop and submit A4 with `unrunnable_reason`.

## What counts as a produced value

`produced_value` must come from code YOU executed in this session. The repository may already contain result files the authors committed. Treat those as untrusted: reading a number out of a file you did not regenerate is not a reproduction, and the location you report is checked against a hash taken of the whole repository before you started. If your run did not regenerate the quantity, set `produced_value` to null, explain in `no_output_reason`, and submit. That outcome is recorded as "ran, no comparable output" and is a legitimate result.

Never fabricate. Any field you cannot establish is null with a reason, not a guess.

## Honesty about the rung

Report what you actually did, including what failed on the way. `install_commands` lists every install you ran, in order. `code_changes` lists every edit you made to the authors' files. An A3 with an empty `code_changes`, or an A0 with a non-empty `install_commands`, is a contradiction and invalidates the submission.

## Submission

Call `submit()` only when you have stopped working — either the analysis ran, or you have established that it cannot be run within the budget. The user message gives the exact schema.
```

### Arm A: user message builder

The submission schema below is appended to every user message.

```python
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
```

````text
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
  "produced_value_location": "<path>:<locator>, or \"stdout: <the exact line>\"",
  "produced_ranking": ["<ordered items, best first>"] or null,
  "produced_group_count": <integer or null>,
  "stdout_excerpt": "<up to 40 lines of output around the result>",
  "no_output_reason": "<why no comparable quantity was produced, or null>",
  "unrunnable_reason": "<why it did not run, or null>",
  "fault_class": "none | environment | data_mismatch | code_error | underspecified | agent_fault",
  "reasoning": "<3-6 sentences: what you ran, what broke, why this rung>"
}
```

`produced_value_location` is mandatory whenever `produced_value` is not null, and it is checked mechanically. Give a path relative to the repository root followed by a colon and a locator inside the file — a JSON key, a CSV cell, a line number. For example `output/results.json:recall_at_5` or `results/metrics.csv:row 3, column recall@5`. If the number only ever appeared on standard output and was written to no file, give `stdout: <the exact line>` instead.

`produced_ranking` and `produced_group_count` are for claims about an ordering or a number of groups. Leave them null for a single headline metric.

`fault_class` describes why the artifact did not produce the quantity. Use `none` when it did. Use `agent_fault` where you believe a competent analyst would have succeeded where you did not; that is a real category and saying so is more useful than blaming the artifact.
````

## Arm B: system prompt (verbatim)

```text
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
performed. The user message gives the exact JSON schema.
```

### Arm B: user message builder

```python
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
```

````text
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
`operationalisation`, and `produced_value_location` are mandatory.
````
