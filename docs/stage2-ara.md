# Stage 2: ARA workflow reconstructability

**Question.** From the paper alone, how completely can its analysis workflow be reconstructed
(data sources → processing steps → reported results)? The score is ARA's reproducibility index
**R**, the geometric mean of a content score **R_c** and a structural score **R_s**.

**Code.** `ara.py` (experiment, runs, scoring), `ara_bridge.py` (runs upstream ARA),
`ara_analysis.py`, `r_analysis.py` · **Commands.** `lakdb ara …`, `lakdb rstats …`

## How the model is called

We do not reimplement ARA. `ara_bridge.py` runs the published `GPTPaperAnalyst` from the ARA
repository, pinned to commit `e015be37d82c57e6da826ad50fe8023e9f1255f5`, inside that
project's own locked environment (`uv run --project .external/ara/src --locked`). ARA is
GPL-3.0 and is therefore not included here; see the README for the clone command.
`ara.py` checks the checkout's commit before every run and refuses to run on a different one.

- **Input:** the original PDF (not our Markdown), uploaded as a file.
- **Model:** `gpt-5.6-terra`, reasoning effort `medium` (injected by the bridge; temperature is
  omitted for reasoning models). This is the only change to the upstream call.
- **Six sequential queries**, each with a strict schema: `header` → `nodes_source` →
  `nodes_sink` (given the source ids) → `nodes_process` (given source and sink ids) →
  `artefacts` → `parameters`. They are dependent, so they run realtime, not through the Batch API.
- **Selection:** papers that Stage 0 labels applicable **and** whose majority claim type is
  quantitative. L0 papers have no workflow to reconstruct and are skipped
  (`--include-non-quantitative` overrides this).
- **Repetitions:** 3 per paper. Scoring (`ara.score_ara_profile`) follows ARA's published
  content, structural and geometric-mean formulas and is computed locally from the returned
  profile.

## Prompts

The six prompts and the shared system instruction are ARA's and are used unchanged. They
are in the upstream file
[`src/ara_pipeline/online_llm_pipelines.py`](https://github.com/AndresLaverdeMarin/agentic_reproducibility_assessment/blob/e015be37d82c57e6da826ad50fe8023e9f1255f5/src/ara_pipeline/online_llm_pipelines.py)
(`SYSTEM_INSTRUCTION`, `PROMPT_HEADER`, `PROMPT_SOURCE_NODES`, `build_sink_nodes_prompt`,
`build_process_nodes_prompt`, `PROMPT_ARTEFACTS`, `PROMPT_PARAMETERS`). We do not copy them here
because of the licence.

| Query | Asks for |
|---|---|
| `header` | title and authors |
| `nodes_source` | every dataset the paper uses, with a quote, description, size and licence |
| `nodes_sink` | every reported result (tables, figures, headline numbers) and the sources it draws on |
| `nodes_process` | the methods and experiments that connect sources to results |
| `artefacts` | code repositories, supplements and external resources the paper mentions |
| `parameters` | parameters, hyperparameters, configuration values, software versions, hardware |

## Reference experiments

| Experiment | Papers | Reps |
|---|---|---|
| `s2-ara-2021_2022-gpt56t-r3` | 2021–2022, applicable and quantitative | 3 |
| `s2-ara-2023_2026-gpt56t-r3` | 2023–2026, applicable and quantitative | 3 |

No threshold tau_R is set (see [protocol §5](protocol.md#5-the-readiness-ladder-rrl)).
`rstats run` reports R as a continuous score and gives a sensitivity curve over tau_R instead.

## Run it

```bash
git clone https://github.com/AndresLaverdeMarin/agentic_reproducibility_assessment.git .external/ara
git -C .external/ara checkout e015be37d82c57e6da826ad50fe8023e9f1255f5

uv run lakdb ara create s2-demo --applicability-experiment s0-demo --model gpt-5.6-terra --reasoning-effort medium --repeats 3
uv run lakdb ara run --experiment s2-demo --paper-id 1
uv run lakdb ara report --experiment s2-demo
uv run lakdb rstats run --experiments s2-demo     # distribution, trend, stability, node deficits
uv run python scripts/ara_g4_report.py --help     # run-to-run stability gate
uv run python scripts/make_stage2_report.py --help
```

Raw ARA profiles are written to `artifacts/ara/`.
