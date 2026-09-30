# Pipeline overview

Each stage writes to the SQLite catalog (`data/lak.sqlite` by default) and reads what earlier
stages wrote. Every model-facing stage runs as a named, frozen experiment. The experiment row
stores the full system prompt, JSON schema, their SHA-256 hashes, the model and the repetition
count, and a changed configuration needs a new experiment. All commands can be resumed: rerunning
one skips work that has already succeeded.

| Stage | LLM | Command group | Main tables | Details |
|---|---|---|---|---|
| Corpus | – | `lakdb ingest`, `parse`, `validate` | `papers`, `file_assets`, `content_versions`, `ingest_issues` | README |
| 0 Applicability | yes | `experiment`, `screen`, `batch`, `haim-align` | `study_experiments`, `applicability_runs`, `applicability_decisions` | [stage0-applicability.md](stage0-applicability.md) |
| 1 Artifact audit | yes (2 prompts) | `artifact`, `resolve`, `fetch` | `artifact_experiments`, `artifact_*`, `artifact_snapshots`, `artifact_environment` | [stage1-artifact-audit.md](stage1-artifact-audit.md) |
| 2 ARA | yes (upstream ARA) | `ara`, `rstats` | `ara_experiments`, `ara_runs` | [stage2-ara.md](stage2-ara.md) |
| 3 Focal claim | yes | `claim` | `claim_experiments`, `claim_runs`, `claim_decisions` | [stage3-focal-claim.md](stage3-focal-claim.md) |
| 4 Execution | yes (agentic) | `arm-a`, `arm-b` | `arm_a_*`, `arm_b_*` | [stage4-execution.md](stage4-execution.md) |
| 5 Calibration | – | `scripts/stage5_observation.py`, `arm-b contrast` | read-only | [stage5-calibration.md](stage5-calibration.md) |

```
PDFs + acm.bib ──ingest/parse──▶ Markdown
      │
      ▼
 Stage 0  applicable? claim type ──(L0: stop)
      │
      ├──▶ Stage 1  what is shared? resolve + fetch artifacts, score E
      │
      ▼
 Stage 2  ARA workflow reconstructability R (papers with a quantitative claim)
      │
      ▼
 Stage 3  focal claim + withheld reported value (papers with data + code retrieved)
      │
      ▼
 Stage 4  Arm A (authors' code)  |  Arm B (paper + data, no code)  →  RRL level
      │
      ▼
 Stage 5  R, R_c, E against execution outcomes (observation)
```

Model everywhere: OpenAI `gpt-5.6-terra` at `medium` reasoning effort. Stage 1 does not depend on
Stage 0's output and can run in parallel with it.

Other commands: `lakdb dashboard` (terminal UI over the catalog), `lakdb usage` (token and
cost totals per experiment, priced from `pricing.py`), `lakdb search` (full-text search over
parsed papers), and `lakdb highlight run` (PDFs annotated with the Stage 0 evidence).
