# Study protocol: LAK reproducibility readiness at scale

This is the frozen protocol that governs the pipeline, with development history and status
notes removed. Section numbers match the original so references in the code (for example
`docs/protocol.md §6`) resolve. Definitions, taxonomy, tolerances, the tier rule and the gate
criteria are unchanged (gates are shown in their final, amended form). Ground-truth layers
that were planned but not collected are omitted from §8, and §12 (change control) is omitted. There is no external pre-registration; the analysis plan was fixed in
this file before the data it governs were seen.

## 1. Purpose

Decide, for each LAK paper, how far it can be taken toward reproduction, using a gated
cascade: applicability → artifact audit → ARA reconstructability → claim extraction →
agentic execution. Report the corpus as a distribution over the readiness ladder (§5), and
test whether cheap document-level signals predict execution-level outcomes.

## 2. Corpus

**Source:** the LAK proceedings as downloaded from the ACM Digital Library, laid out as
`{year}/{long|short}` with the exported `acm.bib` as the sole metadata authority (see the
README for how to rebuild it, and `docs/corpus.csv` for the paper list).

**Window:** LAK 2021–2026. **Unit:** one paper. **Included kinds:** long and short papers;
posters excluded (Haim's LAK study also covered only research articles and short papers).

| Year | Long | Short |
|---|---|---|
| 2021 | 42 | 29 |
| 2022 | 39 | 23 |
| 2023 | 49 | 22 |
| 2024 | 66 | 29 |
| 2025 | 70 | 31 |
| 2026 | 66 | 40 |

**Matching rule (fixed):** PDFs join BibTeX by explicit `file` field, exact DOI-derived
filename, or exact normalized title/prefix (at least 32 normalized characters, unique).
Ambiguous or missing matches are recorded in `ingest_issues`, never guessed.

## 3. Parsing standard

One pinned LiteParse profile: Markdown output, OCR enabled, 300 DPI, small-text
preservation, link extraction. No fallback parser, no pre-parse classifier. Parser name,
version, settings, source hash and outcome are versioned in SQLite.

A paper whose extraction-quality gate fails is recorded as `unclear` and never sent to a
model. Parse quality is a covariate in every analysis, because a badly parsed paper scores
badly for reasons unrelated to reproducibility.

## 4. Experiment discipline

Every model-facing stage runs as a **named, frozen experiment** carrying: provider, model,
reasoning effort or temperature, prompt name and version, prompt/schema hash, repeat count,
and upstream commit where applicable. **A configuration change requires a new experiment,
never a mutation of an existing one.** All runs are resumable and idempotent.

Repetitions: 3 for applicability; 3 for ARA.

**Model, fixed:** OpenAI `gpt-5.6-terra` at **medium** reasoning effort, temperature omitted
(reasoning model), for every model-facing stage. Single provider, single model, by decision.
Robustness is established by repeated runs at fixed configuration, not by model diversity,
and this is stated as a limitation.

## 5. The readiness ladder (RRL)

Each paper occupies exactly one level.

| Level | Definition |
|---|---|
| **L0** | Not applicable: non-computational; no reproduction claim is well-defined |
| **L1** | Opaque: computational, no reachable artifacts, R below tau_R |
| **L2** | Described: no artifacts, but workflow reconstructable from text (R at or above tau_R) |
| **L3** | Available: artifacts resolve, but insufficient to execute (no docs, env, or licence) |
| **L4** | Runnable: artifacts resolve and the agent executes end-to-end |
| **L5** | Recovered: claim reproduced within broad tolerance, or conclusion matches |
| **L6** | Strictly recovered: within strict tolerance |

tau_R is **not** fixed here. It is set only after an external benchmark containing both
human-verified successes and failures is available (§8), and a full threshold-sensitivity
curve is reported alongside any chosen value. Until then papers are reported on continuous
R and the L1/L2 split is omitted.

**"Reproduced" is reserved for execution evidence (L5+). A high R never earns it.**

## 6. Claim taxonomy and tolerances

| Type | Instance | Target | Strict | Broad |
|---|---|---|---|---|
| T1 inferential effect | "condition X raised posttest scores" | standardized effect | abs. delta-d at most 0.05 | at most 0.20 |
| T2 predictive performance | "AUC .82" | headline metric | abs at most 0.01 or rel at most 2% | 0.05 / 10% |
| T3 comparative / ranking | "A beats B, C" | ordering | top-1 claim preserved | Kendall tau at least 0.6 |
| T4 descriptive / structural | "3 learner clusters" | partition | count match + ARI at least 0.7 | count match |
| T5 non-quantitative | qualitative / position | n/a | excluded at Stage 0; exclusion rate reported | |

**Primary cross-type outcome:** conclusion match in {supports, opposite, inconclusive},
majority vote over runs, ties resolved to inconclusive. Type-specific tolerances are
secondary and reported within type.

Tolerances are frozen here. Loosening any of them after seeing Stage 4 results is an
amendment that must be logged and disclosed in any report.

## 7. Outcomes and statistical tier rule

**Primary outcome:** ordinal L-level. **Secondary:** binary `recovered`; and, when the
positive class is thin, binary **L4-vs-below** ("does it run at all").

**Matched conditions:** R is a text-only proxy, so it is analysed against the **cold**
execution condition (paper + data, no author code; Arm B). The **warm** condition (author
code supplied; Arm A) is reported separately.

**Tier rule, keyed to observed event count, chosen before Stage 4 outcomes are seen:**

- **Tier 1, fewer than ~10 events:** no fitted model. Cross-tabulation, AUC as a
  descriptive Mann–Whitney statistic with an exact CI, exact binomial intervals,
  **no corpus extrapolation**.
- **Tier 2, ~10–25 events:** one pre-specified predictor, Firth penalized likelihood,
  pre-specified uniform shrinkage, labelled exploratory.
- **Tier 3, 25 or more events:** full calibrated-proxy estimation with nested bootstrap.

No post-hoc recalibration. Competing predictors are compared explicitly: R vs the mechanical
environment score E vs both, against a page-count baseline and the no-information rate.

**Untestable assumption:** extrapolating from executed to non-executed papers assumes the
outcome is conditionally independent of selection given the predictors.

## 8. Ground truth and benchmarks

| Layer | Source | Use |
|---|---|---|
| G1 | Haim et al. LAK coding (133 papers, 2021–2022) | Stage 0 and Stage 1 validation |
| G3 | our execution round | outcome labels, with mandatory fault adjudication |
| External | a benchmark with human-verified successes and failures | the only admissible source for any tau_R |

## 9. Confirmatory vs exploratory

- **Exploratory:** corpus assembly, artifact audit, executable census, and pilots.
- **Confirmatory:** ARA at corpus scale, claim extraction, execution, and the analysis in §7.
- Any choice made after seeing the outcomes it governs is exploratory by definition and is
  reported that way.

## 10. Acceptance criteria

| Gate | Criterion |
|---|---|
| G1 corpus | zero unresolved `ingest_issues`; parse failures below 5% |
| G2 Stage 0 vs Haim | §10.1: gated on applicable recall and contamination, not on kappa |
| G2b Stage 1 audit | Not a numeric gate. Per field, report accuracy against the majority-class baseline, kappa with a bootstrap interval, and positive recall with an exact interval; no field is described as passing or failing |
| G3 census | branch A at least 25 candidates / B 8–24 / C fewer than 8, declared before Stage 4 |
| G4 ARA stability | SD of R across repeats at or below ARA's reported ceiling of 0.09 on at least 80% of papers, and above 0.09 on no more than 5%. No lower bound |
| G5 validity | methods-strip lowers R_methods; verbosity injection does not raise R; length explains a minority of variance |
| G6 human ceiling | human–human alpha at least 0.6 before any LLM comparison is reported |
| G7 execution | agent-fault failures a minority of failures |
| G8 prediction | CV AUC above the page-count baseline; otherwise reported as a negative result |

### 10.1 Stage 0 acceptance: why kappa is not the criterion

Haim's LAK corpus contains 133 papers of which only 6–7 are not_applicable. At that support
an exact interval on not-applicable recall is about 0.41 wide even for a perfect classifier
(7/7 gives [0.590, 1.000]), and one paper moves Cohen's kappa by about 0.10. No kappa or
macro-F1 bar is testable on this benchmark.

The gate is set on what the benchmark can estimate, weighted by the asymmetry of the two
errors. Excluding a computational paper is silent and unrecoverable, whereas admitting a
non-computational one dilutes the corpus but is visible later.

| Criterion | Bar |
|---|---|
| Applicable recall | exact CI **lower** bound ≥ 0.95 |
| Contamination = FP / predicted applicable | exact CI **upper** bound ≤ 0.10 |
| Unclear rate | ≤ 0.10 |
| Repeated-run agreement | ≥ 0.95 |

Cohen's kappa, macro-F1, balanced accuracy and not-applicable recall are reported with exact
intervals but never gated, always beside the trivial all-applicable baseline and the
negative-class support.

## 11. Reporting rules

- Report every gate outcome, including failures. A failed gate changes the framing of the report,
  never the criterion.
- Never loosen a tolerance or threshold to manufacture a positive class.
- Aggregate results are public; per-paper reports are author-facing or anonymised.
- Cost and wall-clock are reported per stage alongside accuracy.

## 13. Execution order

**Phase A (benchmark window, LAK 2021–2022, 133 papers):** the window with human coding, where
each stage is validated against Haim et al. before scale-out.

**Phase B (LAK 2023–2026, 373 papers):** the unchanged Phase A configuration, with no prompt,
model or threshold changes. A change found necessary in Phase B invalidates the comparison and
requires a re-run across all years.

Claim extraction, the execution round and calibration then run on the pooled corpus under
the tier rule in §7.
