# Stage 5: calibration (observation only)

**Question.** Do the cheap document-level signals from earlier stages point toward the
execution outcomes of Stage 4? The signals are ARA's R and R_c (Stage 2) and the mechanical
environment score E (Stage 1). The outcomes are whether Arm A ran, its intervention rung and
RRL level, and whether Arm B computed the claim.

**No model is called in this stage.** It is a read-only analysis of the database.

With 7 to 9 executed papers, the [tier rule](protocol.md#7-outcomes-and-statistical-tier-rule)
allows no fitted model and no corpus extrapolation. The script therefore reports Spearman's rho
and group means as observations only, and nothing is calibrated. No threshold tau_R is set.

## Run it

```bash
uv run python scripts/stage5_observation.py \
  --arm-a s4-arma-all9-r3 --arm-b s4-armb-7-r3 \
  --ara s2-ara-2021_2022-gpt56t-r3 --ara s2-ara-2023_2026-gpt56t-r3
```

Replace the experiment names with your own when you rerun the pipeline. The script prints one
row per executed paper (R, R_c, E, Arm A ran/rung/RRL, Arm B computed), then the rank
correlations and the mean R of papers where each arm succeeded and where it did not.

Related outputs:

- `lakdb arm-b contrast --arm-a … --arm-b …` gives the per-paper table of both arms side by side.
- `lakdb rstats run` covers the corpus-level analysis of R: distribution, year trend,
  run-to-run stability, R against E, node deficits and the tau_R sensitivity curve.
  `scripts/make_r_analysis_report.py` renders it as Markdown.
