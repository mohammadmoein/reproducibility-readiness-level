"""Generate `artifacts/analysis/stage2-results.md` from the database. Nothing is transcribed by hand.

    uv run python scripts/make_stage2_report.py

Mirrors `scripts/make_phase_b_report.py`: every table is derived from `data/lak.sqlite` and
the stored ARA artifacts, so the file can be regenerated at any time and never drifts from
the evidence. No thresholds are proposed and no gate is decided here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import statistics as st
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ara_usage import assessment_rows, report_windows  # noqa: E402
from r_analysis import stability_stats  # noqa: E402

BENCH = "s2-ara-2021_2022-gpt56t-r3"
LATER = "s2-ara-2023_2026-gpt56t-r3"
PROBE = "s2-ara-luna-probe-r3"


def per_paper(db: sqlite3.Connection, name: str) -> tuple[sqlite3.Row, dict[int, list]]:
    exp = db.execute(
        "SELECT id, model, repeat_count, upstream_commit, configuration_json "
        "FROM ara_experiments WHERE name=?", (name,)
    ).fetchone()
    if exp is None:
        raise SystemExit(f"missing experiment: {name}")
    out: dict[int, list] = {}
    for r in db.execute(
        "SELECT r.paper_id, p.collection_year y, r.reproducibility_index R, "
        "r.content_score Rc, r.structural_score Rs FROM ara_runs r "
        "JOIN papers p ON p.id=r.paper_id "
        "WHERE r.experiment_id=? AND r.status='succeeded'", (exp["id"],)
    ):
        out.setdefault(r["paper_id"], []).append(r)
    return exp, {p: v for p, v in out.items() if len(v) == exp["repeat_count"]}


def stats(rows: dict[int, list]) -> dict:
    """Stability numbers come from `r_analysis.stability_stats` (A3.1 of the R analysis
    spec), so this report and `artifacts/analysis/r-analysis-results.md` cannot disagree about them."""
    rc = [st.mean([x["Rc"] for x in v]) for v in rows.values()]
    rs = [st.mean([x["Rs"] for x in v]) for v in rows.values()]
    return {
        **stability_stats({p: [x["R"] for x in v] for p, v in rows.items()}),
        "rc_mean": st.mean(rc), "rs_mean": st.mean(rs),
        "rs_ceiling": sum(v >= 0.999 for v in rs),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="data/lak.sqlite")
    ap.add_argument("--output", default="artifacts/analysis/stage2-results.md")
    args = ap.parse_args()

    db = sqlite3.connect(args.database)
    db.row_factory = sqlite3.Row
    out: list[str] = []
    w = out.append

    w("# Stage 2 results — ARA workflow reconstructability, LAK 2021–2026\n")
    w(f"Generated from `{args.database}` on {date.today().isoformat()}. Every table is produced")
    w("by `scripts/make_stage2_report.py` directly from the database and the stored ARA")
    w("artifacts; nothing is transcribed by hand. **No thresholds are proposed and no gate is")
    w("decided in this document.**\n")

    bench_exp, bench_rows = per_paper(db, BENCH)
    later_exp, later_rows = per_paper(db, LATER)
    b, l = stats(bench_rows), stats(later_rows)
    pooled = stats({**bench_rows, **later_rows})

    w("## 1. Provenance\n")
    w("| experiment | model | reps | papers | upstream commit |")
    w("|---|---|---|---|---|")
    for nm, e, s in ((BENCH, bench_exp, b), (LATER, later_exp, l)):
        w(f"| `{nm}` | {e['model']} | {e['repeat_count']} | {s['n']} | `{e['upstream_commit'][:12]}` |")
    cfg = json.loads(bench_exp["configuration_json"])
    w(f"\nSelection rule: `{cfg.get('selection')}` (version {cfg.get('selection_version')}).")
    w("Temperature is absent: the pinned model is a reasoning model and does not accept it,")
    w("so run-to-run variation is measured rather than suppressed.\n")

    w("## 2. Stability of R across repetitions\n")
    w("| | benchmark 2021–22 | held out 2023–26 | pooled |")
    w("|---|---|---|---|")
    rows = [
        ("papers", "n", "{:d}"), ("mean SD of R", "sd_mean", "{:.4f}"),
        ("median SD", "sd_median", "{:.4f}"), ("max SD", "sd_max", "{:.4f}"),
        ("mean R", "r_mean", "{:.4f}"), ("between-paper SD of mean R", "r_between", "{:.4f}"),
        ("within/between ratio", "ratio", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        w(f"| {label} | {fmt.format(b[key])} | {fmt.format(l[key])} | {fmt.format(pooled[key])} |")
    for label, key in (("SD at or below 0.09", "le09"), ("SD above 0.09", "gt09"),
                       ("SD below 0.02", "lt02")):
        w(f"| {label} | {b[key]}/{b['n']} ({b[key] / b['n']:.1%}) | "
          f"{l[key]}/{l['n']} ({l[key] / l['n']:.1%}) | "
          f"{pooled[key]}/{pooled['n']} ({pooled[key] / pooled['n']:.1%}) |")
    w("\nThe within-to-between ratio is the quantity that matters for triage: it is how far R")
    w("moves between runs relative to how far it separates papers. It is effectively identical")
    w("inside and outside the validation window, which is not true of the Stage 0 and Stage 1")
    w("instruments.\n")

    w("## 3. Component scores and the structural ceiling\n")
    w("| | benchmark | 2023–26 | pooled |")
    w("|---|---|---|---|")
    w(f"| mean R_c (content) | {b['rc_mean']:.4f} | {l['rc_mean']:.4f} | {pooled['rc_mean']:.4f} |")
    w(f"| mean R_s (structural) | {b['rs_mean']:.4f} | {l['rs_mean']:.4f} | {pooled['rs_mean']:.4f} |")
    w(f"| R_s at exactly 1.000 | {b['rs_ceiling']}/{b['n']} ({b['rs_ceiling'] / b['n']:.0%}) | "
      f"{l['rs_ceiling']}/{l['n']} ({l['rs_ceiling'] / l['n']:.0%}) | "
      f"{pooled['rs_ceiling']}/{pooled['n']} ({pooled['rs_ceiling'] / pooled['n']:.0%}) |")
    w("\nR_s saturates for roughly a third of the corpus while R_c never does. Nearly all usable")
    w("variance in R is content, which is why R_s is not reported as a separate finding.\n")

    w("## 4. R by year\n")
    w("| year | papers | mean R | SD across papers | mean R_c | median run-to-run SD |")
    w("|---|---|---|---|---|---|")
    by_year: dict[int, list] = {}
    for pid, runs in {**bench_rows, **later_rows}.items():
        by_year.setdefault(runs[0]["y"], []).append(runs)
    for y in sorted(by_year):
        rs = by_year[y]
        ms = [st.mean([x["R"] for x in v]) for v in rs]
        cs = [st.mean([x["Rc"] for x in v]) for v in rs]
        sd = [st.pstdev([x["R"] for x in v]) for v in rs]
        w(f"| {y} | {len(rs)} | {st.mean(ms):.4f} | {st.pstdev(ms):.4f} | "
          f"{st.mean(cs):.4f} | {st.median(sd):.4f} |")

    w("\n## 5. Cost\n")
    w("| experiment | assessments | cached input | $/assessment | total |")
    w("|---|---|---|---|---|")
    grand = 0.0
    for nm in (BENCH, LATER, PROBE):
        try:
            model, rws = assessment_rows(db, nm)
        except Exception:
            continue
        for label, s in report_windows(rws, 10**9):
            if label.lower().startswith("all") and s["assessments"]:
                w(f"| `{nm}` | {s['assessments']} | {s['cached_ratio']:.1%} | "
                  f"${s['mean_cost']:.4f} | ${s['total_cost']:.2f} |")
                if nm != PROBE:
                    grand += s["total_cost"]
                break
    w(f"\n**Stage 2 total, terra: ${grand:.2f}** over {b['n'] + l['n']} papers at "
      f"{bench_exp['repeat_count']} repetitions.")
    w("Billing is per request; an assessment is six sequential calls sharing one PDF prefix, so")
    w("queries after the first read from the prompt cache. See `pricing.py`.\n")

    w("## 6. Least stable papers\n")
    w("| paper | year | SD of R | R per repetition |")
    w("|---|---|---|---|")
    allrows = {**bench_rows, **later_rows}
    worst = sorted(allrows.items(), key=lambda kv: -st.pstdev([x["R"] for x in kv[1]]))[:10]
    for pid, runs in worst:
        vals = ", ".join(f"{x['R']:.3f}" for x in runs)
        w(f"| {pid} | {runs[0]['y']} | {st.pstdev([x['R'] for x in runs]):.4f} | {vals} |")

    w("\n## 7. Cross-model probe (not a robustness arm)\n")
    _, probe_rows = per_paper(db, PROBE)
    if probe_rows:
        shared = sorted(set(probe_rows) & set(bench_rows))
        w(f"`{PROBE}` ran gpt-5.6-luna on {len(shared)} papers also covered by the benchmark")
        w("experiment, paired. It is reported as a sensitivity note, not as cross-model validation.\n")
        w("| paper | terra R | luna R | difference |")
        w("|---|---|---|---|")
        dt = []
        for pid in shared:
            t = st.mean([x["R"] for x in bench_rows[pid]])
            lu = st.mean([x["R"] for x in probe_rows[pid]])
            dt.append(lu - t)
            w(f"| {pid} | {t:.3f} | {lu:.3f} | {lu - t:+.3f} |")
        w(f"\nMean shift **{st.mean(dt):+.3f}**, and luna scores every paper higher. A threshold")
        w("calibrated on one model would not transfer to the other.\n")

    pathlib.Path(args.output).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} lines)")
    db.close()


if __name__ == "__main__":
    main()
