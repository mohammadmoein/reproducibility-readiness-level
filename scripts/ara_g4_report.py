"""Run-to-run stability of R for an ARA experiment (the Stage 2 gate, docs/protocol.md).

    uv run python scripts/ara_g4_report.py s2-ara-2021_2022-gpt56t-r3

Reports SD of R, R_c and R_s across repetitions, per paper and pooled, plus the G4 criterion
as amended 2026-09-18. Also reports the R_s ceiling, since a structural score pinned at 1.0
makes R and R_c the same variable.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics as st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--database", default="data/lak.sqlite")
    args = ap.parse_args()

    db = sqlite3.connect(args.database)
    db.row_factory = sqlite3.Row
    exp = db.execute(
        "SELECT id, model, repeat_count FROM ara_experiments WHERE name=?",
        (args.experiment,),
    ).fetchone()
    if exp is None:
        raise SystemExit(f"No such ARA experiment: {args.experiment}")

    per: dict[int, list[sqlite3.Row]] = {}
    for r in db.execute(
        "SELECT paper_id, reproducibility_index R, content_score Rc, structural_score Rs "
        "FROM ara_runs WHERE experiment_id=? AND status='succeeded'",
        (exp["id"],),
    ):
        per.setdefault(r["paper_id"], []).append(r)

    full = {p: rs for p, rs in per.items() if len(rs) == exp["repeat_count"]}
    print(f"\n{args.experiment}  ({exp['model']}, {exp['repeat_count']} repetitions)")
    print(f"  papers with a complete set of repetitions: {len(full)}/{len(per)}")

    sd_r = {p: st.pstdev([x["R"] for x in rs]) for p, rs in full.items()}
    mean_r = {p: st.mean([x["R"] for x in rs]) for p, rs in full.items()}
    sds = list(sd_r.values())
    n = len(sds)

    def pct(k: int) -> str:
        return f"{k:3d}/{n} ({k / n:5.1%})"

    print("\n-- run-to-run SD of R --")
    q = st.quantiles(sds, n=4)
    print(f"  mean {st.mean(sds):.4f}   median {st.median(sds):.4f}   "
          f"quartiles {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}   max {max(sds):.4f}")
    at_or_below = sum(s <= 0.09 for s in sds)
    above = sum(s > 0.09 for s in sds)
    inband = sum(0.02 <= s <= 0.09 for s in sds)
    below = sum(s < 0.02 for s in sds)
    print(f"  at or below 0.09 (stable enough): {pct(at_or_below)}")
    print(f"  above 0.09        (too unstable): {pct(above)}")
    print(f"  strictly inside ARA's 0.02-0.09 band: {pct(inband)}")
    print(f"  BELOW 0.02 (more stable than ARA reported): {pct(below)}")

    print("\n-- Stage 2 stability gate --")
    lit = inband / n >= 0.80 and above / n <= 0.05
    sane = at_or_below / n >= 0.80 and above / n <= 0.05
    print(f"  literal reading  (0.02 <= SD <= 0.09 on >=80%, >0.09 on <=5%): "
          f"{'PASS' if lit else 'FAIL'}")
    print(f"  intended reading (SD <= 0.09 on >=80%, >0.09 on <=5%):         "
          f"{'PASS' if sane else 'FAIL'}")
    if lit != sane:
        print("  NOTE: the two disagree only because papers MORE stable than ARA's"
              "\n        reported floor count as failures under the literal wording."
              "\n        That is a defect in the amendment, not in the instrument.")

    print("\n-- R across papers (the signal SD has to be small relative to) --")
    ms = list(mean_r.values())
    print(f"  mean R {st.mean(ms):.4f}   SD across papers {st.pstdev(ms):.4f}   "
          f"range {min(ms):.3f}-{max(ms):.3f}")
    print(f"  ratio, within-paper SD to between-paper SD: "
          f"{st.mean(sds) / st.pstdev(ms):.3f}")

    print("\n-- component scores --")
    for label, key in (("R_c content", "Rc"), ("R_s structural", "Rs")):
        vals = [st.mean([x[key] for x in rs]) for rs in full.values()]
        ceil = sum(v >= 0.999 for v in vals)
        print(f"  {label:16s} mean {st.mean(vals):.4f}  SD {st.pstdev(vals):.4f}  "
              f"at ceiling(1.0) {ceil}/{len(vals)} ({ceil / len(vals):.0%})")

    worst = sorted(sd_r.items(), key=lambda kv: -kv[1])[:8]
    print("\n-- least stable papers --")
    for pid, s in worst:
        vals = " ".join(f"{x['R']:.3f}" for x in full[pid])
        print(f"  paper {pid:4d}  SD {s:.4f}  R = {vals}")
    db.close()
    print()


if __name__ == "__main__":
    main()
