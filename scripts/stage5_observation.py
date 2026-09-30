"""Stage 5: relate the document-level signals (R, R_c, E) to the Stage 4 execution outcomes.

Observation only. With 7 to 9 papers no estimate or significance claim is possible, so the
script prints Spearman's rho and group means and nothing else.

Per paper:
- R and R_c: mean over every succeeded repetition in the named ARA experiments.
- E: the environment score of the paper's retrieved code snapshot (mean if several).
- Arm A: `ran` (0/1), intervention rung A0..A4 (ordinal), RRL level (ordinal).
- Arm B: `computed` (0/1, majority over repetitions).

Usage:
    uv run python scripts/stage5_observation.py --arm-a s4-arma-all9-r3 --arm-b s4-armb-7-r3 \
        --ara s2-ara-2021_2022-gpt56t-r3 --ara s2-ara-2023_2026-gpt56t-r3
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics as st

from scipy.stats import spearmanr

RUNG = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4}
RRL = {f"L{i}": i for i in range(7)}


def per_paper(db: sqlite3.Connection, sql: str, params: tuple) -> dict[int, float]:
    groups: dict[int, list[float]] = {}
    for paper_id, value in db.execute(sql, params):
        if value is not None:
            groups.setdefault(int(paper_id), []).append(float(value))
    return {pid: st.mean(values) for pid, values in groups.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default="data/lak.sqlite")
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--ara", action="append", required=True, help="ARA experiment; repeat to pool")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    marks = ",".join("?" * len(args.ara))
    ara_sql = (
        "SELECT r.paper_id, r.{col} FROM ara_runs r JOIN ara_experiments e ON e.id=r.experiment_id "
        f"WHERE r.status='succeeded' AND e.name IN ({marks})"
    )
    r_mean = per_paper(db, ara_sql.format(col="reproducibility_index"), tuple(args.ara))
    rc_mean = per_paper(db, ara_sql.format(col="content_score"), tuple(args.ara))
    e_score = per_paper(db, "SELECT paper_id, e_score FROM artifact_environment", ())

    arm_a = {
        int(pid): (ran, RUNG.get(rung), RRL.get(rrl))
        for pid, ran, rung, rrl in db.execute(
            "SELECT d.paper_id, d.ran, d.intervention_rung, d.rrl_level FROM arm_a_decisions d "
            "JOIN arm_a_experiments e ON e.id=d.experiment_id WHERE e.name=?",
            (args.arm_a,),
        )
    }
    arm_b = {
        int(pid): computed
        for pid, computed in db.execute(
            "SELECT d.paper_id, d.computed FROM arm_b_decisions d "
            "JOIN arm_b_experiments e ON e.id=d.experiment_id WHERE e.name=?",
            (args.arm_b,),
        )
    }

    print(f"{'paper':>6} {'R':>6} {'R_c':>6} {'E':>6}  A ran rung RRL   B computed")
    for pid in sorted(set(arm_a) | set(arm_b)):
        a = arm_a.get(pid, (None, None, None))
        fmt = lambda v: f"{v:6.3f}" if v is not None else "     -"  # noqa: E731
        print(f"{pid:>6} {fmt(r_mean.get(pid))} {fmt(rc_mean.get(pid))} {fmt(e_score.get(pid))}"
              f"  {a[0]!s:>5} {a[1]!s:>4} {a[2]!s:>3}   {arm_b.get(pid, '-')}")

    def rho(signal: dict[int, float], outcome: dict[int, float], label: str) -> None:
        ids = sorted(pid for pid in outcome if pid in signal and outcome[pid] is not None)
        if len(ids) < 3:
            print(f"{label:34} n={len(ids)}  too few papers")
            return
        res = spearmanr([signal[i] for i in ids], [outcome[i] for i in ids])
        print(f"{label:34} n={len(ids)}  rho={res.statistic:+.2f}  p={res.pvalue:.2f}")

    print()
    rho(r_mean, arm_b, "R vs Arm B computed")
    rho(rc_mean, arm_b, "R_c vs Arm B computed")
    rho(r_mean, {p: v[2] for p, v in arm_a.items()}, "R vs Arm A RRL")
    rho(e_score, {p: v[1] for p, v in arm_a.items()}, "E vs Arm A rung")
    rho(e_score, {p: v[0] for p, v in arm_a.items()}, "E vs Arm A ran")

    print()
    for label, outcome in (("Arm B computed", arm_b), ("Arm A ran", {p: v[0] for p, v in arm_a.items()})):
        yes = [r_mean[p] for p, v in outcome.items() if v and p in r_mean]
        no = [r_mean[p] for p, v in outcome.items() if not v and p in r_mean]
        if yes and no:
            print(f"mean R, {label}: {st.mean(yes):.3f} ({len(yes)}) vs {st.mean(no):.3f} ({len(no)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
