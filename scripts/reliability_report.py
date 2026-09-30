"""Report Stage 1 repeated-run agreement from the reliability subsample.

Reads the three repetitions stored for each paper in the reliability experiment and reports,
per field and overall, the proportion of papers on which all repetitions agree, with an exact
(Clopper-Pearson) interval.

Usage:
    uv run python scripts/reliability_report.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

from scipy.stats import beta as beta_dist

EXPERIMENT = "s1-claim-reliability-gpt56t-p1.1.0"
FIELDS = [
    "open_data",
    "data_documentation",
    "open_materials",
    "materials_documentation",
    "readme",
    "license",
    "preregistration",
]


def exact_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2, successes, total - successes + 1))
    hi = 1.0 if successes == total else float(beta_dist.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--database", type=Path, default=Path("data/lak.sqlite"))
    args = parser.parse_args()

    con = sqlite3.connect(args.database)
    con.row_factory = sqlite3.Row
    exp = con.execute(
        "SELECT id, repeat_count FROM artifact_experiments WHERE name=?", (args.experiment,)
    ).fetchone()
    if exp is None:
        raise SystemExit(f"unknown experiment: {args.experiment}")

    per_paper: dict[int, list[dict]] = collections.defaultdict(list)
    for r in con.execute(
        "SELECT paper_id, parsed_result_json FROM artifact_runs "
        "WHERE experiment_id=? AND status='succeeded'",
        (exp["id"],),
    ):
        if r["parsed_result_json"]:
            per_paper[r["paper_id"]].append(json.loads(r["parsed_result_json"]))

    complete = {p: v for p, v in per_paper.items() if len(v) >= exp["repeat_count"]}
    print(f"experiment: {args.experiment}")
    print(f"papers with {exp['repeat_count']} completed repetitions: {len(complete)}")
    if not complete:
        raise SystemExit("no completed papers yet")

    print(f"\n{'field':26s} {'unanimous':>10} {'rate':>7}   95% CI")
    disagreements: list[tuple[int, str, list[str]]] = []
    total_ok = total_n = 0
    for field in FIELDS:
        ok = 0
        for paper_id, payloads in complete.items():
            verdicts = [str((p.get(field) or {}).get("verdict")) for p in payloads]
            if len(set(verdicts)) == 1:
                ok += 1
            else:
                disagreements.append((paper_id, field, verdicts))
        lo, hi = exact_ci(ok, len(complete))
        total_ok += ok
        total_n += len(complete)
        print(f"{field:26s} {ok:>6}/{len(complete):<3} {ok/len(complete):7.3f}   [{lo:.3f}, {hi:.3f}]")

    lo, hi = exact_ci(total_ok, total_n)
    print(f"\n{'ALL FIELDS':26s} {total_ok:>6}/{total_n:<3} {total_ok/total_n:7.3f}   [{lo:.3f}, {hi:.3f}]")

    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}):")
        for paper_id, field, verdicts in disagreements:
            doi = con.execute("SELECT canonical_doi FROM papers WHERE id=?", (paper_id,)).fetchone()[0]
            print(f"  {str(doi)[-7:]:8s} {field:26s} {verdicts}")
    else:
        print("\nno disagreements: every field unanimous on every paper")

    out = Path("artifacts/analysis/reliability/agreement.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "papers": len(complete),
                "repeats": exp["repeat_count"],
                "overall_unanimous": total_ok,
                "overall_total": total_n,
                "overall_rate": total_ok / total_n,
                "overall_ci": exact_ci(total_ok, total_n),
                "disagreements": [
                    {"paper_id": p, "field": f, "verdicts": v} for p, f, v in disagreements
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
