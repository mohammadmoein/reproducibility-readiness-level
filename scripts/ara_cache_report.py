"""Prompt-cache and cost report for an ARA experiment.

Usage: uv run python scripts/ara_cache_report.py EXPERIMENT [--recent 25]
The dashboard ARA Cost tab uses the same artifact-backed calculations.
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ara_usage import assessment_rows, report_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--database", default="data/lak.sqlite")
    parser.add_argument("--recent", type=int, default=25)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as db:
        db.row_factory = sqlite3.Row
        model, rows = assessment_rows(db, args.experiment)
    print(f"\n{args.experiment}  ({model})")
    for label, summary in report_windows(rows, args.recent):
        if not summary["assessments"]:
            print(f"  {label}: no data")
            continue
        print(
            f"  {label:<14} n={summary['assessments']:4d}  cached {summary['cached_ratio']:5.1%}  "
            f"write {summary['written_ratio']:5.1%}  ${summary['mean_cost']:.4f}/assessment  "
            f"total ${summary['total_cost']:7.2f}  [{summary['cache_health']}]"
        )
    print()


if __name__ == "__main__":
    main()
