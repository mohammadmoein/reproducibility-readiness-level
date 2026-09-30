"""Draw the Stage 1 reliability subsample and emit the commands to run it.

The Stage 1 corpus run used one repetition, so no
variance estimate exists. Twenty papers are drawn from the 133-paper benchmark window,
stratified by whether the paper has a retrievable artifact and by the benchmark's open_data
code, with a fixed seed so the sample is reproducible.

Usage:
    uv run python scripts/reliability_sample.py
    uv run python scripts/reliability_sample.py --seed 20260918 --size 20
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from pathlib import Path

SEED = 20260918
SIZE = 20
SURVEY = Path("baseline/haim-lak/artifacts/qualtrics-survey-data.csv")
DATABASE = Path("data/lak.sqlite")
CLAIM_EXPERIMENT = "s1-claim-2021_2022-gpt56t-p1.1.0"


def load_survey(path: Path) -> dict[str, str]:
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header = rows[1]
    col = header.index("Open Data")
    out: dict[str, str] = {}
    for row in rows[3:]:
        if row and row[0].startswith("http"):
            doi = row[0].strip().lower().replace("https://doi.org/", "")
            out[doi] = row[col]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--survey", type=Path, default=SURVEY)
    args = parser.parse_args()

    survey = load_survey(args.survey)
    con = sqlite3.connect(args.database)
    con.row_factory = sqlite3.Row

    with_artifact = {
        r[0] for r in con.execute(
            "SELECT DISTINCT paper_id FROM artifact_snapshots"
        )
    }

    papers = []
    for r in con.execute(
        "SELECT id, canonical_doi, collection_year FROM papers "
        "WHERE collection_year IN (2021, 2022) ORDER BY id"
    ):
        doi = (r["canonical_doi"] or "").strip().lower()
        if doi not in survey:
            continue
        papers.append(
            {
                "id": r["id"],
                "doi": doi,
                "year": r["collection_year"],
                "open_data": survey[doi],
                "artifact": r["id"] in with_artifact,
            }
        )

    # Three strata: has a retrieved artifact; benchmark says no data at all; everything else.
    strata: dict[str, list[dict]] = {"artifact": [], "no_data": [], "other": []}
    for p in papers:
        if p["artifact"]:
            strata["artifact"].append(p)
        elif p["open_data"] == "N/A":
            strata["no_data"].append(p)
        else:
            strata["other"].append(p)

    print(f"population: {len(papers)} papers in the benchmark window")
    for name, group in strata.items():
        print(f"  stratum {name:9s} {len(group)}")

    rng = random.Random(args.seed)
    quota = {"artifact": 8, "no_data": 4, "other": args.size - 12}
    chosen: list[dict] = []
    for name, group in strata.items():
        take = min(quota[name], len(group))
        pool = sorted(group, key=lambda p: p["id"])
        rng.shuffle(pool)
        chosen.extend(pool[:take])

    # Top up from whatever remains if a stratum was short.
    if len(chosen) < args.size:
        picked = {p["id"] for p in chosen}
        rest = sorted((p for p in papers if p["id"] not in picked), key=lambda p: p["id"])
        rng.shuffle(rest)
        chosen.extend(rest[: args.size - len(chosen)])

    chosen.sort(key=lambda p: p["id"])
    print(f"\nseed={args.seed}  size={len(chosen)}")
    print(f"{'id':>5}  {'year':>4}  {'artifact':>8}  {'open_data':<26} doi")
    for p in chosen:
        print(f"{p['id']:>5}  {p['year']:>4}  {str(p['artifact']):>8}  {p['open_data']:<26} {p['doi']}")

    out = Path("artifacts/analysis/reliability")
    out.mkdir(parents=True, exist_ok=True)
    sample_path = out / f"sample-seed{args.seed}.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "doi", "year", "open_data", "artifact"])
        writer.writeheader()
        writer.writerows(chosen)
    print(f"\nwrote {sample_path}")

    print("\n--- commands to run ---")
    print(
        "uv run lakdb artifact create s1-claim-reliability-gpt56t-p1.1.0 "
        "--model gpt-5.6-terra --reasoning-effort medium --repeats 3 --freeze"
    )
    for p in chosen:
        print(
            "uv run lakdb artifact run --experiment s1-claim-reliability-gpt56t-p1.1.0 "
            f"--paper-id {p['id']} --no-progress"
        )
    print("\nuv run python scripts/reliability_report.py")


if __name__ == "__main__":
    main()
