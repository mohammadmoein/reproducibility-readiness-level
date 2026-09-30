"""Read-only, artifact-backed ARA prompt-cache and cost summaries."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pricing import PRICES, request_cost_usd


def ara_experiment_names(db: sqlite3.Connection) -> list[str]:
    try:
        rows = db.execute(
            "SELECT e.name FROM ara_experiments e "
            "LEFT JOIN ara_runs r ON r.experiment_id=e.id AND r.status='succeeded' "
            "GROUP BY e.id ORDER BY COUNT(r.id) DESC, e.created_at DESC, e.id DESC"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [str(row[0]) for row in rows]


def assessment_rows(db: sqlite3.Connection, experiment: str) -> tuple[str, list[dict[str, Any]]]:
    """Read saved raw request usage for successful runs of one ARA experiment."""
    row = db.execute("SELECT id,model FROM ara_experiments WHERE name=?", (experiment,)).fetchone()
    if row is None:
        raise ValueError(f"No such ARA experiment: {experiment}")
    model = str(row["model"] if isinstance(row, sqlite3.Row) else row[1])
    if model not in PRICES:
        raise ValueError(f"No recorded pricing for ARA model {model!r}; add it to pricing.py")
    experiment_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
    runs = db.execute(
        "SELECT r.id,r.artifact_path,r.finished_at,r.repetition_index,p.canonical_doi,p.title "
        "FROM ara_runs r JOIN papers p ON p.id=r.paper_id "
        "WHERE r.experiment_id=? AND r.status='succeeded' AND r.artifact_path IS NOT NULL "
        "ORDER BY r.finished_at,r.id",
        (experiment_id,),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for run in runs:
        path = Path(run["artifact_path"] if isinstance(run, sqlite3.Row) else run[1])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cached = written = fresh = output = 0
        cost = 0.0
        for entry in payload.get("responses", []):
            usage = (entry.get("response") or {}).get("usage") or {}
            if not usage:
                continue
            details = usage.get("input_tokens_details") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            cached_tokens = int(details.get("cached_tokens") or 0)
            write_tokens = int(details.get("cache_write_tokens") or 0)
            cached += cached_tokens
            written += write_tokens
            fresh += max(input_tokens - cached_tokens - write_tokens, 0)
            output += int(usage.get("output_tokens") or 0)
            cost += request_cost_usd(model, usage)
        if cached + written + fresh:
            result.append({
                "run_id": int(run["id"] if isinstance(run, sqlite3.Row) else run[0]),
                "finished": run["finished_at"] if isinstance(run, sqlite3.Row) else run[2],
                "repetition": run["repetition_index"] if isinstance(run, sqlite3.Row) else run[3],
                "doi": run["canonical_doi"] if isinstance(run, sqlite3.Row) else run[4],
                "title": run["title"] if isinstance(run, sqlite3.Row) else run[5],
                "artifact_path": str(path), "cached": cached, "written": written,
                "fresh": fresh, "output": output, "cost": cost,
            })
    return model, result


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_input = sum(int(row["cached"]) + int(row["written"]) + int(row["fresh"]) for row in rows)
    cached = sum(int(row["cached"]) for row in rows)
    written = sum(int(row["written"]) for row in rows)
    fresh = sum(int(row["fresh"]) for row in rows)
    output = sum(int(row["output"]) for row in rows)
    total_cost = sum(float(row["cost"]) for row in rows)
    return {
        "assessments": len(rows), "cached": cached, "written": written,
        "fresh": fresh, "output": output, "total_input": total_input,
        "cached_ratio": cached / total_input if total_input else None,
        "written_ratio": written / total_input if total_input else None,
        "total_cost": total_cost,
        "mean_cost": total_cost / len(rows) if rows else None,
        "cache_health": "OK" if total_input and cached / total_input > 0.25 else "CACHE NOT WORKING",
    }


def report_windows(rows: list[dict[str, Any]], recent: int = 25) -> list[tuple[str, dict[str, Any]]]:
    windows = [("All", summarise(rows))]
    if len(rows) > recent:
        windows.extend([(f"Last {recent}", summarise(rows[-recent:])), (f"First {len(rows) - recent}", summarise(rows[:-recent]))])
    return windows
